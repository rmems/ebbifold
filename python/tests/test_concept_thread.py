"""Concept-thread follow-up policy parity units."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ebbifold.scheduler.concept_thread import (
    CONCEPT_THREAD_NATURAL_BREAK_MS,
    infer_clinical_facet,
    normalize_clinical_thread_text,
    prepare_concept_thread_matcher,
)
from ebbifold.scheduler.pipeline import run_session
from ebbifold.scheduler.ranking import QuestionPoolQuery, QuestionRankingOptions, get_questions_from_bulk

SESSIONS = json.loads((Path(__file__).parent / "fixtures" / "unified-scheduler-session.cases.json").read_text(encoding="utf-8"))["sessions"]


def test_concept_thread_normalization_and_facet_inference() -> None:
    assert normalize_clinical_thread_text("  Acute—Renal & Tubular! ") == "acute renal and tubular"
    assert infer_clinical_facet({"text": "What organism caused this?", "topics": []}) == "cause_mechanism"
    assert infer_clinical_facet({"text": "Most likely diagnosis?", "topics": []}) == "diagnosis_investigation"
    assert infer_clinical_facet({"text": "Next best step in management?", "topics": []}) == "management"


def _anchor(now_ms: int) -> dict:
    return {
        "id": "anchor-card", "itemType": "card", "text": "Renal tubular acidosis mechanism",
        "topics": ["Renal Tubular Acidosis"], "conceptIds": ["c-renal"], "variantGroupId": None,
        "anchorEventId": "event-anchor", "answeredAtMs": now_ms - CONCEPT_THREAD_NATURAL_BREAK_MS,
        "interveningExposures": 0,
    }


def _question_query(concept: dict, bulk: dict, **overrides) -> QuestionPoolQuery:
    return QuestionPoolQuery("c-renal", concept, 1, bulk, QuestionRankingOptions(**overrides))


def test_concept_thread_matcher_requires_different_facet_and_shared_specific_topic() -> None:
    now = 1_800_000_000_000
    matcher = prepare_concept_thread_matcher([_anchor(now)], now)
    match = matcher.find_match({
        "id": "candidate-question", "itemType": "question", "text": "Most likely diagnosis of renal tubular acidosis?",
        "topics": ["Renal Tubular Acidosis"], "questionType": "diagnosis", "format": None,
        "conceptIds": ["c-renal"], "variantGroupId": None,
    })
    assert match is not None
    assert match["anchorEventId"] == "event-anchor"
    assert match["anchorFacet"] == "cause_mechanism"
    assert match["targetFacet"] == "diagnosis_investigation"
    assert match["sharedTopic"] == "renal tubular acidosis"

    assert matcher.find_match({
        "id": "same-facet", "itemType": "question", "text": "Another renal tubular acidosis mechanism",
        "topics": ["Renal Tubular Acidosis"], "questionType": "mechanism", "conceptIds": ["c-renal"],
    }) is None


def test_concept_thread_reserves_inside_existing_freshness_tier() -> None:
    now = 1_800_000_000_000
    bulk = {
        "rotation": "rot-main",
        "questionConceptLinks": [{"questionId": "plain", "conceptId": "c-renal", "isPrimary": True}, {"questionId": "followup", "conceptId": "c-renal", "isPrimary": True}],
        "questionMap": {},
        "rotationQuestions": [
            {"id": "plain", "stem": "Most likely diagnosis?", "questionType": "diagnosis",
             "difficulty": "medium", "variantGroupId": None, "variantType": None, "format": None,
             "facilityIndex": None, "totalAttempts": None, "source": "bank", "topics": ["Unrelated"]},
            {"id": "followup", "stem": "Most likely diagnosis of renal tubular acidosis in a child?", "questionType": "diagnosis",
             "difficulty": "medium", "variantGroupId": None, "variantType": None, "format": None,
             "facilityIndex": None, "totalAttempts": None, "source": "bank", "topics": ["Renal Tubular Acidosis"]},
        ],
        "questionScores": {}, "questionExamRelevance": {}, "questionFamiliarity": {},
    }
    bulk["questionMap"] = {q["id"]: q for q in bulk["rotationQuestions"]}
    bulk["conceptThreadMatcher"] = prepare_concept_thread_matcher([_anchor(now)], now)
    concept = {"topics": ["Renal Tubular Acidosis"]}
    query = _question_query(concept, bulk)
    picked = get_questions_from_bulk(query)
    assert picked[0]["id"] == "followup"
    assert picked[0]["conceptThreadMatch"]["anchorEventId"] == "event-anchor"
    assert [row["id"] for row in get_questions_from_bulk(query)] == [row["id"] for row in picked]
    assert all("_conceptThreadMatch" not in row for row in bulk["rotationQuestions"])


def test_concept_thread_match_blocks_review_challenge_difficulty_shift() -> None:
    now = 1_800_000_000_000
    rows = [
        {"id": "followup", "stem": "Most likely diagnosis of renal tubular acidosis in a child?", "questionType": "diagnosis",
         "difficulty": "medium", "variantGroupId": None, "variantType": None, "format": None,
         "facilityIndex": None, "totalAttempts": None, "source": "bank", "topics": ["Renal Tubular Acidosis"]},
        {"id": "shifted-hard", "stem": "Hard renal tubular acidosis question", "questionType": "diagnosis",
         "difficulty": "hard", "variantGroupId": None, "variantType": None, "format": None,
         "facilityIndex": None, "totalAttempts": None, "source": "bank", "topics": ["Renal Tubular Acidosis"]},
    ]
    bulk = {
        "rotation": "rot-main",
        "questionConceptLinks": [{"questionId": row["id"], "conceptId": "c-renal", "isPrimary": True} for row in rows],
        "questionMap": {row["id"]: row for row in rows}, "rotationQuestions": rows,
        "questionScores": {}, "questionExamRelevance": {}, "questionFamiliarity": {},
    }
    concept = {"topics": ["Renal Tubular Acidosis"]}
    matched_bulk = {**bulk, "conceptThreadMatcher": prepare_concept_thread_matcher([_anchor(now)], now)}
    matched = get_questions_from_bulk(_question_query(concept, matched_bulk, review_challenge=1, challenge_slot_index=2))
    assert [row["id"] for row in matched] == ["followup"]

    unmatched = get_questions_from_bulk(_question_query(concept, bulk, review_challenge=1, challenge_slot_index=2))
    assert [row["id"] for row in unmatched] == ["shifted-hard"]


def test_session_marks_one_concept_followup_and_reserves_only_one() -> None:
    base = copy.deepcopy(next(c for c in SESSIONS if c["name"] == "full_session_weak_new_pristine")["snapshot"])
    anchor = _anchor(base["nowMs"])
    anchor.update({"text": "Cardiac output mechanism", "topics": ["Cardiac Output"], "conceptIds": ["c-cardiac"]})
    base["conceptThreadAnchors"] = [anchor]
    result = run_session(base)["result"]
    matches = [item for item in result["items"] if item.get("conceptThreadPolicyApplied")]
    assert len(matches) == 1
    assert matches[0]["interventionReason"] == "concept_followup"
    assert matches[0]["conceptThreadAnchorEventId"] == "event-anchor"



def test_concept_thread_cadence_and_variant_suppression() -> None:
    now = 1_800_000_000_000
    young = _anchor(now)
    young["answeredAtMs"] = now - 60_000
    assert prepare_concept_thread_matcher([young], now).anchor_count == 0

    anchor = _anchor(now)
    anchor["variantGroupId"] = "variant-1"
    matcher = prepare_concept_thread_matcher([anchor], now)
    assert matcher.find_match({
        "id": "candidate-question", "itemType": "question", "text": "Most likely diagnosis of renal tubular acidosis?",
        "topics": ["Renal Tubular Acidosis"], "questionType": "diagnosis", "conceptIds": ["c-renal"],
        "variantGroupId": "variant-1",
    }) is None
