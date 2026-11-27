"""Coverage, backfill, and orphan-rescue selection lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .exam_target import active_targets
from .lane_utils import (
    CardCommitRequest,
    CardPoolRequest,
    QuestionPoolRequest,
    ReasonRequest,
    card_item,
    card_pool,
    commit_card,
    question_pool,
    question_thread_fields,
    record_question_thread_selection,
    resolve_reason,
    thread_trace,
)
from .topics import expand_topic_set


@dataclass(frozen=True)
class CoverageCardRequest:
    inputs: dict[str, Any]
    plan: dict[str, Any]
    build: SessionBuildState
    concept: dict[str, Any]
    state: Any
    threshold: float


@dataclass(frozen=True)
class UncappedCardRequest:
    inputs: dict[str, Any]
    plan: dict[str, Any]
    build: SessionBuildState
    concept: Any
    threshold: float


@dataclass(frozen=True)
class RotationTopUpContext:
    inputs: dict[str, Any]
    build: SessionBuildState
    deficit: int
    familiarity: dict[str, Any]


def _coverage_card_available(card: dict[str, Any], topics: set[str], selected: set[str], deferred: set[str]) -> bool:
    if card["id"] in selected:
        return False
    if card["id"] in deferred:
        return False
    return any(topic in topics for topic in card["topics"])


def _coverage_counts(inputs: dict[str, Any], build: SessionBuildState) -> dict[str, int]:
    bulk = inputs["bulk"]
    counts: dict[str, int] = {}
    for concept in inputs["concepts"]:
        if concept["id"] in build.selected_concept_ids:
            continue
        topics = set(expand_topic_set(concept["topics"]))
        count = sum(
            1 for card in bulk["unseenCards"]
            if _coverage_card_available(card, topics, build.selected_card_ids, bulk["passSetDeferredCardIds"])
        )
        if count > 0:
            counts[concept["id"]] = count
    return counts


def _coverage_rows(inputs: dict[str, Any], plan: dict[str, Any], counts: dict[str, int]) -> list[tuple[dict[str, Any], Any, int]]:
    rows = [(inputs["conceptMap"].get(cid), plan["statesById"].get(cid), count) for cid, count in counts.items()]
    rows = [row for row in rows if row[0] is not None]
    rows.sort(key=lambda row: (-row[2], -(1 if row[0].get("examWeight") is None else row[0]["examWeight"])))
    return rows


def _coverage_reason(request: CoverageCardRequest) -> str:
    recall = request.state.recallOnExamDay if request.state else 0
    return resolve_reason(ReasonRequest(request.inputs, request.plan, request.concept["id"], recall, "reinforcement"))


def _try_coverage_card(request: CoverageCardRequest) -> None:
    recall = request.state.currentRecall if request.state else None
    candidates = card_pool(CardPoolRequest(request.inputs, request.build, request.concept["id"], request.concept, recall))
    pristine = [card for card in candidates if card["id"] in request.plan["unseenCardIds"]]
    if not pristine:
        return
    picked = request.build.pick_card_candidate(pristine, similarity_threshold=request.threshold, max_per_cluster=5)
    if not picked:
        return
    item = card_item(picked, challengePolicyApplied=True, conceptId=request.concept["id"], conceptName=request.concept["name"],
                     priority=request.state.priority if request.state else 0, interventionReason=_coverage_reason(request))
    commit_card(CardCommitRequest(request.build, picked, item, recall))


def fill_coverage_lane(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    target = plan["targetCardCount"]
    if plan["coverageReservation"] <= 0:
        return
    if build.selected_card_count >= target:
        return
    rows = _coverage_rows(inputs, plan, _coverage_counts(inputs, build))
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept, state, _ in rows:
        if build.selected_card_count >= target:
            break
        _try_coverage_card(CoverageCardRequest(inputs, plan, build, concept, state, threshold))


def _unselected_concept_states(plan: dict[str, Any], build: SessionBuildState) -> list[Any]:
    return [state for state in plan["conceptStates"] if state.conceptId not in build.selected_concept_ids]


def _try_backfill_card(inputs: dict[str, Any], build: SessionBuildState,
                       concept: Any, threshold: float) -> None:
    data = inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    candidates = card_pool(CardPoolRequest(inputs, build, concept.conceptId, data, concept.currentRecall, False))
    if not candidates:
        return
    picked = build.pick_card_candidate(candidates, similarity_threshold=threshold, max_per_cluster=5)
    if not picked:
        return
    item = card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                     conceptName=concept.conceptName, priority=concept.priority, interventionReason="reinforcement")
    commit_card(CardCommitRequest(build, picked, item, concept.currentRecall))


def _backfill_any_concept_cards(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept in _unselected_concept_states(plan, build):
        if len(build.selected_items) >= inputs["size"]:
            break
        if build.selected_card_count >= plan["targetCardCount"]:
            break
        _try_backfill_card(inputs, build, concept, threshold)


def _try_backfill_question(inputs: dict[str, Any], build: SessionBuildState, concept: Any) -> None:
    data = inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    questions = question_pool(QuestionPoolRequest(inputs, build, concept.conceptId, data, "medium"))
    if not questions:
        return
    question = questions[0]
    added = build.add_item({"type": "question", "id": question["id"], "conceptId": concept.conceptId,
                            "conceptName": concept.conceptName, "priority": concept.priority,
                            **question_thread_fields(question, "reinforcement"),
                            "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType")})
    record_question_thread_selection(build, question, added)


def _backfill_any_concept_questions(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    for concept in _unselected_concept_states(plan, build):
        if len(build.selected_items) >= inputs["size"]:
            break
        _try_backfill_question(inputs, build, concept)


def backfill_from_any_concept(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    if len(build.selected_items) >= inputs["size"]:
        return
    _backfill_any_concept_cards(inputs, plan, build)
    if len(build.selected_items) >= inputs["size"]:
        return
    _backfill_any_concept_questions(inputs, plan, build)


def _stale_strong_concepts(plan: dict[str, Any], build: SessionBuildState) -> list[Any]:
    stale = [state for state in plan["conceptStates"] if state.recallOnExamDay >= DEFAULTS["target_recall"]
             and state.daysSinceProbe > DEFAULTS["days_since_probe_threshold"] and state.conceptId not in build.selected_concept_ids]
    stale.sort(key=lambda state: -state.daysSinceProbe)
    return stale


def _question_item(question: dict[str, Any], concept: Any, reason: str) -> dict[str, Any]:
    return {"type": "question", "id": question["id"], "conceptId": concept.conceptId,
            "conceptName": concept.conceptName, "priority": concept.priority,
            **question_thread_fields(question, reason),
            "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType")}


def _probe_stale_concept(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState, concept: Any) -> None:
    data = inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    reason = resolve_reason(ReasonRequest(inputs, plan, concept.conceptId, concept.recallOnExamDay, "needs_retest"))
    questions = question_pool(QuestionPoolRequest(
        inputs, build, concept.conceptId, data, "hard", None, reason != "failure_escalation",
    ))
    if not questions:
        return
    added = build.add_item(_question_item(questions[0], concept, reason))
    record_question_thread_selection(build, questions[0], added)


def probe_stale_strong_concepts(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    for concept in _stale_strong_concepts(plan, build):
        if len(build.selected_items) >= inputs["size"]:
            break
        _probe_stale_concept(inputs, plan, build, concept)


def _topup_deficit(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> int:
    item_deficit = inputs["size"] - len(build.selected_items)
    question_deficit = max(0, plan["targetQuestionCount"] - build.selected_question_count)
    return min(item_deficit, question_deficit)


def _topup_row_available(row: dict[str, Any], already_picked: set[str], variant_groups: set[str], key: str | None) -> bool:
    if row["id"] in already_picked:
        return False
    return not key or key not in variant_groups


def _topup_rows(context: RotationTopUpContext) -> list[dict[str, Any]]:
    from .build_state import question_suppression_key

    already_picked = set(context.build.selected_question_ids)
    already_picked.update(item["id"] for item in context.build.selected_items if item["type"] == "question")
    rows = []
    for row in context.inputs.get("rotationTopUp") or []:
        key = question_suppression_key(row)
        if _topup_row_available(row, already_picked, context.build.selected_question_variant_groups, key):
            rows.append(dict(row))
        if len(rows) == context.deficit:
            break
    return rows


def _mark_topup_rows(context: RotationTopUpContext, rows: list[dict[str, Any]]) -> None:
    from .build_state import question_suppression_key
    from .policies import freshness_tier

    mastered = sum(1 for row in rows if freshness_tier(context.familiarity.get(row["id"])) == 2)
    context.build.mastered_reentry_counter["masteredServed"] += mastered
    for row in rows:
        context.build.selected_question_ids.add(row["id"])
        key = question_suppression_key(row)
        if key:
            context.build.selected_question_variant_groups.add(key)


def _topup_item(row: dict[str, Any], rotation: str) -> dict[str, Any]:
    return {"type": "question", "id": row["id"], "rotation": rotation, "conceptId": f"rotation:{row['id']}",
            "conceptName": "Rotation fill", "priority": 0.4, "interventionReason": "needs_retest",
            **thread_trace(), "variantGroupId": row.get("variantGroupId"), "variantType": row.get("variantType")}


def top_up_from_rotation_bank(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    """Rotation top-up. ``inputs['rotationTopUp']`` is the storage adapter: the
    ordered rows ``getQuestionsForRotation`` returns for this deficit."""
    if len(build.selected_items) >= inputs["size"]:
        return
    deficit = _topup_deficit(inputs, plan, build)
    if deficit <= 0:
        return
    context = RotationTopUpContext(inputs, build, deficit, inputs["bulk"]["questionFamiliarity"])
    rows = _topup_rows(context)
    _mark_topup_rows(context, rows)
    for row in rows:
        build.add_item(_topup_item(row, inputs["rotation"]))


def _uncapped_card_item(picked: dict[str, Any], concept: Any, reason: str) -> dict[str, Any]:
    return card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                     conceptName=concept.conceptName, priority=concept.priority, interventionReason=reason)


def _try_uncapped_card(request: UncappedCardRequest) -> None:
    concept = request.concept
    data = request.inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    candidates = card_pool(CardPoolRequest(request.inputs, request.build, concept.conceptId, data, concept.currentRecall, False))
    if not candidates:
        return
    picked = request.build.pick_card_candidate(candidates, similarity_threshold=request.threshold, max_per_cluster=5)
    if not picked:
        return
    reason = resolve_reason(ReasonRequest(request.inputs, request.plan, concept.conceptId, concept.recallOnExamDay, "reinforcement"))
    commit_card(CardCommitRequest(request.build, picked, _uncapped_card_item(picked, concept, reason), concept.currentRecall))


def backfill_cards_uncapped(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept in _unselected_concept_states(plan, build):
        if len(build.selected_items) >= inputs["size"]:
            break
        _try_uncapped_card(UncappedCardRequest(inputs, plan, build, concept, threshold))


def _orphan_attribution(inputs: dict[str, Any], topics: list[str]) -> tuple[str, str]:
    from .policies import has_specific_clinical_topic_overlap

    for concept in inputs["concepts"]:
        if has_specific_clinical_topic_overlap(topics, concept["topics"]):
            return concept["id"], concept["name"]
    return "_unattached", "Unattached"


def _add_orphan_card(inputs: dict[str, Any], build: SessionBuildState, card: dict[str, Any], reason: str) -> None:
    concept_id, name = _orphan_attribution(inputs, card["topics"])
    build.add_item(card_item(card, conceptId=concept_id, conceptName=name, priority=0.2, interventionReason=reason))


def _orphan_nudge(inputs: dict[str, Any], active: dict[str, Any], rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    from .ranking import apply_bounded_exam_target_nudge

    if not inputs.get("applyExamTarget"):
        return rows
    return apply_bounded_exam_target_nudge(rows, lambda row: active[kind].get(row["id"]), active["rankMove"])


def _variant_blocked(build: SessionBuildState, card: dict[str, Any]) -> bool:
    return bool(card.get("variantGroupId") and card["variantGroupId"] in build.selected_card_variant_groups)


def _rescue_seen_cards(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    if len(build.selected_items) >= inputs["size"]:
        return
    rows = [card for card in inputs["bulk"]["seenCards"] if card["id"] not in build.selected_card_ids]
    for card in _orphan_nudge(inputs, active, rows, "card"):
        if len(build.selected_items) >= inputs["size"]:
            break
        if not _variant_blocked(build, card):
            _add_orphan_card(inputs, build, card, "needs_retest")


def _pass_set_fact_key(pass_set: dict[str, Any], card: dict[str, Any]) -> str:
    fact_key = (pass_set.get("factKeyByCardId") or {}).get(card["id"])
    if fact_key:
        return fact_key
    return card.get("variantGroupId") or card["id"]


def _pass_set_fact_keys(pass_set: dict[str, Any], fact_key_set: set[str] | None) -> set[str]:
    if fact_key_set is not None:
        return fact_key_set
    return set(pass_set.get("factKeys") or [])


def _pass_set_rank(bulk: dict[str, Any], card: dict[str, Any], fact_key_set: set[str] | None = None) -> int:
    import sys

    wrapper = bulk.get("passSet") or {}
    pass_set = wrapper.get("passSet") if isinstance(wrapper, dict) else None
    if not pass_set:
        return 0
    if bulk.get("passSetIntakeDecision") not in ("pass-set-first", "supply-exhausted", "unlocked"):
        return 0
    key = _pass_set_fact_key(pass_set, card)
    fact_keys = _pass_set_fact_keys(pass_set, fact_key_set)
    if key not in fact_keys:
        return sys.maxsize
    return (pass_set.get("rankByFactKey") or {}).get(key) or 0


def _rescue_unseen_rows(inputs: dict[str, Any], build: SessionBuildState) -> list[dict[str, Any]]:
    bulk = inputs["bulk"]
    rows = [card for card in bulk["unseenCards"] if card["id"] not in build.selected_card_ids
            and card["id"] not in bulk["passSetDeferredCardIds"]]
    pass_set = ((bulk.get("passSet") or {}).get("passSet") or {})
    fact_key_set = set(pass_set.get("factKeys") or [])
    rows.sort(key=lambda card: (_pass_set_rank(bulk, card, fact_key_set), -(1 if card.get("importance") is None else card["importance"])))
    return rows


def _rescue_unseen_cards(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    for card in _orphan_nudge(inputs, active, _rescue_unseen_rows(inputs, build), "card"):
        if len(build.selected_items) >= inputs["size"]:
            break
        if not _variant_blocked(build, card):
            _add_orphan_card(inputs, build, card, "reinforcement")


def _suppression_group_blocked(build: SessionBuildState, suppression: str | None) -> bool:
    if not suppression:
        return False
    return suppression in build.selected_question_variant_groups


def _question_available(build: SessionBuildState, question: dict[str, Any], familiarity: dict[str, Any]) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    key = question_suppression_key(question)
    if _suppression_group_blocked(build, key):
        return False
    return take_with_reentry_cap(question["id"], familiarity, build.mastered_reentry_counter)


def _rescue_question_item(inputs: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    concept_id, name = _orphan_attribution(inputs, question["topics"])
    return {"type": "question", "id": question["id"], "conceptId": concept_id, "conceptName": name, "priority": 0.2,
            "interventionReason": "reinforcement", **thread_trace(),
            "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType")}


def _rescue_rotation_questions(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    from .policies import prioritize_least_recently_served_contrast_siblings

    bulk = inputs["bulk"]
    familiarity = bulk["questionFamiliarity"]
    questions = prioritize_least_recently_served_contrast_siblings(
        [question for question in bulk["rotationQuestions"] if question["id"] not in build.selected_question_ids],
        lambda question: question["id"], familiarity,
    )
    for question in _orphan_nudge(inputs, active, questions, "question"):
        if len(build.selected_items) >= inputs["size"]:
            break
        if _question_available(build, question, familiarity):
            build.add_item(_rescue_question_item(inputs, question))


def rescue_orphans(inputs: dict[str, Any], _plan: dict[str, Any], build: SessionBuildState) -> None:
    active = active_targets(inputs)
    _rescue_seen_cards(inputs, build, active)
    if len(build.selected_items) >= inputs["size"]:
        return
    if inputs.get("effectiveMaxNewCards") == 0:
        return
    _rescue_unseen_cards(inputs, build, active)
    _rescue_rotation_questions(inputs, build, active)
