"""Coverage, backfill, and orphan-rescue selection lanes."""

from __future__ import annotations

from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .exam_target import active_targets
from .lane_utils import (
    card_item,
    card_pool,
    commit_card,
    question_pool,
    resolve_reason,
    thread_trace,
)
from .topics import expand_topic_set


def _coverage_counts(inputs: dict[str, Any], build: SessionBuildState) -> dict[str, int]:
    bulk = inputs["bulk"]
    counts: dict[str, int] = {}
    for c in inputs["concepts"]:
        if c["id"] in build.selected_concept_ids:
            continue
        topics = set(expand_topic_set(c["topics"]))
        n = sum(
            1 for card in bulk["unseenCards"]
            if card["id"] not in build.selected_card_ids and card["id"] not in bulk["passSetDeferredCardIds"]
            and any(t in topics for t in card["topics"])
        )
        if n > 0:
            counts[c["id"]] = n
    return counts


def _coverage_rows(inputs: dict[str, Any], plan: dict[str, Any], counts: dict[str, int]) -> list[tuple[dict[str, Any], Any, int]]:
    rows = [(inputs["conceptMap"].get(cid), plan["statesById"].get(cid), n) for cid, n in counts.items()]
    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda r: (-r[2], -(1 if r[0].get("examWeight") is None else r[0]["examWeight"])))
    return rows


def _try_coverage_card(
    inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState,
    concept: dict[str, Any], state: Any, threshold: float
) -> None:
    recall = state.currentRecall if state else None
    candidates = card_pool(inputs, build, concept["id"], concept, recall)
    pristine = [c for c in candidates if c["id"] in plan["unseenCardIds"]]
    if not pristine:
        return
    picked = build.pick_card_candidate(pristine, similarity_threshold=threshold, max_per_cluster=5)
    if not picked:
        return
    reason = resolve_reason(inputs, plan, concept["id"], state.recallOnExamDay if state else 0, "reinforcement")
    item = card_item(picked, challengePolicyApplied=True, conceptId=concept["id"], conceptName=concept["name"],
                     priority=state.priority if state else 0, interventionReason=reason)
    commit_card(build, picked, item, recall)


def fill_coverage_lane(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    target = plan["targetCardCount"]
    if plan["coverageReservation"] <= 0 or build.selected_card_count >= target:
        return
    rows = _coverage_rows(inputs, plan, _coverage_counts(inputs, build))
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept, state, _ in rows:
        if build.selected_card_count >= target:
            break
        _try_coverage_card(inputs, plan, build, concept, state, threshold)


def _unselected_concept_states(plan: dict[str, Any], build: SessionBuildState) -> list[Any]:
    return [state for state in plan["conceptStates"] if state.conceptId not in build.selected_concept_ids]


def _try_backfill_card(inputs: dict[str, Any], build: SessionBuildState,
                       concept: Any, threshold: float) -> None:
    data = inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    candidates = card_pool(inputs, build, concept.conceptId, data, concept.currentRecall, apply_review_challenge=False)
    if not candidates:
        return
    picked = build.pick_card_candidate(candidates, similarity_threshold=threshold, max_per_cluster=5)
    if not picked:
        return
    item = card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                     conceptName=concept.conceptName, priority=concept.priority, interventionReason="reinforcement")
    commit_card(build, picked, item, concept.currentRecall)


def _backfill_any_concept_cards(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept in _unselected_concept_states(plan, build):
        if len(build.selected_items) >= size or build.selected_card_count >= plan["targetCardCount"]:
            break
        _try_backfill_card(inputs, build, concept, threshold)


def _try_backfill_question(inputs: dict[str, Any], build: SessionBuildState, concept: Any) -> None:
    data = inputs["conceptMap"].get(concept.conceptId)
    if not data:
        return
    questions = question_pool(inputs, build, concept.conceptId, data, "medium")
    if not questions:
        return
    question = questions[0]
    build.add_item({"type": "question", "id": question["id"], "conceptId": concept.conceptId,
                    "conceptName": concept.conceptName, "priority": concept.priority,
                    "interventionReason": "reinforcement", **thread_trace(),
                    "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType")})


def _backfill_any_concept_questions(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    for concept in _unselected_concept_states(plan, build):
        if len(build.selected_items) >= size:
            break
        _try_backfill_question(inputs, build, concept)


def backfill_from_any_concept(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    if len(build.selected_items) >= inputs["size"]:
        return
    _backfill_any_concept_cards(inputs, plan, build)
    if len(build.selected_items) >= inputs["size"]:
        return
    _backfill_any_concept_questions(inputs, plan, build)


def probe_stale_strong_concepts(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    if len(build.selected_items) >= size:
        return
    stale = [s for s in plan["conceptStates"] if s.recallOnExamDay >= DEFAULTS["target_recall"]
             and s.daysSinceProbe > DEFAULTS["days_since_probe_threshold"] and s.conceptId not in build.selected_concept_ids]
    stale.sort(key=lambda s: -s.daysSinceProbe)
    for concept in stale:
        if len(build.selected_items) >= size:
            break
        data = inputs["conceptMap"].get(concept.conceptId)
        if not data:
            continue
        reason = resolve_reason(inputs, plan, concept.conceptId, concept.recallOnExamDay, "needs_retest")
        questions = question_pool(inputs, build, concept.conceptId, data, "hard")
        if questions:
            q = questions[0]
            build.add_item({"type": "question", "id": q["id"], "conceptId": concept.conceptId,
                            "conceptName": concept.conceptName, "priority": concept.priority,
                            "interventionReason": reason, **thread_trace(),
                            "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")})


def top_up_from_rotation_bank(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    """Rotation top-up. ``inputs['rotationTopUp']`` is the storage adapter: the
    ordered rows ``getQuestionsForRotation`` returns for this deficit."""
    size = inputs["size"]
    if len(build.selected_items) >= size:
        return
    deficit = min(size - len(build.selected_items), max(0, plan["targetQuestionCount"] - build.selected_question_count))
    if deficit <= 0:
        return
    from .build_state import question_suppression_key
    from .policies import freshness_tier

    familiarity = inputs["bulk"]["questionFamiliarity"]
    already_picked = set(build.selected_question_ids)
    already_picked.update(i["id"] for i in build.selected_items if i["type"] == "question")
    rows = []
    for row in inputs.get("rotationTopUp") or []:
        key = question_suppression_key(row)
        if row["id"] in already_picked or (key and key in build.selected_question_variant_groups):
            continue
        rows.append(dict(row))
        if len(rows) == deficit:
            break
    build.mastered_reentry_counter["masteredServed"] += sum(1 for r in rows if freshness_tier(familiarity.get(r["id"])) == 2)
    for r in rows:
        build.selected_question_ids.add(r["id"])
        key = question_suppression_key(r)
        if key:
            build.selected_question_variant_groups.add(key)
    for q in rows:
        build.add_item({"type": "question", "id": q["id"], "rotation": inputs["rotation"], "conceptId": f"rotation:{q['id']}",
                        "conceptName": "Rotation fill", "priority": 0.4, "interventionReason": "needs_retest",
                        **thread_trace(), "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")})


def backfill_cards_uncapped(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    if len(build.selected_items) >= size:
        return
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept in [s for s in plan["conceptStates"] if s.conceptId not in build.selected_concept_ids]:
        if len(build.selected_items) >= size:
            break
        data = inputs["conceptMap"].get(concept.conceptId)
        if not data:
            continue
        candidates = card_pool(inputs, build, concept.conceptId, data, concept.currentRecall, apply_review_challenge=False)
        if not candidates:
            continue
        picked = build.pick_card_candidate(candidates, similarity_threshold=threshold, max_per_cluster=5)
        if not picked:
            continue
        reason = resolve_reason(inputs, plan, concept.conceptId, concept.recallOnExamDay, "reinforcement")
        item = card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                         conceptName=concept.conceptName, priority=concept.priority, interventionReason=reason)
        commit_card(build, picked, item, concept.currentRecall)


def _orphan_attribution(inputs: dict[str, Any], topics: list[str]) -> tuple[str, str]:
    from .policies import has_specific_clinical_topic_overlap

    for c in inputs["concepts"]:
        if has_specific_clinical_topic_overlap(topics, c["topics"]):
            return c["id"], c["name"]
    return "_unattached", "Unattached"


def _add_orphan_card(inputs: dict[str, Any], build: SessionBuildState, card: dict[str, Any], reason: str) -> None:
    cid, name = _orphan_attribution(inputs, card["topics"])
    build.add_item(card_item(card, conceptId=cid, conceptName=name, priority=0.2, interventionReason=reason))


def _orphan_nudge(inputs: dict[str, Any], active: dict[str, Any], rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    from .ranking import apply_bounded_exam_target_nudge

    if not inputs.get("applyExamTarget"):
        return rows
    return apply_bounded_exam_target_nudge(rows, lambda r: active[kind].get(r["id"]), active["rankMove"])


def _rescue_seen_cards(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    size, bulk = inputs["size"], inputs["bulk"]
    if len(build.selected_items) >= size:
        return
    rows = [c for c in bulk["seenCards"] if c["id"] not in build.selected_card_ids]
    for card in _orphan_nudge(inputs, active, rows, "card"):
        if len(build.selected_items) >= size:
            break
        if card.get("variantGroupId") and card["variantGroupId"] in build.selected_card_variant_groups:
            continue
        _add_orphan_card(inputs, build, card, "needs_retest")


def _pass_set_rank(bulk: dict[str, Any], card: dict[str, Any], fact_key_set: set[str] | None = None) -> int:
    import sys

    wrapper = bulk.get("passSet") or {}
    pass_set = wrapper.get("passSet") if isinstance(wrapper, dict) else None
    if not pass_set or bulk.get("passSetIntakeDecision") not in ("pass-set-first", "supply-exhausted", "unlocked"):
        return 0
    key = (pass_set.get("factKeyByCardId") or {}).get(card["id"]) or card.get("variantGroupId") or card["id"]
    fact_keys = fact_key_set if fact_key_set is not None else set(pass_set.get("factKeys") or [])
    if key not in fact_keys:
        return sys.maxsize
    return (pass_set.get("rankByFactKey") or {}).get(key) or 0


def _rescue_unseen_cards(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    size, bulk = inputs["size"], inputs["bulk"]
    unseen = [c for c in bulk["unseenCards"] if c["id"] not in build.selected_card_ids
              and c["id"] not in bulk["passSetDeferredCardIds"]]
    pass_set = ((bulk.get("passSet") or {}).get("passSet") or {})
    fact_key_set = set(pass_set.get("factKeys") or [])
    unseen.sort(key=lambda c: (_pass_set_rank(bulk, c, fact_key_set), -(1 if c.get("importance") is None else c["importance"])))
    for card in _orphan_nudge(inputs, active, unseen, "card"):
        if len(build.selected_items) >= size:
            break
        if card.get("variantGroupId") and card["variantGroupId"] in build.selected_card_variant_groups:
            continue
        _add_orphan_card(inputs, build, card, "reinforcement")


def _rescue_rotation_questions(inputs: dict[str, Any], build: SessionBuildState, active: dict[str, Any]) -> None:
    from .build_state import question_suppression_key
    from .policies import prioritize_least_recently_served_contrast_siblings, take_with_reentry_cap

    size, bulk = inputs["size"], inputs["bulk"]
    familiarity = bulk["questionFamiliarity"]
    questions = prioritize_least_recently_served_contrast_siblings(
        [q for q in bulk["rotationQuestions"] if q["id"] not in build.selected_question_ids], lambda q: q["id"], familiarity
    )
    for q in _orphan_nudge(inputs, active, questions, "question"):
        if len(build.selected_items) >= size:
            break
        key = question_suppression_key(q)
        if key and key in build.selected_question_variant_groups:
            continue
        if not take_with_reentry_cap(q["id"], familiarity, build.mastered_reentry_counter):
            continue
        cid, name = _orphan_attribution(inputs, q["topics"])
        build.add_item({"type": "question", "id": q["id"], "conceptId": cid, "conceptName": name, "priority": 0.2,
                        "interventionReason": "reinforcement", **thread_trace(),
                        "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")})


def rescue_orphans(inputs: dict[str, Any], _plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    active = active_targets(inputs)
    _rescue_seen_cards(inputs, build, active)
    if len(build.selected_items) >= size or inputs.get("effectiveMaxNewCards") == 0:
        return
    _rescue_unseen_cards(inputs, build, active)
    _rescue_rotation_questions(inputs, build, active)
