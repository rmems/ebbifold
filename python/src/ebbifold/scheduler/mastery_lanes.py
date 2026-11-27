"""Exam-target mastery reservation lanes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .exam_target import active_targets, allocate_mastery_seats
from .lane_utils import (
    CardCommitRequest,
    _core_card_resolver,
    _core_question_resolver,
    card_item,
    commit_card,
    thread_trace,
)


@dataclass(frozen=True)
class CoreReservationContext:
    inputs: dict[str, Any]
    build: SessionBuildState
    cards: dict[str, dict[str, Any]]
    active: dict[str, float]
    resolve: Any


@dataclass(frozen=True)
class SurplusReservationContext:
    inputs: dict[str, Any]
    build: SessionBuildState
    bulk: dict[str, Any]
    cards: dict[str, dict[str, Any]]
    resolve_card: Any
    resolve_question: Any


@dataclass(frozen=True)
class UnearnedStageContext:
    assignments: dict[str, Any]
    session_rotation: str
    eligible: set[str]
    quota_satisfied: bool


def _concept_value(concept: Any, attribute: str, fallback: Any) -> Any:
    if concept:
        return getattr(concept, attribute)
    return fallback


def _is_weak_recall(concept: Any) -> bool:
    if not concept:
        return False
    return concept.recallOnExamDay < DEFAULTS["recall_threshold"]


def _intervention_for_weak(weak: bool, weak_reason: str) -> str:
    if weak:
        return weak_reason
    return "reinforcement"


def _mastery_candidates(evidence: dict[str, Any], stages: set[str]) -> list[dict[str, Any]]:
    rows = []
    for unit in evidence.get("units") or []:
        if unit.get("stage") not in stages:
            continue
        for key in (unit.get("safeMetadata") or {}).get("candidateItemKeys") or []:
            rows.append({"itemKey": key, "unitId": unit.get("unitId"), "stage": unit.get("stage")})
    return rows


def _core_unit_candidates(unit: dict[str, Any], cards: dict[str, dict[str, Any]], active: dict[str, float]) -> list[dict[str, Any]]:
    candidates = []
    for key in (unit.get("safeMetadata") or {}).get("candidateItemKeys") or []:
        if not key.startswith("card:"):
            continue
        card = cards.get(key[5:])
        if card is not None:
            candidates.append(card)
    candidates.sort(key=lambda card: (-(active.get(card["id"]) or 0), -(card.get("importance") or 1), card["id"]))
    return candidates


def _core_reservation_item(picked: dict[str, Any], concept: Any, unit: dict[str, Any]) -> dict[str, Any]:
    weak = _is_weak_recall(concept)
    return card_item(
        picked, conceptId=_concept_value(concept, "conceptId", "_unattached"),
        conceptName=_concept_value(concept, "conceptName", "Scheduled atomic core"),
        priority=max(1, _concept_value(concept, "priority", 0)),
        interventionReason=_intervention_for_weak(weak, "weak_recall"),
        examTargetMasteryStage="scheduled-atomic-core", examTargetMasteryUnitId=unit.get("unitId"),
    )


def _reserve_core_unit(context: CoreReservationContext, unit: dict[str, Any]) -> bool:
    candidates = _core_unit_candidates(unit, context.cards, context.active)
    picked = context.build.pick_card_candidate(
        candidates, similarity_threshold=context.inputs["interferenceThreshold"], max_per_cluster=3,
    )
    if not picked:
        return False
    item = _core_reservation_item(picked, context.resolve(picked), unit)
    return commit_card(CardCommitRequest(context.build, picked, item, record_recall=False))


def _reserve_core_units(context: CoreReservationContext, ordered_units: list[dict[str, Any]], quota: int) -> None:
    for unit in ordered_units:
        if context.build.selected_core_target_seats >= quota:
            break
        if _reserve_core_unit(context, unit):
            context.build.selected_core_target_seats += 1


def _has_scheduled_core_work(inputs: dict[str, Any], evidence: dict[str, Any] | None, workload: dict[str, Any] | None) -> bool:
    if not inputs.get("applyExamTarget"):
        return False
    if not evidence:
        return False
    if not workload:
        return False
    return (workload.get("coreTargetSeats") or 0) > 0


def _core_allocation(evidence: dict[str, Any], eligible_core: set[str]) -> dict[str, Any]:
    return allocate_mastery_seats(
        candidates=_mastery_candidates(evidence, {"scheduled-atomic-core"}),
        eligible_unit_ids_by_stage={"scheduled-atomic-core": list(eligible_core)},
        core_target_seats=len(eligible_core), surplus_seats=0,
    )


def _ordered_core_units(evidence: dict[str, Any], allocation: dict[str, Any]) -> list[dict[str, Any]]:
    planned = {row["unitId"] for row in allocation["selectedCandidates"]}
    units = []
    for unit in evidence.get("units") or []:
        if unit.get("stage") != "scheduled-atomic-core":
            continue
        if unit.get("unitId") in planned:
            units.append(unit)
    return units


def reserve_scheduled_core(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    evidence, workload = inputs.get("masteryEvidence"), inputs.get("targetWorkload")
    if not _has_scheduled_core_work(inputs, evidence, workload):
        return
    bulk = inputs["bulk"]
    eligible_core = set(inputs.get("eligibleCoreUnitIds") or [])
    allocation = _core_allocation(evidence, eligible_core)
    context = CoreReservationContext(
        inputs=inputs, build=build, cards={card["id"]: card for card in bulk["unseenCards"] + bulk["seenCards"]},
        active=active_targets(inputs)["card"], resolve=_core_card_resolver(inputs, plan),
    )
    _reserve_core_units(context, _ordered_core_units(evidence, allocation), workload.get("coreTargetSeats") or 0)
    workload_core = workload.get("coreTargetSeats")
    requirement = inputs["size"] if workload_core is None else workload_core
    build.core_target_quota_satisfied = build.selected_core_target_seats >= requirement


def _surplus_card_item(picked: dict[str, Any], concept: Any, planned: dict[str, Any]) -> dict[str, Any]:
    weak = _is_weak_recall(concept)
    return card_item(
        picked, conceptId=_concept_value(concept, "conceptId", "_unattached"),
        conceptName=_concept_value(concept, "conceptName", "Exam-target surplus"),
        priority=_concept_value(concept, "priority", 0.5),
        interventionReason=_intervention_for_weak(weak, "weak_recall"),
        examTargetMasteryStage=planned["stage"], examTargetMasteryUnitId=planned["unitId"],
    )


def _reserve_surplus_card(context: SurplusReservationContext, card: dict[str, Any], planned: dict[str, Any]) -> bool:
    picked = context.build.pick_card_candidate(
        [card], similarity_threshold=context.inputs["interferenceThreshold"], max_per_cluster=3,
    )
    if not picked:
        return False
    item = _surplus_card_item(picked, context.resolve_card(picked), planned)
    return commit_card(CardCommitRequest(context.build, picked, item, record_recall=False))


def _suppression_group_blocked(build: SessionBuildState, suppression: str | None) -> bool:
    if not suppression:
        return False
    return suppression in build.selected_question_variant_groups


def _surplus_question_available(inputs: dict[str, Any], build: SessionBuildState, question: dict[str, Any]) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    suppression = question_suppression_key(question)
    if question["id"] in build.selected_question_ids:
        return False
    if _suppression_group_blocked(build, suppression):
        return False
    return take_with_reentry_cap(question["id"], inputs["bulk"]["questionFamiliarity"], build.mastered_reentry_counter)


def _surplus_question_item(question: dict[str, Any], concept: Any, planned: dict[str, Any]) -> dict[str, Any]:
    weak = _is_weak_recall(concept)
    return {
        "type": "question", "id": question["id"],
        "conceptId": _concept_value(concept, "conceptId", f"rotation:{question['id']}"),
        "conceptName": _concept_value(concept, "conceptName", "Exam-target application"),
        "priority": _concept_value(concept, "priority", 0.5),
        "interventionReason": _intervention_for_weak(weak, "needs_retest"),
        **thread_trace(), "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType"),
        "examTargetMasteryStage": planned["stage"], "examTargetMasteryUnitId": planned["unitId"],
    }


def _reserve_surplus_question(context: SurplusReservationContext, question: dict[str, Any], planned: dict[str, Any]) -> bool:
    if not _surplus_question_available(context.inputs, context.build, question):
        return False
    concept = context.resolve_question(question)
    return context.build.add_item(_surplus_question_item(question, concept, planned))


def _reserve_planned_surplus(context: SurplusReservationContext, planned: dict[str, Any]) -> bool:
    key = planned["itemKey"]
    if key.startswith("card:"):
        card = context.cards.get(key[5:])
        if card is not None:
            return _reserve_surplus_card(context, card, planned)
        return False
    if key.startswith("question:"):
        question = context.bulk["questionMap"].get(key[9:])
        if question is not None:
            return _reserve_surplus_question(context, question, planned)
    return False


def _has_applied_surplus_work(inputs: dict[str, Any], evidence: dict[str, Any] | None,
                              workload: dict[str, Any] | None, quota_satisfied: bool) -> bool:
    if not inputs.get("applyExamTarget"):
        return False
    if not evidence:
        return False
    if not workload:
        return False
    if not quota_satisfied:
        return False
    return (workload.get("surplusSeats") or 0) > 0


def _surplus_allocation(evidence: dict[str, Any], eligible: set[str]) -> dict[str, Any]:
    return allocate_mastery_seats(
        candidates=_mastery_candidates(evidence, {"applied-distinction", "breadth-exploration"}),
        eligible_unit_ids_by_stage={"applied-distinction": list(eligible), "breadth-exploration": []},
        core_target_seats=0, surplus_seats=len(eligible),
    )


def _applied_candidate_available(build: SessionBuildState, planned: dict[str, Any], quota: int) -> bool:
    if planned["stage"] != "applied-distinction":
        return False
    return build.selected_surplus_target_seats < quota


def _reserve_applied_candidate(context: SurplusReservationContext, planned: dict[str, Any], quota: int) -> bool:
    if not _applied_candidate_available(context.build, planned, quota):
        return False
    return _reserve_planned_surplus(context, planned)


def reserve_applied_surplus(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    evidence, workload = inputs.get("masteryEvidence"), inputs.get("targetWorkload")
    if not _has_applied_surplus_work(inputs, evidence, workload, build.core_target_quota_satisfied):
        return
    bulk = inputs["bulk"]
    allocation = _surplus_allocation(evidence, set(inputs.get("eligibleAppliedUnitIds") or []))
    context = SurplusReservationContext(
        inputs=inputs, build=build, bulk=bulk,
        cards={card["id"]: card for card in bulk["unseenCards"] + bulk["seenCards"]},
        resolve_card=_core_card_resolver(inputs, plan), resolve_question=_core_question_resolver(inputs, plan),
    )
    quota = workload.get("surplusSeats") or 0
    for planned in allocation["selectedCandidates"]:
        if _reserve_applied_candidate(context, planned, quota):
            build.selected_surplus_target_seats += 1


def _cross_rotation_denied(context: UnearnedStageContext, source_rotation: str | None) -> bool:
    if context.quota_satisfied:
        return False
    return source_rotation != context.session_rotation


def _applied_stage_denied(context: UnearnedStageContext, assignment: dict[str, Any]) -> bool:
    if assignment.get("unitId") not in context.eligible:
        return True
    return not context.quota_satisfied


def _stage_denied(context: UnearnedStageContext, assignment: dict[str, Any]) -> bool:
    stage = assignment.get("stage")
    if stage == "applied-distinction":
        return _applied_stage_denied(context, assignment)
    if stage == "breadth-exploration":
        return not context.quota_satisfied
    return False


def _unearned_stage_denied(context: UnearnedStageContext, assignment: dict[str, Any] | None,
                           source_rotation: str | None) -> bool:
    if _cross_rotation_denied(context, source_rotation):
        return True
    if not assignment:
        return False
    return _stage_denied(context, assignment)


def _hold_back_cards(inputs: dict[str, Any], build: SessionBuildState, context: UnearnedStageContext) -> None:
    for card in inputs["bulk"]["unseenCards"] + inputs["bulk"]["seenCards"]:
        assignment = context.assignments.get(f"card:{card['id']}")
        if _unearned_stage_denied(context, assignment, card.get("rotation")):
            build.selected_card_ids.add(card["id"])


def _hold_back_questions(inputs: dict[str, Any], build: SessionBuildState, context: UnearnedStageContext) -> None:
    for question in inputs["bulk"]["rotationQuestions"]:
        assignment = context.assignments.get(f"question:{question['id']}")
        if _unearned_stage_denied(context, assignment, question.get("rotation")):
            build.selected_question_ids.add(question["id"])


def hold_back_unearned_stages(inputs: dict[str, Any], build: SessionBuildState) -> None:
    evidence = inputs.get("masteryEvidence")
    if not inputs.get("applyExamTarget"):
        return
    if not evidence:
        return
    context = UnearnedStageContext(
        assignments=evidence.get("itemStageMap") or {}, session_rotation=inputs["rotation"],
        eligible=set(inputs.get("eligibleAppliedUnitIds") or []), quota_satisfied=build.core_target_quota_satisfied,
    )
    _hold_back_cards(inputs, build, context)
    _hold_back_questions(inputs, build, context)
