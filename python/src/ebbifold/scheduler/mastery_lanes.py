"""Exam-target mastery reservation lanes."""

from __future__ import annotations

from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .exam_target import active_targets
from .lane_utils import (
    _core_card_resolver,
    _core_question_resolver,
    card_item,
    commit_card,
    thread_trace,
)


def _mastery_candidates(evidence: dict[str, Any], stages: set[str]) -> list[dict[str, Any]]:
    rows = []
    for unit in evidence.get("units") or []:
        if unit.get("stage") not in stages:
            continue
        for key in (unit.get("safeMetadata") or {}).get("candidateItemKeys") or []:
            rows.append({"itemKey": key, "unitId": unit.get("unitId"), "stage": unit.get("stage")})
    return rows


def _reserve_core_unit(inputs: dict[str, Any], build: SessionBuildState, unit: dict[str, Any],
                       cards: dict[str, dict[str, Any]], active: dict[str, float], resolve: Any) -> bool:
    candidates = [
        cards[key[5:]] for key in (unit.get("safeMetadata") or {}).get("candidateItemKeys") or []
        if key.startswith("card:") and key[5:] in cards
    ]
    candidates.sort(key=lambda card: (-(active.get(card["id"]) or 0), -(card.get("importance") or 1), card["id"]))
    picked = build.pick_card_candidate(candidates, similarity_threshold=inputs["interferenceThreshold"], max_per_cluster=3)
    if not picked:
        return False
    concept = resolve(picked)
    weak = concept and concept.recallOnExamDay < DEFAULTS["recall_threshold"]
    item = card_item(
        picked, conceptId=concept.conceptId if concept else "_unattached",
        conceptName=concept.conceptName if concept else "Scheduled atomic core",
        priority=max(1, concept.priority if concept else 0),
        interventionReason="weak_recall" if weak else "reinforcement",
        examTargetMasteryStage="scheduled-atomic-core", examTargetMasteryUnitId=unit.get("unitId"),
    )
    return commit_card(build, picked, item, record_recall=False)


def _reserve_core_units(inputs: dict[str, Any], build: SessionBuildState, ordered_units: list[dict[str, Any]],
                        cards: dict[str, dict[str, Any]], active: dict[str, Any], resolve: Any, quota: int) -> None:
    for unit in ordered_units:
        if build.selected_core_target_seats >= quota:
            break
        if _reserve_core_unit(inputs, build, unit, cards, active, resolve):
            build.selected_core_target_seats += 1


def reserve_scheduled_core(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    from .exam_target import allocate_mastery_seats

    evidence, workload = inputs.get("masteryEvidence"), inputs.get("targetWorkload")
    if not (inputs.get("applyExamTarget") and evidence and workload and (workload.get("coreTargetSeats") or 0) > 0):
        return
    bulk = inputs["bulk"]
    cards = {c["id"]: c for c in bulk["unseenCards"] + bulk["seenCards"]}
    eligible_core = set(inputs.get("eligibleCoreUnitIds") or [])
    allocation = allocate_mastery_seats(
        candidates=_mastery_candidates(evidence, {"scheduled-atomic-core"}),
        eligible_unit_ids_by_stage={"scheduled-atomic-core": list(eligible_core)},
        core_target_seats=len(eligible_core), surplus_seats=0,
    )
    planned = {row["unitId"] for row in allocation["selectedCandidates"]}
    ordered_units = [u for u in evidence.get("units") or [] if u.get("stage") == "scheduled-atomic-core" and u.get("unitId") in planned]
    active = active_targets(inputs)["card"]
    resolve = _core_card_resolver(inputs, plan)
    quota = workload.get("coreTargetSeats") or 0
    _reserve_core_units(inputs, build, ordered_units, cards, active, resolve, quota)
    workload_core = workload.get("coreTargetSeats")
    requirement = inputs["size"] if workload_core is None else workload_core
    build.core_target_quota_satisfied = build.selected_core_target_seats >= requirement


def _reserve_surplus_card(inputs: dict[str, Any], build: SessionBuildState, card: dict[str, Any],
                          planned: dict[str, Any], resolve: Any) -> bool:
    picked = build.pick_card_candidate([card], similarity_threshold=inputs["interferenceThreshold"], max_per_cluster=3)
    if not picked:
        return False
    concept = resolve(picked)
    weak = concept and concept.recallOnExamDay < DEFAULTS["recall_threshold"]
    item = card_item(picked, conceptId=concept.conceptId if concept else "_unattached",
                     conceptName=concept.conceptName if concept else "Exam-target surplus",
                     priority=concept.priority if concept else 0.5,
                     interventionReason="weak_recall" if weak else "reinforcement",
                     examTargetMasteryStage=planned["stage"], examTargetMasteryUnitId=planned["unitId"])
    return commit_card(build, picked, item, record_recall=False)


def _reserve_surplus_question(inputs: dict[str, Any], build: SessionBuildState, question: dict[str, Any],
                              planned: dict[str, Any], resolve: Any) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    suppression = question_suppression_key(question)
    if question["id"] in build.selected_question_ids:
        return False
    if suppression and suppression in build.selected_question_variant_groups:
        return False
    if not take_with_reentry_cap(question["id"], inputs["bulk"]["questionFamiliarity"], build.mastered_reentry_counter):
        return False
    concept = resolve(question)
    weak = concept and concept.recallOnExamDay < DEFAULTS["recall_threshold"]
    return build.add_item({
        "type": "question", "id": question["id"],
        "conceptId": concept.conceptId if concept else f"rotation:{question['id']}",
        "conceptName": concept.conceptName if concept else "Exam-target application",
        "priority": concept.priority if concept else 0.5,
        "interventionReason": "needs_retest" if weak else "reinforcement",
        **thread_trace(), "variantGroupId": question.get("variantGroupId"), "variantType": question.get("variantType"),
        "examTargetMasteryStage": planned["stage"], "examTargetMasteryUnitId": planned["unitId"],
    })


def _reserve_planned_surplus(inputs: dict[str, Any], build: SessionBuildState, bulk: dict[str, Any],
                             cards: dict[str, dict[str, Any]], planned: dict[str, Any], resolve_card: Any,
                             resolve_question: Any) -> bool:
    key = planned["itemKey"]
    if key.startswith("card:") and key[5:] in cards:
        return _reserve_surplus_card(inputs, build, cards[key[5:]], planned, resolve_card)
    if key.startswith("question:") and key[9:] in bulk["questionMap"]:
        return _reserve_surplus_question(inputs, build, bulk["questionMap"][key[9:]], planned, resolve_question)
    return False


def reserve_applied_surplus(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    from .exam_target import allocate_mastery_seats

    evidence, workload = inputs.get("masteryEvidence"), inputs.get("targetWorkload")
    if not (inputs.get("applyExamTarget") and evidence and workload and (workload.get("surplusSeats") or 0) > 0
            and build.core_target_quota_satisfied):
        return
    bulk = inputs["bulk"]
    eligible = set(inputs.get("eligibleAppliedUnitIds") or [])
    allocation = allocate_mastery_seats(
        candidates=_mastery_candidates(evidence, {"applied-distinction", "breadth-exploration"}),
        eligible_unit_ids_by_stage={"applied-distinction": list(eligible), "breadth-exploration": []},
        core_target_seats=0, surplus_seats=len(eligible),
    )
    cards = {c["id"]: c for c in bulk["unseenCards"] + bulk["seenCards"]}
    resolve_card = _core_card_resolver(inputs, plan)
    resolve_question = _core_question_resolver(inputs, plan)

    quota = workload.get("surplusSeats") or 0
    for planned in allocation["selectedCandidates"]:
        if planned["stage"] != "applied-distinction" or build.selected_surplus_target_seats >= quota:
            continue
        if _reserve_planned_surplus(inputs, build, bulk, cards, planned, resolve_card, resolve_question):
            build.selected_surplus_target_seats += 1


def _unearned_stage_denied(assignment: dict[str, Any] | None, source_rotation: str | None,
                           *, session_rotation: str, eligible: set[str], quota_satisfied: bool) -> bool:
    if not quota_satisfied and source_rotation != session_rotation:
        return True
    if not assignment:
        return False
    stage = assignment.get("stage")
    if stage == "applied-distinction":
        return assignment.get("unitId") not in eligible or not quota_satisfied
    return stage == "breadth-exploration" and not quota_satisfied


def hold_back_unearned_stages(inputs: dict[str, Any], build: SessionBuildState) -> None:
    evidence = inputs.get("masteryEvidence")
    if not (inputs.get("applyExamTarget") and evidence):
        return
    eligible = set(inputs.get("eligibleAppliedUnitIds") or [])
    assignments = evidence.get("itemStageMap") or {}
    quota = build.core_target_quota_satisfied
    rotation = inputs["rotation"]

    def denied(key: str, source_rotation: str | None) -> bool:
        return _unearned_stage_denied(assignments.get(key), source_rotation,
                                      session_rotation=rotation, eligible=eligible, quota_satisfied=quota)

    for card in inputs["bulk"]["unseenCards"] + inputs["bulk"]["seenCards"]:
        if denied(f"card:{card['id']}", card.get("rotation")):
            build.selected_card_ids.add(card["id"])
    for question in inputs["bulk"]["rotationQuestions"]:
        if denied(f"question:{question['id']}", question.get("rotation")):
            build.selected_question_ids.add(question["id"])
