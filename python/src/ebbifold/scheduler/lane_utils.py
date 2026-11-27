"""Shared helpers for scheduler selection lanes."""

from __future__ import annotations

import math
from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .policies import has_demonstrated_learning_gap
from .exam_target import active_targets
from .ranking import get_cards_from_bulk, get_questions_from_bulk
from .scoring import build_cluster_round_robin, normalize_review_challenge_level
from .scoring import resolve_intervention_reason as _resolve_reason
from .topics import expand_topic_set


def resolve_reason(inputs: dict[str, Any], plan: dict[str, Any], concept_id: str, recall: float, base: str) -> str:
    return _resolve_reason(
        concept_id, recall, base, inputs["recentFailureConceptIds"], plan["conceptHasPristine"], DEFAULTS["target_recall"]
    )


def demonstrated_gap(inputs: dict[str, Any], concept_id: str) -> bool:
    state = inputs["stateMap"].get(concept_id) or {}
    return has_demonstrated_learning_gap(
        probe_count=state.get("probeCount"),
        recall_probability=state.get("recallProbability"),
        recent_fail_rate=state.get("recentFailRate"),
        last_probe_at_ms=state.get("lastProbeAt"),
        recent_failure=concept_id in inputs["recentFailureConceptIds"],
        now_ms=inputs["nowMs"],
    )


def card_pool(inputs: dict[str, Any], build: SessionBuildState, concept_id: str, concept: dict[str, Any],
              recall: float | None, *, apply_review_challenge: bool = True):
    """Backfill lanes (any-concept, uncapped) pass no review challenge in TypeScript."""
    options = inputs["options"]
    challenge = (options.get("reviewChallenge") or 0) if apply_review_challenge else 0
    active = active_targets(inputs)
    return get_cards_from_bulk(
        concept_id, concept, 20, inputs["bulk"], build.selected_card_ids, inputs.get("penaltyContext"), recall,
        options.get("currentTeachingWeek"), active["card"] if inputs.get("applyExamTarget") else None,
        active["rankMove"], options.get("topicTeachingWeeks"),
        options.get("recentFigureExposures"), inputs["nowMs"], challenge,
        options.get("topicRecencyDays"),
    )


def question_pool(inputs: dict[str, Any], build: SessionBuildState, concept_id: str, concept: dict[str, Any],
                  difficulty: str, slot: int | None = None) -> list[dict[str, Any]]:
    options = inputs["options"]
    exam: dict[str, Any] = {}
    if inputs.get("applyExamTarget"):
        active = active_targets(inputs)
        exam = {"exam_target_scores": active["question"], "exam_target_max_rank_move": active["rankMove"]}
    return get_questions_from_bulk(
        concept_id, concept, 1, inputs["bulk"],
        selected_question_ids=build.selected_question_ids,
        selected_question_variant_groups=build.selected_question_variant_groups,
        difficulty_plan=[difficulty],
        review_challenge=options.get("reviewChallenge"),
        review_challenge_gap=demonstrated_gap(inputs, concept_id),
        challenge_slot_index=slot or 0,
        current_teaching_week=options.get("currentTeachingWeek"),
        topic_teaching_weeks=options.get("topicTeachingWeeks"),
        topic_recency_days=options.get("topicRecencyDays"),
        mastered_reentry_counter=build.mastered_reentry_counter,
        recent_formats_by_concept=inputs.get("recentFormatsByConcept"),
        **exam,
    )


def card_item(picked: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {"type": "card", "id": picked["id"], **fields, "variantGroupId": picked.get("variantGroupId"),
            "variantIndex": picked.get("variantIndex"), "variantType": picked.get("variantType")}


def commit_card(build: SessionBuildState, picked: dict[str, Any], item: dict[str, Any], recall: float | None = None,
                record_recall: bool = True) -> bool:
    added = build.add_item(item)
    if added:
        if record_recall:
            build.record_applied_challenge_recall(picked["id"], recall)
        build.record_cluster(picked.get("clusterId"))
        build.record_card_neighborhood(picked)
    return added


def _minimum_first_sight_items(size: int, options: dict[str, Any]) -> int:
    level_floor = math.ceil(size * 0.8) if normalize_review_challenge_level(options.get("reviewChallenge")) == 2 else 0
    return min(size, max(0, math.floor(max(options.get("minFirstSightItems") or 0, level_floor))))


def _core_target_requirement(inputs: dict[str, Any], size: int) -> int:
    if not inputs.get("applyExamTarget"):
        return 0
    workload_core = (inputs.get("targetWorkload") or {}).get("coreTargetSeats")
    return size if workload_core is None else workload_core


def _block_items_without_mastery_evidence(bulk: dict[str, Any], build: SessionBuildState) -> None:
    for card in bulk["unseenCards"] + bulk["seenCards"]:
        build.selected_card_ids.add(card["id"])
    for question in bulk["rotationQuestions"]:
        build.selected_question_ids.add(question["id"])
    for video in bulk["rotationVideos"]:
        build.selected_video_ids.add(video["id"])


def create_session_build(inputs: dict[str, Any], plan: dict[str, Any]) -> SessionBuildState:
    size, bulk = inputs["size"], inputs["bulk"]
    build = SessionBuildState(
        size=size, min_first_sight_items=_minimum_first_sight_items(size, inputs["options"]),
        max_new_cards=plan["maxNewCards"],
        max_cross_source_items=plan["maxCrossSourceItems"], unseen_card_ids=plan["unseenCardIds"],
        cross_source_card_ids=plan["crossSourceCardIds"], cross_source_question_ids=plan["crossSourceQuestionIds"],
        question_familiarity=bulk["questionFamiliarity"].keys(), excluded_card_ids=inputs["excludedCardIds"],
        excluded_question_ids=inputs["excludedQuestionIds"], excluded_video_ids=inputs["excludedVideoIds"],
        weak_concepts=plan["weakConcepts"], concept_thread_anchors=inputs.get("conceptThreadAnchors") or [],
    )
    build.mastered_reentry_counter = {"masteredServed": 0}
    build.selected_core_target_seats = 0
    build.selected_surplus_target_seats = 0
    requirement = _core_target_requirement(inputs, size)
    build.core_target_quota_satisfied = requirement == 0
    if inputs.get("applyExamTarget") and requirement > 0 and inputs.get("masteryEvidence") is None:
        _block_items_without_mastery_evidence(bulk, build)
    return build


def _expanded_topics(inputs: dict[str, Any], concept_id: str) -> set[str]:
    concept = inputs["conceptMap"].get(concept_id)
    return set(expand_topic_set((concept or {}).get("topics") or []))


def _topic_fallback_state(inputs: dict[str, Any], plan: dict[str, Any], topics: list[str]):
    matches = [state for state in plan["conceptStates"] if any(topic in _expanded_topics(inputs, state.conceptId) for topic in topics)]
    matches.sort(key=lambda state: (-state.priority, state.conceptId))
    return matches[0] if matches else None


def _scored_card_state(inputs: dict[str, Any], plan: dict[str, Any], card: dict[str, Any]):
    best = None
    for state in plan["conceptStates"]:
        score = (inputs["bulk"]["cardScores"].get(state.conceptId) or {}).get(card["id"])
        if score is None:
            continue
        if best is None or score > best[1] or (score == best[1] and state.conceptId < best[0].conceptId):
            best = (state, score)
    return best[0] if best else None


def _core_card_resolver(inputs: dict[str, Any], plan: dict[str, Any]):
    def resolve(card: dict[str, Any]):
        return _scored_card_state(inputs, plan, card) or _topic_fallback_state(inputs, plan, card.get("topics") or [])
    return resolve

def _linked_question_state(inputs: dict[str, Any], plan: dict[str, Any], question: dict[str, Any]):
    states = plan["statesById"]
    links = [link for link in inputs["bulk"]["questionConceptLinks"] if link["questionId"] == question["id"] and link["conceptId"] in states]
    links.sort(key=lambda link: (-(1 if link.get("isPrimary") else 0), -states[link["conceptId"]].priority, link["conceptId"]))
    return states[links[0]["conceptId"]] if links else None


def _core_question_resolver(inputs: dict[str, Any], plan: dict[str, Any]):
    def resolve(question: dict[str, Any]):
        return _linked_question_state(inputs, plan, question) or _topic_fallback_state(inputs, plan, question.get("topics") or [])
    return resolve

def _round_robin(plan: dict[str, Any]) -> list[Any]:
    ranks = plan.get("allocatedConceptRanks")
    rank_fn = (lambda c: ranks.get(c.conceptId)) if ranks is not None else None
    return build_cluster_round_robin(plan["weakConcepts"], plan["getConceptCluster"], rank_fn)


def _simulation_registry(inputs: dict[str, Any]):
    from .simulation import SimulationRegistry

    if "_simulationRegistry" not in inputs:
        inputs["_simulationRegistry"] = SimulationRegistry(inputs.get("simulationItemMap") or [])
    return inputs["_simulationRegistry"]

CONCEPT_THREAD_POLICY_VERSION = "clinical-thread-facets-v1"


def thread_trace() -> dict[str, Any]:
    return {"conceptThreadPolicyVersion": CONCEPT_THREAD_POLICY_VERSION, "conceptThreadPolicyApplied": False}
