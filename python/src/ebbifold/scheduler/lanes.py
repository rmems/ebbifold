"""Selection lanes of unified-scheduler-phase-select.ts over explicit snapshots.

The concept-driven, intervention, simulation-focus, exam-target, rescue and
rotation-bank lanes are ported; storage access arrives through explicit
snapshot adapters.
"""

from __future__ import annotations

import math
from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .ranking import get_videos_from_bulk
from .scoring import classify_teaching_state
from .simulation import prefer_reviewed_simulation_cards
from .lane_utils import (
    CardCommitRequest,
    CardPoolRequest,
    QuestionPoolRequest,
    ReasonRequest,
    _round_robin,
    _simulation_registry,
    card_item,
    card_pool,
    commit_card,
    create_session_build,
    question_pool,
    resolve_reason,
    thread_trace,
)
from .mastery_lanes import hold_back_unearned_stages, reserve_applied_surplus, reserve_scheduled_core
from .prep_lanes import reserve_session_prep_blend
from .rescue_lanes import (
    backfill_cards_uncapped,
    backfill_from_any_concept,
    fill_coverage_lane,
    probe_stale_strong_concepts,
    rescue_orphans,
    top_up_from_rotation_bank,
)


def pre_teach_videos(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    if not inputs.get("includeVideos"):
        return
    for concept in [c for c in plan["weakConcepts"] if c.recallOnExamDay < 0.4 and c.confidence < 0.6]:
        if len(build.selected_items) >= inputs["size"]:
            break
        if build.videos_per_concept.get(concept.conceptId, 0) >= 1:
            continue
        data = inputs["conceptMap"].get(concept.conceptId)
        if not data:
            continue
        videos = get_videos_from_bulk(concept.conceptId, data, 1, inputs["bulk"], build.selected_video_ids)
        if videos:
            v = videos[0]
            item = {"type": "video", "id": v["id"], "conceptId": concept.conceptId, "conceptName": concept.conceptName,
                    "priority": concept.priority, "interventionReason": "pre_teach", "videoTitle": v.get("title"),
                    "videoThumbnailUrl": v.get("thumbnailR2Key"), "videoR2Key": v.get("r2Key")}
            if v.get("durationSecs") is not None:
                item["videoDuration"] = v["durationSecs"]
            if v.get("creatorName") is not None:
                item["creatorName"] = v["creatorName"]
            build.add_item(item)


def _topics_overlap(a: list[str] | None, b: list[str]) -> bool:
    if not a:
        return False
    lower = {t.lower() for t in b}
    return any(t.lower() in lower for t in a)


def pick_naive_pre_teach_cards(naive: list[dict[str, Any]], bulk: dict[str, Any], max_pre_teach: float,
                               already: set[str], prefer_card: Any = None) -> list[dict[str, Any]]:
    if max_pre_teach <= 0 or not naive or not bulk["unseenCards"]:
        return []
    used = set(already)
    scaffolds = [c for c in bulk["unseenCards"] if c.get("complexity") == 1]
    if not scaffolds:
        return []
    ordered = sorted(naive, key=lambda c: (-c["priority"], c["conceptName"]))
    if prefer_card is not None:
        scaffolds.sort(key=lambda c: not prefer_card(c))
        preferred = [c for c in scaffolds if c["id"] not in used and prefer_card(c)]
        ordered.sort(key=lambda concept: not any(_topics_overlap(concept["topics"], c["topics"]) for c in preferred))
    picks: list[dict[str, Any]] = []
    for concept in ordered:
        if len(picks) >= max_pre_teach:
            break
        card = next((c for c in scaffolds if c["id"] not in used and _topics_overlap(concept["topics"], c["topics"])), None)
        if card is None:
            continue
        used.add(card["id"])
        picks.append({**{k: concept[k] for k in ("conceptId", "conceptName", "priority")}, "card": card})
    return picks


def pre_teach_naive_concepts(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    naive = []
    for cs in plan["weakConcepts"]:
        if classify_teaching_state(cs.exposureCount, cs.recallOnExamDay, cs.confidence) != "naive":
            continue
        data = inputs["conceptMap"].get(cs.conceptId)
        if data:
            naive.append({"conceptId": cs.conceptId, "conceptName": cs.conceptName, "topics": data["topics"],
                          "priority": cs.priority})
    max_new = plan["maxNewCards"]
    budget = math.inf if max_new == math.inf else max(0, max_new - build.new_card_count)
    target = plan["targetCardCount"]
    if budget > 0 and naive and build.selected_card_count < target:
        focus = inputs["options"].get("simulationFocus")
        registry = _simulation_registry(inputs)
        prefer = (lambda c: registry.evidence(c.get("stableId"), focus) is not None) if focus else None
        picks = pick_naive_pre_teach_cards(naive, inputs["bulk"], min(3, target - build.selected_card_count, budget),
                                           build.selected_card_ids, prefer)
        for pick in picks:
            if build.selected_card_count >= target:
                break
            build.add_item(card_item(pick["card"], conceptId=pick["conceptId"], conceptName=pick["conceptName"],
                                     priority=pick["priority"], interventionReason="pre_teach_naive"))

def fill_simulation_preparation(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    from .simulation import order_unrecalled_simulation_preparation

    rows = inputs["bulk"].get("simulationPreparation") or []
    if not inputs["options"].get("simulationFocus") or not rows:
        return
    for row in order_unrecalled_simulation_preparation(rows, inputs["nowMs"]):
        if len(build.selected_items) >= inputs["size"]:
            break
        if any(item["id"] == row["id"] for item in build.selected_items):
            continue
        build.selected_card_ids.discard(row["id"])
        card = row.get("card") or {}
        build.add_item({
            "type": "card", "id": row["id"], "conceptId": "_unattached", "conceptName": "Simulation preparation",
            "priority": 1, "interventionReason": "simulation_preparation",
            "variantGroupId": card.get("variantGroupId"), "variantIndex": card.get("variantIndex"),
            "variantType": card.get("variantType"),
        }, bypass_novelty_reservation=True)


def fill_questions(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    from .difficulty import question_request_difficulty

    target = plan["targetQuestionCount"]
    if target <= 0:
        return
    concepts = _round_robin(plan)
    pass_no = 0
    while build.selected_question_count < target:
        allowed = DEFAULTS["max_questions_per_concept"] + pass_no
        added_this_pass = 0
        for concept in concepts:
            if build.selected_question_count >= target:
                break
            already = build.questions_per_concept.get(concept.conceptId, 0)
            if already >= allowed:
                continue
            if build.concept_items_so_far(concept.conceptId) >= build.teaching_cap_for(concept.conceptId):
                continue
            if concept.intervention == "probe":
                low = concept.confidence < DEFAULTS["confidence_threshold"] or concept.exposureCount < 3
                base = "low_confidence" if low else "needs_retest"
            else:
                base = "needs_retest" if concept.intervention == "remediate" else "reinforcement"
            reason = resolve_reason(ReasonRequest(inputs, plan, concept.conceptId, concept.recallOnExamDay, base))
            difficulty = question_request_difficulty(inputs["examPressure"], base, already)
            data = inputs["conceptMap"].get(concept.conceptId)
            if not data:
                continue
            questions = question_pool(QuestionPoolRequest(
                inputs, build, concept.conceptId, data, difficulty, build.selected_question_count,
            ))
            if not questions:
                continue
            q = questions[0]
            if build.add_item({"type": "question", "id": q["id"], "conceptId": concept.conceptId,
                               "conceptName": concept.conceptName, "priority": concept.priority,
                               "interventionReason": reason, **thread_trace(),
                               "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")}):
                added_this_pass += 1
        if added_this_pass == 0:
            break
        pass_no += 1


def fill_cards(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    target = plan["targetWeakCardCount"]
    if target <= 0:
        return
    hard_max = 5
    concepts = _round_robin(plan)
    threshold = inputs["interferenceThreshold"]
    recent_failures = set(inputs.get("recentFailureConceptIds", []))
    bulk = inputs["bulk"]
    stable_ids = {c["id"]: c.get("stableId") for c in [*bulk["unseenCards"], *bulk["seenCards"]]}
    for _ in range(hard_max):
        if build.selected_card_count >= target:
            break
        added_this_pass = 0
        for concept in concepts:
            if build.selected_card_count >= target:
                break
            already = build.cards_per_concept.get(concept.conceptId, 0)
            if already >= hard_max:
                continue
            if build.concept_items_so_far(concept.conceptId) >= build.teaching_cap_for(concept.conceptId):
                continue
            data = inputs["conceptMap"].get(concept.conceptId)
            if not data:
                continue
            candidates = card_pool(CardPoolRequest(inputs, build, concept.conceptId, data, concept.currentRecall))
            if not candidates:
                continue
            sim = threshold if already == 0 else min(0.97, threshold + 0.1)
            focus = None if concept.conceptId in recent_failures else inputs["options"].get("simulationFocus")
            ordered = prefer_reviewed_simulation_cards(list(candidates), lambda c: stable_ids.get(c["id"]), focus,
                                                       _simulation_registry(inputs))
            picked = build.pick_card_candidate(ordered, similarity_threshold=sim,
                                               max_per_cluster=3 if already == 0 else 5)
            if not picked:
                continue
            base = "weak_recall" if concept.recallOnExamDay < DEFAULTS["recall_threshold"] else "reinforcement"
            item = card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                             conceptName=concept.conceptName, priority=concept.priority,
                             interventionReason=resolve_reason(ReasonRequest(
                                 inputs, plan, concept.conceptId, concept.recallOnExamDay, base,
                             )))
            if commit_card(CardCommitRequest(build, picked, item, concept.currentRecall)):
                added_this_pass += 1
        if added_this_pass == 0:
            break

def run_selection(inputs: dict[str, Any], plan: dict[str, Any]) -> SessionBuildState:
    """Facade steps 10-26: build state, every selection lane, first-sight release."""
    build = create_session_build(inputs, plan)
    fill_simulation_preparation(inputs, plan, build)
    reserve_scheduled_core(inputs, plan, build)
    reserve_applied_surplus(inputs, plan, build)
    hold_back_unearned_stages(inputs, build)
    reserve_session_prep_blend(inputs, plan, build)
    for lane in (pre_teach_videos, pre_teach_naive_concepts, fill_questions, fill_cards, fill_coverage_lane,
                 backfill_from_any_concept, probe_stale_strong_concepts, top_up_from_rotation_bank,
                 backfill_cards_uncapped, rescue_orphans):
        lane(inputs, plan, build)
        build.trace.append({"op": "lane", "id": lane.__name__, "result": len(build.selected_items)})
    if len(build.selected_items) < inputs["size"] and build.deferred_for_novelty:
        build.release_first_sight_reservation()
    return build
