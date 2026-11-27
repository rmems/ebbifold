"""Selection lanes of unified-scheduler-phase-select.ts over explicit snapshots.

Lanes that only act under exam-target treatment, simulation focus or a
session-prep blend are rejected up front by ``concepts.assert_supported``; the
remaining no-op branches are kept so the phase order matches the facade.
"""

from __future__ import annotations

import math
from typing import Any

from .build_state import SessionBuildState
from .config import DEFAULTS
from .policies import has_demonstrated_learning_gap
from .exam_target import active_targets
from .simulation import prefer_reviewed_simulation_cards
from .ranking import get_cards_from_bulk, get_questions_from_bulk, get_videos_from_bulk
from .scoring import build_cluster_round_robin, classify_teaching_state, normalize_review_challenge_level
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


def create_session_build(inputs: dict[str, Any], plan: dict[str, Any]) -> SessionBuildState:
    size, options, bulk = inputs["size"], inputs["options"], inputs["bulk"]
    level_floor = math.ceil(size * 0.8) if normalize_review_challenge_level(options.get("reviewChallenge")) == 2 else 0
    min_first_sight = min(size, max(0, math.floor(max(options.get("minFirstSightItems") or 0, level_floor))))
    build = SessionBuildState(
        size=size, min_first_sight_items=min_first_sight, max_new_cards=plan["maxNewCards"],
        max_cross_source_items=plan["maxCrossSourceItems"], unseen_card_ids=plan["unseenCardIds"],
        cross_source_card_ids=plan["crossSourceCardIds"], cross_source_question_ids=plan["crossSourceQuestionIds"],
        question_familiarity=bulk["questionFamiliarity"].keys(), excluded_card_ids=inputs["excludedCardIds"],
        excluded_question_ids=inputs["excludedQuestionIds"], excluded_video_ids=inputs["excludedVideoIds"],
        weak_concepts=plan["weakConcepts"], concept_thread_anchors=inputs.get("conceptThreadAnchors") or [],
    )
    build.mastered_reentry_counter = {"masteredServed": 0}
    build.selected_core_target_seats = 0
    build.core_target_quota_satisfied = build.selected_core_target_seats >= 0
    return build


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


def _round_robin(plan: dict[str, Any]) -> list[Any]:
    ranks = plan.get("allocatedConceptRanks")
    rank_fn = (lambda c: ranks.get(c.conceptId)) if ranks is not None else None
    return build_cluster_round_robin(plan["weakConcepts"], plan["getConceptCluster"], rank_fn)


def _simulation_registry(inputs: dict[str, Any]):
    from .simulation import SimulationRegistry

    if "_simulationRegistry" not in inputs:
        inputs["_simulationRegistry"] = SimulationRegistry(inputs.get("simulationItemMap") or [])
    return inputs["_simulationRegistry"]


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
            reason = resolve_reason(inputs, plan, concept.conceptId, concept.recallOnExamDay, base)
            difficulty = question_request_difficulty(inputs["examPressure"], base, already)
            data = inputs["conceptMap"].get(concept.conceptId)
            if not data:
                continue
            questions = question_pool(inputs, build, concept.conceptId, data, difficulty, build.selected_question_count)
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
            candidates = card_pool(inputs, build, concept.conceptId, data, concept.currentRecall)
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
                             interventionReason=resolve_reason(inputs, plan, concept.conceptId, concept.recallOnExamDay, base))
            if commit_card(build, picked, item, concept.currentRecall):
                added_this_pass += 1
        if added_this_pass == 0:
            break


CONCEPT_THREAD_POLICY_VERSION = "clinical-thread-facets-v1"


def thread_trace() -> dict[str, Any]:
    return {"conceptThreadPolicyVersion": CONCEPT_THREAD_POLICY_VERSION, "conceptThreadPolicyApplied": False}


def fill_coverage_lane(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    target = plan["targetCardCount"]
    if plan["coverageReservation"] <= 0 or build.selected_card_count >= target:
        return
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
    rows = [(inputs["conceptMap"].get(cid), plan["statesById"].get(cid), n) for cid, n in counts.items()]
    rows = [r for r in rows if r[0] is not None]
    rows.sort(key=lambda r: (-r[2], -(1 if r[0].get("examWeight") is None else r[0]["examWeight"])))
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept, state, _ in rows:
        if build.selected_card_count >= target:
            break
        recall = state.currentRecall if state else None
        candidates = card_pool(inputs, build, concept["id"], concept, recall)
        pristine = [c for c in candidates if c["id"] in plan["unseenCardIds"]]
        if not pristine:
            continue
        picked = build.pick_card_candidate(pristine, similarity_threshold=threshold, max_per_cluster=5)
        if not picked:
            continue
        reason = resolve_reason(inputs, plan, concept["id"], state.recallOnExamDay if state else 0, "reinforcement")
        item = card_item(picked, challengePolicyApplied=True, conceptId=concept["id"], conceptName=concept["name"],
                         priority=state.priority if state else 0, interventionReason=reason)
        commit_card(build, picked, item, recall)


def backfill_from_any_concept(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    if len(build.selected_items) >= size:
        return
    threshold = min(0.97, inputs["interferenceThreshold"] + 0.1)
    for concept in [s for s in plan["conceptStates"] if s.conceptId not in build.selected_concept_ids]:
        if len(build.selected_items) >= size or build.selected_card_count >= plan["targetCardCount"]:
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
        item = card_item(picked, challengePolicyApplied=True, conceptId=concept.conceptId,
                         conceptName=concept.conceptName, priority=concept.priority, interventionReason="reinforcement")
        commit_card(build, picked, item, concept.currentRecall)
    if len(build.selected_items) >= size:
        return
    for concept in [s for s in plan["conceptStates"] if s.conceptId not in build.selected_concept_ids]:
        if len(build.selected_items) >= size:
            break
        data = inputs["conceptMap"].get(concept.conceptId)
        if not data:
            continue
        questions = question_pool(inputs, build, concept.conceptId, data, "medium")
        if questions:
            q = questions[0]
            build.add_item({"type": "question", "id": q["id"], "conceptId": concept.conceptId,
                            "conceptName": concept.conceptName, "priority": concept.priority,
                            "interventionReason": "reinforcement", **thread_trace(),
                            "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")})


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
    rows = [dict(r) for r in (inputs.get("rotationTopUp") or [])][:deficit]
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


def rescue_orphans(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    from .build_state import question_suppression_key
    from .errors import UnsupportedSchedulerInput
    from .ranking import apply_bounded_exam_target_nudge
    from .policies import (
        has_specific_clinical_topic_overlap,
        prioritize_least_recently_served_contrast_siblings,
        take_with_reentry_cap,
    )

    size, bulk = inputs["size"], inputs["bulk"]
    if bulk.get("passSet") and bulk.get("passSetIntakeDecision") in ("pass-set-first", "supply-exhausted", "unlocked"):
        raise UnsupportedSchedulerInput("pass-set orphan ordering is not ported in Phase 3a")

    def attribution(topics: list[str]) -> tuple[str, str]:
        for c in inputs["concepts"]:
            if has_specific_clinical_topic_overlap(topics, c["topics"]):
                return c["id"], c["name"]
        return "_unattached", "Unattached"

    def add_card(card: dict[str, Any], reason: str) -> None:
        cid, name = attribution(card["topics"])
        build.add_item(card_item(card, conceptId=cid, conceptName=name, priority=0.2, interventionReason=reason))

    active = active_targets(inputs)

    def nudge(rows: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
        if not inputs.get("applyExamTarget"):
            return rows
        return apply_bounded_exam_target_nudge(rows, lambda r: active[kind].get(r["id"]), active["rankMove"])

    if len(build.selected_items) < size:
        for card in nudge([c for c in bulk["seenCards"] if c["id"] not in build.selected_card_ids], "card"):
            if len(build.selected_items) >= size:
                break
            if card.get("variantGroupId") and card["variantGroupId"] in build.selected_card_variant_groups:
                continue
            add_card(card, "needs_retest")
    if len(build.selected_items) >= size or inputs.get("effectiveMaxNewCards") == 0:
        return
    unseen = [c for c in bulk["unseenCards"] if c["id"] not in build.selected_card_ids
              and c["id"] not in bulk["passSetDeferredCardIds"]]
    unseen.sort(key=lambda c: -(1 if c.get("importance") is None else c["importance"]))
    for card in nudge(unseen, "card"):
        if len(build.selected_items) >= size:
            break
        if card.get("variantGroupId") and card["variantGroupId"] in build.selected_card_variant_groups:
            continue
        add_card(card, "reinforcement")
    familiarity = bulk["questionFamiliarity"]
    questions = prioritize_least_recently_served_contrast_siblings(
        [q for q in bulk["rotationQuestions"] if q["id"] not in build.selected_question_ids], lambda q: q["id"], familiarity
    )
    for q in nudge(questions, "question"):
        if len(build.selected_items) >= size:
            break
        key = question_suppression_key(q)
        if key and key in build.selected_question_variant_groups:
            continue
        if not take_with_reentry_cap(q["id"], familiarity, build.mastered_reentry_counter):
            continue
        cid, name = attribution(q["topics"])
        build.add_item({"type": "question", "id": q["id"], "conceptId": cid, "conceptName": name, "priority": 0.2,
                        "interventionReason": "reinforcement", **thread_trace(),
                        "variantGroupId": q.get("variantGroupId"), "variantType": q.get("variantType")})


def run_selection(inputs: dict[str, Any], plan: dict[str, Any]) -> SessionBuildState:
    """Facade steps 10-26: build state, every selection lane, first-sight release."""
    build = create_session_build(inputs, plan)
    fill_simulation_preparation(inputs, plan, build)
    for lane in (pre_teach_videos, pre_teach_naive_concepts, fill_questions, fill_cards, fill_coverage_lane,
                 backfill_from_any_concept, probe_stale_strong_concepts, top_up_from_rotation_bank,
                 backfill_cards_uncapped, rescue_orphans):
        lane(inputs, plan, build)
        build.trace.append({"op": "lane", "id": lane.__name__, "result": len(build.selected_items)})
    if len(build.selected_items) < inputs["size"] and build.deferred_for_novelty:
        build.release_first_sight_reservation()
    return build
