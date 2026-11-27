"""Selection lanes of unified-scheduler-phase-select.ts over explicit snapshots.

The concept-driven, intervention, simulation-focus, exam-target, rescue and
rotation-bank lanes are ported; storage access arrives through explicit
snapshot adapters. ``concepts.assert_supported`` still rejects inputs whose
session-prep blend or other unported upstream lanes would change behavior.
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
    build.selected_surplus_target_seats = 0
    workload_core = (inputs.get("targetWorkload") or {}).get("coreTargetSeats")
    if inputs.get("applyExamTarget"):
        requirement = size if workload_core is None else workload_core
    else:
        requirement = 0
    build.core_target_quota_satisfied = requirement == 0
    if inputs.get("applyExamTarget") and requirement > 0 and inputs.get("masteryEvidence") is None:
        for card in bulk["unseenCards"] + bulk["seenCards"]:
            build.selected_card_ids.add(card["id"])
        for question in bulk["rotationQuestions"]:
            build.selected_question_ids.add(question["id"])
        for video in bulk["rotationVideos"]:
            build.selected_video_ids.add(video["id"])
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


_PROTECTED_PREP_REASONS = frozenset({
    "failure_escalation", "weak_recall", "pre_teach", "chronic_stuck_mcq", "stuck_intervention",
    "mcq_bridge_card", "preemptive_scaffold", "pre_teach_naive",
})


def _partition_prep_candidates(items: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    prep_keys = set(plan.get("itemIds") or [])
    weak_keys = set(plan.get("weakItemIds") or [])
    unseen_keys = set(plan.get("unseenItemIds") or [])
    buckets = {"protected": [], "ordinary": [], "weak": [], "unseen": [], "other_prep": []}
    for item in items:
        key = f"{item['type']}:{item['id']}"
        if item.get("interventionReason") in _PROTECTED_PREP_REASONS:
            bucket = "protected"
        elif key not in prep_keys:
            bucket = "ordinary"
        elif key in weak_keys:
            bucket = "weak"
        elif key in unseen_keys:
            bucket = "unseen"
        else:
            bucket = "other_prep"
        buckets[bucket].append(item)
    return buckets


def _interleave_prep_items(weak: list[dict[str, Any]], unseen: list[dict[str, Any]], share: float) -> list[dict[str, Any]]:
    prep = []
    for index in range(max(len(weak), len(unseen))):
        if share < 0.25 and index < len(unseen):
            prep.append(unseen[index])
        if index < len(weak):
            prep.append(weak[index])
        if share >= 0.25 and index < len(unseen):
            prep.append(unseen[index])
    return prep


def _seat_prep_candidates(protected: list[dict[str, Any]], ordinary: list[dict[str, Any]],
                          prep: list[dict[str, Any]], size: int, share: float) -> list[dict[str, Any]]:
    seats = max(0, size - len(protected))
    wanted = min(len(prep), seats, max(0, int(seats * min(0.5, share))))
    prefix = list(protected)
    prep_index = ordinary_index = 0
    for slot in range(seats):
        prep_due = int((slot + 1) * wanted / seats) > prep_index if seats else False
        prep_available = prep_index < len(prep)
        if (prep_due or ordinary_index >= len(ordinary)) and prep_available:
            prefix.append(prep[prep_index])
            prep_index += 1
        elif ordinary_index < len(ordinary):
            prefix.append(ordinary[ordinary_index])
            ordinary_index += 1
        else:
            break
    return prefix + ordinary[ordinary_index:] + prep[prep_index:]


def _order_prep_candidates(items: list[dict[str, Any]], plan: dict[str, Any], size: int) -> list[dict[str, Any]]:
    share = plan.get("share") or 0
    buckets = _partition_prep_candidates(items, plan)
    prep = _interleave_prep_items(buckets["weak"], buckets["unseen"], share)
    prep.extend(buckets["other_prep"])
    return _seat_prep_candidates(buckets["protected"], buckets["ordinary"], prep, size, share)


def _interleave_prep_keys(keys: list[str], weak: set[str], unseen: set[str], share: float) -> list[str]:
    weak_keys = [key for key in keys if key in weak]
    unseen_keys = [key for key in keys if key in unseen]
    out = []
    for index in range(max(len(weak_keys), len(unseen_keys))):
        if share < 0.25 and index < len(unseen_keys):
            out.append(unseen_keys[index])
        if index < len(weak_keys):
            out.append(weak_keys[index])
        if share >= 0.25 and index < len(unseen_keys):
            out.append(unseen_keys[index])
    return out + [key for key in keys if key not in weak and key not in unseen]


def _prep_session_queues(keys: list[str], sessions: list[dict[str, Any]], weak: set[str],
                         unseen: set[str], share: float) -> list[dict[str, Any]]:
    queues = []
    for session in sessions:
        ids = set(session.get("itemIds") or [])
        session_keys = [key for key in keys if key in ids]
        queues.append({"share": session["share"], "taken": 0,
                       "keys": _interleave_prep_keys(session_keys, weak, unseen, share)})
    return queues


def _drop_used_prep_keys(queues: list[dict[str, Any]], used: set[str]) -> None:
    for queue in queues:
        while queue["keys"] and queue["keys"][0] in used:
            queue["keys"].pop(0)


def order_prep_keys(keys: list[str], plan: dict[str, Any]) -> list[str]:
    weak = set(plan.get("weakItemIds") or [])
    unseen = set(plan.get("unseenItemIds") or [])
    share = plan.get("share") or 0
    sessions = [session for session in plan.get("sessions") or [] if (session.get("share") or 0) > 0]
    if len(sessions) < 2:
        return _interleave_prep_keys(keys, weak, unseen, share)
    queues = _prep_session_queues(keys, sessions, weak, unseen, share)
    used, out = set(), []
    while True:
        _drop_used_prep_keys(queues, used)
        available = [queue for queue in queues if queue["keys"]]
        if not available:
            break
        queue = min(available, key=lambda item: (item["taken"] + 1) / item["share"])
        key = queue["keys"].pop(0)
        queue["taken"] += 1
        used.add(key)
        out.append(key)
    return out + [key for key in keys if key not in used]


def _linked_question_state(inputs: dict[str, Any], plan: dict[str, Any], question: dict[str, Any]):
    states = plan["statesById"]
    links = [link for link in inputs["bulk"]["questionConceptLinks"] if link["questionId"] == question["id"] and link["conceptId"] in states]
    links.sort(key=lambda link: (-(1 if link.get("isPrimary") else 0), -states[link["conceptId"]].priority, link["conceptId"]))
    return states[links[0]["conceptId"]] if links else None


def _core_question_resolver(inputs: dict[str, Any], plan: dict[str, Any]):
    def resolve(question: dict[str, Any]):
        return _linked_question_state(inputs, plan, question) or _topic_fallback_state(inputs, plan, question.get("topics") or [])
    return resolve


def _prep_card_concept_id(topics: list[str], inputs: dict[str, Any]) -> str:
    topic_set = set(topics)
    match = next((concept for concept in inputs["concepts"] if any(topic in topic_set for topic in concept.get("topics") or [])), None)
    return match["id"] if match else "_unattached"


def _prep_question_concept_id(question_id: str, inputs: dict[str, Any]) -> str:
    links = [link for link in inputs["bulk"]["questionConceptLinks"] if link["questionId"] == question_id]
    primary = next((link for link in links if link.get("isPrimary")), None)
    link = primary or (links[0] if links else None)
    return link["conceptId"] if link else "_unattached"


def _prep_maps(inputs: dict[str, Any], item_ids: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    bulk, wanted = inputs["bulk"], set(item_ids)
    cards = [card for card in bulk["unseenCards"] + bulk["seenCards"] if f"card:{card['id']}" in wanted]
    questions = [question for question in bulk["rotationQuestions"] if f"question:{question['id']}" in wanted]
    return ({f"card:{card['id']}": card for card in cards},
            {f"question:{question['id']}": question for question in questions})


def _prep_question_allowed(build: SessionBuildState, question: dict[str, Any], familiarity: dict[str, Any], reentry: dict[str, int]) -> bool:
    from .build_state import question_suppression_key
    from .policies import take_with_reentry_cap

    suppression = question_suppression_key(question)
    if question["id"] in build.selected_question_ids:
        return False
    if suppression and suppression in build.selected_question_variant_groups:
        return False
    return take_with_reentry_cap(question["id"], familiarity, reentry)


def _prep_item(inputs: dict[str, Any], card: dict[str, Any] | None, question: dict[str, Any] | None) -> dict[str, Any]:
    source = card or question
    concept_id = (_prep_card_concept_id(card.get("topics") or [], inputs) if card
                  else _prep_question_concept_id(question["id"], inputs))
    concept = inputs["conceptMap"].get(concept_id)
    item = {
        "type": "card" if card else "question", "id": source["id"], "conceptId": concept_id,
        "conceptName": (concept or {}).get("name") or "Class preparation", "priority": 1,
        "interventionReason": "pre_teach", "topics": source.get("topics") or [],
        "variantGroupId": source.get("variantGroupId"), "variantType": source.get("variantType"),
        "rotation": source.get("rotation"), "challengePolicyApplied": False,
    }
    if card:
        item.update({"variantIndex": card.get("variantIndex"), "complexity": card.get("complexity"), "clusterId": card.get("clusterId")})
    return item


def _add_prep_candidate(inputs: dict[str, Any], build: SessionBuildState,
                        card: dict[str, Any] | None, question: dict[str, Any] | None) -> bool:
    picked = None
    if card:
        picked = build.pick_card_candidate([card], similarity_threshold=inputs["interferenceThreshold"], max_per_cluster=3)
        if not picked:
            return False
    reentry = dict(build.mastered_reentry_counter)
    if question and not _prep_question_allowed(build, question, inputs["bulk"]["questionFamiliarity"], reentry):
        return False
    added = build.add_item(_prep_item(inputs, card, question))
    if added and card:
        build.record_cluster(card.get("clusterId"))
        build.record_card_neighborhood(card)
    if added and question:
        build.mastered_reentry_counter["masteredServed"] = reentry["masteredServed"]
    return added


def reserve_session_prep_blend(inputs: dict[str, Any], _plan: dict[str, Any], build: SessionBuildState) -> None:
    from .queue import shuffle_with_seed

    prep = inputs["options"].get("sessionPrepBlend")
    item_ids = list((prep or {}).get("itemIds") or [])
    share_value = (prep or {}).get("share")
    if not prep or not isinstance(share_value, (int, float)) or share_value <= 0 or not item_ids:
        return
    share = min(0.5, max(0.15, share_value))
    discretionary = max(0, inputs["size"] - len(build.selected_items))
    seats = min(discretionary, int(discretionary * share))
    if seats == 0:
        return

    card_by_key, question_by_key = _prep_maps(inputs, item_ids)
    available = [key for key in item_ids if key in card_by_key or key in question_by_key]
    shuffled = shuffle_with_seed(available, f"{inputs['servingRotationSeed']}:class-prep")
    ordered_ids = order_prep_keys(shuffled, prep)
    stop_at = min(inputs["size"], len(build.selected_items) + seats)

    for key in ordered_ids:
        if len(build.selected_items) >= stop_at:
            break
        card, question = card_by_key.get(key), question_by_key.get(key)
        if card or question:
            _add_prep_candidate(inputs, build, card, question)


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


def rescue_orphans(inputs: dict[str, Any], plan: dict[str, Any], build: SessionBuildState) -> None:
    size = inputs["size"]
    active = active_targets(inputs)
    _rescue_seen_cards(inputs, build, active)
    if len(build.selected_items) >= size or inputs.get("effectiveMaxNewCards") == 0:
        return
    _rescue_unseen_cards(inputs, build, active)
    _rescue_rotation_questions(inputs, build, active)


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
