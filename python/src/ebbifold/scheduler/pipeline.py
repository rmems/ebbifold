"""Phase 3a scheduler entry points over explicit input snapshots.

``plan_concepts`` runs unified-scheduler.ts phases 4-6 (rank, jitter-sort,
pristine detection, weak-concept selection, quota) and returns every value the
later lanes consume. ``run_build_script`` drives ``SessionBuildState`` through
a scripted sequence of add/pick/release operations so the bookkeeping rules
(first-sight reservation, teaching caps, cross-source caps, interference,
cluster caps, variant suppression, new-card budget) can be checked one decision
at a time.
"""

from __future__ import annotations

import math
from typing import Any

from .build_state import SessionBuildState
from .concepts import (
    ConceptState,
    _json_number,
    find_pristine_concepts,
    plan_card_quota,
    rank_concepts,
    select_weak_concepts,
    sort_concepts_by_priority,
)
from .scoring import build_cluster_round_robin


def _planner_inputs(snapshot: dict[str, Any]) -> dict[str, Any]:
    inputs = dict(snapshot)
    inputs["options"] = dict(snapshot.get("options") or {})
    inputs["conceptMap"] = dict(
        snapshot.get("conceptMap") or {c["id"]: c for c in snapshot.get("concepts", [])}
    )
    for key in ("recentFailureConceptIds", "excludedCardIds", "excludedQuestionIds", "excludedVideoIds"):
        inputs[key] = set(snapshot.get(key) or [])
    return inputs


def plan_concepts(snapshot: dict[str, Any]) -> dict[str, Any]:
    inputs = _planner_inputs(snapshot)
    ranking = rank_concepts(inputs)
    ranked = [s.to_json() for s in ranking.conceptStates]
    clusters = {s.conceptId: ranking.getConceptCluster(s) for s in ranking.conceptStates}
    sort_concepts_by_priority(inputs, ranking)
    pristine = find_pristine_concepts(inputs)
    selection = select_weak_concepts(inputs, ranking, pristine)
    quota = plan_card_quota(inputs, selection.cardRatio)
    allocated_ranks = selection.allocatedConceptRanks
    target_rank = None
    if allocated_ranks is not None:
        target_rank = lambda concept: allocated_ranks.get(concept.conceptId)
    round_robin = build_cluster_round_robin(
        selection.weakConcepts, ranking.getConceptCluster, target_rank
    )
    return {
        "rankedStates": ranked,
        "conceptClusters": clusters,
        "sortedConceptIds": [s.conceptId for s in ranking.conceptStates],
        "sortedPriorities": [s.priority for s in ranking.conceptStates],
        "pristineConceptIds": sorted(pristine),
        "weakConceptIds": [s.conceptId for s in selection.weakConcepts],
        "cardRatio": selection.cardRatio,
        "quota": quota,
        "clusterRoundRobin": [s.conceptId for s in round_robin],
        "unseenCardIds": sorted(ranking.unseenCardIds),
        "crossSourceCardIds": sorted(ranking.crossSourceCardIds),
        "crossSourceQuestionIds": sorted(ranking.crossSourceQuestionIds),
        "maxCrossSourceItems": ranking.maxCrossSourceItems,
        "maxNewCards": _json_number(ranking.maxNewCards),
        "unknownItemRotation": ranking.unknownItemRotation,
    }


def _concept_state(raw: dict[str, Any]) -> ConceptState:
    return ConceptState(**{k: raw[k] for k in ConceptState.__dataclass_fields__})


def run_build_script(script: dict[str, Any]) -> dict[str, Any]:
    weak = [_concept_state(w) for w in script["weakConcepts"]]
    max_new = script.get("maxNewCards")
    build = SessionBuildState(
        size=script["size"],
        min_first_sight_items=script["minFirstSightItems"],
        max_new_cards=math.inf if max_new is None else max_new,
        max_cross_source_items=script.get("maxCrossSourceItems") or 0,
        unseen_card_ids=script.get("unseenCardIds", []),
        cross_source_card_ids=script.get("crossSourceCardIds", []),
        cross_source_question_ids=script.get("crossSourceQuestionIds", []),
        question_familiarity=script.get("questionFamiliarity", []),
        excluded_card_ids=script.get("excludedCardIds", []),
        excluded_question_ids=script.get("excludedQuestionIds", []),
        excluded_video_ids=script.get("excludedVideoIds", []),
        weak_concepts=weak,
    )
    results: list[dict[str, Any]] = []
    for op in script["ops"]:
        if op["op"] == "add":
            added = build.add_item(dict(op["item"]), bypass_novelty_reservation=bool(op.get("bypassNoveltyReservation")))
            results.append({"op": "add", "id": op["item"]["id"], "added": added})
        elif op["op"] == "pick":
            picked = build.pick_card_candidate(
                op["candidates"], similarity_threshold=op["similarityThreshold"], max_per_cluster=op["maxPerCluster"]
            )
            added = None
            if picked is not None:
                added = build.add_item({
                    "type": "card", "id": picked["id"], "conceptId": op["conceptId"], "conceptName": op["conceptId"],
                    "priority": 1, "interventionReason": "weak_recall", "variantGroupId": picked.get("variantGroupId"),
                })
                if added:
                    build.record_cluster(picked.get("clusterId"))
                    build.record_card_neighborhood(picked)
            results.append({"op": "pick", "picked": picked["id"] if picked else None, "added": added})
        elif op["op"] == "release":
            build.release_first_sight_reservation()
            results.append({"op": "release"})
        else:
            raise ValueError(f"unknown build-script op {op['op']!r}")
    return {
        "results": results,
        "selectedIds": [f"{i['type']}:{i['id']}" for i in build.selected_items],
        "deferredForNovelty": list(build.deferred_for_novelty),
        "selectedCardCount": build.selected_card_count,
        "selectedQuestionCount": build.selected_question_count,
        "newCardCount": build.new_card_count,
        "firstSightSelected": build.first_sight_selected,
        "selectedCrossSourceItems": build.selected_cross_source_items,
        "enforceFirstSightReservation": build.enforce_first_sight_reservation,
        "teachingCaps": {w.conceptId: build.teaching_cap_for(w.conceptId) for w in weak},
        "cardsPerConcept": build.cards_per_concept,
        "questionsPerConcept": build.questions_per_concept,
        "trace": build.trace,
    }


def normalize_inputs(snapshot: dict[str, Any]) -> dict[str, Any]:
    from .ranking import normalize_bulk

    inputs = dict(snapshot)
    inputs["options"] = dict(snapshot.get("options") or {})
    inputs["bulk"] = normalize_bulk({"rotation": snapshot.get("rotation", ""), **(snapshot.get("bulk") or {})})
    inputs["concepts"] = list(snapshot.get("concepts") or [])
    inputs["conceptMap"] = {c["id"]: c for c in inputs["concepts"]}
    inputs["stateMap"] = dict(snapshot.get("stateMap") or {})
    for key in ("recentFailureConceptIds", "excludedCardIds", "excludedQuestionIds", "excludedVideoIds"):
        inputs[key] = set(snapshot.get(key) or [])
    if snapshot.get("penaltyContext") is not None:
        pc = dict(snapshot["penaltyContext"])
        pc["recentCardIds"] = set(pc.get("recentCardIds") or [])
        pc["recentFailureConceptIds"] = set(pc.get("recentFailureConceptIds") or [])
        inputs["penaltyContext"] = pc
    inputs.setdefault("interferenceThreshold", 0.85)
    return inputs


def run_session(snapshot: dict[str, Any]) -> dict[str, Any]:
    """End-to-end constructUnifiedSession over an explicit snapshot (steps 2-30)."""
    from .cluster_session import construct_cluster_session
    from .errors import UnsupportedSchedulerInput
    from .finalize import assemble_session_result, attach_delivery_telemetry
    from .lanes import run_selection
    from .queue import apply_struggle_to_selection, order_session_items

    inputs = normalize_inputs(snapshot)
    if not inputs["concepts"]:
        return {"result": construct_cluster_session(inputs), "trace": []}
    if inputs["bulk"].get("cardGradeById"):
        raise UnsupportedSchedulerInput("repetition-slot shadow telemetry is not ported in Phase 3a")
    ranking = rank_concepts(inputs)
    sort_concepts_by_priority(inputs, ranking)
    pristine = find_pristine_concepts(inputs)
    selection = select_weak_concepts(inputs, ranking, pristine)
    quota = plan_card_quota(inputs, selection.cardRatio)
    plan = {
        "conceptStates": ranking.conceptStates, "statesById": {s.conceptId: s for s in ranking.conceptStates},
        "getConceptCluster": ranking.getConceptCluster, "unseenCardIds": ranking.unseenCardIds,
        "maxNewCards": ranking.maxNewCards, "crossSourceCardIds": ranking.crossSourceCardIds,
        "crossSourceQuestionIds": ranking.crossSourceQuestionIds, "maxCrossSourceItems": ranking.maxCrossSourceItems,
        "unknownItemRotation": ranking.unknownItemRotation, "weakConcepts": selection.weakConcepts,
        "cardRatio": selection.cardRatio, **quota,
        "allocatedConceptRanks": selection.allocatedConceptRanks, "conceptHasPristine": pristine,
    }
    build = run_selection(inputs, plan)
    lane_counts = [{"id": t["id"], "result": t["result"]} for t in build.trace if t["op"] == "lane"]
    selected_ids = [f"{i['type']}:{i['id']}" for i in build.selected_items]
    items = apply_struggle_to_selection(inputs, build)
    final, meta = order_session_items(inputs, plan, build, items)
    attach_delivery_telemetry(inputs, plan, build, final, meta)
    return {"result": assemble_session_result(inputs, plan, build, final), "laneCounts": lane_counts,
            "selectedIds": selected_ids, "trace": build.trace}
