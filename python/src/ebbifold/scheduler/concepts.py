"""Concept ranking, weak-concept selection and quota planning.

Ports unified-scheduler-phase-concepts.ts (rankConcepts, sortConceptsByPriority,
findPristineConcepts, selectWeakConcepts, planCardQuota). The input is an
explicit snapshot dict shaped like the TypeScript ``SchedulerInputs`` (camelCase
keys); storage reads happen upstream and are out of scope.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Callable

from . import math as smath
from .config import DEFAULTS, RECALL_RANKING_HORIZON_DAYS, stable_target_random
from .errors import UnsupportedSchedulerInput
from .exam_target import (
    allocate_exam_target_concept_order,
    assert_exam_target_supported,
    resolve_learner_policy,
)
from .scoring import (
    PRACTICE_MISS_BOOST,
    cluster_exposure_dampening,
    compute_recall_soft_penalty,
    normalize_review_challenge_level,
)
from .topics import (
    dated_topic_recency_concept_boost,
    expand_topic_set,
    teaching_cadence_concept_boost,
    teaching_week_for_topics,
)

MS_PER_DAY = 1000 * 60 * 60 * 24


@dataclass
class ConceptState:
    conceptId: str
    conceptName: str
    currentRecall: float
    recallOnExamDay: float
    confidence: float
    exposureCount: float
    daysSinceProbe: float
    daysSinceExposure: float
    priority: float
    intervention: str

    def to_json(self) -> dict[str, Any]:
        return {k: _json_number(v) for k, v in asdict(self).items()}


def _json_number(value: Any) -> Any:
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def assert_supported(inputs: dict[str, Any]) -> None:
    """Fail closed on lanes Phase 3a does not port."""
    assert_exam_target_supported(inputs)


@dataclass
class ConceptRanking:
    conceptStates: list[ConceptState]
    statesById: dict[str, ConceptState]
    getConceptCluster: Callable[[ConceptState], str]
    unseenCardIds: set[str]
    maxNewCards: float
    crossSourceCardIds: set[str]
    crossSourceQuestionIds: set[str]
    maxCrossSourceItems: int
    unknownItemRotation: str


def _days_since(now_ms: float, at_ms: float | None) -> float:
    return math.inf if at_ms is None else (now_ms - at_ms) / MS_PER_DAY


def _concept_intervention(
    confidence: float, exposure_count: int, recall_on_exam_day: float, days_since_probe: float
) -> str:
    if confidence < DEFAULTS["confidence_threshold"] or exposure_count < 3:
        return "probe"
    if recall_on_exam_day < DEFAULTS["recall_threshold"]:
        return "remediate"
    if days_since_probe > DEFAULTS["days_since_probe_threshold"]:
        return "probe"
    return "reinforce"


def _concept_gap_alignment(inputs: dict[str, Any], concept_id: str, exam_pressure: float) -> float:
    embeddings = inputs.get("conceptEmbeddings", {})
    gap_direction = inputs.get("gapDirection")
    raw_emb = embeddings.get(concept_id)
    concept_emb = smath.truncate_to_manifold_dim(raw_emb) if raw_emb else None
    if inputs.get("applyExamTarget") or not concept_emb or not gap_direction:
        return 0
    return max(0, smath.cosine_similarity(concept_emb, gap_direction)) * (0.15 + exam_pressure * 0.15)


def _initial_concept_state(inputs: dict[str, Any], concept: dict[str, Any]) -> ConceptState:
    options = inputs.get("options", {})
    now_ms = inputs["nowMs"]
    state = inputs.get("stateMap", {}).get(concept["id"])
    days_since_probe = _days_since(now_ms, state.get("lastProbeAt") if state else None)
    days_since_exposure = _days_since(now_ms, state.get("lastExposureAt") if state else None)
    stored_recall = state.get("recallProbability", 0) if state else 0
    confidence = state.get("confidence", 0) if state else 0
    exposure_count = state.get("exposureCount", 0) if state else 0
    current_recall = (
        smath.apply_decay(stored_recall, days_since_exposure, confidence)
        if days_since_exposure < math.inf
        else 0
    )
    days_to_exam = inputs["daysToExam"]
    recall_on_exam_day = smath.project_recall_to_exam_day(current_recall, days_to_exam, confidence)
    ranking_recall = smath.project_recall_to_exam_day(
        current_recall, days_to_exam, confidence, RECALL_RANKING_HORIZON_DAYS
    )
    intervention = _concept_intervention(confidence, exposure_count, recall_on_exam_day, days_since_probe)

    gap_score = max(0, DEFAULTS["target_recall"] - ranking_recall)
    confidence_boost = 0.2 if confidence < DEFAULTS["confidence_threshold"] else 0
    stale_boost = 0.1 if days_since_probe > DEFAULTS["days_since_probe_threshold"] else 0
    gap_alignment = _concept_gap_alignment(inputs, concept["id"], inputs["examPressure"])
    liked_boost = 0.15 if concept["id"] in set(inputs.get("likedConceptIds", [])) else 0
    failure_boost = 0.25 if concept["id"] in set(inputs.get("recentFailureConceptIds", [])) else 0
    practice_boost = PRACTICE_MISS_BOOST if concept["id"] in set(inputs.get("practiceMissConceptIds", [])) else 0
    cadence = teaching_cadence_concept_boost(
        teaching_week_for_topics(concept["topics"], options.get("topicTeachingWeeks")),
        options.get("currentTeachingWeek"),
    )
    dated = dated_topic_recency_concept_boost(concept["topics"], options.get("topicRecencyDays"))
    exam_weight_multiplier = 1 if inputs.get("applyExamTarget") else (concept.get("examWeight") or 1) / 3
    soft_penalty = compute_recall_soft_penalty(
        current_recall, confidence, is_chronic_failure=concept["id"] in set(inputs.get("chronicFailureConceptIds", []))
    )
    priority = (
        (gap_score + confidence_boost + stale_boost + gap_alignment + liked_boost + failure_boost
         + practice_boost + max(cadence, dated))
        * exam_weight_multiplier
        * soft_penalty
    )
    return ConceptState(
        conceptId=concept["id"], conceptName=concept["name"], currentRecall=current_recall,
        recallOnExamDay=recall_on_exam_day, confidence=confidence, exposureCount=exposure_count,
        daysSinceProbe=days_since_probe, daysSinceExposure=days_since_exposure,
        priority=priority, intervention=intervention,
    )


def _boost_one_prerequisite_chain(
    concept: dict[str, Any], state: ConceptState, states_by_id: dict[str, ConceptState],
    concept_by_id: dict[str, dict[str, Any]],
) -> None:
    base_boost = state.priority * 0.85
    if base_boost <= 0:
        return
    visited = {concept["id"]}
    queue = [(pid, 1) for pid in concept.get("prerequisiteIds", [])]
    while queue:
        next_id, depth = queue.pop(0)
        if next_id in visited:
            continue
        visited.add(next_id)
        prereq = states_by_id.get(next_id)
        if prereq is None:
            continue
        boosted = base_boost * math.pow(0.7, depth - 1)
        solid = (
            prereq.recallOnExamDay >= DEFAULTS["target_recall"]
            and prereq.confidence >= DEFAULTS["confidence_threshold"]
        )
        if not solid:
            prereq.priority = max(prereq.priority, boosted)
        if depth < 3:
            prereq_concept = concept_by_id.get(next_id)
            if prereq_concept is None:
                continue
            for pid in prereq_concept.get("prerequisiteIds", []):
                if pid not in visited:
                    queue.append((pid, depth + 1))


def _apply_prerequisite_boosts(
    concepts: list[dict[str, Any]], states_by_id: dict[str, ConceptState],
    concept_by_id: dict[str, dict[str, Any]],
) -> None:
    for concept in concepts:
        state = states_by_id.get(concept["id"])
        if state is None or state.recallOnExamDay >= DEFAULTS["target_recall"]:
            continue
        _boost_one_prerequisite_chain(concept, state, states_by_id, concept_by_id)


def _apply_hub_readiness_dampening(
    concepts: list[dict[str, Any]], states: list[ConceptState], states_by_id: dict[str, ConceptState]
) -> None:
    prereq_recalls = {s.conceptId: s.recallOnExamDay for s in states}
    for concept in concepts:
        prereqs = concept.get("prerequisiteIds", [])
        if not prereqs:
            continue
        state = states_by_id.get(concept["id"])
        if state is None:
            continue
        ready, dampening = smath.compute_hub_readiness(prereqs, prereq_recalls)
        if not ready:
            state.priority *= dampening


def _topic_to_cluster(cards: list[dict[str, Any]]) -> dict[str, str]:
    topic_to_cluster: dict[str, str] = {}
    for card in cards:
        if not card.get("clusterId"):
            continue
        for topic in card["topics"]:
            topic_to_cluster.setdefault(topic, card["clusterId"])
    return topic_to_cluster


def _cluster_lookup(concept_by_id: dict[str, dict[str, Any]], topic_to_cluster: dict[str, str]) -> Any:
    def get_concept_cluster(cs: ConceptState) -> str:
        data = concept_by_id.get(cs.conceptId)
        if data is None:
            return cs.conceptId
        for topic in data["topics"]:
            cluster = topic_to_cluster.get(topic)
            if cluster:
                return cluster
        return cs.conceptId

    return get_concept_cluster


def rank_concepts(inputs: dict[str, Any]) -> ConceptRanking:
    assert_supported(inputs)
    options = inputs.get("options", {})
    bulk = inputs["bulk"]
    concepts = inputs["concepts"]
    states = [_initial_concept_state(inputs, concept) for concept in concepts]

    resolve_learner_policy(inputs, states)
    states_by_id = {s.conceptId: s for s in states}
    concept_by_id = {c["id"]: c for c in concepts}

    # Prerequisite boost: weak concepts lift their unsolid prerequisites (BFS, depth <= 3).
    _apply_prerequisite_boosts(concepts, states_by_id, concept_by_id)

    _apply_hub_readiness_dampening(concepts, states, states_by_id)

    all_cards = [*bulk.get("unseenCards", []), *bulk.get("seenCards", [])]
    get_concept_cluster = _cluster_lookup(concept_by_id, _topic_to_cluster(all_cards))
    recent_cluster = inputs.get("recentClusterExposures") or {}
    if recent_cluster and sum(recent_cluster.values()) > 0:
        for state in states:
            state.priority *= cluster_exposure_dampening(recent_cluster.get(get_concept_cluster(state), 0))

    cross_rotations = set(options.get("crossSourceRotations") or [])
    card_rotation = {c["id"]: c.get("rotation") for c in all_cards}
    question_rotation = {q["id"]: q.get("rotation") for q in bulk.get("rotationQuestions", [])}
    max_new = inputs.get("effectiveMaxNewCards")
    return ConceptRanking(
        conceptStates=states,
        statesById=states_by_id,
        getConceptCluster=get_concept_cluster,
        unseenCardIds={c["id"] for c in bulk.get("unseenCards", [])},
        maxNewCards=math.inf if max_new is None else max_new,
        crossSourceCardIds={cid for cid, rot in card_rotation.items() if rot in cross_rotations},
        crossSourceQuestionIds={qid for qid, rot in question_rotation.items() if rot in cross_rotations},
        maxCrossSourceItems=max(0, math.floor(options.get("maxCrossSourceItems") or 0)),
        unknownItemRotation=(
            inputs.get("rotation") if not cross_rotations else "__outside-authorized-candidate-scope__"
        ),
    )


def sort_concepts_by_priority(inputs: dict[str, Any], ranking: ConceptRanking) -> None:
    seed = inputs.get("options", {}).get("examTargetTieBreakSeed")
    for state in ranking.conceptStates:
        jitter = (stable_target_random(seed, state.conceptId) - 0.5) * 0.6 * max(state.priority, 0.05)
        state.priority += jitter
    ranking.conceptStates.sort(key=lambda s: -s.priority)


def find_pristine_concepts(inputs: dict[str, Any]) -> set[str]:
    bulk = inputs["bulk"]
    deferred = set(bulk.get("passSetDeferredCardIds") or [])
    pristine: set[str] = set()
    for concept in inputs["concepts"]:
        topic_set = set(expand_topic_set(concept["topics"]))
        for card in bulk.get("unseenCards", []):
            if card["id"] in deferred:
                continue
            if any(t in topic_set for t in card["topics"]):
                pristine.add(concept["id"])
                break
    return pristine


@dataclass
class ConceptSelection:
    weakConcepts: list[ConceptState]
    cardRatio: float
    maintenanceShift: float
    allocatedConceptRanks: dict[str, int] | None = None
    allocationChangedConceptMembershipCount: int = 0
    allocationCoverageDebtDomainCodes: list[str] | None = None


def select_weak_concepts(
    inputs: dict[str, Any], ranking: ConceptRanking, concept_has_pristine: set[str]
) -> ConceptSelection:
    options = inputs.get("options", {})
    card_ratio = inputs["cardRatio"]
    target = DEFAULTS["target_recall"]
    weak = [
        s for s in ranking.conceptStates
        if s.recallOnExamDay < target or s.conceptId in concept_has_pristine
    ]
    budget_filtered = [
        c for c in weak
        if smath.compute_exposures_needed(c.currentRecall, target, c.confidence, inputs["daysToExam"]) > 0
        or c.conceptId in concept_has_pristine
    ]
    if budget_filtered:
        weak = budget_filtered
    allocated_ranks = None
    allocation_changed = 0
    coverage_debt: list[str] = []
    policy = inputs.get("learnerTargetPolicyResolved")
    workload = inputs.get("targetWorkload") or {}
    if (inputs.get("applyExamTarget") and policy and inputs.get("runtimeTargetSnapshot")
            and (workload.get("surplusSeats") or 0) > 0 and weak):
        desired = {
            code: (domain or {}).get("desiredShare", 0)
            for code, domain in (policy.get("desiredDomains") or {}).items()
        }
        snapshot_influence = ((inputs.get("runtimeTargetSnapshot") or {}).get("definition") or {}).get("influence") or {}
        allocated = allocate_exam_target_concept_order(
            concepts=weak, protected_concept_ids=set(inputs.get("recentFailureConceptIds") or []),
            requested_discretionary_size=workload.get("surplusSeats") or 0,
            concept_mappings=(inputs.get("conceptTargetScores") or {}).get("scores") or {},
            desired_shares=desired,
            observed_domain_counts=inputs.get("observedExamTargetDomainCounts") or {},
            influence=inputs.get("effectiveTargetInfluence") or snapshot_influence,
        )
        weak = allocated["concepts"]
        allocated_ranks = {state.conceptId: rank for rank, state in enumerate(weak)}
        allocation_changed = allocated["changedMembershipCount"]
        coverage_debt = allocated["coverageDebtDomainCodes"]

    focus = options.get("simulationFocus")
    if focus:
        from .simulation import prefer_simulation_topics

        recent = set(inputs.get("recentFailureConceptIds", []))
        concept_map = inputs["conceptMap"]
        preferred = iter(prefer_simulation_topics(
            [s for s in weak if s.conceptId not in recent],
            lambda s: (concept_map.get(s.conceptId) or {}).get("topics", []), focus))
        weak = [s if s.conceptId in recent else next(preferred) for s in weak]
        allocated_ranks = {s.conceptId: rank for rank, s in enumerate(weak)}
    shift = 0.0
    if options.get("cardRatio") is None and normalize_review_challenge_level(options.get("reviewChallenge")) == 0:
        _, breadth_ratio = smath.compute_knowledge_breadth([s.recallOnExamDay for s in ranking.conceptStates])
        shift = smath.compute_maintenance_shift(breadth_ratio, len(ranking.conceptStates))
        if shift > 0:
            card_ratio = min(0.9, card_ratio + shift)
    return ConceptSelection(weakConcepts=weak, cardRatio=card_ratio, maintenanceShift=shift,
                            allocatedConceptRanks=allocated_ranks,
                            allocationChangedConceptMembershipCount=allocation_changed,
                            allocationCoverageDebtDomainCodes=coverage_debt)


def plan_card_quota(inputs: dict[str, Any], card_ratio: float) -> dict[str, int]:
    options = inputs.get("options", {})
    size = inputs["size"]
    challenge = normalize_review_challenge_level(options.get("reviewChallenge"))
    if challenge == 2:
        target_cards = 0
    elif challenge == -2:
        target_cards = size
    else:
        core_seats = (inputs.get("targetWorkload") or {}).get("coreTargetSeats")
        target_core = (size if core_seats is None else core_seats) if inputs.get("applyExamTarget") else 0
        target_cards = min(size, max(math.ceil(size * card_ratio), target_core))
    coverage = (
        0
        if options.get("mode") == "crunch" or inputs.get("effectiveMaxNewCards") == 0
        else min(2, max(0, math.floor(target_cards * 0.3)))
    )
    return {
        "targetCardCount": target_cards,
        "targetQuestionCount": size - target_cards,
        "coverageReservation": coverage,
        "targetWeakCardCount": max(0, target_cards - coverage),
    }
