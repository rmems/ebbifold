"""Finalize phase (unified-scheduler-phase-finalize.ts): delivery telemetry and
result assembly. Exam-target decisions only exist under exam-target treatment,
which ``concepts.assert_supported`` rejects; serve conditioning needs a rating
reliability record, which must be absent from the snapshot."""

from __future__ import annotations

import math
from typing import Any

from .difficulty import get_effective_question_difficulty
from .errors import UnsupportedSchedulerInput
from .exam_target import exam_target_decision, exam_target_stats
from .policies import recent_neighbor_signal
from .queue import compute_concept_pairing_rate
from .scoring import is_calibrated_review_item, is_synthetic_concept_attribution

CHALLENGE_POLICY_VERSION = "frontier-then-stepdown-v1"
RECENT_NEIGHBOR_POLICY_VERSION = "recent-card-neighbor-v1"
ITEM_RECALL_MODEL_VERSION = "item-facility-v1"
ITEM_RECALL_MODEL_STATUS = "telemetry-only-unvalidated"
_CHALLENGE_BYPASS = frozenset({"failure_escalation", "stuck_intervention", "pre_teach", "pre_teach_naive",
                               "chronic_stuck_mcq", "mcq_bridge_card", "preemptive_scaffold"})
_TIER_RANK = {"scaffolding": 0, "standard": 1, "stretch": 2}


def tier_from_complexity(complexity: Any) -> str | None:
    if complexity is None:
        return None
    return "scaffolding" if complexity <= 1 else "standard" if complexity == 2 else "stretch"


def tier_from_question_difficulty(difficulty: Any) -> str | None:
    if not isinstance(difficulty, str):
        return None
    return {"easy": "scaffolding", "medium": "standard", "hard": "stretch"}.get(difficulty.strip().lower())


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def estimate_item_recall(*, concept_recall: float, concept_exposure_count: float, item_type: str,
                         facility_index: Any = None, sample_size: Any = None, difficulty: Any = None) -> dict[str, Any]:
    recall = max(0, min(1, concept_recall if _finite(concept_recall) else 0))
    exposures = max(0, concept_exposure_count if _finite(concept_exposure_count) else 0)
    if item_type == "card":
        fallback = 0.46
    elif item_type == "question":
        fallback = 0.71 if isinstance(difficulty, str) and difficulty.lower() == "easy" else 0.59
    else:
        fallback = 0.5
    valid = _finite(facility_index) and 0 <= facility_index <= 1
    samples = max(0, sample_size) if valid and _finite(sample_size) else 0
    evidence = samples / (samples + 20)
    facility = fallback + evidence * (facility_index - fallback) if valid and samples > 0 else fallback
    concept_evidence = exposures / (exposures + 3)
    weight = min(0.75, 0.35 + 0.30 * evidence + 0.10 * (1 - concept_evidence))
    predicted = max(0.05, min(0.95, recall * (1 - weight) + facility * weight))
    return {"predictedRecall": predicted, "source": "empirical" if valid and samples > 0 else "fallback"}


def attach_delivery_telemetry(inputs: dict[str, Any], plan: dict[str, Any], build: Any,
                              final: list[dict[str, Any]], card_meta: dict[str, dict[str, Any]]) -> None:
    if inputs.get("ratingReliability"):
        raise UnsupportedSchedulerInput("serve conditioning (rating reliability) is not ported in Phase 3a")
    bulk = inputs["bulk"]
    unseen = {c["id"] for c in bulk["unseenCards"]}
    recent_ids = (inputs.get("penaltyContext") or {}).get("recentCardIds")
    for item in final:
        raw = inputs["stateMap"].get(item["conceptId"]) or {}
        state = plan["statesById"].get(item["conceptId"])
        serve_recall = state.currentRecall if state else raw.get("recallProbability", 0)
        serve_exposures = state.exposureCount if state else raw.get("exposureCount", 0)
        q_meta = bulk["questionMap"].get(item["id"]) if item["type"] == "question" else None
        c_meta = card_meta.get(item["id"]) if item["type"] == "card" else None
        applied = item.get("challengePolicyApplied") is True
        item.pop("challengePolicyApplied", None)
        descriptive = state.currentRecall if state else raw.get("recallProbability")
        challenge_recall = build.challenge_recall_at_selection.get(item["id"]) if applied else descriptive
        if item["type"] == "card":
            tier = tier_from_complexity(c_meta["complexity"] if c_meta and c_meta.get("complexity") is not None
                                        else item.get("complexity"))
        elif item["type"] == "question" and q_meta:
            tier = tier_from_question_difficulty(get_effective_question_difficulty(
                q_meta.get("difficulty"), q_meta.get("facilityIndex"), q_meta.get("totalAttempts")))
        else:
            tier = None
        item["difficultyTier"] = tier
        synthetic = is_synthetic_concept_attribution(item["conceptId"])
        bypass = item.get("interventionReason") in _CHALLENGE_BYPASS
        if item["type"] in ("card", "question") and not synthetic and not bypass:
            target = "stretch" if _finite(challenge_recall) else None
            if target and tier:
                item["challengePolicyVersion"] = CHALLENGE_POLICY_VERSION
                item["challengeTargetTier"] = target
                item["challengeDistance"] = abs(_TIER_RANK[tier] - _TIER_RANK[target])
                item["challengePolicyApplied"] = item["type"] == "card" and applied
        if (item["type"] == "card" and applied and not synthetic and item["id"] in unseen
                and item["conceptId"] not in inputs["recentFailureConceptIds"] and not bypass):
            novelty = recent_neighbor_signal((c_meta or {}).get("similarCards"), recent_ids)
            item["noveltyPolicyVersion"] = RECENT_NEIGHBOR_POLICY_VERSION
            item["recentNeighborSimilarity"] = novelty["maxSimilarity"]
            item["noveltyPenalty"] = novelty["penalty"]
        complexity = (c_meta or {}).get("complexity", item.get("complexity"))
        if is_calibrated_review_item(item["type"], item.get("complexity")):
            facility = (c_meta or {}).get("facilityIndex")
            if facility is None:
                facility = (q_meta or {}).get("facilityIndex")
            samples = (c_meta or {}).get("sampleSize")
            if samples is None:
                samples = (q_meta or {}).get("totalAttempts")
            est = estimate_item_recall(concept_recall=serve_recall, concept_exposure_count=serve_exposures,
                                       item_type=item["type"], facility_index=facility, sample_size=samples,
                                       difficulty=(q_meta or {}).get("difficulty") or item.get("difficulty"))
            del complexity
            item["predictedRecall"] = est["predictedRecall"]
            item["predictedRecallModel"] = ITEM_RECALL_MODEL_VERSION
            item["predictedRecallSource"] = est["source"]
            item["predictedRecallStatus"] = ITEM_RECALL_MODEL_STATUS


def assemble_session_result(inputs: dict[str, Any], plan: dict[str, Any], build: Any,
                            final: list[dict[str, Any]]) -> dict[str, Any]:
    items = [{**item, "firstSightAtSelection": build.is_first_sight_item(item)} for item in final]
    avg = sum(i["priority"] for i in final) / len(final) if final else 0
    budget = inputs.get("budget") or {"dailyThroughput": 0, "totalRemaining": 0, "sessionsRemaining": 0}
    decision = exam_target_decision(inputs, plan, final)
    result = {
        "items": items,
        "repetitionSlotContext": None,
        "noveltyQuota": {"required": build.min_first_sight_items,
                         "selected": sum(1 for i in items if i["firstSightAtSelection"])},
        "stats": {
            "totalConcepts": len(inputs["concepts"]),
            "weakConcepts": len(plan["weakConcepts"]),
            "selectedConcepts": len(build.selected_concept_ids),
            "cardCount": sum(1 for i in items if i["type"] == "card"),
            "questionCount": sum(1 for i in items if i["type"] == "question"),
            "averagePriority": js_round(avg * 1000) / 1000,
            "examPressure": js_round(inputs["examPressure"] * 100) / 100,
            "conceptPairingRate": compute_concept_pairing_rate(final),
            "dailyThroughput": js_round(budget["dailyThroughput"]),
            "totalBudgetRemaining": js_round(budget["totalRemaining"]),
            "sessionsRemaining": budget["sessionsRemaining"],
        },
    }
    if decision is not None:
        result["stats"].update(exam_target_stats(inputs, decision))
        result["examTargetDecision"] = decision
    return result


def js_round(value: float) -> float:
    return math.floor(value + 0.5)
