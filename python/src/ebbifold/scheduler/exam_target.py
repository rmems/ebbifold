"""Controlled exam-target treatment (unified-scheduler-phase-concepts/select/
backfill/finalize.ts).

Storage and exam-target-library derivations arrive as explicit snapshot inputs:
``runtimeExamTarget``, ``runtimeTargetSnapshot``, ``effectiveTargetInfluence``,
``targetWorkload`` (computeExamTargetWorkload), ``learnerTargetPolicy``
(buildExamTargetLearnerPolicy output) and ``bulk.examTargetItemScores``.
Mastery-seat lanes (core/surplus target seats, mastery evidence) and the
surplus concept allocator are rejected."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from .errors import UnsupportedSchedulerInput

EMPTY_ACTIVE = {"card": {}, "question": {}, "rankMove": 0}
MASTERY_STAGES = frozenset({"scheduled-atomic-core", "applied-distinction", "breadth-exploration"})


def allocate_mastery_seats(*, candidates: list[dict[str, Any]], eligible_unit_ids_by_stage: dict[str, list[str]],
                           core_target_seats: int = 0, surplus_seats: int = 0) -> dict[str, Any]:
    """Port of exam-target/mastery-seat-allocation.ts over JSON-shaped snapshots."""
    eligible = {stage: set(eligible_unit_ids_by_stage.get(stage) or []) for stage in MASTERY_STAGES}
    selected: list[dict[str, Any]] = []
    used_items: set[str] = set()
    used_units: set[str] = set()

    def select(stage: str, limit: int) -> int:
        count = 0
        for candidate in candidates:
            if count >= limit:
                break
            if (
                candidate.get("stage") != stage
                or candidate.get("unitId") not in eligible[stage]
                or candidate.get("itemKey") in used_items
                or candidate.get("unitId") in used_units
            ):
                continue
            row = {"itemKey": candidate["itemKey"], "unitId": candidate["unitId"], "stage": stage}
            selected.append(row)
            used_items.add(row["itemKey"])
            used_units.add(row["unitId"])
            count += 1
        return count

    core = select("scheduled-atomic-core", max(0, core_target_seats))
    applied = select("applied-distinction", max(0, surplus_seats))
    breadth = select("breadth-exploration", max(0, surplus_seats - applied))
    return {
        "selectedCandidates": selected,
        "coreSelectedCount": core,
        "appliedSelectedCount": applied,
        "breadthSelectedCount": breadth,
        "surplusSelectedCount": applied + breadth,
    }


def assert_exam_target_supported(inputs: dict[str, Any]) -> None:
    if not inputs.get("applyExamTarget"):
        return
    if inputs.get("targetWorkload") is None:
        raise UnsupportedSchedulerInput("exam-target treatment requires an explicit targetWorkload snapshot")


def resolve_learner_policy(inputs: dict[str, Any], states: list[Any]) -> dict[str, Any] | None:
    """Step 4c-4d: learner policy, concept multipliers and active item target scores."""
    policy = None
    if inputs.get("targetSidecarsValid") and inputs.get("runtimeTargetSnapshot"):
        policy = inputs.get("learnerTargetPolicy")
        if policy is None:
            raise UnsupportedSchedulerInput("valid exam-target sidecars require a learnerTargetPolicy snapshot")
    applied = bool(inputs.get("applyExamTarget"))
    active = {"card": {}, "question": {}, "rankMove": 0}
    if applied and policy:
        multipliers = policy.get("conceptMultipliers") or {}
        for state in states:
            value = multipliers.get(state.conceptId)
            state.priority *= 1 if value is None else value
        for key, score in (policy.get("itemPersonalizedScores") or {}).items():
            if key.startswith("card:"):
                active["card"][key[5:]] = score
            elif key.startswith("question:"):
                active["question"][key[9:]] = score
    if applied and inputs.get("runtimeTargetSnapshot"):
        move = (inputs.get("effectiveTargetInfluence") or {}).get("maxItemRankMove")
        active["rankMove"] = 0 if move is None else move
    inputs["learnerTargetPolicyResolved"] = policy
    inputs["examTargetActive"] = active
    return policy


def active_targets(inputs: dict[str, Any]) -> dict[str, Any]:
    return inputs.get("examTargetActive") or EMPTY_ACTIVE


def js_number(value: float) -> str:
    """JSON.stringify for a finite JS number."""
    if not math.isfinite(value):
        raise TypeError("artifact numbers must be finite")
    if value == int(value) and abs(value) < 1e21:
        return str(int(value))
    text = repr(float(value))
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    exp = int(exponent)
    if -7 < exp < 21:
        return f"{float(value):.{max(0, len(mantissa.replace('-', '').replace('.', '')) - 1 - exp)}f}"
    return f"{mantissa}e{'+' if exp > 0 else '-'}{abs(exp)}"


def canonical_json(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        import json

        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return js_number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{canonical_json(k)}:{canonical_json(value[k])}" for k in sorted(value)) + "}"
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def hash_exam_target_artifact(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _bypass_reason(inputs: dict[str, Any], mode: str, assignment: Any) -> str | None:
    if not inputs.get("targetSidecarsValid"):
        return "invalid-or-missing-target-sidecars"
    allocator = inputs["runtimeTargetSnapshot"]["definition"]["influence"].get("allocator")
    if allocator == "shadow" and not inputs.get("examTargetEvaluationOnly"):
        return "target-authority-shadow-only"
    if mode == "shadow":
        return "shadow-serves-control"
    if assignment == "control":
        return "control-assignment"
    return None if inputs.get("applyExamTarget") else "target-not-applied"


def _trace_fields(score: dict[str, Any], domain: dict[str, Any], personalized: float) -> dict[str, Any]:
    return {
        "targetDomainCode": score["domainCode"],
        "embeddingHash": score["embeddingHash"],
        "examRelevancePct": score["fitPercentile"],
        "examDomainWeight": score["effectiveDomainWeight"],
        "userDomainGap": domain["userDomainGap"],
        "contentTargetScore": score["itemTargetIndex"],
        "personalizedTargetScore": personalized,
        "targetWeightProvenance": score["weightProvenance"],
    }


def _lookup(policy: dict[str, Any] | None, scores: dict[str, Any], key: str):
    score = scores.get(key)
    domain = (policy.get("desiredDomains") or {}).get(score["domainCode"]) if score and policy else None
    personalized = (policy.get("itemPersonalizedScores") or {}).get(key) if policy else None
    return score, domain, personalized


def _item_traces(policy: dict[str, Any] | None, scores: dict[str, Any], final: list[dict[str, Any]]) -> dict[str, Any]:
    traces: dict[str, Any] = {}
    if not policy:
        return traces
    for item in final:
        if item["type"] not in ("card", "question"):
            continue
        key = f"{item['type']}:{item['id']}"
        score, domain, personalized = _lookup(policy, scores, key)
        if score and domain and personalized is not None:
            fields = _trace_fields(score, domain, personalized)
            traces[key] = {"targetDomainCode": fields.pop("targetDomainCode"),
                           "sourceRotation": score["sourceRotation"], **fields}
    return traces


def _candidate_pool(policy: dict[str, Any] | None, scores: dict[str, Any], bulk: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [(f"card:{c['id']}", c.get("rotation")) for c in [*bulk["unseenCards"], *bulk["seenCards"]]]
    candidates += [(f"question:{q['id']}", q.get("rotation")) for q in bulk["rotationQuestions"]]
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    empty = dict.fromkeys(("targetDomainCode", "embeddingHash", "examRelevancePct", "examDomainWeight",
                           "userDomainGap", "contentTargetScore", "personalizedTargetScore",
                           "targetWeightProvenance"))
    for key, rotation in candidates:
        if key in seen:
            continue
        seen.add(key)
        score, domain, personalized = _lookup(policy, scores, key)
        eligible = bool(score and domain and personalized is not None)
        fields = _trace_fields(score, domain, personalized) if eligible else empty
        pool.append({"itemKey": key, "sourceRotation": rotation, "baseRank": len(pool),
                     "targetEligible": eligible, **fields})
    return pool


def _allocation_error(inputs: dict[str, Any], policy: dict[str, Any] | None, scores: dict[str, Any],
                      final: list[dict[str, Any]]) -> float | None:
    if not inputs.get("applyExamTarget") or not policy:
        return None
    discretionary = [i for i in final if i["type"] in ("card", "question")
                     and i.get("examTargetMasteryStage") != "scheduled-atomic-core"]
    if not discretionary:
        return None
    observed: dict[str, int] = {}
    for item in discretionary:
        score = scores.get(f"{item['type']}:{item['id']}")
        if score:
            observed[score["domainCode"]] = observed.get(score["domainCode"], 0) + 1
    error = sum(abs(observed.get(code, 0) / len(discretionary) - domain["desiredShare"])
                for code, domain in (policy.get("desiredDomains") or {}).items()) / 2
    return min(1, max(0, math.floor(error * 1_000_000 + 0.5) / 1_000_000))


def _pressure_bucket(pressure: float) -> str:
    if pressure >= 0.8:
        return "crunch"
    if pressure >= 0.5:
        return "near"
    return "building" if pressure >= 0.2 else "far"


def exam_target_decision(inputs: dict[str, Any], plan: dict[str, Any],
                         final: list[dict[str, Any]], build: Any = None) -> dict[str, Any] | None:
    runtime = inputs.get("runtimeExamTarget") or {}
    snapshot = inputs.get("runtimeTargetSnapshot")
    if not (inputs.get("targetComputeRequested") and snapshot and runtime.get("schedulerVersion")
            and runtime.get("policyDigest") and runtime.get("activationRevision") is not None):
        return None
    policy = inputs.get("learnerTargetPolicyResolved")
    scores = inputs["bulk"].get("examTargetItemScores") or {}
    mode = "active" if runtime["resolved"].get("effectiveMode") == "active" else "shadow"
    assignment = runtime["resolved"].get("assignment")
    workload = inputs.get("targetWorkload")
    influence = inputs.get("effectiveTargetInfluence") or {}
    learner_state = {
        "schema": "md3.exam-target-learner-state/v1",
        "concepts": sorted(({"conceptId": s.conceptId, "recallOnExamDay": s.recallOnExamDay,
                             "confidence": s.confidence} for s in plan["conceptStates"]),
                           key=lambda c: c["conceptId"]),
        "completedCoreWorkToday": (inputs.get("masteryEvidence") or {}).get("completedCoreWorkToday") or 0,
        "remainingTargetWork": (inputs.get("masteryEvidence") or {}).get("remainingTargetWork"),
    }
    move = influence.get("maxItemRankMove")
    return {
        "targetSnapshotId": snapshot["id"], "targetVersion": snapshot["targetVersion"],
        "targetBasis": snapshot["targetBasis"], "targetScorerVersion": snapshot["scorerVersion"],
        "schedulerVersion": runtime["schedulerVersion"], "policyDigest": runtime["policyDigest"],
        "activationRevision": runtime["activationRevision"], "mode": mode, "assignment": assignment,
        "applied": bool(inputs.get("applyExamTarget")), "bypassReason": _bypass_reason(inputs, mode, assignment),
        "maxItemRankMove": 0 if move is None else move,
        "unmappedDomainCodes": list((policy or {}).get("unmappedDomainCodes") or []),
        "masteryPolicyVersion": ((inputs.get("masteryEvidence") or {}).get("masteryPlan") or {}).get("policyVersion"),
        "workload": dict(workload) if workload else None,
        "coreTargetSeatsSelected": getattr(build, "selected_core_target_seats", 0),
        "coreTargetSeatShortfall": max(0, ((workload or {}).get("coreTargetSeats") or 0) - getattr(build, "selected_core_target_seats", 0)),
        "surplusTargetSeatsSelected": getattr(build, "selected_surplus_target_seats", 0),
        "allocationChangedConceptMembershipCount": 0,
        "allocationCoverageDebtDomainCodes": [],
        "targetAllocationError": _allocation_error(inputs, policy, scores, final),
        "evaluationOnly": bool(inputs.get("examTargetEvaluationOnly")), "curriculumCoverageDebtCount": None,
        "masteryLoadWarnings": [], "daysToExam": max(0, math.ceil(inputs["daysToExam"])),
        "pressureBucket": _pressure_bucket(inputs["examPressure"]),
        "learnerStateVersion": hash_exam_target_artifact(learner_state),
        "candidatePool": _candidate_pool(policy, scores, inputs["bulk"]),
        "itemTraces": _item_traces(policy, scores, final),
    }


def exam_target_stats(inputs: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    workload = decision["workload"] or {}
    return {
        "examTargetComputed": inputs.get("learnerTargetPolicyResolved") is not None,
        "examTargetApplied": decision["applied"],
        "examTargetVersion": decision["targetVersion"],
        "examTargetBypassReason": decision["bypassReason"],
        "examTargetCoreSeatsRequired": workload.get("coreTargetSeats") or 0,
        "examTargetCoreSeatsSelected": decision["coreTargetSeatsSelected"],
        "examTargetCoreSeatShortfall": decision["coreTargetSeatShortfall"],
        "examTargetSurplusSeats": workload.get("surplusSeats") or 0,
        "examTargetRemainingWork": (inputs.get("masteryEvidence") or {}).get("remainingTargetWork"),
        "examTargetWorkloadShortfall": workload.get("shortfall"),
    }
