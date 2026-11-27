"""Pure scoring helpers (unified-scheduler-scoring.ts, intervention-reason.ts,
concept-teaching-state.ts, synthetic-concept.ts)."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .topics import js_trim

CALIBRATION_ITEM_PENALTY_THRESHOLD = 0.9
PRACTICE_MISS_BOOST = 0.2
HIGH_RECALL_PRIORITY_MULTIPLIER = 0.4
CLUSTER_FREE_EXPOSURES = 4
CLUSTER_DAMPEN_FLOOR = 0.25


def resolve_max_new_cards(explicit: float | None = None, mode: str | None = None) -> float:
    if explicit is not None:
        return explicit
    if mode == "crunch":
        return 0
    return math.inf


def compute_exam_pressure(days_to_exam: float) -> float:
    return 1 / (1 + math.exp(-0.15 * (21 - days_to_exam)))


def is_calibrated_review_item(item_type: str, complexity: int | None = None) -> bool:
    if item_type == "question":
        return True
    if item_type == "card":
        return complexity != 1
    return False


def is_synthetic_concept_attribution(concept_id: str) -> bool:
    normalized = js_trim(concept_id).lower()
    return (
        normalized in ("", "_unattached", "unattached")
        or normalized.startswith("rotation:")
        or normalized.startswith("question:")
    )


def compute_recall_soft_penalty(
    current_recall: float, confidence: float, is_chronic_failure: bool = False
) -> float:
    if is_chronic_failure:
        return 1
    if current_recall <= CALIBRATION_ITEM_PENALTY_THRESHOLD:
        return 1
    if confidence < 0.4:
        return 1
    return HIGH_RECALL_PRIORITY_MULTIPLIER


def cluster_exposure_dampening(cluster_count: float) -> float:
    excess = max(0, cluster_count - CLUSTER_FREE_EXPOSURES)
    return max(CLUSTER_DAMPEN_FLOOR, 0.82**excess)


def build_cluster_round_robin(
    concepts: Sequence[Any],
    get_cluster: Callable[[Any], str],
    target_order_rank: Callable[[Any], int | None] | None = None,
) -> list[Any]:
    groups: dict[str, list[Any]] = {}
    for concept in concepts:
        groups.setdefault(get_cluster(concept), []).append(concept)

    import functools

    def compare(a: list[Any], b: list[Any]) -> float:
        if target_order_rank is not None:
            left, right = target_order_rank(a[0]), target_order_rank(b[0])
            if left is not None and right is not None and left != right:
                return left - right
        return b[0].priority - a[0].priority

    ordered = sorted((g for g in groups.values() if g), key=functools.cmp_to_key(compare))
    result: list[Any] = []
    depth = 0
    while True:
        has_more = False
        for group in ordered:
            if depth < len(group):
                result.append(group[depth])
                if depth + 1 < len(group):
                    has_more = True
        depth += 1
        if not has_more:
            break
    return result


def _remove_teaching_where(
    indexes: list[int], predicate: Callable[[int], bool], removed: set[int], remaining: int,
) -> int:
    for index in reversed(indexes):
        if remaining <= 0:
            break
        if index in removed or not predicate(index):
            continue
        removed.add(index)
        remaining -= 1
    return remaining


def _ordinary_teaching_removable(
    items: Sequence[Mapping[str, Any]], index: int, scaffold: set[int], mastery_core: set[int],
    protected_ids: set[Any], preserve: Callable[[Mapping[str, Any]], bool] | None,
) -> bool:
    if index in scaffold or index in mastery_core:
        return False
    preserved = bool(preserve and preserve(items[index]))
    return items[index]["id"] not in protected_ids and not preserved


def _teaching_cap_removals(
    items: Sequence[Mapping[str, Any]], indexes: list[int], cap_value: float,
    preserve: Callable[[Mapping[str, Any]], bool] | None = None,
) -> set[int]:
    cap = max(0, math.floor(cap_value))
    remaining = len(indexes) - cap
    if remaining <= 0:
        return set()

    def struggle(i: int) -> Mapping[str, Any]:
        return items[i].get("struggleIntervention") or {}

    scaffold = {i for i in indexes if struggle(i).get("isScaffold") is True}
    mastery_core = {i for i in indexes if items[i].get("examTargetMasteryStage") == "scheduled-atomic-core"}
    protected_ids = {struggle(i).get("targetCardId") for i in scaffold} - {None, ""}
    removed: set[int] = set()
    removal_stages = (
        lambda index: _ordinary_teaching_removable(items, index, scaffold, mastery_core, protected_ids, preserve),
        lambda index: index in scaffold,
        lambda index: index not in mastery_core,
        lambda _index: True,
    )
    for stage in removal_stages:
        remaining = _remove_teaching_where(indexes, stage, removed, remaining)
    return removed


def enforce_concept_teaching_caps(
    items: Sequence[Mapping[str, Any]], cap_for: Callable[[str], float],
    preserve: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[Mapping[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        concept_id = item["conceptId"]
        if is_synthetic_concept_attribution(concept_id):
            continue
        groups.setdefault(concept_id, []).append(index)

    removed: set[int] = set()
    for concept_id, indexes in groups.items():
        removed.update(_teaching_cap_removals(items, indexes, cap_for(concept_id), preserve))
    return [item for index, item in enumerate(items) if index not in removed]


@dataclass(frozen=True)
class InterventionReasonContext:
    concept_id: str
    recall_on_exam_day: float
    base_reason: str
    recent_failure_concept_ids: set[str] | frozenset[str]
    concept_has_pristine: set[str] | frozenset[str]
    target_recall: float


def resolve_intervention_reason(context: InterventionReasonContext) -> str:
    if context.concept_id in context.recent_failure_concept_ids:
        return "failure_escalation"
    if context.recall_on_exam_day >= context.target_recall and context.concept_id in context.concept_has_pristine:
        return "strong_pristine"
    return context.base_reason


TEACHING_ARC_TARGET_ITEMS = {"naive": 2, "learning": 3, "consolidating": 3, "mastered": 1}


def classify_teaching_state(exposure_count: float, recall_on_exam_day: float, confidence: float) -> str:
    if exposure_count <= 0:
        return "naive"
    if recall_on_exam_day < 0.5:
        return "learning"
    if recall_on_exam_day < 0.85:
        return "consolidating"
    if confidence < 0.7:
        return "consolidating"
    return "mastered"


def normalize_review_challenge_level(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return 0
    rounded = math.floor(value + 0.5)  # JS Math.round
    closest = 0
    for level in (-2, -1, 0, 1, 2):
        if abs(level - rounded) < abs(closest - rounded):
            closest = level
    return closest
