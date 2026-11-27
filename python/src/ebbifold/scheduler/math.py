"""Recall decay, projection and the scheduler signals.

Ports state.ts (applyDecay, projectRecallToExamDay), throughput.ts
(computeExposuresNeeded), scheduler-signals.ts, math/vector-math.ts
(cosineSimilarity) and the manifold truncation helper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

BASE_HALF_LIFE_DAYS = 7
CONFIDENCE_MULTIPLIER = 2
RUNTIME_MANIFOLD_DIM = 1024


def _half_life(confidence: float) -> float:
    return BASE_HALF_LIFE_DAYS * (1 + confidence * CONFIDENCE_MULTIPLIER)


def apply_decay(probability: float, days_since_last_exposure: float, confidence: float) -> float:
    if days_since_last_exposure <= 0:
        return probability
    decayed = probability * math.pow(0.5, days_since_last_exposure / _half_life(confidence))
    return max(0.0, min(1.0, decayed))


def project_recall_to_exam_day(
    current_probability: float,
    days_to_exam: float,
    confidence: float,
    max_horizon_days: float = math.inf,
) -> float:
    if days_to_exam <= 0:
        return current_probability
    return apply_decay(current_probability, min(days_to_exam, max_horizon_days), confidence)


def compute_exposures_needed(
    current_recall: float, target_recall: float, confidence: float, days_to_exam: float
) -> int:
    half_life = _half_life(confidence)
    projected = (
        current_recall * math.pow(0.5, days_to_exam / half_life) if days_to_exam > 0 else current_recall
    )
    if projected >= target_recall:
        return 0
    exposure_gain = 0.3
    max_exposures = 20
    for n in range(1, max_exposures + 1):
        recall = current_recall
        interval = days_to_exam / (n + 1)
        for _ in range(n):
            recall = recall * math.pow(0.5, interval / half_life)
            recall = recall + exposure_gain * (1 - recall)
            recall = min(1.0, recall)
        recall = recall * math.pow(0.5, interval / half_life)
        if recall >= target_recall:
            return n
    return max_exposures


def compute_hub_readiness(
    prerequisite_ids: Sequence[str], prereq_recalls: Mapping[str, float], threshold: float = 0.5
) -> tuple[bool, float]:
    """Return ``(ready, dampening)``."""
    if not prerequisite_ids:
        return True, 1.0
    min_recall = min(prereq_recalls.get(pid, 0) for pid in prerequisite_ids)
    ready = min_recall >= threshold
    return ready, 1.0 if ready else min(1.0, min_recall / threshold)


def compute_knowledge_breadth(recalls: Sequence[float]) -> tuple[float, float]:
    """Return ``(d_eff, breadth_ratio)`` — the participation ratio of recalls."""
    if not recalls:
        return 0.0, 0.0
    total = 0.0
    total_sq = 0.0
    for r in recalls:
        clamped = max(0.0, r)
        total += clamped
        total_sq += clamped * clamped
    if total_sq == 0:
        return 0.0, 0.0
    d_eff = (total * total) / total_sq
    return d_eff, d_eff / len(recalls)


def compute_maintenance_shift(breadth_ratio: float, total_concepts: int) -> float:
    if total_concepts < 5 or breadth_ratio >= 0.5:
        return 0.0
    return min(0.3, max(0.0, 0.3 * (1 - breadth_ratio / 0.5)))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"Embedding dimensions must match: got {len(a)} vs {len(b)}")
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    denominator = math.sqrt(norm_a) * math.sqrt(norm_b)
    return 0.0 if denominator == 0 else dot / denominator


def l2_normalize(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        return [0.0 for _ in v]
    return [x / norm for x in v]


def truncate_to_manifold_dim(v: Sequence[float]) -> list[float]:
    return l2_normalize(list(v)[:RUNTIME_MANIFOLD_DIM])
