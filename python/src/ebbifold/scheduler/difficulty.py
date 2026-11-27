"""Question difficulty normalization (candidate-ranking.ts, question-analytics.ts)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .topics import js_trim

QUESTION_DIFFICULTIES = ("easy", "medium", "hard")
MIN_ATTEMPTS_FOR_DIFFICULTY_CALIBRATION = 20


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_difficulty_plan(pattern: Sequence[str], length: int) -> list[str]:
    base = [d for d in pattern if d in QUESTION_DIFFICULTIES] or ["medium"]
    return [base[i % len(base)] for i in range(length)]


def normalize_question_difficulty(value: Any) -> str:
    if not isinstance(value, str):
        return "medium"
    normalized = js_trim(value).lower()
    return normalized if normalized in QUESTION_DIFFICULTIES else "medium"


def calibrate_difficulty_from_facility_index(facility_index: float) -> str:
    if facility_index >= 0.7:
        return "easy"
    if facility_index >= 0.4:
        return "medium"
    return "hard"


def get_effective_question_difficulty(
    difficulty: Any, facility_index: Any, total_attempts: Any
) -> str:
    if (
        _is_number(facility_index)
        and _is_number(total_attempts)
        and total_attempts >= MIN_ATTEMPTS_FOR_DIFFICULTY_CALIBRATION
    ):
        return calibrate_difficulty_from_facility_index(facility_index)
    return normalize_question_difficulty(difficulty)



def question_request_difficulty(exam_pressure: float, reason: str, index_for_concept: int) -> str:
    """fillQuestions' per-concept difficulty ladder."""
    if exam_pressure >= 0.5:
        plan = ["medium", "hard", "hard", "hard"]
    elif reason == "low_confidence":
        plan = ["easy", "medium", "medium", "hard"]
    elif reason == "needs_retest":
        plan = ["medium", "hard", "hard", "hard"]
    else:
        plan = ["medium", "hard"]
    return plan[min(index_for_concept, len(plan) - 1)]
