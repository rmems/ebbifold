"""Domain-neutral review engine joining corpus adapters to both ported schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ebbifold.adaptive import AdaptiveHistoryRow, AdaptiveQuestion, rank_adaptive
from ebbifold.corpus import CorpusSnapshot
from ebbifold.corpus.models import CorpusQuestion
from ebbifold.scheduler.concepts import rank_concepts

MS_PER_SECOND = 1000


@dataclass(frozen=True)
class ReviewQuestion:
    id: str
    stem: str
    options: tuple[Any, ...]
    difficulty: str | None
    concept_id: str
    concept_priority: float
    record: CorpusQuestion


def _concept_id(question: CorpusQuestion) -> str:
    return question.topics[0] if question.topics else "unclassified"


def _questions(snapshot: CorpusSnapshot) -> list[CorpusQuestion]:
    return [*snapshot.step1_questions, *snapshot.module_questions]


def _concept_inputs(questions: list[CorpusQuestion]) -> list[dict[str, Any]]:
    concepts: dict[str, set[str]] = {}
    for question in questions:
        concept = _concept_id(question)
        concepts.setdefault(concept, set()).update(question.topics)
    return [
        {"id": concept, "name": concept.replace("-", " "), "topics": sorted(topics), "prerequisiteIds": []}
        for concept, topics in sorted(concepts.items())
    ]


def _attempt_time_ms(value: str) -> float:
    return datetime.fromisoformat(value).timestamp() * MS_PER_SECOND


def _state_map(questions: list[CorpusQuestion], attempts: list[Any]) -> dict[str, dict[str, Any]]:
    concept_by_item = {question.id: _concept_id(question) for question in questions}
    rows: dict[str, list[Any]] = {}
    for attempt in attempts:
        concept = concept_by_item.get(attempt["item_id"])
        if concept:
            rows.setdefault(concept, []).append(attempt)
    states: dict[str, dict[str, Any]] = {}
    for concept, concept_attempts in rows.items():
        correct = sum(1 for row in concept_attempts if row["is_correct"])
        count = len(concept_attempts)
        latest_ms = max(_attempt_time_ms(row["answered_at"]) for row in concept_attempts)
        states[concept] = {
            "recallProbability": correct / count,
            "confidence": min(1.0, count / 3),
            "exposureCount": count,
            "lastProbeAt": latest_ms,
            "lastExposureAt": latest_ms,
        }
    return states


class ReviewEngine:
    """Corpus-agnostic question engine.

    The concept-space scheduler ranks topic concepts from persisted learner
    state; the adaptive scheduler performs tiered item selection inside that
    ordering. Neither layer knows what corpus or subject supplied the items.
    """

    def __init__(self, snapshot: CorpusSnapshot, *, now_ms: float | None = None, days_to_exam: float = 30) -> None:
        self.snapshot = snapshot
        self.question_records = {question.id: question for question in _questions(snapshot)}
        self.now_ms = now_ms if now_ms is not None else datetime.now().timestamp() * MS_PER_SECOND
        self.days_to_exam = days_to_exam

    def concept_priorities(self, attempts: list[Any]) -> dict[str, float]:
        concepts = _concept_inputs(list(self.question_records.values()))
        ranking = rank_concepts(
            {
                "nowMs": self.now_ms,
                "daysToExam": self.days_to_exam,
                "examPressure": 0.5,
                "options": {},
                "concepts": concepts,
                "stateMap": _state_map(list(self.question_records.values()), attempts),
                "bulk": {},
            }
        )
        return {state.conceptId: state.priority for state in ranking.conceptStates}

    def next_question(self, session_attempts: list[Any], all_attempts: list[Any] | None = None) -> ReviewQuestion | None:
        priorities = self.concept_priorities(all_attempts if all_attempts is not None else session_attempts)
        adaptive_questions = []
        for question in self.question_records.values():
            concept = _concept_id(question)
            adaptive_questions.append(
                AdaptiveQuestion(
                    id=question.id,
                    difficulty=question.difficulty,
                    domain=concept,
                    preference_rank=-priorities.get(concept, 0.0),
                    ladder_id=question.ladder_id,
                )
            )
        history = [
            AdaptiveHistoryRow(
                question_id=row["item_id"],
                is_correct=bool(row["is_correct"]),
                created_at=datetime.fromisoformat(row["answered_at"]),
            )
            for row in session_attempts
        ]
        ranked = rank_adaptive(adaptive_questions, history, 1)
        if not ranked:
            return None
        record = self.question_records[ranked[0].id]
        concept = _concept_id(record)
        return ReviewQuestion(
            id=record.id,
            stem=record.stem,
            options=tuple(record.options),
            difficulty=record.difficulty,
            concept_id=concept,
            concept_priority=priorities.get(concept, 0.0),
            record=record,
        )

    @staticmethod
    def grade(question: ReviewQuestion, selected_label: str) -> bool:
        selected = selected_label.strip().upper()
        return any(option.label.upper() == selected and option.is_correct for option in question.options)
