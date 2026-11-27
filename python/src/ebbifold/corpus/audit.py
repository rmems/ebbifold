"""Corpus audit types and answer-key integrity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ANSWER_LABELS = ("A", "B", "C", "D", "E")
MINIMUM_AUDIT_SIZE = 10
FULL_COVERAGE_SIZE = 20
MINIMUM_SMALL_CORPUS_COVERAGE = 4
MAXIMUM_LABEL_SHARE = 0.4
MAXIMUM_SAME_LABEL_STREAK = 5


@dataclass(frozen=True)
class AnswerKeyAudit:
    question_count: int
    represented_label_count: int
    maximum_label_count: int
    maximum_label_share: float
    longest_same_label_streak: int
    issues: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusCounts:
    json_files: int
    step1_questions: int
    baseline_questions: int
    module_questions: int
    original_module_questions: int
    cloze_cards: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CorpusAudit:
    counts: CorpusCounts
    step1_answer_key: AnswerKeyAudit
    baseline_answer_key: AnswerKeyAudit
    citations_checked: int
    release_members_checked: int
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors and not self.step1_answer_key.issues and not self.baseline_answer_key.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts.to_dict(),
            "step1AnswerKey": self.step1_answer_key.to_dict(),
            "baselineAnswerKey": self.baseline_answer_key.to_dict(),
            "citationsChecked": self.citations_checked,
            "releaseMembersChecked": self.release_members_checked,
            "errors": self.errors,
            "ok": self.ok,
        }


def _correct_label(question: Any, index: int) -> str:
    correct = [option for option in question.options if option.is_correct]
    if len(correct) != 1:
        raise ValueError(
            f"question at index {index} must have exactly one correct option to audit its raw answer key"
        )
    label = correct[0].label.strip().upper()
    if label not in ANSWER_LABELS:
        raise ValueError(
            f"question at index {index} has unsupported raw correct label {label!r}; expected A-E"
        )
    return label


def _longest_streak(labels: list[str]) -> int:
    longest = 0
    current = 0
    previous = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label
    return longest


def _required_coverage(question_count: int) -> int:
    if question_count >= FULL_COVERAGE_SIZE:
        return len(ANSWER_LABELS)
    if question_count >= MINIMUM_AUDIT_SIZE:
        return MINIMUM_SMALL_CORPUS_COVERAGE
    return 0


@dataclass(frozen=True)
class _AnswerKeyMetrics:
    question_count: int
    represented: int
    maximum_count: int
    maximum_share: float
    longest: int


def _answer_key_issues(metrics: _AnswerKeyMetrics) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    required_coverage = _required_coverage(metrics.question_count)
    if metrics.represented < required_coverage:
        issues.append(
            {"code": "label-coverage", "actual": metrics.represented, "required": required_coverage}
        )
    if (
        metrics.question_count >= MINIMUM_AUDIT_SIZE
        and metrics.maximum_count > metrics.question_count * MAXIMUM_LABEL_SHARE
    ):
        issues.append(
            {
                "code": "label-concentration",
                "actualCount": metrics.maximum_count,
                "actualShare": metrics.maximum_share,
                "maximumShare": MAXIMUM_LABEL_SHARE,
            }
        )
    if metrics.question_count >= MINIMUM_AUDIT_SIZE and metrics.longest > MAXIMUM_SAME_LABEL_STREAK:
        issues.append(
            {
                "code": "label-streak",
                "actual": metrics.longest,
                "maximum": MAXIMUM_SAME_LABEL_STREAK,
            }
        )
    return issues


def audit_answer_key(questions: list[Any] | tuple[Any, ...]) -> AnswerKeyAudit:
    """Port of upstream's deliberately corpus-level raw answer-position floor."""
    labels = [_correct_label(question, index) for index, question in enumerate(questions)]
    counts = dict.fromkeys(ANSWER_LABELS, 0)
    for label in labels:
        counts[label] += 1

    question_count = len(labels)
    represented = sum(1 for label in ANSWER_LABELS if counts[label] > 0)
    maximum_count = max(counts.values(), default=0)
    maximum_share = maximum_count / question_count if question_count else 0.0
    longest = _longest_streak(labels)
    metrics = _AnswerKeyMetrics(
        question_count=question_count,
        represented=represented,
        maximum_count=maximum_count,
        maximum_share=maximum_share,
        longest=longest,
    )
    return AnswerKeyAudit(
        question_count=question_count,
        represented_label_count=represented,
        maximum_label_count=maximum_count,
        maximum_label_share=maximum_share,
        longest_same_label_streak=longest,
        issues=_answer_key_issues(metrics),
    )
