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


def audit_answer_key(questions: list[Any] | tuple[Any, ...]) -> AnswerKeyAudit:
    """Port of upstream's deliberately corpus-level raw answer-position floor."""
    counts = dict.fromkeys(ANSWER_LABELS, 0)
    labels: list[str] = []
    for index, question in enumerate(questions):
        correct = [option for option in question.options if option.is_correct]
        if len(correct) != 1:
            raise ValueError(
                f"question at index {index} must have exactly one correct option to audit its raw answer key"
            )
        label = correct[0].label.strip().upper()
        if label not in counts:
            raise ValueError(
                f"question at index {index} has unsupported raw correct label {label!r}; expected A-E"
            )
        counts[label] += 1
        labels.append(label)

    longest = 0
    current = 0
    previous = None
    for label in labels:
        current = current + 1 if label == previous else 1
        longest = max(longest, current)
        previous = label

    question_count = len(labels)
    represented = sum(1 for label in ANSWER_LABELS if counts[label] > 0)
    maximum_count = max(counts.values(), default=0)
    maximum_share = maximum_count / question_count if question_count else 0.0
    issues: list[dict[str, Any]] = []
    required_coverage = 0
    if question_count >= FULL_COVERAGE_SIZE:
        required_coverage = len(ANSWER_LABELS)
    elif question_count >= MINIMUM_AUDIT_SIZE:
        required_coverage = MINIMUM_SMALL_CORPUS_COVERAGE
    if represented < required_coverage:
        issues.append(
            {"code": "label-coverage", "actual": represented, "required": required_coverage}
        )
    if question_count >= MINIMUM_AUDIT_SIZE and maximum_count > question_count * MAXIMUM_LABEL_SHARE:
        issues.append(
            {
                "code": "label-concentration",
                "actualCount": maximum_count,
                "actualShare": maximum_share,
                "maximumShare": MAXIMUM_LABEL_SHARE,
            }
        )
    if question_count >= MINIMUM_AUDIT_SIZE and longest > MAXIMUM_SAME_LABEL_STREAK:
        issues.append(
            {"code": "label-streak", "actual": longest, "maximum": MAXIMUM_SAME_LABEL_STREAK}
        )
    return AnswerKeyAudit(
        question_count=question_count,
        represented_label_count=represented,
        maximum_label_count=maximum_count,
        maximum_label_share=maximum_share,
        longest_same_label_streak=longest,
        issues=issues,
    )
