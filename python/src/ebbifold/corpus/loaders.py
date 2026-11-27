"""Fail-closed loaders for the frozen ``open-content/`` corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .audit import CorpusAudit, CorpusCounts, audit_answer_key
from .cards import load_cloze_card_corpus, load_cloze_cards
from .loader_types import CorpusSnapshot
from .modules import load_module_corpus, load_module_questions
from .questions import admit_attempt_payload, admit_question_payload
from .step1 import load_step1_bank, load_step1_baseline, load_step1_corpus
from .schema import CorpusAdmissionError
from .validation import JSON_GLOB, _read_json, open_content_root

__all__ = [
    "CorpusSnapshot",
    "admit_attempt_payload",
    "admit_question_payload",
    "load_cloze_card_corpus",
    "load_cloze_cards",
    "load_corpus",
    "load_module_corpus",
    "load_module_questions",
    "load_step1_bank",
    "load_step1_baseline",
    "load_step1_corpus",
    "open_content_root",
]

EXPECTED_CORPUS_COUNTS = {
    "json_files": 1357,
    "step1_questions": 556,
    "baseline_questions": 25,
    "module_questions": 392,
    "cloze_cards": 2136,
}


def _json_parse_errors(content_root: Path, json_files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in json_files:
        try:
            _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            source = path.relative_to(content_root.parent).as_posix()
            errors.append(f"{source}: invalid JSON ({error})")
    return errors


def _load_corpus_parts(content_root: Path):
    errors: list[str] = []
    parts = []
    for loader in (load_step1_corpus, load_module_corpus, load_cloze_card_corpus):
        try:
            parts.append(loader(content_root))
        except CorpusAdmissionError as error:
            errors.extend(error.errors)
            parts.append(None)
    return tuple(parts), errors


def _answer_key_errors(step1_answer_key, baseline_answer_key) -> list[str]:
    return [
        f"step1 answer-key integrity: {issue['code']}" for issue in step1_answer_key.issues
    ] + [
        f"baseline answer-key integrity: {issue['code']}" for issue in baseline_answer_key.issues
    ]


def _count_errors(counts: CorpusCounts) -> list[str]:
    return [
        f"corpus count {field}: expected {expected}, found {getattr(counts, field)}"
        for field, expected in EXPECTED_CORPUS_COUNTS.items()
        if getattr(counts, field) != expected
    ]


def load_corpus(root: str | Path | None = None) -> CorpusSnapshot:
    """Load and audit the complete frozen corpus, fail-closed.

    Every JSON file under ``open-content/`` is parsed. Question, card, manifest,
    citation, release-membership, fingerprint, count, and answer-key gates must
    all pass before a snapshot is returned.
    """
    content_root = open_content_root(root)
    json_files = sorted(content_root.rglob(JSON_GLOB))
    parse_errors = _json_parse_errors(content_root, json_files)
    if parse_errors:
        raise CorpusAdmissionError(parse_errors)

    (step1, modules, cards), loader_errors = _load_corpus_parts(content_root)
    if loader_errors or step1 is None or modules is None or cards is None:
        raise CorpusAdmissionError(loader_errors)

    step1_by_id = {question.id: question for question in step1.questions}
    baseline_questions = tuple(step1_by_id[question_id] for question_id in step1.baseline_question_ids)
    step1_answer_key = audit_answer_key(list(step1.questions))
    baseline_answer_key = audit_answer_key(list(baseline_questions))
    counts = CorpusCounts(
        json_files=len(json_files),
        step1_questions=len(step1.questions),
        baseline_questions=len(baseline_questions),
        module_questions=len(modules.questions),
        original_module_questions=len(modules.original_questions),
        cloze_cards=len(cards.cards),
    )
    audit_errors = [
        *_answer_key_errors(step1_answer_key, baseline_answer_key),
        *_count_errors(counts),
    ]
    audit = CorpusAudit(
        counts=counts,
        step1_answer_key=step1_answer_key,
        baseline_answer_key=baseline_answer_key,
        citations_checked=len(step1.questions) + len(modules.questions) + len(modules.original_questions),
        release_members_checked=len(step1.questions)
        + len(modules.questions)
        + len(modules.original_questions)
        + len(cards.cards),
        errors=audit_errors,
    )
    if audit_errors:
        raise CorpusAdmissionError(audit_errors)
    return CorpusSnapshot(
        step1_questions=step1.questions,
        baseline_questions=baseline_questions,
        module_questions=modules.questions,
        original_module_questions=modules.original_questions,
        cloze_cards=cards.cards,
        audit=audit,
    )
