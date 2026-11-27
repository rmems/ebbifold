"""Fail-closed loaders for the frozen ``open-content/`` corpus."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .audit import CorpusAudit, CorpusCounts, audit_answer_key
from .cards import load_cloze_card_corpus, load_cloze_cards
from .loader_types import ClozeCardCorpus, CorpusSnapshot, ModuleCorpus, Step1Corpus
from .modules import load_module_corpus, load_module_questions
from .questions import admit_attempt_payload, admit_question_payload
from .step1 import load_step1_bank, load_step1_baseline, load_step1_corpus
from .schema import CorpusAdmissionError
from .validation import JSON_GLOB, _read_json, open_content_root

__all__ = [
    "ClozeCardCorpus",
    "CorpusSnapshot",
    "ModuleCorpus",
    "Step1Corpus",
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


@dataclass(frozen=True)
class _CorpusParts:
    step1: Step1Corpus
    modules: ModuleCorpus
    cards: ClozeCardCorpus


def _require_corpus_parts(parts: tuple, loader_errors: list[str]) -> _CorpusParts:
    step1, modules, cards = parts
    missing = []
    if step1 is None:
        missing.append("Step 1 corpus")
    if modules is None:
        missing.append("module corpus")
    if cards is None:
        missing.append("cloze-card corpus")
    if loader_errors or missing:
        errors = list(loader_errors)
        errors.extend(f"{name} did not load" for name in missing if not loader_errors)
        raise CorpusAdmissionError(errors)
    return _CorpusParts(step1=step1, modules=modules, cards=cards)


def _baseline_questions(step1) -> tuple:
    step1_by_id = {question.id: question for question in step1.questions}
    return tuple(step1_by_id[question_id] for question_id in step1.baseline_question_ids)


def _corpus_counts(parts: _CorpusParts, baseline_questions: tuple, json_count: int) -> CorpusCounts:
    return CorpusCounts(
        json_files=json_count,
        step1_questions=len(parts.step1.questions),
        baseline_questions=len(baseline_questions),
        module_questions=len(parts.modules.questions),
        original_module_questions=len(parts.modules.original_questions),
        cloze_cards=len(parts.cards.cards),
    )


def _corpus_audit(parts: _CorpusParts, baseline_questions: tuple, counts: CorpusCounts) -> CorpusAudit:
    step1_answer_key = audit_answer_key(list(parts.step1.questions))
    baseline_answer_key = audit_answer_key(list(baseline_questions))
    audit_errors = [
        *_answer_key_errors(step1_answer_key, baseline_answer_key),
        *_count_errors(counts),
    ]
    return CorpusAudit(
        counts=counts,
        step1_answer_key=step1_answer_key,
        baseline_answer_key=baseline_answer_key,
        citations_checked=len(parts.step1.questions)
        + len(parts.modules.questions)
        + len(parts.modules.original_questions),
        release_members_checked=len(parts.step1.questions)
        + len(parts.modules.questions)
        + len(parts.modules.original_questions)
        + len(parts.cards.cards),
        errors=audit_errors,
    )


def _snapshot(parts: _CorpusParts, baseline_questions: tuple, audit: CorpusAudit) -> CorpusSnapshot:
    return CorpusSnapshot(
        step1_questions=parts.step1.questions,
        baseline_questions=baseline_questions,
        module_questions=parts.modules.questions,
        original_module_questions=parts.modules.original_questions,
        cloze_cards=parts.cards.cards,
        audit=audit,
    )


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

    raw_parts, loader_errors = _load_corpus_parts(content_root)
    parts = _require_corpus_parts(raw_parts, loader_errors)

    baseline_questions = _baseline_questions(parts.step1)
    counts = _corpus_counts(parts, baseline_questions, len(json_files))
    audit = _corpus_audit(parts, baseline_questions, counts)
    if audit.errors:
        raise CorpusAdmissionError(audit.errors)
    return _snapshot(parts, baseline_questions, audit)
