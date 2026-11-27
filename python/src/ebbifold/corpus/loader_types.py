"""Shared result types for corpus loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .audit import CorpusAudit
from .models import (
    ClozeCardRecord,
    CorpusQuestion,
    ModuleQuestionRecord,
    Step1QuestionRecord,
)


@dataclass(frozen=True)
class LoadedQuestion:
    record: CorpusQuestion
    raw: dict[str, Any]
    source_file: str

@dataclass(frozen=True)
class Step1Corpus:
    questions: tuple[Step1QuestionRecord, ...]
    baseline_question_ids: tuple[str, ...]
    release_question_ids: tuple[str, ...]
    raw_questions: tuple[dict[str, Any], ...]

@dataclass(frozen=True)
class ModuleCorpus:
    questions: tuple[ModuleQuestionRecord, ...]
    original_questions: tuple[ModuleQuestionRecord, ...]
    release_question_ids: tuple[str, ...]
    original_release_question_ids: tuple[str, ...]

@dataclass(frozen=True)
class ClozeCardCorpus:
    cards: tuple[ClozeCardRecord, ...]
    release_card_ids: tuple[str, ...]

@dataclass(frozen=True)
class CorpusSnapshot:
    step1_questions: tuple[Step1QuestionRecord, ...]
    baseline_questions: tuple[Step1QuestionRecord, ...]
    module_questions: tuple[ModuleQuestionRecord, ...]
    original_module_questions: tuple[ModuleQuestionRecord, ...]
    cloze_cards: tuple[ClozeCardRecord, ...]
    audit: CorpusAudit
