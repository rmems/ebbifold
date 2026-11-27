"""Question admission and per-record audit gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .question_audit import _question_audit_errors
from .loader_types import LoadedQuestion
from .models import (
    AttemptRecord,
    CorpusQuestion,
    ModuleQuestionRecord,
    Step1QuestionRecord,
)
from .schema import (
    ATTEMPT_SCHEMA,
    QUESTION_SCHEMA,
    CorpusAdmissionError,
    CorpusSchemaError,
    validate_against_schema,
)
from .validation import _read_json, _repo_relative, _validation_error_messages, JSON_GLOB

def _validate_question_schema(payload: dict[str, Any], source: str, repo_root: Path | None) -> None:
    try:
        validate_against_schema(payload, QUESTION_SCHEMA, repo_root=repo_root, source=source)
    except CorpusSchemaError as error:
        raise CorpusAdmissionError(str(error).splitlines()) from error


def _validated_question_model(payload: dict[str, Any], source: str) -> CorpusQuestion:
    model = Step1QuestionRecord if "publicUsmle" in payload else ModuleQuestionRecord
    try:
        return model.model_validate(payload)
    except ValidationError as error:
        raise CorpusAdmissionError(_validation_error_messages(source, error)) from error


def _validate_question_audit(record: CorpusQuestion, payload: dict[str, Any], source: str) -> None:
    errors = _question_audit_errors(record, payload, source)
    if errors:
        raise CorpusAdmissionError(errors)


def admit_question_payload(
    payload: dict[str, Any],
    *,
    source: str = "<memory>",
    repo_root: Path | None = None,
) -> CorpusQuestion:
    """Validate one raw question JSON object through schema, model, and audit.

    This is the corpus admission gate: callers receive a typed record only when
    every layer accepts the exact bytes-shaped document. Any failure raises
    ``CorpusAdmissionError``; there is no partial or best-effort admission.
    """
    _validate_question_schema(payload, source, repo_root)
    record = _validated_question_model(payload, source)
    _validate_question_audit(record, payload, source)
    return record

def admit_attempt_payload(
    payload: dict[str, Any],
    *,
    source: str = "<memory>",
    repo_root: Path | None = None,
) -> AttemptRecord:
    """Validate one learner-attempt record against its versioned schema."""
    errors: list[str] = []
    try:
        validate_against_schema(payload, ATTEMPT_SCHEMA, repo_root=repo_root, source=source)
    except CorpusSchemaError as error:
        errors.extend(str(error).splitlines())
    if errors:
        raise CorpusAdmissionError(errors)
    try:
        return AttemptRecord.model_validate(payload)
    except ValidationError as error:
        raise CorpusAdmissionError(_validation_error_messages(source, error)) from error

def _load_question_file(path: Path, repo_root: Path) -> tuple[LoadedQuestion | None, list[str]]:
    source = _repo_relative(path, repo_root)
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{source}: invalid JSON ({error})"]
    try:
        record = admit_question_payload(raw, source=source, repo_root=repo_root)
    except CorpusAdmissionError as error:
        return None, error.errors
    return LoadedQuestion(record=record, raw=raw, source_file=source), []


def _duplicate_question_errors(loaded: list[LoadedQuestion]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for item in loaded:
        if item.record.id in seen:
            errors.append(f"{item.source_file}: duplicate question id {item.record.id}")
        seen.add(item.record.id)
    return errors


def _load_question_tree(
    directory: Path,
    *,
    repo_root: Path,
) -> tuple[list[LoadedQuestion], list[str]]:
    errors: list[str] = []
    loaded: list[LoadedQuestion] = []
    for path in sorted(directory.rglob(JSON_GLOB)):
        item, item_errors = _load_question_file(path, repo_root)
        errors.extend(item_errors)
        if item is not None:
            loaded.append(item)
    errors.extend(_duplicate_question_errors(loaded))
    return loaded, errors
