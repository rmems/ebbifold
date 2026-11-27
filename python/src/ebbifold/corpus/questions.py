"""Question admission and per-record audit gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .fingerprint import module_item_content_hash
from .loader_types import LoadedQuestion
from .models import (
    AttemptRecord,
    CorpusQuestion,
    ModuleQuestionRecord,
    PassageEvidence,
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

COHORT_DISCIPLINES = frozenset(
    {
        "aged",
        "anaes",
        "anatomy",
        "cardio",
        "clinical-skills",
        "derm",
        "ebm",
        "em",
        "endo",
        "ent",
        "gi",
        "gp",
        "gynae",
        "haem",
        "icu",
        "id",
        "immuno",
        "msk",
        "neuro",
        "obs",
        "onc",
        "ophth",
        "paeds",
        "paramedicine",
        "pathology",
        "pharm",
        "physiology",
        "psych",
        "renal",
        "resp",
        "surgery",
        "neurosurgery",
        "radiology",
    }
)
PUBLIC_TEXT_LICENCES = frozenset({"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0"})
PUBLIC_EVIDENCE_LICENCE_IDS = frozenset(
    {
        "cc0",
        "cc0-1.0",
        "public-domain",
        "us-gov",
        "cc-by",
        "cc-by-2.0",
        "cc-by-2.5",
        "cc-by-3.0",
        "cc-by-4.0",
        "cc-by-sa",
        "cc-by-sa-2.0",
        "cc-by-sa-2.5",
        "cc-by-sa-3.0",
        "cc-by-sa-4.0",
    }
)


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
    errors: list[str] = []
    try:
        validate_against_schema(payload, QUESTION_SCHEMA, repo_root=repo_root, source=source)
    except CorpusSchemaError as error:
        errors.extend(str(error).splitlines())

    record: CorpusQuestion | None = None
    if not errors:
        model = Step1QuestionRecord if "publicUsmle" in payload else ModuleQuestionRecord
        try:
            record = model.model_validate(payload)
        except ValidationError as error:
            errors.extend(_validation_error_messages(source, error))

    if record is not None:
        errors.extend(_question_audit_errors(record, payload, source))
    if errors:
        raise CorpusAdmissionError(errors)
    if record is None:
        raise CorpusAdmissionError([f"{source}: question admission failed before model construction"])
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

def _answer_key_errors(record: CorpusQuestion, source: str) -> list[str]:
    errors: list[str] = []
    labels = [option.label for option in record.options]
    if len(labels) != len(set(labels)):
        errors.append(f"{source}: answer-key integrity: duplicate option label")
    correct_count = sum(1 for option in record.options if option.is_correct)
    if correct_count != 1:
        errors.append(
            f"{source}: answer-key integrity: expected exactly one correct option, found {correct_count}"
        )
    if any(not option.explanation or not option.explanation.strip() for option in record.options):
        errors.append(f"{source}: every option requires an explanation")
    return errors

def _step1_evidence_errors(record: Step1QuestionRecord, source: str) -> list[str]:
    evidence = record.public_usmle.evidence
    if not isinstance(evidence, PassageEvidence):
        return [f"{source}: citation presence: public Step 1 evidence must be a passage"]
    licence = evidence.licence
    if licence.cls != "foss" or licence.id.lower() not in PUBLIC_EVIDENCE_LICENCE_IDS:
        return [f"{source}: evidence licence {licence.cls}/{licence.id} is not approved FOSS"]
    return []

def _step1_media_errors(record: Step1QuestionRecord, source: str) -> list[str]:
    media = record.public_usmle.media
    if not record.image_url:
        if media is not None and media.kind == "asset":
            return [f"{source}: media asset declared without imageUrl"]
        return []
    if media is None or media.kind != "asset":
        return [f"{source}: image-bearing item lacks asset media provenance"]
    licence = media.licence
    if licence.cls != "foss" or licence.id.lower() not in PUBLIC_EVIDENCE_LICENCE_IDS:
        return [f"{source}: media licence {licence.cls}/{licence.id} is not approved FOSS"]
    return []

def _step1_question_errors(record: Step1QuestionRecord, source: str) -> list[str]:
    errors: list[str] = []
    if not record.cite.strip():
        errors.append(f"{source}: citation presence: Step 1 item has no cite")
    provenance = record.public_usmle
    if provenance.origin not in {"authored", "generated"}:
        errors.append(f"{source}: provenance origin {provenance.origin} is not public")
    if provenance.item_text.licence not in PUBLIC_TEXT_LICENCES:
        errors.append(
            f"{source}: item text licence {provenance.item_text.licence} is not distributable"
        )
    errors.extend(_step1_evidence_errors(record, source))
    errors.extend(_step1_media_errors(record, source))
    return errors

def _module_question_errors(
    record: ModuleQuestionRecord, raw: dict[str, Any], source: str
) -> list[str]:
    errors: list[str] = []
    provenance = record.public_module
    if provenance.discipline not in COHORT_DISCIPLINES:
        errors.append(f"{source}: unknown module discipline {provenance.discipline}")
    if f"cohort/{provenance.discipline}" not in record.module_nodes:
        errors.append(f"{source}: moduleNodes missing cohort/{provenance.discipline}")
    if record.image_url:
        errors.append(f"{source}: module questions do not admit images in this release")
    if provenance.review.verdict != "passed":
        errors.append(f"{source}: module review has not passed")
    if provenance.review.content_hash != module_item_content_hash(raw):
        errors.append(f"{source}: module review content hash is stale")
    return errors

def _question_audit_errors(
    record: CorpusQuestion, raw: dict[str, Any], source: str
) -> list[str]:
    errors = _answer_key_errors(record, source)
    if isinstance(record, Step1QuestionRecord):
        errors.extend(_step1_question_errors(record, source))
    elif isinstance(record, ModuleQuestionRecord):
        errors.extend(_module_question_errors(record, raw, source))
    return errors

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
