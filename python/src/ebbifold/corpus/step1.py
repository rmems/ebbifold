"""Fail-closed Step 1 corpus loading and release validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fingerprint import (
    BASELINE_MODULE,
    open_usmle_quote_set_sha256,
    open_usmle_registry_sha256,
    public_serving_fingerprint,
)
from .loader_types import LoadedQuestion, Step1Corpus
from .models import PassageEvidence, Step1QuestionRecord
from .questions import _load_question_tree
from .schema import CorpusAdmissionError
from .validation import (
    _expect_manifest_object,
    _read_json,
    _read_question_manifest,
    open_content_root,
)


def _read_step1_registry(registry_path: Path) -> tuple[list[Any] | None, list[str]]:
    try:
        registry = _read_json(registry_path)
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"{registry_path}: invalid source registry ({error})"]
    errors = _expect_manifest_object(registry, registry_path, 1)
    sources = registry.get("sources") if isinstance(registry, dict) else None
    if not isinstance(sources, list):
        return None, [*errors, f"{registry_path}: sources must be an array"]
    if registry.get("quoteSetSha256") != open_usmle_quote_set_sha256(sources):
        errors.append(f"{registry_path}: quote-set digest does not match sources")
    if registry.get("registrySha256") != open_usmle_registry_sha256(
        registry.get("verifiedAt", ""), sources
    ):
        errors.append(f"{registry_path}: registry digest does not match sources")
    return sources, errors

def _passage_id(passage: Any) -> str | None:
    if isinstance(passage, dict) and isinstance(passage.get("id"), str):
        return passage["id"]
    return None


def _index_step1_passages(
    registry_path: Path,
    source: dict[str, Any],
    source_id: str,
    passage_keys: set[str],
) -> list[str]:
    errors: list[str] = []
    passage_ids: set[str] = set()
    for passage in source.get("passages", []):
        passage_id = _passage_id(passage)
        if passage_id is None:
            continue
        if passage_id in passage_ids:
            errors.append(
                f"{registry_path}: duplicate passage id {passage_id} for source {source_id}"
            )
        passage_ids.add(passage_id)
        passage_keys.add(f"{source_id}:{passage_id}")
    return errors


def _index_step1_source(
    registry_path: Path,
    source: Any,
    source_by_id: dict[str, dict[str, Any]],
    passage_keys: set[str],
) -> list[str]:
    if not isinstance(source, dict) or not isinstance(source.get("id"), str):
        return [f"{registry_path}: source entry missing id"]
    source_id = source["id"]
    errors: list[str] = []
    if source_id in source_by_id:
        errors.append(f"{registry_path}: duplicate source id {source_id}")
    source_by_id[source_id] = source
    errors.extend(_index_step1_passages(registry_path, source, source_id, passage_keys))
    return errors


def _index_step1_sources(
    registry_path: Path, sources: list[Any]
) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
    errors: list[str] = []
    source_by_id: dict[str, dict[str, Any]] = {}
    passage_keys: set[str] = set()
    for source in sources:
        errors.extend(_index_step1_source(registry_path, source, source_by_id, passage_keys))
    return source_by_id, passage_keys, errors

def _registered_passage(source: dict[str, Any], passage_id: str) -> dict[str, Any]:
    return next(
        passage
        for passage in source.get("passages", [])
        if isinstance(passage, dict) and passage.get("id") == passage_id
    )

def _step1_citation_errors(
    item: LoadedQuestion,
    evidence: PassageEvidence,
    source_by_id: dict[str, dict[str, Any]],
    passage_keys: set[str],
) -> list[str]:
    errors: list[str] = []
    record = item.record
    expected_cite = f"{evidence.source_id}#{evidence.passage_id}"
    if record.cite != expected_cite:
        errors.append(
            f"{item.source_file}: citation presence: cite {record.cite!r} does not match evidence pointer {expected_cite!r}"
        )
    source = source_by_id.get(evidence.source_id)
    if source is None:
        errors.append(f"{item.source_file}: unknown evidence source {evidence.source_id}")
        return errors

    registered = source.get("licence", {})
    if evidence.licence.cls != registered.get("cls") or evidence.licence.id.lower() != str(
        registered.get("id", "")
    ).lower():
        errors.append(f"{item.source_file}: evidence licence differs from source registry")
    key = f"{evidence.source_id}:{evidence.passage_id}"
    if key not in passage_keys:
        errors.append(f"{item.source_file}: unknown evidence passage {evidence.passage_id}")
    elif not _registered_passage(source, evidence.passage_id).get("quote"):
        errors.append(f"{item.source_file}: evidence passage has no checked-in quote")
    return errors

def _validate_step1_citations(
    questions: list[LoadedQuestion],
    source_by_id: dict[str, dict[str, Any]],
    passage_keys: set[str],
) -> tuple[int, list[str]]:
    checked = 0
    errors: list[str] = []
    for item in questions:
        record = item.record
        if not isinstance(record, Step1QuestionRecord) or not isinstance(
            record.public_usmle.evidence, PassageEvidence
        ):
            continue
        checked += 1
        errors.extend(
            _step1_citation_errors(item, record.public_usmle.evidence, source_by_id, passage_keys)
        )
    return checked, errors

def _validate_step1_registry(
    registry_path: Path, questions: list[LoadedQuestion]
) -> tuple[int, list[str]]:
    sources, errors = _read_step1_registry(registry_path)
    if sources is None:
        return 0, errors
    source_by_id, passage_keys, index_errors = _index_step1_sources(registry_path, sources)
    errors.extend(index_errors)
    checked, citation_errors = _validate_step1_citations(questions, source_by_id, passage_keys)
    errors.extend(citation_errors)
    return checked, errors

def _step1_membership_errors(
    baseline_path: Path,
    release_path: Path,
    question_ids: set[str],
    baseline_ids: list[str],
    release_ids: list[str],
) -> list[str]:
    release_id_set = set(release_ids)
    return [
        *(f"{baseline_path}: unknown baseline question {question_id}"
          for question_id in baseline_ids if question_id not in question_ids),
        *(f"{release_path}: unknown release question {question_id}"
          for question_id in release_ids if question_id not in question_ids),
        *(f"{release_path}: unreleased source question {question_id}"
          for question_id in sorted(question_ids) if question_id not in release_id_set),
    ]

def _step1_fingerprint_errors(
    release_raw: Any,
    release_path: Path,
    loaded: list[LoadedQuestion],
    release_id_set: set[str],
    baseline_ids: list[str],
) -> list[str]:
    fingerprints = release_raw.get("questionFingerprints") if isinstance(release_raw, dict) else None
    if not isinstance(fingerprints, dict):
        return [f"{release_path}: questionFingerprints must be an object"]

    errors: list[str] = []
    baseline_id_set = set(baseline_ids)
    for item in loaded:
        if item.record.id not in release_id_set:
            continue
        expected = fingerprints.get(item.record.id)
        if not expected:
            errors.append(f"{release_path}: missing release fingerprint {item.record.id}")
            continue
        actual = public_serving_fingerprint(
            item.raw,
            source_file=item.source_file,
            baseline_question_ids=baseline_id_set,
        )
        if actual != expected:
            errors.append(f"{release_path}: release fingerprint differs from source {item.record.id}")
    return errors

def _project_step1_questions(
    loaded: list[LoadedQuestion], baseline_ids: list[str]
) -> list[Step1QuestionRecord]:
    baseline_id_set = set(baseline_ids)
    questions: list[Step1QuestionRecord] = []
    for item in loaded:
        if not isinstance(item.record, Step1QuestionRecord):
            continue
        record = item.record
        if record.id in baseline_id_set and BASELINE_MODULE not in record.module_nodes:
            record = record.model_copy(
                update={"module_nodes": [*record.module_nodes, BASELINE_MODULE]}
            )
        questions.append(record)
    return questions

def load_step1_corpus(root: str | Path | None = None) -> Step1Corpus:
    content_root = open_content_root(root)
    repo_root = content_root.parent
    step1_root = content_root / "usmle" / "step1"
    loaded, errors = _load_question_tree(step1_root / "questions", repo_root=repo_root)

    baseline_path = step1_root / "baseline-v1.json"
    release_path = step1_root / "release-v1.json"
    _, baseline_ids, baseline_errors = _read_question_manifest(
        baseline_path, 1, "baseline manifest"
    )
    release_raw, release_ids, release_errors = _read_question_manifest(
        release_path, 2, "release manifest"
    )
    errors.extend(baseline_errors)
    errors.extend(release_errors)

    question_ids = {item.record.id for item in loaded}
    errors.extend(
        _step1_membership_errors(
            baseline_path, release_path, question_ids, baseline_ids, release_ids
        )
    )
    errors.extend(
        _step1_fingerprint_errors(
            release_raw, release_path, loaded, set(release_ids), baseline_ids
        )
    )

    citations_checked, registry_errors = _validate_step1_registry(
        step1_root / "sources.json", loaded
    )
    errors.extend(registry_errors)
    if citations_checked != len(loaded):
        errors.append("citation presence: not every Step 1 question resolved a passage citation")

    if errors:
        raise CorpusAdmissionError(errors)
    return Step1Corpus(
        questions=tuple(_project_step1_questions(loaded, baseline_ids)),
        baseline_question_ids=tuple(baseline_ids),
        release_question_ids=tuple(release_ids),
        raw_questions=tuple(item.raw for item in loaded),
    )

def load_step1_bank(root: str | Path | None = None) -> list[Step1QuestionRecord]:
    """Load the 556-question public Step 1 bank."""
    return list(load_step1_corpus(root).questions)

def load_step1_baseline(root: str | Path | None = None) -> list[Step1QuestionRecord]:
    """Load the 25-item baseline set in manifest order."""
    corpus = load_step1_corpus(root)
    by_id = {question.id: question for question in corpus.questions}
    return [by_id[question_id] for question_id in corpus.baseline_question_ids]
