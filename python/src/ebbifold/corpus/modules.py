"""Fail-closed clinical-module corpus loading and release validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fingerprint import module_source_fingerprint, public_serving_fingerprint
from .loader_types import LoadedQuestion, ModuleCorpus
from .models import ModuleQuestionRecord
from .questions import _load_question_tree
from .schema import CorpusAdmissionError
from .validation import (
    _read_json,
    _read_question_manifest,
    open_content_root,
)


def _read_module_registry(registry_path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        registry = _read_json(registry_path)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [f"{registry_path}: invalid source registry ({error})"]
    sources = registry.get("sources")
    if registry.get("schemaVersion") != 1 or not isinstance(sources, dict):
        return {}, [f"{registry_path}: expected schemaVersion 1 with a sources map"]
    return sources, []

def _module_membership_errors(
    release_path: Path, on_disk: set[str], release_ids: list[str]
) -> list[str]:
    released = set(release_ids)
    return [
        *(f"{release_path}: {question_id} is on disk but not in the release"
          for question_id in sorted(on_disk) if question_id not in released),
        *(f"{release_path}: {question_id} is released but its file is missing"
          for question_id in release_ids if question_id not in on_disk),
    ]

def _module_release_maps(
    release: Any, release_path: Path
) -> tuple[dict[str, Any], dict[str, Any], Any, list[str]]:
    fingerprints = release.get("questionFingerprints") if isinstance(release, dict) else None
    release_sources = release.get("questionSources") if isinstance(release, dict) else None
    source_fingerprints = (
        release.get("questionSourceFingerprints") if isinstance(release, dict) else None
    )
    errors: list[str] = []
    if not isinstance(fingerprints, dict):
        errors.append(f"{release_path}: questionFingerprints must be an object")
        fingerprints = {}
    if not isinstance(release_sources, dict):
        errors.append(f"{release_path}: questionSources must be an object")
        release_sources = {}
    return fingerprints, release_sources, source_fingerprints, errors

def _module_source_error(source: Any) -> bool:
    return (
        not isinstance(source, dict)
        or not str(source.get("url", "")).startswith("https://")
        or not source.get("verifiedAt")
    )

@dataclass(frozen=True)
class _ModuleLaneContext:
    release_path: Path
    released: set[str]
    fingerprints: dict[str, Any]
    release_sources: dict[str, Any]
    source_fingerprints: Any
    sources: dict[str, Any]
    original: bool


def _module_release_item_errors(
    record: ModuleQuestionRecord, item: LoadedQuestion, context: _ModuleLaneContext
) -> list[str]:
    errors: list[str] = []
    actual = public_serving_fingerprint(item.raw, source_file=item.source_file)
    if record.id in context.released and context.fingerprints.get(record.id) != actual:
        errors.append(
            f"{context.release_path}: {record.id} serving fingerprint does not match its file"
        )
    cited = record.public_module.evidence.source_id
    if record.id in context.released and context.release_sources.get(record.id) != cited:
        errors.append(f"{context.release_path}: {record.id} release source differs from its file")
    return errors


def _module_source_item_errors(
    record: ModuleQuestionRecord, item: LoadedQuestion, context: _ModuleLaneContext
) -> list[str]:
    cited = record.public_module.evidence.source_id
    source = context.sources.get(cited)
    if _module_source_error(source):
        return [f"{item.source_file}: cited source is absent from the checked-in registry"]
    if not context.original:
        return []
    expected = (
        context.source_fingerprints.get(record.id)
        if isinstance(context.source_fingerprints, dict)
        else None
    )
    if expected != module_source_fingerprint(cited, source):
        return [f"{context.release_path}: {record.id} citation differs from its reviewed release"]
    return []


def _module_item_errors(item: LoadedQuestion, context: _ModuleLaneContext) -> list[str]:
    record = item.record
    if not isinstance(record, ModuleQuestionRecord):
        return []
    return [
        *_module_release_item_errors(record, item, context),
        *_module_source_item_errors(record, item, context),
    ]

def _validate_module_lane(
    items: list[LoadedQuestion],
    *,
    release_path: Path,
    registry_path: Path,
    original: bool,
) -> tuple[list[str], list[str]]:
    sources, errors = _read_module_registry(registry_path)
    release, release_ids, manifest_errors = _read_question_manifest(
        release_path, 1, "release manifest"
    )
    errors.extend(manifest_errors)

    on_disk = {item.record.id for item in items}
    released = set(release_ids)
    errors.extend(_module_membership_errors(release_path, on_disk, release_ids))
    fingerprints, release_sources, source_fingerprints, map_errors = _module_release_maps(
        release, release_path
    )
    errors.extend(map_errors)

    context = _ModuleLaneContext(
        release_path=release_path,
        released=released,
        fingerprints=fingerprints,
        release_sources=release_sources,
        source_fingerprints=source_fingerprints,
        sources=sources,
        original=original,
    )
    for item in items:
        errors.extend(_module_item_errors(item, context))
    return release_ids, errors

def load_module_corpus(root: str | Path | None = None) -> ModuleCorpus:
    content_root = open_content_root(root)
    repo_root = content_root.parent
    module_root = content_root / "modules"
    mirrored, errors = _load_question_tree(module_root / "questions", repo_root=repo_root)
    originals, original_errors = _load_question_tree(
        module_root / "original-questions", repo_root=repo_root
    )
    errors.extend(original_errors)

    release_ids, lane_errors = _validate_module_lane(
        mirrored,
        release_path=module_root / "release-v1.json",
        registry_path=module_root / "sources.json",
        original=False,
    )
    errors.extend(lane_errors)
    original_release_ids, original_lane_errors = _validate_module_lane(
        originals,
        release_path=module_root / "original-questions-release-v1.json",
        registry_path=module_root / "original-questions-sources.json",
        original=True,
    )
    errors.extend(original_lane_errors)

    if errors:
        raise CorpusAdmissionError(errors)
    return ModuleCorpus(
        questions=tuple(item.record for item in mirrored if isinstance(item.record, ModuleQuestionRecord)),
        original_questions=tuple(
            item.record for item in originals if isinstance(item.record, ModuleQuestionRecord)
        ),
        release_question_ids=tuple(release_ids),
        original_release_question_ids=tuple(original_release_ids),
    )

def load_module_questions(
    root: str | Path | None = None, *, include_original: bool = False
) -> list[ModuleQuestionRecord]:
    """Load the 392 released clinical-module questions.

    The 36 authored original-lane questions are validated by the corpus gate but
    are not part of the published 392 count unless ``include_original`` is true.
    """
    corpus = load_module_corpus(root)
    questions = list(corpus.questions)
    if include_original:
        questions.extend(corpus.original_questions)
    return questions
