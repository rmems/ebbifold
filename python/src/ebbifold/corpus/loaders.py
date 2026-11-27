"""Fail-closed loaders for the frozen ``open-content/`` corpus."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .audit import CorpusAudit, CorpusCounts, audit_answer_key
from .fingerprint import (
    BASELINE_MODULE,
    cohort_card_serving_fingerprint,
    module_item_content_hash,
    module_source_fingerprint,
    open_usmle_quote_set_sha256,
    open_usmle_registry_sha256,
    public_serving_fingerprint,
)
from .models import (
    AttemptRecord,
    ClozeCardRecord,
    ClozeCardShard,
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
    find_repo_root,
    validate_against_schema,
)

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
EXPECTED_CORPUS_COUNTS = {
    "json_files": 1357,
    "step1_questions": 556,
    "baseline_questions": 25,
    "module_questions": 392,
    "cloze_cards": 2136,
}
JSON_GLOB = "*.json"
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


def open_content_root(root: str | Path | None = None) -> Path:
    if root is None:
        return find_repo_root() / "open-content"
    path = Path(root).resolve()
    if (path / "open-content").is_dir():
        return path / "open-content"
    if (path / "usmle").is_dir() and (path / "modules").is_dir():
        return path
    raise FileNotFoundError(f"{path} is not an ebbifold repo root or open-content directory")


def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_error_messages(source: str, error: ValidationError) -> list[str]:
    messages = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", ())) or "(root)"
        messages.append(f"{source}: {location}: {issue.get('msg', 'invalid value')}")
    return messages


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


def _load_question_tree(
    directory: Path,
    *,
    repo_root: Path,
) -> tuple[list[LoadedQuestion], list[str]]:
    errors: list[str] = []
    loaded: list[LoadedQuestion] = []
    for path in sorted(directory.rglob(JSON_GLOB)):
        source = _repo_relative(path, repo_root)
        try:
            raw = _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{source}: invalid JSON ({error})")
            continue
        try:
            record = admit_question_payload(raw, source=source, repo_root=repo_root)
        except CorpusAdmissionError as error:
            errors.extend(error.errors)
            continue
        loaded.append(LoadedQuestion(record=record, raw=raw, source_file=source))

    seen: set[str] = set()
    for item in loaded:
        if item.record.id in seen:
            errors.append(f"{item.source_file}: duplicate question id {item.record.id}")
        seen.add(item.record.id)
    return loaded, errors


def _expect_manifest_object(raw: Any, path: Path, version: int) -> list[str]:
    if not isinstance(raw, dict):
        return [f"{path}: expected manifest object"]
    if raw.get("schemaVersion") != version:
        return [f"{path}: expected schemaVersion {version}"]
    return []


def _id_list(raw: Any, key: str, path: Path) -> tuple[list[str], list[str]]:
    if not isinstance(raw, dict):
        return [], [f"{path}: {key} is unavailable because the manifest is not an object"]
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        return [], [f"{path}: {key} must be a non-empty string array"]
    duplicates = sorted({item for item in value if value.count(item) > 1})
    errors = [f"{path}: duplicate {key} entry {item}" for item in duplicates]
    return list(value), errors


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


def _index_step1_sources(
    registry_path: Path, sources: list[Any]
) -> tuple[dict[str, dict[str, Any]], set[str], list[str]]:
    errors: list[str] = []
    source_by_id: dict[str, dict[str, Any]] = {}
    passage_keys: set[str] = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str):
            errors.append(f"{registry_path}: source entry missing id")
            continue
        source_id = source["id"]
        if source_id in source_by_id:
            errors.append(f"{registry_path}: duplicate source id {source_id}")
        source_by_id[source_id] = source
        passage_ids: set[str] = set()
        for passage in source.get("passages", []):
            if not isinstance(passage, dict) or not isinstance(passage.get("id"), str):
                continue
            passage_id = passage["id"]
            if passage_id in passage_ids:
                errors.append(
                    f"{registry_path}: duplicate passage id {passage_id} for source {source_id}"
                )
            passage_ids.add(passage_id)
            passage_keys.add(f"{source_id}:{passage_id}")
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


def _read_question_manifest(
    path: Path, version: int, invalid_label: str
) -> tuple[Any, list[str], list[str]]:
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [], [f"{path}: invalid {invalid_label} ({error})"]
    errors = _expect_manifest_object(raw, path, version)
    question_ids, id_errors = _id_list(raw, "questionIds", path)
    errors.extend(id_errors)
    return raw, question_ids, errors


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


def _module_item_errors(
    item: LoadedQuestion,
    *,
    release_path: Path,
    released: set[str],
    fingerprints: dict[str, Any],
    release_sources: dict[str, Any],
    source_fingerprints: Any,
    sources: dict[str, Any],
    original: bool,
) -> list[str]:
    record = item.record
    if not isinstance(record, ModuleQuestionRecord):
        return []
    errors: list[str] = []
    actual = public_serving_fingerprint(item.raw, source_file=item.source_file)
    if record.id in released and fingerprints.get(record.id) != actual:
        errors.append(f"{release_path}: {record.id} serving fingerprint does not match its file")
    cited = record.public_module.evidence.source_id
    if record.id in released and release_sources.get(record.id) != cited:
        errors.append(f"{release_path}: {record.id} release source differs from its file")
    source = sources.get(cited)
    if _module_source_error(source):
        errors.append(f"{item.source_file}: cited source is absent from the checked-in registry")
    elif original:
        expected = (
            source_fingerprints.get(record.id) if isinstance(source_fingerprints, dict) else None
        )
        if expected != module_source_fingerprint(cited, source):
            errors.append(f"{release_path}: {record.id} citation differs from its reviewed release")
    return errors


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

    for item in items:
        errors.extend(
            _module_item_errors(
                item,
                release_path=release_path,
                released=released,
                fingerprints=fingerprints,
                release_sources=release_sources,
                source_fingerprints=source_fingerprints,
                sources=sources,
                original=original,
            )
        )
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


def _card_shard_suffix(card: ClozeCardRecord) -> str:
    match = re.search(r":c-([0-9a-f])", card.id)
    if not match:
        raise ValueError(f"unexpected public card id {card.id}")
    return f"open-content/modules/cards/{card.discipline}/{match.group(1)}.json"


def _card_placement_errors(
    card: ClozeCardRecord, shard: ClozeCardShard, source: str
) -> list[str]:
    errors: list[str] = []
    try:
        expected_suffix = _card_shard_suffix(card)
    except ValueError as error:
        errors.append(f"{source}: {card.id}: {error}")
        expected_suffix = None
    if card.discipline != shard.discipline:
        errors.append(f"{source}: {card.id}: discipline does not match shard")
    if expected_suffix is not None and source != expected_suffix:
        errors.append(f"{source}: {card.id}: belongs in {expected_suffix}")
    return errors


def _load_card_shard(
    path: Path, repo_root: Path
) -> tuple[list[tuple[ClozeCardRecord, dict[str, Any], str]], list[str]]:
    source = _repo_relative(path, repo_root)
    try:
        raw_shard = _read_json(path)
        shard = ClozeCardShard.model_validate(raw_shard)
    except (OSError, json.JSONDecodeError) as error:
        return [], [f"{source}: invalid JSON ({error})"]
    except ValidationError as error:
        return [], _validation_error_messages(source, error)

    errors: list[str] = []
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]] = []
    for raw_card, card in zip(raw_shard.get("cards", []), shard.cards, strict=True):
        errors.extend(_card_placement_errors(card, shard, source))
        cards.append((card, raw_card, source))
    return cards, errors


def _read_card_release(release_path: Path) -> tuple[Any, list[str], list[str]]:
    try:
        release = _read_json(release_path)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [], [f"{release_path}: invalid card release ({error})"]
    errors = _expect_manifest_object(release, release_path, 1)
    release_ids, id_errors = _id_list(release, "cardIds", release_path)
    errors.extend(id_errors)
    return release, release_ids, errors


def _index_cards(
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]]
) -> tuple[dict[str, tuple[ClozeCardRecord, dict[str, Any], str]], list[str]]:
    errors: list[str] = []
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]] = {}
    for card, raw, source in cards:
        if card.id in on_disk:
            errors.append(f"{source}: duplicate card id {card.id}")
        on_disk[card.id] = (card, raw, source)
    return on_disk, errors


def _card_membership_errors(
    release_path: Path,
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]],
    release_ids: list[str],
) -> list[str]:
    released = set(release_ids)
    return [
        *(f"{release_path}: {card_id} is on disk but not in the release"
          for card_id in sorted(on_disk) if card_id not in released),
        *(f"{release_path}: {card_id} is released but its file is missing"
          for card_id in release_ids if card_id not in on_disk),
    ]


def _card_fingerprint_errors(
    release_path: Path,
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]],
    release_ids: list[str],
    fingerprints: dict[str, Any],
) -> list[str]:
    released = set(release_ids)
    return [
        f"{release_path}: {card_id} serving fingerprint does not match its file"
        for card_id, (_, raw, _) in on_disk.items()
        if card_id in released and fingerprints.get(card_id) != cohort_card_serving_fingerprint(raw)
    ]


def load_cloze_card_corpus(root: str | Path | None = None) -> ClozeCardCorpus:
    content_root = open_content_root(root)
    repo_root = content_root.parent
    module_root = content_root / "modules"
    release_path = module_root / "cards-release-v1.json"
    errors: list[str] = []
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]] = []

    for path in sorted((module_root / "cards").rglob(JSON_GLOB)):
        shard_cards, shard_errors = _load_card_shard(path, repo_root)
        cards.extend(shard_cards)
        errors.extend(shard_errors)

    release, release_ids, release_errors = _read_card_release(release_path)
    errors.extend(release_errors)
    fingerprints = release.get("cardFingerprints") if isinstance(release, dict) else None
    if not isinstance(fingerprints, dict):
        errors.append(f"{release_path}: cardFingerprints must be an object")
        fingerprints = {}

    on_disk, index_errors = _index_cards(cards)
    errors.extend(index_errors)
    errors.extend(_card_membership_errors(release_path, on_disk, release_ids))
    errors.extend(_card_fingerprint_errors(release_path, on_disk, release_ids, fingerprints))

    if errors:
        raise CorpusAdmissionError(errors)
    return ClozeCardCorpus(
        cards=tuple(card for card, _, _ in cards), release_card_ids=tuple(release_ids)
    )


def load_cloze_cards(root: str | Path | None = None) -> list[ClozeCardRecord]:
    return list(load_cloze_card_corpus(root).cards)


def load_corpus(root: str | Path | None = None) -> CorpusSnapshot:
    """Load and audit the complete frozen corpus, fail-closed.

    Every JSON file under ``open-content/`` is parsed. Question, card, manifest,
    citation, release-membership, fingerprint, count, and answer-key gates must
    all pass before a snapshot is returned.
    """
    content_root = open_content_root(root)
    json_files = sorted(content_root.rglob(JSON_GLOB))
    parse_errors: list[str] = []
    for path in json_files:
        try:
            _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            parse_errors.append(f"{path.relative_to(content_root.parent).as_posix()}: invalid JSON ({error})")
    if parse_errors:
        raise CorpusAdmissionError(parse_errors)

    loader_errors: list[str] = []
    try:
        step1 = load_step1_corpus(content_root)
    except CorpusAdmissionError as error:
        loader_errors.extend(error.errors)
        step1 = None
    try:
        modules = load_module_corpus(content_root)
    except CorpusAdmissionError as error:
        loader_errors.extend(error.errors)
        modules = None
    try:
        cards = load_cloze_card_corpus(content_root)
    except CorpusAdmissionError as error:
        loader_errors.extend(error.errors)
        cards = None
    if loader_errors or step1 is None or modules is None or cards is None:
        raise CorpusAdmissionError(loader_errors)

    step1_by_id = {question.id: question for question in step1.questions}
    baseline_questions = tuple(step1_by_id[question_id] for question_id in step1.baseline_question_ids)
    step1_answer_key = audit_answer_key(list(step1.questions))
    baseline_answer_key = audit_answer_key(list(baseline_questions))
    audit_errors = [
        f"step1 answer-key integrity: {issue['code']}" for issue in step1_answer_key.issues
    ] + [
        f"baseline answer-key integrity: {issue['code']}" for issue in baseline_answer_key.issues
    ]
    counts = CorpusCounts(
        json_files=len(json_files),
        step1_questions=len(step1.questions),
        baseline_questions=len(baseline_questions),
        module_questions=len(modules.questions),
        original_module_questions=len(modules.original_questions),
        cloze_cards=len(cards.cards),
    )
    for field, expected in EXPECTED_CORPUS_COUNTS.items():
        actual = getattr(counts, field)
        if actual != expected:
            audit_errors.append(f"corpus count {field}: expected {expected}, found {actual}")
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
