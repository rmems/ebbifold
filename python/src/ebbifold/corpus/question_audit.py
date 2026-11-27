"""Per-record corpus audit gates."""

from __future__ import annotations

from typing import Any

from .fingerprint import module_item_content_hash
from .models import CorpusQuestion, ModuleQuestionRecord, PassageEvidence, Step1QuestionRecord

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
