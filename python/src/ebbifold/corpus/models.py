"""Pydantic records for the frozen open-content corpus.

The JSON files use camelCase because they are the TypeScript corpus contract.
Python fields use snake_case aliases, but validation remains schema-first:
loaders validate the raw JSON document before constructing these models.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class CorpusModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class QuestionOption(CorpusModel):
    label: NonEmptyStr
    text: NonEmptyStr
    is_correct: bool = Field(alias="isCorrect")
    misconception: str | None = None
    explanation: str | None = None


class EvidenceLicence(CorpusModel):
    cls: Literal["foss", "verify", "unknown"]
    id: NonEmptyStr


class NoEvidence(CorpusModel):
    kind: Literal["none"]


class ReferenceEvidence(CorpusModel):
    kind: Literal["reference"]
    source_id: NonEmptyStr = Field(alias="sourceId")
    licence: EvidenceLicence


class PassageEvidence(CorpusModel):
    kind: Literal["passage"]
    source_id: NonEmptyStr = Field(alias="sourceId")
    passage_id: NonEmptyStr = Field(alias="passageId")
    licence: EvidenceLicence


PublicUsmleEvidence = Annotated[
    NoEvidence | ReferenceEvidence | PassageEvidence,
    Field(discriminator="kind"),
]


class NoMedia(CorpusModel):
    kind: Literal["none"]


class AssetMedia(CorpusModel):
    kind: Literal["asset"]
    asset_id: NonEmptyStr = Field(alias="assetId")
    job: Literal[
        "pathway",
        "mechanism",
        "anatomy",
        "compare-table",
        "finding-exemplar",
        "smear",
        "histo",
        "imaging",
        "trace",
    ]
    show_when: Literal["after-reveal", "always"] = Field(alias="showWhen")
    licence: EvidenceLicence
    attribution: NonEmptyStr
    source_url: str | None = Field(default=None, alias="sourceUrl")
    content_hash: Annotated[str, StringConstraints(pattern=r"^sha256:[a-f0-9]{64}$")] | None = Field(
        default=None, alias="contentHash"
    )
    clinical_condition_ids: list[NonEmptyStr] | None = Field(
        default=None, alias="clinicalConditionIds"
    )
    clinical_key_findings: list[NonEmptyStr] | None = Field(
        default=None, alias="clinicalKeyFindings"
    )


PublicUsmleMedia = Annotated[NoMedia | AssetMedia, Field(discriminator="kind")]


class ItemTextRights(CorpusModel):
    licence: Literal["CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "restricted", "unknown"]
    attribution: NonEmptyStr


class PublicUsmleProvenance(CorpusModel):
    schema_version: Literal[1] = Field(alias="schemaVersion")
    origin: Literal[
        "authored",
        "generated",
        "anking",
        "y3g-reshape",
        "legacy-instagram",
        "first-aid-derived",
        "source-derived",
        "unknown",
    ]
    item_text: ItemTextRights = Field(alias="itemText")
    evidence: PublicUsmleEvidence
    media: PublicUsmleMedia | None = None


class ModuleItemTextRights(CorpusModel):
    licence: Literal["CC-BY-4.0"]
    attribution: NonEmptyStr


class ModuleEvidence(CorpusModel):
    kind: Literal["reference"]
    source_id: NonEmptyStr = Field(alias="sourceId")


class ModuleReview(CorpusModel):
    verdict: Literal["passed", "needs-fix"]
    content_hash: Sha256 = Field(alias="contentHash")


class PublicModuleProvenance(CorpusModel):
    schema_version: Literal[1] = Field(alias="schemaVersion")
    origin: Literal["mirrored", "original"]
    discipline: NonEmptyStr
    mirrored_from: NonEmptyStr | None = Field(default=None, alias="mirroredFrom")
    original_id: NonEmptyStr | None = Field(default=None, alias="originalId")
    item_text: ModuleItemTextRights = Field(alias="itemText")
    evidence: ModuleEvidence
    review: ModuleReview

    @model_validator(mode="after")
    def origin_trace_is_present(self) -> "PublicModuleProvenance":
        if self.origin == "mirrored" and not self.mirrored_from:
            raise ValueError("mirrored items require mirroredFrom")
        if self.origin == "original" and not self.original_id:
            raise ValueError("original items require originalId")
        return self


class QuestionRecord(CorpusModel):
    id: NonEmptyStr
    rotation: NonEmptyStr
    topics: list[NonEmptyStr]
    module_nodes: list[NonEmptyStr] = Field(alias="moduleNodes")
    ladder_id: NonEmptyStr | None = Field(default=None, alias="ladderId")
    question_type: NonEmptyStr | None = Field(alias="questionType")
    difficulty: NonEmptyStr | None
    stem: NonEmptyStr
    options: list[QuestionOption]
    context: str | None
    image_url: NonEmptyStr | None = Field(default=None, alias="imageUrl")
    image_caption: NonEmptyStr | None = Field(default=None, alias="imageCaption")


class Step1QuestionRecord(QuestionRecord):
    rotation: Literal["usmle-step1-open"]
    cite: NonEmptyStr
    public_usmle: PublicUsmleProvenance = Field(alias="publicUsmle")


class ModuleQuestionRecord(QuestionRecord):
    rotation: Literal["cohort-open"]
    public_module: PublicModuleProvenance = Field(alias="publicModule")


CorpusQuestion = Step1QuestionRecord | ModuleQuestionRecord


class AttemptRecord(CorpusModel):
    schema_version: Literal[1] = Field(alias="schemaVersion")
    attempt_id: NonEmptyStr = Field(alias="attemptId")
    session_id: NonEmptyStr = Field(alias="sessionId")
    question_id: NonEmptyStr = Field(alias="questionId")
    selected_label: NonEmptyStr | None = Field(alias="selectedLabel")
    is_correct: bool = Field(alias="isCorrect")
    attempt_number: int = Field(alias="attemptNumber", ge=1)
    answered_at: datetime = Field(alias="answeredAt")
    time_to_answer_ms: int | None = Field(default=None, alias="timeToAnswerMs", ge=0)
    difficulty_at_attempt: NonEmptyStr | None = Field(default=None, alias="difficultyAtAttempt")
    domain_at_attempt: NonEmptyStr | None = Field(default=None, alias="domainAtAttempt")


class CardReference(CorpusModel):
    title: NonEmptyStr
    publisher: str | None
    url: NonEmptyStr


class ClozeCardRecord(CorpusModel):
    id: NonEmptyStr
    origin: NonEmptyStr
    discipline: NonEmptyStr
    card_type: Literal["cloze"] = Field(alias="cardType")
    source_component: NonEmptyStr = Field(alias="sourceComponent")
    front: NonEmptyStr
    back: NonEmptyStr
    context: str | None
    complexity: int
    importance: Literal[1, 2, 3]
    variant_group: str | None = Field(alias="variantGroup")
    variant_index: int | None = Field(alias="variantIndex")
    image_url: NonEmptyStr | None = Field(default=None, alias="imageUrl")
    image_caption: NonEmptyStr | None = Field(default=None, alias="imageCaption")
    reference: CardReference | None

    @model_validator(mode="after")
    def single_blank_cloze(self) -> "ClozeCardRecord":
        if len(self.front.split("[___]")) != 2:
            raise ValueError("cloze front must contain exactly one [___] blank")
        if not self.back.strip():
            raise ValueError("cloze back must not be empty")
        return self


class ClozeCardShard(CorpusModel):
    schema_version: Literal[1] = Field(alias="schemaVersion")
    discipline: NonEmptyStr
    licence: Literal["CC-BY-4.0"]
    attribution: Literal["MD3 contributors"]
    cards: list[ClozeCardRecord]
