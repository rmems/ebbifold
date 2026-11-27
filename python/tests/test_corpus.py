"""Phase 2 corpus-layer gate tests.

These tests intentionally read the real frozen corpus. The corpus itself is the
contract: if one byte of content, release membership, citation provenance, or a
reviewed fingerprint drifts, admission must fail closed rather than partially
load a bank.
"""

import json
from pathlib import Path

import pytest

from ebbifold.corpus import (
    CorpusAdmissionError,
    admit_attempt_payload,
    admit_question_payload,
    load_cloze_cards,
    load_corpus,
    load_module_questions,
    load_step1_bank,
    load_step1_baseline,
    load_step1_corpus,
)
from ebbifold.corpus.fingerprint import BASELINE_MODULE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def snapshot():
    return load_corpus(REPO_ROOT)


def test_full_open_content_tree_loads_with_zero_schema_failures(snapshot) -> None:
    assert snapshot.audit.counts.json_files == 1357
    assert snapshot.audit.errors == []
    assert snapshot.audit.ok


def test_published_corpus_counts_match_upstream(snapshot) -> None:
    counts = snapshot.audit.counts
    assert counts.step1_questions == 556
    assert counts.baseline_questions == 25
    assert counts.module_questions == 392
    assert counts.cloze_cards == 2136


def test_loaders_expose_step1_baseline_modules_and_cards() -> None:
    assert len(load_step1_bank(REPO_ROOT)) == 556
    assert len(load_step1_baseline(REPO_ROOT)) == 25
    assert len(load_module_questions(REPO_ROOT)) == 392
    assert len(load_module_questions(REPO_ROOT, include_original=True)) == 428
    assert len(load_cloze_cards(REPO_ROOT)) == 2136


def test_baseline_records_keep_manifest_order_and_module_tag() -> None:
    corpus = load_step1_corpus(REPO_ROOT)
    baseline = load_step1_baseline(REPO_ROOT)
    assert [question.id for question in baseline] == list(corpus.baseline_question_ids)
    assert all(BASELINE_MODULE in question.module_nodes for question in baseline)


def test_citation_and_answer_key_audit(snapshot) -> None:
    question_count = (
        len(snapshot.step1_questions)
        + len(snapshot.module_questions)
        + len(snapshot.original_module_questions)
    )
    assert snapshot.audit.citations_checked == question_count
    assert snapshot.audit.release_members_checked == (
        len(snapshot.step1_questions)
        + len(snapshot.module_questions)
        + len(snapshot.original_module_questions)
        + len(snapshot.cloze_cards)
    )
    assert snapshot.audit.step1_answer_key.issues == []
    assert snapshot.audit.baseline_answer_key.issues == []


def test_corrupted_fixture_record_is_rejected_by_admission_gate() -> None:
    payload = json.loads(
        (FIXTURES / "corrupted-question.v1.json").read_text(encoding="utf-8")
    )
    with pytest.raises(CorpusAdmissionError) as raised:
        admit_question_payload(
            payload,
            source="python/tests/fixtures/corrupted-question.v1.json",
            repo_root=REPO_ROOT,
        )
    assert "exactly one correct option" in str(raised.value)


def test_attempt_record_schema_and_model_admit_a_valid_attempt() -> None:
    payload = {
        "schemaVersion": 1,
        "attemptId": "attempt-1",
        "sessionId": "session-1",
        "questionId": "bank:usmle-step1:biochem-acetyl-coa-gate:v1",
        "selectedLabel": "A",
        "isCorrect": True,
        "attemptNumber": 1,
        "answeredAt": "2026-11-27T06:00:00Z",
        "timeToAnswerMs": 4500,
    }
    attempt = admit_attempt_payload(payload, repo_root=REPO_ROOT)
    assert attempt.question_id == payload["questionId"]
    assert attempt.attempt_number == 1


def test_image_media_requires_public_foss_licence() -> None:
    payload = json.loads(
        (FIXTURES / "corrupted-question.v1.json").read_text(encoding="utf-8")
    )
    payload["options"][0]["isCorrect"] = True
    payload["imageUrl"] = "/figures/usmle/step1/example.svg"
    payload["publicUsmle"]["media"] = {
        "kind": "asset",
        "assetId": "example",
        "job": "imaging",
        "showWhen": "always",
        "licence": {"cls": "verify", "id": "cc-by-4.0"},
        "attribution": "Example attribution",
    }
    with pytest.raises(CorpusAdmissionError) as raised:
        admit_question_payload(payload, repo_root=REPO_ROOT)
    assert "media licence verify/cc-by-4.0 is not approved FOSS" in str(raised.value)


def test_attempt_record_schema_rejects_timezone_free_timestamp() -> None:
    payload = {
        "schemaVersion": 1,
        "attemptId": "attempt-1",
        "sessionId": "session-1",
        "questionId": "bank:usmle-step1:biochem-acetyl-coa-gate:v1",
        "selectedLabel": "A",
        "isCorrect": True,
        "attemptNumber": 1,
        "answeredAt": "2026-11-27T06:00:00",
    }
    with pytest.raises(CorpusAdmissionError) as raised:
        admit_attempt_payload(payload, repo_root=REPO_ROOT)
    assert "date-time" in str(raised.value)


def test_attempt_record_schema_rejects_missing_answer_result() -> None:
    payload = {
        "schemaVersion": 1,
        "attemptId": "attempt-1",
        "sessionId": "session-1",
        "questionId": "bank:usmle-step1:biochem-acetyl-coa-gate:v1",
        "selectedLabel": "A",
        "attemptNumber": 1,
        "answeredAt": "2026-11-27T06:00:00Z",
    }
    with pytest.raises(CorpusAdmissionError):
        admit_attempt_payload(payload, repo_root=REPO_ROOT)
