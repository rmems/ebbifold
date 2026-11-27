"""Fixture-pipeline tests for Phase 0.

The golden fixtures under tests/fixtures/ are generated once by
python/tools/generate_step1_adaptive_fixtures.ts (a TypeScript script run
against the upstream original) and committed. Python never executes
TypeScript; these tests prove the committed artifacts are loadable and
structurally sound, so Phase 1's parity tests can rely on them.

The invariants checked here hold of rankAdaptive's *output* regardless of
implementation, so they stay true after the Python port exists.
"""

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "step1-adaptive.cases.json"

REQUIRED_QUESTION_KEYS = {"id"}
REQUIRED_HISTORY_KEYS = {"questionId", "isCorrect", "createdAt"}


@pytest.fixture(scope="module")
def cases() -> list[dict]:
    with FIXTURE_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["source"] == "src/lib/usmle/step1-adaptive.ts"
    return data["cases"]


def test_fixture_file_exists() -> None:
    assert FIXTURE_PATH.is_file(), (
        "fixtures missing — regenerate with "
        "`node --import tsx python/tools/generate_step1_adaptive_fixtures.ts` "
        "from the repository root"
    )


def test_cases_are_present(cases: list[dict]) -> None:
    names = [c["name"] for c in cases]
    assert len(names) == len(set(names)), "case names must be unique"
    assert len(cases) >= 8, "scaffold shipped with 9 scenarios; do not drop coverage"


def test_case_structure(cases: list[dict]) -> None:
    for case in cases:
        questions = case["input"]["questions"]
        assert questions, f"{case['name']}: empty question pool"
        ids = [q["id"] for q in questions]
        assert len(ids) == len(set(ids)), f"{case['name']}: duplicate question ids"
        assert REQUIRED_QUESTION_KEYS <= questions[0].keys()
        for row in case["input"]["history"]:
            assert REQUIRED_HISTORY_KEYS <= row.keys()
            # ISO 8601 strings, not epoch numbers — Date is not JSON-portable.
            assert "T" in row["createdAt"], f"{case['name']}: createdAt is not ISO 8601"
        assert case["input"]["size"] >= 1


def test_expected_order_is_well_formed(cases: list[dict]) -> None:
    for case in cases:
        order = case["expected"]["order"]
        pool_ids = {q["id"] for q in case["input"]["questions"]}
        assert len(order) == len(set(order)), f"{case['name']}: duplicate ids in output"
        assert set(order) <= pool_ids, f"{case['name']}: output ids outside the pool"
        assert len(order) <= case["input"]["size"], (
            f"{case['name']}: output longer than requested size"
        )


def test_answered_questions_never_reappear(cases: list[dict]) -> None:
    for case in cases:
        answered = {row["questionId"] for row in case["input"]["history"]}
        assert not answered & set(case["expected"]["order"]), (
            f"{case['name']}: an answered question was served again"
        )


def test_output_length_matches_available_pool(cases: list[dict]) -> None:
    """rankAdaptive serves min(size, unanswered) questions — never a dead end."""
    for case in cases:
        answered = {row["questionId"] for row in case["input"]["history"]}
        unanswered = sum(
            1 for q in case["input"]["questions"] if q["id"] not in answered
        )
        expected_len = min(case["input"]["size"], unanswered)
        assert len(case["expected"]["order"]) == expected_len, case["name"]
