"""Phase 3a parity: unified concept-space scheduler vs TypeScript golden fixtures.

Fixtures come from python/tools/generate_unified_scheduler_fixtures.ts (run once
against the TypeScript original, committed). These tests never execute
TypeScript and fail, rather than skip, when the fixture file is missing,
empty, or malformed.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from ebbifold.scheduler import (
    SchedulerDeterminismError,
    UnsupportedSchedulerInput,
    plan_concepts,
    run_build_script,
)
from ebbifold.scheduler import attribution, config, difficulty, scoring, topics

FIXTURE = Path(__file__).parent / "fixtures" / "unified-scheduler.cases.json"
REL_TOL = 1e-12


def _load() -> dict[str, Any]:
    if not FIXTURE.is_file():
        pytest.fail(f"golden fixture missing: {FIXTURE}; regenerate, do not skip")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for section in ("conceptPlan", "buildScripts", "units"):
        if not data.get(section):
            pytest.fail(f"golden fixture section {section!r} is empty")
    return data


DATA = _load()


def assert_parity(actual: Any, expected: Any, path: str = "$") -> None:
    """Exact on structure, strings, bools and ordering; floats within 1e-12."""
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float)) and not isinstance(actual, bool), f"{path}: {actual!r}"
        assert math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=REL_TOL), f"{path}: {actual!r} != {expected!r}"
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"{path}: {actual!r} != {expected!r}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_parity(a, e, f"{path}[{i}]")
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: {actual!r}"
        assert sorted(actual) == sorted(expected), f"{path}: keys {sorted(actual)} != {sorted(expected)}"
        for key in expected:
            assert_parity(actual[key], expected[key], f"{path}.{key}")
    else:  # pragma: no cover - fixtures are JSON
        pytest.fail(f"{path}: unsupported fixture value {expected!r}")


@pytest.mark.parametrize("case", DATA["conceptPlan"], ids=lambda c: c["name"])
def test_concept_plan_parity(case: dict[str, Any]) -> None:
    assert_parity(plan_concepts(copy.deepcopy(case["snapshot"])), case["expected"])


@pytest.mark.parametrize("case", DATA["buildScripts"], ids=lambda c: c["name"])
def test_build_state_parity(case: dict[str, Any]) -> None:
    actual = run_build_script(copy.deepcopy(case["script"]))
    trace = actual.pop("trace")
    assert_parity(actual, case["expected"])
    added = [r["id"] for r in trace if r["op"] == "add" and r["result"] == "added"]
    assert added == [s.split(":", 1)[1] for s in case["expected"]["selectedIds"]]


@pytest.mark.parametrize("case", DATA["conceptPlan"], ids=lambda c: c["name"])
def test_repeated_construction_is_deterministic(case: dict[str, Any]) -> None:
    runs = [json.dumps(plan_concepts(copy.deepcopy(case["snapshot"])), sort_keys=True) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("case", DATA["buildScripts"], ids=lambda c: c["name"])
def test_repeated_build_script_is_deterministic(case: dict[str, Any]) -> None:
    runs = [json.dumps(run_build_script(copy.deepcopy(case["script"])), sort_keys=True) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


UNITS = DATA["units"]


def test_stable_target_random_parity() -> None:
    for row in UNITS["stableTargetRandom"]:
        assert_parity(config.stable_target_random(DATA["seed"], row["key"]), row["value"], row["key"])


def test_scalar_scoring_parity() -> None:
    for row in UNITS["examPressure"]:
        assert_parity(scoring.compute_exam_pressure(row["daysToExam"]), row["value"])
    for row in UNITS["resolveMaxNewCards"]:
        value = scoring.resolve_max_new_cards(row["explicit"], row["mode"])
        assert_parity("Infinity" if value == math.inf else value, row["value"])
    for row in UNITS["recallSoftPenalty"]:
        assert_parity(scoring.compute_recall_soft_penalty(row["recall"], row["confidence"], row["chronic"]), row["value"])
    for row in UNITS["clusterDampening"]:
        assert_parity(scoring.cluster_exposure_dampening(row["count"]), row["value"])
    for row in UNITS["calibratedReviewItem"]:
        assert scoring.is_calibrated_review_item(row["type"], row["complexity"]) is row["value"]
    for row in UNITS["interventionReason"]:
        actual = scoring.resolve_intervention_reason(
            row["conceptId"], row["recall"], row["base"], {"c-f"}, {"c-p", "c-f"}, 0.8
        )
        assert actual == row["value"]
    for row in UNITS["teachingState"]:
        assert scoring.classify_teaching_state(row["exposureCount"], row["recall"], row["confidence"]) == row["value"]


def test_teaching_cap_enforcement_parity() -> None:
    caps = UNITS["teachingCaps"]
    for variant in caps["variants"]:
        cap_a = variant["capA"]
        kept = scoring.enforce_concept_teaching_caps(
            caps["items"], lambda cid: cap_a if cid == "c-a" else caps["caps"].get(cid, 3)
        )
        assert [i["id"] for i in kept] == variant["result"], cap_a


def test_cluster_round_robin_parity() -> None:
    rr = UNITS["clusterRoundRobin"]
    states = [plan_state(c) for c in rr["concepts"]]
    by_priority = scoring.build_cluster_round_robin(states, lambda s: rr["clusters"][s.conceptId])
    assert [s.conceptId for s in by_priority] == rr["byPriority"]
    by_rank = scoring.build_cluster_round_robin(
        states, lambda s: rr["clusters"][s.conceptId], lambda s: rr["ranks"].get(s.conceptId)
    )
    assert [s.conceptId for s in by_rank] == rr["byRank"]


def plan_state(raw: dict[str, Any]) -> Any:
    from ebbifold.scheduler.pipeline import _concept_state

    return _concept_state(raw)


def test_difficulty_parity() -> None:
    for row in UNITS["difficultyPlan"]:
        assert difficulty.build_difficulty_plan(row["pattern"], row["length"]) == row["value"]
    for row in UNITS["normalizeDifficulty"]:
        assert difficulty.normalize_question_difficulty(row["input"]) == row["value"], row["input"]
    for row in UNITS["effectiveDifficulty"]:
        actual = difficulty.get_effective_question_difficulty(
            row["difficulty"], row["facilityIndex"], row["totalAttempts"]
        )
        assert actual == row["value"], row


def test_attribution_parity() -> None:
    attr = UNITS["attribution"]
    index = attribution.build_concept_topic_index(attr["concepts"])
    current = set(attr["currentConceptIds"])
    for card in attr["cards"]:
        assert attribution.resolve_card_concept_id(card, current, index) == card["value"], card


def test_topic_normalization_parity() -> None:
    for row in UNITS["topics"]:
        assert topics.normalize_topic(row["input"]) == row["normalized"], row["input"]
        assert topics.expand_topic_set([row["input"]]) == row["expanded"], row["input"]


# --- fail-closed behavior -------------------------------------------------


def _base_snapshot() -> dict[str, Any]:
    return copy.deepcopy(DATA["conceptPlan"][0]["snapshot"])


@pytest.mark.parametrize("seed", [None, "", "abc", "A" * 64, "a" * 63, "a" * 64 + "\n"])
def test_unseeded_or_malformed_seed_fails_closed(seed: str | None) -> None:
    snapshot = _base_snapshot()
    snapshot["options"]["examTargetTieBreakSeed"] = seed
    with pytest.raises(SchedulerDeterminismError):
        plan_concepts(snapshot)


_UNPORTED = {
    "exam_target": lambda s: s.update(applyExamTarget=True),
    "learner_policy": lambda s: s.update(targetSidecarsValid=True, runtimeTargetSnapshot={"definition": {}}),
    "simulation_focus": lambda s: s["options"].update(simulationFocus={"topics": ["x"]}),
    "prep_blend": lambda s: s["options"].update(sessionPrepBlend={"items": []}),
}


@pytest.mark.parametrize("lane", sorted(_UNPORTED))
def test_unported_lanes_fail_closed(lane: str) -> None:
    snapshot = _base_snapshot()
    _UNPORTED[lane](snapshot)
    with pytest.raises(UnsupportedSchedulerInput):
        plan_concepts(snapshot)


def test_concept_thread_anchors_fail_closed() -> None:
    from ebbifold.scheduler.build_state import SessionBuildState

    with pytest.raises(UnsupportedSchedulerInput):
        SessionBuildState(
            size=1, min_first_sight_items=0, max_new_cards=0, max_cross_source_items=0,
            unseen_card_ids=[], concept_thread_anchors=[{"anchorEventId": "e"}],
        )


def test_unknown_build_op_fails_closed() -> None:
    script = copy.deepcopy(DATA["buildScripts"][0]["script"])
    script["ops"] = [{"op": "teleport"}]
    with pytest.raises(ValueError, match="unknown build-script op"):
        run_build_script(script)


def test_embedding_dimension_mismatch_fails_closed() -> None:
    snapshot = _base_snapshot()
    snapshot["gapDirection"] = [1.0, 0.0]
    with pytest.raises(ValueError, match="dimensions must match"):
        plan_concepts(snapshot)


def test_parity_comparator_rejects_drift() -> None:
    expected = DATA["conceptPlan"][0]["expected"]
    drifted = copy.deepcopy(expected)
    drifted["sortedConceptIds"] = list(reversed(drifted["sortedConceptIds"]))
    with pytest.raises(AssertionError):
        assert_parity(drifted, expected)
    nudged = copy.deepcopy(expected)
    nudged["sortedPriorities"][0] += 1e-9
    with pytest.raises(AssertionError):
        assert_parity(nudged, expected)


def test_fixture_covers_required_behaviors() -> None:
    names = {c["name"] for c in DATA["conceptPlan"]} | {c["name"] for c in DATA["buildScripts"]}
    for required in (
        "weak_new_pristine_mix", "cluster_dampening", "empty_pools", "crunch_no_new_cards",
        "maintenance_shift", "interference_cluster_variant_budget", "teaching_caps_and_synthetic",
        "first_sight_reservation_release", "cross_source_cap_and_session_full",
    ):
        assert required in names
    assert DATA["generator"] == "python/tools/generate_unified_scheduler_fixtures.ts"
    assert (Path(__file__).parents[1] / "tools" / "generate_unified_scheduler_fixtures.ts").is_file()
