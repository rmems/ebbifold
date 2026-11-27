"""Phase 3a parity: unified concept-space scheduler vs TypeScript golden fixtures.

Fixtures were generated once against the TypeScript original and committed as
self-contained JSON artifacts. These tests never execute TypeScript and fail,
rather than skip, when the fixture file is missing, empty, or malformed.
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
        pytest.fail(f"golden fixture missing: {FIXTURE}; restore the committed artifact")
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for section in ("conceptPlan", "buildScripts", "units"):
        if not data.get(section):
            pytest.fail(f"golden fixture section {section!r} is empty")
    return data


DATA = _load()


def assert_parity(actual: Any, expected: Any, path: str = "$") -> None:
    """Exact on structure, strings, bools and ordering; floats within 1e-12."""
    if isinstance(expected, bool):
        assert isinstance(actual, bool) and actual == expected, f"{path}: {actual!r} != {expected!r}"
    elif expected is None or isinstance(expected, str):
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
    "exam_target_without_workload": lambda s: s.update(applyExamTarget=True),
    "learner_policy_missing": lambda s: s.update(targetSidecarsValid=True, runtimeTargetSnapshot={"definition": {}}),
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
    with pytest.raises(AssertionError):
        assert_parity(1, True)
    with pytest.raises(AssertionError):
        assert_parity(0, False)


def test_fixture_covers_required_behaviors() -> None:
    names = {c["name"] for c in DATA["conceptPlan"]} | {c["name"] for c in DATA["buildScripts"]}
    for required in (
        "weak_new_pristine_mix", "cluster_dampening", "empty_pools", "crunch_no_new_cards",
        "maintenance_shift", "interference_cluster_variant_budget", "teaching_caps_and_synthetic",
        "first_sight_reservation_release", "cross_source_cap_and_session_full",
    ):
        assert required in names
    assert DATA["generator"] == "python/tools/generate_unified_scheduler_fixtures.ts"
    assert FIXTURE.is_file()


# --- end-to-end session construction --------------------------------------

from ebbifold.scheduler.pipeline import run_session  # noqa: E402

SESSION_FIXTURE = Path(__file__).parent / "fixtures" / "unified-scheduler-session.cases.json"
if not SESSION_FIXTURE.is_file():
    pytest.fail(f"session golden fixture missing: {SESSION_FIXTURE}; restore the committed artifact", pytrace=False)
SESSIONS = json.loads(SESSION_FIXTURE.read_text(encoding="utf-8"))["sessions"]
if not SESSIONS:
    pytest.fail("session golden fixture has no cases", pytrace=False)


def _session(case: dict[str, Any]) -> dict[str, Any]:
    out = run_session(copy.deepcopy(case["snapshot"]))
    out.pop("trace")
    return out


@pytest.mark.parametrize("case", SESSIONS, ids=lambda c: c["name"])
def test_session_parity(case: dict[str, Any]) -> None:
    assert_parity(_session(case), case["expected"])


@pytest.mark.parametrize("case", SESSIONS, ids=lambda c: c["name"])
def test_session_construction_is_deterministic(case: dict[str, Any]) -> None:
    runs = [json.dumps(_session(case), sort_keys=True) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_session_fixture_covers_required_behaviors() -> None:
    names = {c["name"] for c in SESSIONS}
    for required in (
        "full_session_weak_new_pristine", "empty_pools", "backfill_and_orphan_rescue", "exhausted_backfill",
        "rotation_bank_top_up", "first_sight_reservation_release", "interference_tight_threshold",
        "cluster_dampening", "struggle_step_down_scaffolds", "review_challenge_harder",
        "empty_concepts_cluster_fallback",
    ):
        assert required in names
    reasons = {i["interventionReason"] for c in SESSIONS for i in c["expected"]["result"]["items"]}
    assert {"preemptive_scaffold", "pre_teach", "pre_teach_naive", "needs_retest", "reinforcement"} <= reasons


def _base_session() -> dict[str, Any]:
    return copy.deepcopy(SESSIONS[0]["snapshot"])


def test_plan_concepts_derives_concept_map_and_uses_simulation_ranks() -> None:
    case = next(c for c in SESSIONS if c["name"] == "simulation_focus_preparation")
    snapshot = copy.deepcopy(case["snapshot"])
    snapshot.pop("conceptMap", None)
    result = plan_concepts(snapshot)
    assert result["weakConceptIds"][:4] == ["c-pe", "c-renal", "c-ecl", "c-naive"]
    assert result["clusterRoundRobin"] == result["weakConceptIds"]


def test_bridge_candidates_filter_eligibility_before_window() -> None:
    from ebbifold.scheduler.struggle import _bridge_candidates

    now = 1_000_000
    pool = [
        {"id": f"cooldown-{i}", "progress": {"lastReview": now - 1, "nextDueAt": 0}}
        for i in range(12)
    ]
    pool.append({"id": "eligible", "progress": None})
    assert [row["id"] for row in _bridge_candidates(pool, set(), now)] == ["eligible"]


def test_serve_conditioning_and_repetition_slot_telemetry() -> None:
    from ebbifold.scheduler.delivery import build_serve_conditioning, substitute_due_slot

    overlay = {"cards": {"stable-1": [{"q": "q-tight", "sim": 0.9}, {"q": "q-loose", "sim": 0.8}, {"q": "q-noise", "sim": 0.7}]}}
    record = {"verdict": "trusted", "questionOutcomes": {"q-tight": {"n": 2, "correct": 1}, "q-loose": {"n": 1, "correct": 1}}, "calibration": None, "computedAt": "2026-01-15T00:00:00Z"}
    conditioning = build_serve_conditioning(record, overlay, "stable-1")
    assert conditioning["evidence"] == {"tight": {"n": 2, "correct": 1}, "loose": {"n": 1, "correct": 1}}

    due = {"cardId": "due", "stableId": "stable-1", "variantGroupId": "vg", "variantType": "original", "complexity": 2, "lastQuality": 5, "correctCount": 2}
    pools = {"siblings": [], "seenCardIds": {"due"}, "proximity": {"stable-1": overlay["cards"]["stable-1"]}, "seenQuestionIds": set(), "questionComplexity": {}}
    decision = substitute_due_slot(due, pools)
    assert decision == {"kind": "substitute", "itemType": "question", "id": "q-tight", "step": 2, "reason": "same-fact-mcq"}


def test_mastery_seat_allocator_and_prep_ordering() -> None:
    from ebbifold.scheduler.exam_target import allocate_mastery_seats
    from ebbifold.scheduler.lanes import _order_prep_candidates

    candidates = [
        {"itemKey": "card:one", "unitId": "u1", "stage": "scheduled-atomic-core"},
        {"itemKey": "card:two", "unitId": "u1", "stage": "scheduled-atomic-core"},
        {"itemKey": "question:three", "unitId": "u2", "stage": "applied-distinction"},
    ]
    allocation = allocate_mastery_seats(
        candidates=candidates,
        eligible_unit_ids_by_stage={"scheduled-atomic-core": ["u1"], "applied-distinction": ["u2"]},
        core_target_seats=2,
        surplus_seats=1,
    )
    assert allocation["coreSelectedCount"] == 1
    assert allocation["surplusSelectedCount"] == 1
    assert [row["itemKey"] for row in allocation["selectedCandidates"]] == ["card:one", "question:three"]

    items = [
        {"type": "question", "id": "ordinary"},
        {"type": "card", "id": "weak"},
        {"type": "card", "id": "unseen"},
    ]
    ordered = _order_prep_candidates(items, {
        "share": 0.5,
        "itemIds": ["card:weak", "card:unseen"],
        "weakItemIds": ["card:weak"],
        "unseenItemIds": ["card:unseen"],
    }, 2)
    assert [row["id"] for row in ordered[:2]] == ["ordinary", "weak"]
    assert ordered[-1]["id"] == "unseen"


def test_pass_set_and_composed_deck_source_identity() -> None:
    from ebbifold.scheduler.lanes import _pass_set_rank
    from ebbifold.scheduler.queue import source_corpus_for

    bulk = {
        "passSetIntakeDecision": "pass-set-first",
        "passSet": {"passSet": {"factKeyByCardId": {"c1": "fact-a"}, "factKeys": ["fact-a"], "rankByFactKey": {"fact-a": 7}}},
    }
    assert _pass_set_rank(bulk, {"id": "c1", "variantGroupId": None}) == 7
    assert _pass_set_rank(bulk, {"id": "c2", "variantGroupId": None}) > 7
    assert source_corpus_for("anatomy") == "bluelink"
    assert source_corpus_for("surgical-sciences", "kubie-neuro.mdx") == "kubie"
    assert source_corpus_for("surgical-sciences", "plates.mdx") == "plates"
    assert source_corpus_for("neurosurg") == "kubie"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.update(struggleHistory={"cardProgress": {}}, userId=None),
        lambda s: s.update(conceptThreadAnchors=[{"anchorEventId": "e"}]),
        lambda s: s.update(concepts=[], deckStrandMix={"strands": []}),
    ],
    ids=["struggle", "concept_thread", "cluster_strands"],
)
def test_session_unported_adapters_fail_closed(mutate: Any) -> None:
    snapshot = _base_session()
    mutate(snapshot)
    with pytest.raises(UnsupportedSchedulerInput):
        run_session(snapshot)


def test_session_missing_seed_fails_closed() -> None:
    snapshot = _base_session()
    snapshot["options"].pop("examTargetTieBreakSeed")
    with pytest.raises(SchedulerDeterminismError):
        run_session(snapshot)
