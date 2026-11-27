"""Cluster fallback eligibility, strand, and prep lane coverage."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from ebbifold.scheduler.deck_strand_mix import DeckStrandOrderOptions, order_by_deck_strands
from ebbifold.scheduler.pipeline import run_session

SESSIONS = json.loads((Path(__file__).parent / "fixtures" / "unified-scheduler-session.cases.json").read_text(encoding="utf-8"))["sessions"]


def _cluster_snapshot() -> dict:
    return copy.deepcopy(next(c for c in SESSIONS if c["name"] == "cluster_present_fallback")["snapshot"])


def test_cluster_media_lane_filters_cards_and_questions() -> None:
    snapshot = _cluster_snapshot()
    snapshot["options"] = {**snapshot["options"], "media": "image"}
    eligible_cards = {"k-renal-1", "k-card-1"}
    for rows in snapshot["clusterCards"].values():
        for row in rows:
            if row["id"] in eligible_cards:
                row["imageUrl"] = "figure.png"
    result = run_session(snapshot)["result"]
    assert result["items"]
    assert {item["id"] for item in result["items"] if item["type"] == "card"} <= eligible_cards
    assert not any(item["type"] == "question" for item in result["items"])


def _mark_rot_other(rows: list[dict], eligible: set[str]) -> None:
    for row in rows:
        if row["id"] in eligible:
            row.update({"rotation": "rot-other", "week": 2})


def test_cluster_cross_source_and_week_lanes_filter_candidates() -> None:
    snapshot = _cluster_snapshot()
    snapshot["week"] = 2
    snapshot["options"] = {**snapshot["options"], "crossSourceRotations": ["rot-other"]}
    eligible = {"k-renal-1", "k-card-1", "cq-1"}
    for rows in snapshot["clusterCards"].values():
        _mark_rot_other(rows, eligible)
    _mark_rot_other(snapshot["clusterQuestions"], eligible)
    result = run_session(snapshot)["result"]
    assert result["items"]
    assert {item["id"] for item in result["items"]} <= eligible


def test_deck_strand_ordering_preserves_items_and_spreads_strands() -> None:
    items = [
        {"id": f"clinical-{index}", "rotation": "clinical", "type": "card"} for index in range(4)
    ] + [
        {"id": f"anatomy-{index}", "rotation": "anatomy", "type": "card"} for index in range(2)
    ]
    mix = {"strands": [
        {"id": "anatomy", "share": 0.5, "matchRotations": ["anatomy"]},
        {"id": "clinical", "share": 0.5},
    ]}
    options = DeckStrandOrderOptions(
        seed="test", share_basis="target",
        identify=lambda item: {"rotation": item["rotation"], "sourceFile": None},
    )
    ordered = order_by_deck_strands(items, mix, options)
    assert sorted(item["id"] for item in ordered) == sorted(item["id"] for item in items)
    assert len(ordered) == len(items)
    assert ordered is not items
    anatomy_positions = [index for index, item in enumerate(ordered) if item["rotation"] == "anatomy"]
    assert anatomy_positions[0] < 3
    assert anatomy_positions[-1] - anatomy_positions[0] > 1


def test_cluster_prep_lane_prefers_planned_items() -> None:
    snapshot = _cluster_snapshot()
    prep_key = "card:k-renal-4"
    snapshot["options"] = {**snapshot["options"], "sessionPrepBlend": {
        "share": 0.5, "itemIds": [prep_key], "weakItemIds": [prep_key], "unseenItemIds": [],
    }}
    baseline_ids = [item["id"] for item in run_session(_cluster_snapshot())["result"]["items"]]
    result = run_session(snapshot)["result"]
    ids = [item["id"] for item in result["items"]]
    assert "k-renal-4" not in baseline_ids
    assert "k-renal-4" in ids
