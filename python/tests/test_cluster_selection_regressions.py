"""Regression coverage for cluster card selection edge cases."""

from ebbifold.scheduler.cluster_card_selection import select_cluster_cards


def _row(row_id: str) -> dict:
    return {
        "id": row_id, "clusterId": "cl", "rotation": "rot-main", "topics": [],
        "complexity": 2, "firstSightAtSelection": True,
    }


def _inputs(rows: list[dict], excluded: set[str] | None = None, mix: dict | None = None) -> dict:
    return {
        "rotation": "rot-main", "size": 2, "options": {}, "excludedCardIds": excluded or set(),
        "clusterCards": {"cl": rows}, "deckStrandMix": mix,
    }


def _state() -> list[dict]:
    return [{"clusterId": "cl", "clusterName": "Cluster", "priority": 1.0, "mastery": 1.0}]


def test_cluster_selection_empty_strand_mix_terminates_without_cards() -> None:
    selected, _clusters, _identities = select_cluster_cards(
        _inputs([_row("c1")], mix={"strands": []}), _state(), 1,
    )
    assert selected == []


def test_cluster_card_window_applies_exclusions_before_limit() -> None:
    rows = [_row(f"excluded-{index}") for index in range(12)] + [_row("eligible")]
    selected, _clusters, _identities = select_cluster_cards(
        _inputs(rows, {row["id"] for row in rows[:12]}), _state(), 1,
    )
    assert [row["id"] for row in selected] == ["eligible"]
