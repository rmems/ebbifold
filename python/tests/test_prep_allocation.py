"""Direct coverage for session-prep allocation helpers."""

from ebbifold.scheduler.prep_allocation import order_prep_keys, order_session_prep_candidates, prep_modality_budgets


def test_order_prep_keys_low_share_puts_unseen_before_weak() -> None:
    keys = ["card:weak", "card:unseen", "card:other"]
    ordered = order_prep_keys(keys, {"share": 0.2, "weakItemIds": ["card:weak"], "unseenItemIds": ["card:unseen"]})
    assert ordered == ["card:unseen", "card:weak", "card:other"]


def test_order_prep_keys_high_share_and_overlapping_sessions_are_stable() -> None:
    keys = ["card:w1", "card:u1", "card:w2", "card:u2", "card:x"]
    plan = {
        "share": 0.5,
        "weakItemIds": ["card:w1", "card:w2"],
        "unseenItemIds": ["card:u1", "card:u2"],
        "sessions": [
            {"share": 0.7, "itemIds": ["card:w1", "card:u1", "card:x"]},
            {"share": 0.3, "itemIds": ["card:w1", "card:w2", "card:u2"]},
        ],
    }
    ordered = order_prep_keys(keys, plan)
    assert sorted(ordered) == sorted(keys)
    assert len(ordered) == len(keys)
    assert ordered[0] in {"card:w1", "card:u1", "card:w2", "card:u2"}


def test_prep_modality_budgets_caps_share_and_supply() -> None:
    assert prep_modality_budgets(0.5, 6, 4, 20, 20) == {"cards": 3, "questions": 2}
    assert prep_modality_budgets(0.5, 6, 4, 1, 20)["cards"] == 1
    assert prep_modality_budgets(0, 6, 4, 20, 20) == {"cards": 0, "questions": 0}


def test_order_session_prep_candidates_partitions_and_preserves_membership() -> None:
    def item(key: str, reason: str = "reinforcement") -> dict:
        return {"id": key, "type": key.split(":", 1)[0], "interventionReason": reason}

    items = [
        item("card:protected", "failure_escalation"), item("card:ordinary"), item("card:weak"),
        item("card:unseen"), item("card:otherprep"),
    ]
    prep_ids = ["card:weak", "card:unseen", "card:otherprep"]
    plan = {"share": 0.5, "itemIds": prep_ids, "weakItemIds": ["card:weak"], "unseenItemIds": ["card:unseen"]}
    ordered = order_session_prep_candidates(
        items, plan, size=5, key=lambda row: f"{row['type']}:{row['id'].split(':', 1)[1]}",
        protected=lambda row: row["interventionReason"] == "failure_escalation",
    )
    assert sorted(row["id"] for row in ordered) == sorted(row["id"] for row in items)
    assert ordered[0]["id"] == "card:protected"


def test_order_session_prep_candidates_noop_and_multi_session_paths() -> None:
    items = [{"type": "card", "id": "a"}, {"type": "card", "id": "b"}]
    assert order_session_prep_candidates(items, None, size=2, key=lambda row: f"card:{row['id']}") is items
    plan = {
        "share": 0.4, "itemIds": ["card:a", "card:b"], "weakItemIds": [], "unseenItemIds": [],
        "sessions": [{"share": 0.6, "itemIds": ["card:a"]}, {"share": 0.4, "itemIds": ["card:b"]}],
    }
    ordered = order_session_prep_candidates(items, plan, size=2, key=lambda row: f"card:{row['id']}")
    assert sorted(row["id"] for row in ordered) == ["a", "b"]
