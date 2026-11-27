"""Simulation focus and preparation (study/simulation-focus.ts,
simulation-item-relevance.ts, simulation-preparation.ts).

The reviewed simulation item registry (src/lib/study/simulation-item-map.json)
arrives as the explicit snapshot input ``simulationItemMap`` (its records)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any

LABELS: dict[str, tuple[str, ...]] = {
    "pre-eclampsia": ("pre-eclampsia", "preeclampsia", "eclampsia", "hypertensive deterioration", "magnesium toxicity"),
    "postpartum-haemorrhage": ("postpartum haemorrhage", "postpartum hemorrhage", "uterine atony", "retained placenta"),
    "maternal-collapse": ("maternal collapse", "maternal cardiac arrest", "maternal collapse and arrest"),
}


def _normal(value: str) -> str:
    return " ".join(value.strip().lower().split())


def is_simulation_topic(topics: Iterable[str], focus: Mapping[str, Any] | None) -> bool:
    if not focus:
        return False
    labels = {_normal(label) for topic in focus["topics"] for label in LABELS.get(topic, ())}
    return any(_normal(t) in labels for t in topics)


def prefer_simulation_topics(items: list[Any], topics: Callable[[Any], Iterable[str]], focus: Mapping[str, Any] | None) -> list[Any]:
    if not focus:
        return list(items)
    focused = [i for i in items if is_simulation_topic(topics(i), focus)]
    maintenance = [i for i in items if not is_simulation_topic(topics(i), focus)]
    ordered: list[Any] = []
    f = m = 0
    while f < len(focused) or m < len(maintenance):
        for _ in range(2):
            if f < len(focused):
                ordered.append(focused[f])
                f += 1
        if m < len(maintenance):
            ordered.append(maintenance[m])
            m += 1
    return ordered


class SimulationRegistry:
    def __init__(self, records: Iterable[Mapping[str, Any]]) -> None:
        self.by_stable_id: dict[str, Mapping[str, Any]] = {}
        self.by_content_hash: dict[str, Mapping[str, Any]] = {}
        for record in records:
            self.by_stable_id.setdefault(record["stableId"], record)
            self.by_content_hash.setdefault(record.get("contentHash", ""), record)

    def evidence(self, stable_id: str | None, focus: Mapping[str, Any] | None, content_hash: str | None = None):
        if not focus:
            return None
        record = self.by_stable_id.get(stable_id) if stable_id else (
            self.by_content_hash.get(content_hash) if content_hash else None)
        if record is None:
            return None
        record_topics = record.get("topics") or [record["topic"]]
        topic = next((t for t in focus["topics"] if t in record_topics), None)
        return {**record, "topic": topic} if topic else None


def prefer_reviewed_simulation_cards(items: list[dict[str, Any]], stable_id: Callable[[dict[str, Any]], str | None],
                                     focus: Mapping[str, Any] | None, registry: SimulationRegistry) -> list[dict[str, Any]]:
    if not focus:
        return list(items)
    lanes: dict[Any, list[dict[str, Any]]] = {}
    for item in items:
        lanes.setdefault(item.get("complexity"), []).append(item)
    preparation = set(focus.get("preparationStableIds") or [])

    def priority(item: dict[str, Any]) -> int:
        sid = stable_id(item)
        if not registry.evidence(sid, focus):
            return 0
        return 2 if sid and sid in preparation else 1

    for key, lane in lanes.items():
        lanes[key] = sorted(lane, key=lambda i: -priority(i))
    offsets: dict[Any, int] = {}
    out = []
    for item in items:
        k = item.get("complexity")
        off = offsets.get(k, 0)
        offsets[k] = off + 1
        out.append(lanes[k][off])
    return out


def _recall_state(row: Mapping[str, Any], now_ms: float) -> str:
    last = row.get("lastReview")
    if row["totalReviews"] < 1 or row.get("lastQuality") is None or last is None or last > now_ms:
        return "unknown"
    if row["lastQuality"] < 3:
        return "weak"
    due = row.get("nextDueAt")
    if due is None or due <= now_ms:
        return "due"
    return "recalled"


def order_unrecalled_simulation_preparation(rows: list[dict[str, Any]], now_ms: float) -> list[dict[str, Any]]:
    rank = {"weak": 0, "due": 1, "unknown": 2}
    keep = [r for r in rows if _recall_state(r, now_ms) != "recalled"]
    return sorted(keep, key=lambda r: (rank[_recall_state(r, now_ms)],
                                       99 if r.get("lastQuality") is None else r["lastQuality"],
                                       r.get("lastReview") or 0))
