"""Struggle interventions (knowledge/struggle-interventions.ts, intervention.ts,
struggle.ts, bridge-card-eligibility.ts).

The decision logic (stuck detection, fail counts, the deterministic strategy
roll, fallbacks to ``continue``, bridge eligibility, seeded bridge choice and
item assembly) is ported. What the TypeScript reads from Postgres arrives as
the explicit snapshot ``struggleHistory``:

- ``cardProgress``: {cardId: {recentFailCount, recentFailWindowStart, totalReviews, correctCount}}
- ``questionAttempts``: {questionId: {attempts, correct}}
- ``questions``: [{id, topics, rotation}] in storage order (bridge lookup)
- ``bridgePools``: {questionId: {"primary": rows, "keyPoint": rows}} where a row is
  {id, progress: {status, lastReview, nextDueAt} | None}, ordered by feedScore
- ``bridgeCards``: the rotation bridge cards findBridgeCards returns (scaffold strategy)
- ``contextConcepts``: {cardId: {id, name, description}} (context strategy)
- ``variantQuestions``: {cardId: {id, rotation}} (retreat strategy)

Times are epoch milliseconds."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .errors import UnsupportedSchedulerInput
from .queue import shuffle_with_seed

STRUGGLE_THRESHOLD = 3
STRUGGLE_WINDOW_HOURS = 24
CHRONIC_STUCK_MIN_REVIEWS = 3
CHRONIC_STUCK_MAX_ACCURACY = 0.3
ACUTE_STUCK_MIN_REVIEWS = 3
ACUTE_STUCK_MAX_ACCURACY = 0.5
BRIDGE_CANDIDATE_WINDOW = 12
BRIDGE_CARD_REPEAT_COOLDOWN_MS = 24 * 60 * 60 * 1000
INTERVENTION_PROBABILITIES = (("continue", 0.30), ("scaffold", 0.35), ("context", 0.20), ("retreat", 0.15))


def is_card_stuck(recent_fail_count: int, window_start_ms: float | None, now_ms: float) -> bool:
    if window_start_ms is None or recent_fail_count < STRUGGLE_THRESHOLD:
        return False
    return (now_ms - window_start_ms) / 3_600_000 <= STRUGGLE_WINDOW_HOURS


def is_chronically_stuck(total: int, correct: int) -> bool:
    return total >= CHRONIC_STUCK_MIN_REVIEWS and correct / total <= CHRONIC_STUCK_MAX_ACCURACY


def is_acutely_stuck(total: int, correct: int) -> bool:
    return total >= ACUTE_STUCK_MIN_REVIEWS and correct / total < ACUTE_STUCK_MAX_ACCURACY


def deterministic_roll(seed: str) -> float:
    h = 0x811C9DC5
    for unit in seed.encode("utf-16-le").decode("utf-16-le"):
        h = ((h ^ ord(unit)) * 0x01000193) & 0xFFFFFFFF
    return h / 0x100000000


def _day_key(now_ms: float) -> str:
    return datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def is_bridge_card_eligible(progress: dict[str, Any] | None, now_ms: float) -> bool:
    if not progress:
        return True
    last = progress.get("lastReview")
    if last is not None and last >= now_ms - BRIDGE_CARD_REPEAT_COOLDOWN_MS:
        return False
    return progress["nextDueAt"] <= now_ms


def pick_bridge_card(candidates: list[dict[str, Any]], seed: str | None) -> dict[str, Any] | None:
    if not candidates:
        return None
    return candidates[0] if not seed else shuffle_with_seed(candidates, seed)[0]


def _bridge_candidates(pool: list[dict[str, Any]], used: set[str], now_ms: float) -> list[dict[str, Any]]:
    window = [row for row in pool if row["id"] not in used][:BRIDGE_CANDIDATE_WINDOW]
    return [row for row in window if is_bridge_card_eligible(row.get("progress"), now_ms)]


def _find_bridge_cards(history: dict[str, Any], stuck: set[str], used: set[str], now_ms: float,
                       seed: str | None) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for q in history.get("questions", []):
        if q["id"] not in stuck or not q.get("topics"):
            continue
        pools = history.get("bridgePools", {}).get(q["id"], {})
        candidates = _bridge_candidates(pools.get("primary", []), used, now_ms) or \
            _bridge_candidates(pools.get("keyPoint", []), used, now_ms)
        chosen = pick_bridge_card(candidates, f"{seed}:{q['id']}" if seed else None)
        if chosen:
            result[q["id"]] = {"cardId": chosen["id"], "rotation": q["rotation"]}
            used.add(chosen["id"])
    return result


def _tag_chronic_stuck_mcqs(items: list[dict[str, Any]], history: dict[str, Any], excluded: set[str],
                            now_ms: float, seed: str | None) -> list[dict[str, Any]]:
    attempts = history.get("questionAttempts", {})
    stuck = set()
    for item in items:
        if item["type"] == "question":
            stats = attempts.get(item["id"], {})
            total, correct = stats.get("attempts", 0), stats.get("correct", 0)
            if is_chronically_stuck(total, correct) or is_acutely_stuck(total, correct):
                stuck.add(item["id"])
    if not stuck:
        return items
    used = set(excluded) | {i["id"] for i in items if i["type"] == "card"}
    bridges = _find_bridge_cards(history, stuck, used, now_ms, seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        if it["type"] != "question" or it["id"] not in stuck:
            out.append(it)
            continue
        bridge = bridges.get(it["id"])
        if bridge and bridge["cardId"] not in seen:
            out.append({"type": "card", "id": bridge["cardId"], "rotation": bridge["rotation"],
                        "conceptId": it["conceptId"], "conceptName": it["conceptName"], "priority": it["priority"] + 0.1,
                        "interventionReason": "mcq_bridge_card",
                        "struggleIntervention": {"strategy": "scaffold", "isScaffold": True, "targetCardId": it["id"]}})
            seen.add(bridge["cardId"])
        out.append({**it, "interventionReason": "chronic_stuck_mcq"})
    return out


def _stuck_fail_count(progress: dict[str, Any], now_ms: float) -> int | None:
    total, correct = progress["totalReviews"], progress["correctCount"]
    recent = progress["recentFailCount"]
    window = is_card_stuck(recent, progress.get("recentFailWindowStart"), now_ms)
    lifetime = is_chronically_stuck(total, correct) or is_acutely_stuck(total, correct)
    if not (window or lifetime):
        return None
    return max(total - correct, recent) if lifetime else recent


def select_intervention(user_id: str, card_id: str, now_ms: float, history: dict[str, Any]) -> dict[str, Any]:
    roll = deterministic_roll(f"{user_id}|{card_id}|{_day_key(now_ms)}")
    strategy, cumulative = "continue", 0.0
    for name, probability in INTERVENTION_PROBABILITIES:
        cumulative += probability
        if roll < cumulative:
            strategy = name
            break
    decision: dict[str, Any] = {"strategy": strategy, "triggerCardId": card_id}
    if strategy == "scaffold":
        cards = history.get("bridgeCards") or []
        if cards:
            decision["scaffoldCards"] = cards
        else:
            decision["strategy"] = "continue"
    elif strategy == "context":
        concept = history.get("contextConcepts", {}).get(card_id)
        if not (concept and concept.get("description")):
            decision["strategy"] = "continue"
    elif strategy == "retreat":
        variant = history.get("variantQuestions", {}).get(card_id)
        if variant:
            decision["variantQuestionId"] = variant["id"]
            decision["variantQuestionRotation"] = variant["rotation"]
    return decision


def _apply_decision(item: dict[str, Any], decision: dict[str, Any], rotation: str) -> list[dict[str, Any]]:
    strategy = decision["strategy"]
    if strategy == "retreat":
        if not decision.get("variantQuestionId"):
            return []
        return [{"type": "question", "id": decision["variantQuestionId"],
                 "rotation": decision.get("variantQuestionRotation") or rotation, "conceptId": item["conceptId"],
                 "conceptName": item["conceptName"], "priority": item["priority"],
                 "interventionReason": "stuck_intervention",
                 "struggleIntervention": {"strategy": "retreat", "targetCardId": item["id"]}}]
    out = []
    for bridge in decision.get("scaffoldCards", []) if strategy == "scaffold" else []:
        out.append({"type": "card", "id": bridge["id"], "rotation": rotation, "conceptId": item["conceptId"],
                    "conceptName": item["conceptName"], "priority": item["priority"] + 0.1,
                    "interventionReason": "stuck_intervention",
                    "struggleIntervention": {"strategy": "scaffold", "isScaffold": True, "targetCardId": item["id"]}})
    out.append({**item, "interventionReason": "stuck_intervention", "struggleIntervention": {"strategy": strategy}})
    return out


def apply_struggle_interventions(inputs: dict[str, Any], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    history = inputs.get("struggleHistory") or {}
    if not history:
        return items
    user_id = inputs.get("userId")
    if not isinstance(user_id, str) or not user_id:
        raise UnsupportedSchedulerInput("struggleHistory requires the snapshot userId (intervention roll seed)")
    now_ms = inputs["nowMs"]
    tagged = _tag_chronic_stuck_mcqs(items, history, set(inputs.get("excludedCardIds") or []), now_ms,
                                     inputs.get("servingRotationSeed"))
    progress = history.get("cardProgress", {})
    decisions: dict[str, dict[str, Any]] = {}
    for item in tagged:
        if item["type"] == "card" and item["id"] in progress and item["id"] not in decisions:
            if _stuck_fail_count(progress[item["id"]], now_ms) is not None:
                decisions[item["id"]] = select_intervention(user_id, item["id"], now_ms, history)
    if not decisions:
        return tagged
    out: list[dict[str, Any]] = []
    for item in tagged:
        if item["type"] == "card" and item["id"] in decisions:
            out.extend(_apply_decision(item, decisions[item["id"]], inputs["rotation"]))
        else:
            out.append(item)
    return out
