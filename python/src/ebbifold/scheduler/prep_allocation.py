"""Pure session-prep ordering over an eligible pool (session-prep-allocation.ts)."""

from __future__ import annotations

from typing import Any, Callable


def _interleave_keys(pool: list[str], weak: set[str], unseen: set[str], share: float) -> list[str]:
    weak_pool = [key for key in pool if key in weak]
    unseen_pool = [key for key in pool if key in unseen]
    out: list[str] = []
    for index in range(max(len(weak_pool), len(unseen_pool))):
        if share < 0.25 and index < len(unseen_pool):
            out.append(unseen_pool[index])
        if index < len(weak_pool):
            out.append(weak_pool[index])
        if share >= 0.25 and index < len(unseen_pool):
            out.append(unseen_pool[index])
    return out + [key for key in pool if key not in weak and key not in unseen]


def _prep_interleaver(plan: dict[str, Any]):
    weak = set(plan.get("weakItemIds") or [])
    unseen = set(plan.get("unseenItemIds") or [])
    share = float(plan.get("share") or 0)
    return lambda pool: _interleave_keys(pool, weak, unseen, share)


def _session_queues(keys: list[str], sessions: list[dict[str, Any]], interleave) -> list[dict[str, Any]]:
    return [
        {"share": float(session["share"]), "taken": 0, "keys": interleave([k for k in keys if k in set(session.get("itemIds") or [])])}
        for session in sessions
    ]


def _drop_used_heads(queues: list[dict[str, Any]], used: set[str]) -> None:
    for queue in queues:
        while queue["keys"] and queue["keys"][0] in used:
            queue["keys"].pop(0)


def order_prep_keys(keys: list[str], plan: dict[str, Any]) -> list[str]:
    interleave = _prep_interleaver(plan)
    sessions = [s for s in (plan.get("sessions") or []) if float(s.get("share") or 0) > 0]
    if len(sessions) < 2:
        return interleave(keys)
    queues = _session_queues(keys, sessions, interleave)
    used: set[str] = set()
    out: list[str] = []
    while True:
        _drop_used_heads(queues, used)
        available = [queue for queue in queues if queue["keys"]]
        if not available:
            break
        available.sort(key=lambda queue: (queue["taken"] + 1) / queue["share"])
        queue = available[0]
        key = queue["keys"].pop(0)
        queue["taken"] += 1
        used.add(key)
        out.append(key)
    return out + [key for key in keys if key not in used]


def prep_modality_budgets(share: float, card_seats: int, question_seats: int, card_supply: int, question_supply: int) -> dict[str, int]:
    capped = max(0.0, min(0.5, share))
    total = int((card_seats + question_seats) * capped)
    cards = min(card_supply, card_seats, int(card_seats * capped))
    questions = min(question_supply, question_seats, total - cards)
    cards += min(max(0, card_supply - cards), max(0, card_seats - cards), total - cards - questions)
    questions += min(max(0, question_supply - questions), max(0, question_seats - questions), total - cards - questions)
    return {"cards": cards, "questions": questions}


def _partition_prep_items(items, plan, key, protected):
    groups = {"protected": [], "ordinary": [], "weak": [], "unseen": [], "other": []}
    prep_keys = set(plan.get("itemIds") or [])
    weak_keys = set(plan.get("weakItemIds") or [])
    unseen_keys = set(plan.get("unseenItemIds") or [])
    for item in items:
        item_key = key(item)
        if protected and protected(item):
            groups["protected"].append(item)
        elif item_key not in prep_keys:
            groups["ordinary"].append(item)
        elif item_key in weak_keys:
            groups["weak"].append(item)
        elif item_key in unseen_keys:
            groups["unseen"].append(item)
        else:
            groups["other"].append(item)
    return groups


def _ordered_prep_items(groups: dict[str, list[dict[str, Any]]], share: float) -> list[dict[str, Any]]:
    prep: list[dict[str, Any]] = []
    weak, unseen = groups["weak"], groups["unseen"]
    for index in range(max(len(weak), len(unseen))):
        if share < 0.25 and index < len(unseen):
            prep.append(unseen[index])
        if index < len(weak):
            prep.append(weak[index])
        if share >= 0.25 and index < len(unseen):
            prep.append(unseen[index])
    return prep + groups["other"]


def _session_order_prep(prep: list[dict[str, Any]], plan: dict[str, Any], key) -> list[dict[str, Any]]:
    if len(plan.get("sessions") or []) <= 1:
        return prep
    by_key = {key(item): item for item in prep}
    return [by_key[item_key] for item_key in order_prep_keys([key(item) for item in prep], plan)]


def _spread_prep(protected_items, ordinary, prep, seats, wanted):
    prefix = list(protected_items)
    prep_at = ordinary_at = 0
    for slot in range(seats):
        prep_due = int((slot + 1) * wanted / seats) > prep_at
        if (prep_due or ordinary_at >= len(ordinary)) and prep_at < len(prep):
            prefix.append(prep[prep_at])
            prep_at += 1
        elif ordinary_at < len(ordinary):
            prefix.append(ordinary[ordinary_at])
            ordinary_at += 1
        else:
            break
    return prefix, ordinary_at, prep_at


def order_session_prep_candidates(
    items: list[dict[str, Any]], plan: dict[str, Any] | None, *, size: int,
    key: Callable[[dict[str, Any]], str], protected: Callable[[dict[str, Any]], bool] | None = None,
    prep_seats: int | None = None,
) -> list[dict[str, Any]]:
    if not plan or not isinstance(plan.get("share"), (int, float)) or plan["share"] <= 0:
        return items
    groups = _partition_prep_items(items, plan, key, protected)
    share = float(plan["share"])
    prep = _session_order_prep(_ordered_prep_items(groups, share), plan, key)
    ordinary = groups["ordinary"]
    seats = max(0, int(size) - len(groups["protected"]))
    default_prep_seats = seats * min(0.5, share)
    wanted = min(len(prep), seats, max(0, int(prep_seats if prep_seats is not None else default_prep_seats)))
    prefix, ordinary_at, prep_at = _spread_prep(groups["protected"], ordinary, prep, seats, wanted)
    return prefix + ordinary[ordinary_at:] + prep[prep_at:]
