"""Pure session-prep ordering over an eligible pool (session-prep-allocation.ts)."""

from __future__ import annotations

from typing import Any, Callable


def order_prep_keys(keys: list[str], plan: dict[str, Any]) -> list[str]:
    weak = set(plan.get("weakItemIds") or [])
    unseen = set(plan.get("unseenItemIds") or [])

    def interleave(pool: list[str]) -> list[str]:
        weak_pool = [key for key in pool if key in weak]
        unseen_pool = [key for key in pool if key in unseen]
        out: list[str] = []
        share = float(plan.get("share") or 0)
        for index in range(max(len(weak_pool), len(unseen_pool))):
            if share < 0.25 and index < len(unseen_pool):
                out.append(unseen_pool[index])
            if index < len(weak_pool):
                out.append(weak_pool[index])
            if share >= 0.25 and index < len(unseen_pool):
                out.append(unseen_pool[index])
        return out + [key for key in pool if key not in weak and key not in unseen]

    sessions = [s for s in (plan.get("sessions") or []) if float(s.get("share") or 0) > 0]
    if len(sessions) < 2:
        return interleave(keys)
    queues = []
    for session in sessions:
        ids = set(session.get("itemIds") or [])
        queues.append({"share": float(session["share"]), "taken": 0, "keys": interleave([k for k in keys if k in ids])})
    used: set[str] = set()
    out: list[str] = []
    while True:
        for queue in queues:
            while queue["keys"] and queue["keys"][0] in used:
                queue["keys"].pop(0)
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


def order_session_prep_candidates(
    items: list[dict[str, Any]], plan: dict[str, Any] | None, *, size: int,
    key: Callable[[dict[str, Any]], str], protected: Callable[[dict[str, Any]], bool] | None = None,
    prep_seats: int | None = None,
) -> list[dict[str, Any]]:
    if not plan or not isinstance(plan.get("share"), (int, float)) or plan["share"] <= 0:
        return items
    prep_keys = set(plan.get("itemIds") or [])
    weak_keys = set(plan.get("weakItemIds") or [])
    unseen_keys = set(plan.get("unseenItemIds") or [])
    protected_items: list[dict[str, Any]] = []
    ordinary: list[dict[str, Any]] = []
    weak: list[dict[str, Any]] = []
    unseen: list[dict[str, Any]] = []
    other_prep: list[dict[str, Any]] = []
    for item in items:
        item_key = key(item)
        if protected and protected(item):
            protected_items.append(item)
        elif item_key not in prep_keys:
            ordinary.append(item)
        elif item_key in weak_keys:
            weak.append(item)
        elif item_key in unseen_keys:
            unseen.append(item)
        else:
            other_prep.append(item)
    prep: list[dict[str, Any]] = []
    share = float(plan["share"])
    for index in range(max(len(weak), len(unseen))):
        if share < 0.25 and index < len(unseen):
            prep.append(unseen[index])
        if index < len(weak):
            prep.append(weak[index])
        if share >= 0.25 and index < len(unseen):
            prep.append(unseen[index])
    prep.extend(other_prep)
    if len(plan.get("sessions") or []) > 1:
        by_key = {key(item): item for item in prep}
        prep = [by_key[item_key] for item_key in order_prep_keys([key(item) for item in prep], plan)]
    seats = max(0, int(size) - len(protected_items))
    wanted = min(len(prep), seats, max(0, int(prep_seats if prep_seats is not None else seats * min(0.5, share))))
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
    return prefix + ordinary[ordinary_at:] + prep[prep_at:]
