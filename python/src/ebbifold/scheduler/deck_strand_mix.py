"""Serializable deck-strand mixing (deck-strand-mix.ts)."""

from __future__ import annotations

from typing import Any, Callable


def hash_string(value: str) -> int:
    hashed = 0
    for char in value:
        hashed = ((hashed << 5) - hashed + ord(char)) & 0xFFFFFFFF
        if hashed & 0x80000000:
            hashed -= 0x100000000
    return abs(hashed)


def _basename(source_file: str | None) -> str:
    return (source_file or "").rsplit("/", 1)[-1]


def _matches(identity: dict[str, Any], strand: dict[str, Any]) -> bool:
    rotations = strand.get("matchRotations") or strand.get("rotations") or []
    prefixes = strand.get("matchSourcePrefixes") or strand.get("sourceFilePrefixes") or []
    matched = identity.get("rotation") in rotations
    basename = _basename(identity.get("sourceFile"))
    return matched or any(basename.startswith(prefix) for prefix in prefixes) or (not rotations and not prefixes)


def deck_strand_of(mix: dict[str, Any], identity: dict[str, Any]) -> str:
    for strand in mix.get("strands") or []:
        if _matches(identity, strand):
            return strand["id"]
    strands = mix.get("strands") or []
    return strands[-1]["id"] if strands else "default"


def _strand_units(items, mix, identify, keep_with_previous):
    units = []
    for item in items:
        if units and keep_with_previous and keep_with_previous(item):
            units[-1]["items"].append(item)
        else:
            units.append({"strand": deck_strand_of(mix, identify(item)), "items": [item]})
    return units


def _queues(strands, units):
    queues = {strand["id"]: [] for strand in strands}
    for unit in units:
        queues.setdefault(unit["strand"], []).append(unit)
    return queues


def _present_counts(strands, units):
    present = {strand["id"]: 0 for strand in strands}
    for unit in units:
        present[unit["strand"]] = present.get(unit["strand"], 0) + len(unit["items"])
    return present


def _choose_strand(strands, shares, emitted, total, available):
    best, best_score = None, float("-inf")
    for strand in strands:
        sid = strand["id"]
        if not available(sid):
            continue
        score = shares(sid) * (total + 1) - emitted.get(sid, 0)
        if score > best_score + 1e-9:
            best, best_score = sid, score
    return best


def _apply_phase(strands, shares, emitted, seed):
    total = 0
    for _ in range(hash_string(f"{seed}\0deck-strand-phase") % 10):
        strand = _choose_strand(strands, shares, emitted, total, lambda _sid: True)
        if strand is None:
            break
        emitted[strand] = emitted.get(strand, 0) + 1
        total += 1
    return total


def order_by_deck_strands(
    items: list[dict[str, Any]], mix: dict[str, Any] | None, *, seed: str,
    identify: Callable[[dict[str, Any]], dict[str, Any]],
    keep_with_previous: Callable[[dict[str, Any]], bool] | None = None,
    share_basis: str = "target",
) -> list[dict[str, Any]]:
    strands = (mix or {}).get("strands") or []
    if len(items) < 2 or not strands:
        return items
    units = _strand_units(items, mix, identify, keep_with_previous)
    queues = _queues(strands, units)
    if sum(1 for queue in queues.values() if queue) < 2:
        return items
    present = _present_counts(strands, units)
    emitted = {strand["id"]: 0 for strand in strands}

    def shares(sid: str) -> float:
        if share_basis == "present":
            return present.get(sid, 0) / len(items)
        strand = next(s for s in strands if s["id"] == sid)
        return float(strand.get("share") or 0)

    total = _apply_phase(strands, shares, emitted, seed)
    out: list[dict[str, Any]] = []
    cursor = {strand["id"]: 0 for strand in strands}
    def available(sid: str) -> bool:
        return cursor.get(sid, 0) < len(queues.get(sid, []))

    while len(out) < len(items):
        strand = _choose_strand(strands, shares, emitted, total, available)
        if strand is None:
            break
        pos = cursor.get(strand, 0)
        unit = queues[strand][pos]
        cursor[strand] = pos + 1
        for item in unit["items"]:
            out.append(item)
            item_strand = deck_strand_of(mix, identify(item))
            emitted[item_strand] = emitted.get(item_strand, 0) + 1
            total += 1
    return out
