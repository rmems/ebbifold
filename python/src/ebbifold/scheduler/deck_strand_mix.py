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
    matched = False
    if identity.get("rotation") in rotations:
        matched = True
    basename = _basename(identity.get("sourceFile"))
    if any(basename.startswith(prefix) for prefix in prefixes):
        matched = True
    return matched or (not rotations and not prefixes)


def deck_strand_of(mix: dict[str, Any], identity: dict[str, Any]) -> str:
    for strand in mix.get("strands") or []:
        if _matches(identity, strand):
            return strand["id"]
    strands = mix.get("strands") or []
    return strands[-1]["id"] if strands else "default"


def order_by_deck_strands(
    items: list[dict[str, Any]], mix: dict[str, Any] | None, *, seed: str,
    identify: Callable[[dict[str, Any]], dict[str, Any]],
    keep_with_previous: Callable[[dict[str, Any]], bool] | None = None,
    share_basis: str = "target",
) -> list[dict[str, Any]]:
    strands = (mix or {}).get("strands") or []
    if len(items) < 2 or not strands:
        return items
    units: list[dict[str, Any]] = []
    for item in items:
        if units and keep_with_previous and keep_with_previous(item):
            units[-1]["items"].append(item)
        else:
            units.append({"strand": deck_strand_of(mix, identify(item)), "items": [item]})
    queues = {strand["id"]: [] for strand in strands}
    for unit in units:
        queues.setdefault(unit["strand"], []).append(unit)
    if sum(1 for queue in queues.values() if queue) < 2:
        return items

    present = {strand["id"]: 0 for strand in strands}
    for unit in units:
        present[unit["strand"]] = present.get(unit["strand"], 0) + len(unit["items"])
    emitted = {strand["id"]: 0 for strand in strands}
    total = 0

    def share_of(strand: dict[str, Any]) -> float:
        if share_basis == "present":
            return present.get(strand["id"], 0) / len(items)
        return float(strand.get("share") or 0)

    def choose(available) -> str | None:
        best, best_score = None, float("-inf")
        for strand in strands:
            sid = strand["id"]
            if not available(sid):
                continue
            score = share_of(strand) * (total + 1) - emitted.get(sid, 0)
            if score > best_score + 1e-9:
                best, best_score = sid, score
        return best

    for _ in range(hash_string(f"{seed}\0deck-strand-phase") % 10):
        strand = choose(lambda _sid: True)
        if strand is None:
            break
        emitted[strand] = emitted.get(strand, 0) + 1
        total += 1

    out: list[dict[str, Any]] = []
    cursor = {strand["id"]: 0 for strand in strands}
    while len(out) < len(items):
        strand = choose(lambda sid: cursor.get(sid, 0) < len(queues.get(sid, [])))
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
