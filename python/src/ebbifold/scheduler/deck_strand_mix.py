"""Serializable deck-strand mixing (deck-strand-mix.ts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class DeckStrandOrderOptions:
    seed: str
    identify: Callable[[dict[str, Any]], dict[str, Any]]
    keep_with_previous: Callable[[dict[str, Any]], bool] | None = None
    share_basis: str = "target"


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


def strand_seat_targets(mix: dict[str, Any] | None, seats: int) -> dict[str, int]:
    strands = (mix or {}).get("strands") or []
    if seats <= 0 or not strands:
        return {}
    exact = [(strand["id"], seats * float(strand.get("share") or 0)) for strand in strands]
    targets = {sid: int(value) for sid, value in exact}
    remaining = seats - sum(targets.values())
    for sid, _value in sorted(exact, key=lambda pair: pair[1] - int(pair[1]), reverse=True)[:remaining]:
        targets[sid] += 1
    return targets


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


@dataclass(frozen=True)
class StrandChoiceRequest:
    strands: list[dict[str, Any]]
    shares: Callable[[str], float]
    emitted: dict[str, int]
    total: int
    available: Callable[[str], bool]


def _choose_strand(request: StrandChoiceRequest):
    best, best_score = None, float("-inf")
    for strand in request.strands:
        sid = strand["id"]
        if not request.available(sid):
            continue
        score = request.shares(sid) * (request.total + 1) - request.emitted.get(sid, 0)
        if score > best_score + 1e-9:
            best, best_score = sid, score
    return best


class _DeckStrandOrdering:
    def __init__(self, items: list[dict[str, Any]], mix: dict[str, Any], options: DeckStrandOrderOptions) -> None:
        self.items = items
        self.mix = mix
        self.options = options
        self.strands = mix.get("strands") or []
        self.units = _strand_units(items, mix, options.identify, options.keep_with_previous)
        self.queues = _queues(self.strands, self.units)
        self.present = _present_counts(self.strands, self.units)
        self.emitted = {strand["id"]: 0 for strand in self.strands}
        self.cursor = {strand["id"]: 0 for strand in self.strands}
        self.total = 0
        self.out: list[dict[str, Any]] = []

    def has_multiple_strands(self) -> bool:
        return sum(1 for queue in self.queues.values() if queue) >= 2

    def share(self, sid: str) -> float:
        if self.options.share_basis == "present":
            return self.present.get(sid, 0) / len(self.items)
        strand = next(s for s in self.strands if s["id"] == sid)
        return float(strand.get("share") or 0)

    def available(self, sid: str) -> bool:
        return self.cursor.get(sid, 0) < len(self.queues.get(sid, []))

    def choose(self, available=None) -> str | None:
        request = StrandChoiceRequest(
            self.strands, self.share, self.emitted, self.total, available or (lambda _sid: True),
        )
        return _choose_strand(request)

    def apply_seed_phase(self) -> None:
        for _ in range(hash_string(f"{self.options.seed}\0deck-strand-phase") % 10):
            strand = self.choose()
            if strand is None:
                break
            self.emitted[strand] = self.emitted.get(strand, 0) + 1
            self.total += 1

    def emit_unit(self, unit: dict[str, Any]) -> None:
        for item in unit["items"]:
            self.out.append(item)
            item_strand = deck_strand_of(self.mix, self.options.identify(item))
            self.emitted[item_strand] = self.emitted.get(item_strand, 0) + 1
            self.total += 1

    def emit_next(self) -> bool:
        strand = self.choose(self.available)
        if strand is None:
            return False
        position = self.cursor.get(strand, 0)
        self.cursor[strand] = position + 1
        self.emit_unit(self.queues[strand][position])
        return True

    def run(self) -> list[dict[str, Any]]:
        self.apply_seed_phase()
        while len(self.out) < len(self.items):
            if not self.emit_next():
                break
        return self.out


def order_by_deck_strands(
    items: list[dict[str, Any]], mix: dict[str, Any] | None, options: DeckStrandOrderOptions,
) -> list[dict[str, Any]]:
    strands = (mix or {}).get("strands") or []
    if len(items) < 2 or not strands:
        return items
    ordering = _DeckStrandOrdering(items, mix or {}, options)
    if not ordering.has_multiple_strands():
        return items
    return ordering.run()
