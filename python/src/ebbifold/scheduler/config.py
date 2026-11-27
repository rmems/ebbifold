"""Scheduler defaults and seeded randomness (unified-scheduler-config.ts)."""

from __future__ import annotations

import hashlib
import re

from .errors import SchedulerDeterminismError

DEFAULTS = {
    "size": 20,
    "card_ratio": 0.7,
    "interference_threshold": 0.85,
    "max_cards_per_concept": 1,
    "max_questions_per_concept": 1,
    "confidence_threshold": 0.3,
    "recall_threshold": 0.6,
    "target_recall": 0.8,
    "days_since_probe_threshold": 3,
}

RECALL_RANKING_HORIZON_DAYS = 21

_SEED_RE = re.compile(r"^[a-f0-9]{64}$")


def stable_target_random(seed: str | None, key: str) -> float:
    """sha256(seed + NUL + key), first 4 bytes big-endian, scaled to [0, 1).

    Upstream falls back to ``Math.random()`` for a missing or malformed seed.
    The Python core is fixture-driven and must be deterministic, so it fails
    closed instead.
    """
    if not seed or not _SEED_RE.fullmatch(seed) or seed.endswith("\n"):
        raise SchedulerDeterminismError(
            "examTargetTieBreakSeed must be 64 lowercase hex characters; "
            "the Python core never falls back to unseeded randomness"
        )
    digest = hashlib.sha256(seed.encode() + b"\0" + key.encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0x1_0000_0000
