"""Adaptive difficulty for the Step 1 reviewer — Python port.

Ported from src/lib/usmle/step1-adaptive.ts (141 lines, pure and
dependency-free). Behavior must match the TypeScript original exactly; the
golden fixtures under tests/fixtures/ are the proof. Do not "improve" the
rules here — divergence is a bug, not a feature.

The rule: three tiers; two consecutive correct answers at a tier move up, one
miss drops a tier immediately. After a miss, follow-ups come from the missed
concept's teaching ladder before widening to its domain, then everything else.
"""

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

DifficultyTier = Literal["easy", "medium", "hard"]

DIFFICULTY_TIERS: tuple[DifficultyTier, ...] = ("easy", "medium", "hard")

# Consecutive correct answers required at a tier before climbing. Promoting on
# one produces a sawtooth; demotion stays immediate — nobody should have to
# fail twice to be helped. The asymmetry is deliberate.
PROMOTION_STREAK = 2


@dataclass(frozen=True)
class AdaptiveQuestion:
    id: str
    difficulty: str | None = None
    domain: str | None = None
    # Lower wins only after tier and missed-concept scaffolding signals tie.
    preference_rank: float | None = None
    # Groups rungs that build to one concept.
    ladder_id: str | None = None


@dataclass(frozen=True)
class AdaptiveHistoryRow:
    question_id: str
    is_correct: bool
    created_at: datetime


def normalise_difficulty(raw: str | None) -> DifficultyTier:
    """Map a raw corpus difficulty label onto a tier.

    The open corpus labels the middle rung both 'medium' and 'moderate', and
    anything unrecognised lands in the middle too: guessing 'easy' strands a
    strong reader, guessing 'hard' buries a weak one.
    """
    # Match ECMAScript String.trim, not Python's default whitespace set:
    # JS strips U+FEFF but not U+0085; Python does the reverse.
    js_whitespace = (
        "\t\n\v\f\r "
        "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005"
        "\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f"
        "\u3000\ufeff"
    )
    value = (raw or "").strip(js_whitespace).lower()
    if value == "easy":
        return "easy"
    if value == "hard":
        return "hard"
    return "medium"


def next_tier(current: DifficultyTier, was_correct: bool) -> DifficultyTier:
    """Climb on success, step down on failure, and clamp at both ends."""
    i = DIFFICULTY_TIERS.index(current)
    moved = i + 1 if was_correct else i - 1
    # Clamping at the floor is the load-bearing half: a learner who keeps
    # missing easy questions must keep getting easy ones rather than falling
    # out of the ladder entirely.
    return DIFFICULTY_TIERS[min(len(DIFFICULTY_TIERS) - 1, max(0, moved))]


def _preference_rank(question: AdaptiveQuestion) -> float:
    """Mirror Number.isFinite(q.preferenceRank) ? q.preferenceRank : 0."""
    rank = question.preference_rank
    if rank is None or not math.isfinite(rank):
        return 0.0
    return rank


def _epoch(row: AdaptiveHistoryRow) -> float:
    """Instant comparison, matching JS epoch-millisecond semantics.

    Comparing Python datetimes directly is wrong across a repeated local hour
    when both share one tzinfo object (wall times compare, folds collapse).
    JavaScript Dates always compare as instants, so convert to epoch seconds.
    """
    return row.created_at.timestamp()


def _resolve_target_and_scaffold(
    history: list[AdaptiveHistoryRow],
    by_id: dict[str, AdaptiveQuestion],
) -> tuple[DifficultyTier, str | None, str | None]:
    """Compute the target tier and the missed-concept scaffold signals.

    Returns (target, scaffold_ladder, scaffold_domain). The scaffold pair is
    set only when the learner just got something WRONG: stepping down to an
    easier question is not teaching — they need groundwork for what they
    missed. On the way UP the concept is proved, so breadth beats drilling.
    """
    # JS: history.reduce((a, b) => (b.createdAt >= a.createdAt ? b : a))
    # — on a timestamp tie the LATER row in the array wins.
    latest = history[0]
    for row in history[1:]:
        if _epoch(row) >= _epoch(latest):
            latest = row
    asked = by_id.get(latest.question_id)
    asked_tier = normalise_difficulty(asked.difficulty if asked else None)
    # Count the run of consecutive correct answers ending here, at this tier.
    # A miss, or an answer at a different tier, ends the run. Iterate the
    # ascending stable sort backwards, exactly like the TS original — among
    # equal timestamps the later-indexed row is visited first.
    streak = 0
    ordered = sorted(history, key=_epoch)
    for row in ordered[::-1]:
        if not row.is_correct:
            break
        tier = normalise_difficulty(
            (by_id.get(row.question_id).difficulty)
            if row.question_id in by_id
            else None
        )
        if tier != asked_tier:
            break
        streak += 1
    target = (
        asked_tier
        if latest.is_correct and streak < PROMOTION_STREAK
        else next_tier(asked_tier, latest.is_correct)
    )
    if latest.is_correct:
        return target, None, None
    scaffold_ladder = asked.ladder_id if asked else None
    scaffold_domain = asked.domain if asked else None
    return target, scaffold_ladder, scaffold_domain


def rank_adaptive(
    questions: list[AdaptiveQuestion],
    history: list[AdaptiveHistoryRow],
    size: int,
) -> list[AdaptiveQuestion]:
    """Order unanswered questions for this learner.

    Distance from the target tier is the sort key, so an exhausted tier falls
    back to the nearest neighbour rather than ending the session.
    """
    by_id = {q.id: q for q in questions}
    answered = {row.question_id for row in history}

    if history:
        target, scaffold_ladder, scaffold_domain = _resolve_target_and_scaffold(
            history, by_id
        )
    else:
        target, scaffold_ladder, scaffold_domain = "medium", None, None
    target_index = DIFFICULTY_TIERS.index(target)

    def sort_key(q: AdaptiveQuestion) -> tuple[float, int, int, float, str]:
        distance = abs(
            DIFFICULTY_TIERS.index(normalise_difficulty(q.difficulty)) - target_index
        )
        # Preferences, not filters. An exhausted ladder or domain must not end
        # the session, so everything still ranks — just later.
        off_ladder = 1 if scaffold_ladder is not None and q.ladder_id != scaffold_ladder else 0
        off_domain = 1 if scaffold_domain is not None and q.domain != scaffold_domain else 0
        # Tier, then the ladder that was missed, then its domain, then id.
        # Deterministic: same learner, same pool, same order.
        return (distance, off_ladder, off_domain, _preference_rank(q), q.id)

    # Python's sort is stable, matching the deterministic tie-breaking of the
    # TS original once the id key is included.
    ranked = sorted((q for q in questions if q.id not in answered), key=sort_key)
    return ranked[:size]
