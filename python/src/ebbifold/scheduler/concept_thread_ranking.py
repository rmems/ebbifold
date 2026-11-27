"""Question-ranking integration for concept-thread matches."""

from __future__ import annotations

from typing import Any, Mapping

from .policies import freshness_tier


def find_thread_matches(by_difficulty: dict[str, list[dict[str, Any]]], matcher: Any, concept_id: str) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    if matcher is None or not matcher.anchor_count:
        return matches
    for rows in by_difficulty.values():
        for candidate in rows:
            match = matcher.find_match({
                "id": candidate["id"], "itemType": "question", "text": candidate.get("stem") or "",
                "topics": candidate.get("topics") or [], "questionType": candidate.get("questionType"),
                "format": candidate.get("format"), "conceptIds": [concept_id],
                "variantGroupId": candidate.get("variantGroupId"),
            })
            if match:
                matches[candidate["id"]] = match
    return matches


def _tier_bounds(ranked: list[dict[str, Any]], familiarity: Mapping[str, Any], tier: int) -> tuple[int, int]:
    start = next((i for i, c in enumerate(ranked) if freshness_tier(familiarity.get(c["id"])) == tier), -1)
    if start < 0:
        return -1, -1
    end = start
    while end < len(ranked) and freshness_tier(familiarity.get(ranked[end]["id"])) == tier:
        end += 1
    return start, end


def _best_thread_winner(ranked: list[dict[str, Any]], matches: dict[str, dict[str, Any]], start: int, end: int) -> int:
    winner_index = -1
    winner_match = None
    for index in range(start, end):
        match = matches.get(ranked[index]["id"])
        if match and (winner_match is None or match["targetPreference"] < winner_match["targetPreference"]):
            winner_index = index
            winner_match = match
    return winner_index


def _promote_in_tier(ranked: list[dict[str, Any]], matches: dict[str, dict[str, Any]], familiarity: Mapping[str, Any], tier: int) -> None:
    start, end = _tier_bounds(ranked, familiarity, tier)
    if start < 0:
        return
    winner_index = _best_thread_winner(ranked, matches, start, end)
    if winner_index > start:
        ranked.insert(start, ranked.pop(winner_index))


def promote_thread_matches(
    by_difficulty: dict[str, list[dict[str, Any]]], matches: dict[str, dict[str, Any]], familiarity: Mapping[str, Any]
) -> None:
    if not matches:
        return
    for rows in by_difficulty.values():
        ranked = list(rows)
        for tier in (0, 1, 2):
            _promote_in_tier(ranked, matches, familiarity, tier)
        rows[:] = ranked


def attach_thread_matches(selected: list[dict[str, Any]], matches: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    for picked in selected:
        if picked["id"] in matches:
            picked["conceptThreadMatch"] = matches[picked["id"]]
    return selected
