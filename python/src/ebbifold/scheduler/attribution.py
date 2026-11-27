"""Card-to-concept attribution (scheduler-attribution.ts)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .topics import normalize_topic


def build_concept_topic_index(concepts: Iterable[Mapping[str, Any]]) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for concept in concepts:
        for topic in concept["topics"]:
            normalized = normalize_topic(topic)
            if normalized:
                index.setdefault(normalized, set()).add(concept["id"])
    return index


def concept_ids_for_topics(topics: Iterable[str], index: Mapping[str, set[str]]) -> set[str]:
    ids: set[str] = set()
    for topic in topics:
        normalized = normalize_topic(topic)
        if normalized:
            ids.update(index.get(normalized, ()))
    return ids


def resolve_card_concept_id(
    card: Mapping[str, Any], current_concept_ids: set[str], index: Mapping[str, set[str]]
) -> str | None:
    if card.get("conceptId") and card["conceptId"] in current_concept_ids:
        return card["conceptId"]
    matches = concept_ids_for_topics(card.get("topics", []), index)
    return next(iter(matches)) if len(matches) == 1 else None
