"""Fail-closed cloze-card shard loading and release validation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .fingerprint import cohort_card_serving_fingerprint
from .loader_types import ClozeCardCorpus
from .models import ClozeCardRecord, ClozeCardShard
from .schema import CorpusAdmissionError
from .validation import (
    JSON_GLOB,
    _expect_manifest_object,
    _id_list,
    _read_json,
    _repo_relative,
    _validation_error_messages,
    open_content_root,
)


def _card_shard_suffix(card: ClozeCardRecord) -> str:
    match = re.search(r":c-([0-9a-f])", card.id)
    if not match:
        raise ValueError(f"unexpected public card id {card.id}")
    return f"open-content/modules/cards/{card.discipline}/{match.group(1)}.json"

def _card_placement_errors(
    card: ClozeCardRecord, shard: ClozeCardShard, source: str
) -> list[str]:
    errors: list[str] = []
    try:
        expected_suffix = _card_shard_suffix(card)
    except ValueError as error:
        errors.append(f"{source}: {card.id}: {error}")
        expected_suffix = None
    if card.discipline != shard.discipline:
        errors.append(f"{source}: {card.id}: discipline does not match shard")
    if expected_suffix is not None and source != expected_suffix:
        errors.append(f"{source}: {card.id}: belongs in {expected_suffix}")
    return errors

def _load_card_shard(
    path: Path, repo_root: Path
) -> tuple[list[tuple[ClozeCardRecord, dict[str, Any], str]], list[str]]:
    source = _repo_relative(path, repo_root)
    try:
        raw_shard = _read_json(path)
        shard = ClozeCardShard.model_validate(raw_shard)
    except (OSError, json.JSONDecodeError) as error:
        return [], [f"{source}: invalid JSON ({error})"]
    except ValidationError as error:
        return [], _validation_error_messages(source, error)

    errors: list[str] = []
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]] = []
    for raw_card, card in zip(raw_shard.get("cards", []), shard.cards, strict=True):
        errors.extend(_card_placement_errors(card, shard, source))
        cards.append((card, raw_card, source))
    return cards, errors

def _read_card_release(release_path: Path) -> tuple[Any, list[str], list[str]]:
    try:
        release = _read_json(release_path)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [], [f"{release_path}: invalid card release ({error})"]
    errors = _expect_manifest_object(release, release_path, 1)
    release_ids, id_errors = _id_list(release, "cardIds", release_path)
    errors.extend(id_errors)
    return release, release_ids, errors

def _index_cards(
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]]
) -> tuple[dict[str, tuple[ClozeCardRecord, dict[str, Any], str]], list[str]]:
    errors: list[str] = []
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]] = {}
    for card, raw, source in cards:
        if card.id in on_disk:
            errors.append(f"{source}: duplicate card id {card.id}")
        on_disk[card.id] = (card, raw, source)
    return on_disk, errors

def _card_membership_errors(
    release_path: Path,
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]],
    release_ids: list[str],
) -> list[str]:
    released = set(release_ids)
    return [
        *(f"{release_path}: {card_id} is on disk but not in the release"
          for card_id in sorted(on_disk) if card_id not in released),
        *(f"{release_path}: {card_id} is released but its file is missing"
          for card_id in release_ids if card_id not in on_disk),
    ]

def _card_fingerprint_errors(
    release_path: Path,
    on_disk: dict[str, tuple[ClozeCardRecord, dict[str, Any], str]],
    release_ids: list[str],
    fingerprints: dict[str, Any],
) -> list[str]:
    released = set(release_ids)
    return [
        f"{release_path}: {card_id} serving fingerprint does not match its file"
        for card_id, (_, raw, _) in on_disk.items()
        if card_id in released and fingerprints.get(card_id) != cohort_card_serving_fingerprint(raw)
    ]

def load_cloze_card_corpus(root: str | Path | None = None) -> ClozeCardCorpus:
    content_root = open_content_root(root)
    repo_root = content_root.parent
    module_root = content_root / "modules"
    release_path = module_root / "cards-release-v1.json"
    errors: list[str] = []
    cards: list[tuple[ClozeCardRecord, dict[str, Any], str]] = []

    for path in sorted((module_root / "cards").rglob(JSON_GLOB)):
        shard_cards, shard_errors = _load_card_shard(path, repo_root)
        cards.extend(shard_cards)
        errors.extend(shard_errors)

    release, release_ids, release_errors = _read_card_release(release_path)
    errors.extend(release_errors)
    fingerprints = release.get("cardFingerprints") if isinstance(release, dict) else None
    if not isinstance(fingerprints, dict):
        errors.append(f"{release_path}: cardFingerprints must be an object")
        fingerprints = {}

    on_disk, index_errors = _index_cards(cards)
    errors.extend(index_errors)
    errors.extend(_card_membership_errors(release_path, on_disk, release_ids))
    errors.extend(_card_fingerprint_errors(release_path, on_disk, release_ids, fingerprints))

    if errors:
        raise CorpusAdmissionError(errors)
    return ClozeCardCorpus(
        cards=tuple(card for card, _, _ in cards), release_card_ids=tuple(release_ids)
    )

def load_cloze_cards(root: str | Path | None = None) -> list[ClozeCardRecord]:
    return list(load_cloze_card_corpus(root).cards)
