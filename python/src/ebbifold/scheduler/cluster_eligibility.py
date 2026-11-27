"""Cluster fallback candidate eligibility filters."""

from __future__ import annotations

from typing import Any


def restricted(inputs: dict[str, Any]) -> bool:
    options = inputs["options"]
    return bool(
        options.get("media") or options.get("crossSourceRotations")
        or inputs.get("week") is not None or options.get("week") is not None
    )


def rotation_eligible(inputs: dict[str, Any], row: dict[str, Any]) -> bool:
    rotations = [inputs["rotation"], *(inputs["options"].get("crossSourceRotations") or [])]
    return len(rotations) == 1 or row.get("rotation") in rotations


def week_eligible(inputs: dict[str, Any], row: dict[str, Any]) -> bool:
    week = inputs.get("week", inputs["options"].get("week"))
    return week is None or row.get("week") == week


def media_eligible(media: str | None, row: dict[str, Any]) -> bool:
    if not media:
        return True
    image = bool((row.get("imageUrl") or "").strip())
    video = bool((row.get("clipId") or "").strip() or (row.get("clip") or {}).get("url"))
    if media == "image":
        return image
    if media == "video":
        return video
    return media == "any" and (image or video)


def eligible_row(inputs: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        rotation_eligible(inputs, row)
        and week_eligible(inputs, row)
        and media_eligible(inputs["options"].get("media"), row)
    )


def eligible_rows(inputs: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if eligible_row(inputs, row)]
