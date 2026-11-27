"""Shared parsing and manifest helpers for corpus loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schema import find_repo_root

JSON_GLOB = "*.json"


def open_content_root(root: str | Path | None = None) -> Path:
    if root is None:
        return find_repo_root() / "open-content"
    path = Path(root).resolve()
    if (path / "open-content").is_dir():
        return path / "open-content"
    if (path / "usmle").is_dir() and (path / "modules").is_dir():
        return path
    raise FileNotFoundError(f"{path} is not an ebbifold repo root or open-content directory")

def _repo_relative(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _validation_error_messages(source: str, error: ValidationError) -> list[str]:
    messages = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue.get("loc", ())) or "(root)"
        messages.append(f"{source}: {location}: {issue.get('msg', 'invalid value')}")
    return messages

def _expect_manifest_object(raw: Any, path: Path, version: int) -> list[str]:
    if not isinstance(raw, dict):
        return [f"{path}: expected manifest object"]
    if raw.get("schemaVersion") != version:
        return [f"{path}: expected schemaVersion {version}"]
    return []

def _id_list(raw: Any, key: str, path: Path) -> tuple[list[str], list[str]]:
    if not isinstance(raw, dict):
        return [], [f"{path}: {key} is unavailable because the manifest is not an object"]
    value = raw.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        return [], [f"{path}: {key} must be a non-empty string array"]
    duplicates = sorted({item for item in value if value.count(item) > 1})
    errors = [f"{path}: duplicate {key} entry {item}" for item in duplicates]
    return list(value), errors

def _read_question_manifest(
    path: Path, version: int, invalid_label: str
) -> tuple[Any, list[str], list[str]]:
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return {}, [], [f"{path}: invalid {invalid_label} ({error})"]
    errors = _expect_manifest_object(raw, path, version)
    question_ids, id_errors = _id_list(raw, "questionIds", path)
    errors.extend(id_errors)
    return raw, question_ids, errors
