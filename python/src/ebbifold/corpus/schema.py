"""Schema-first validation for corpus records."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonschema

QUESTION_SCHEMA = "question-record-v1.json"
ATTEMPT_SCHEMA = "attempt-record-v1.json"


class CorpusSchemaError(ValueError):
    """A corpus document violates its versioned JSON Schema."""


class CorpusAdmissionError(ValueError):
    """Fail-closed admission refused one or more corpus records."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        detail = "\n".join(errors[:20])
        if len(errors) > 20:
            detail += f"\n... and {len(errors) - 20} more error(s)"
        super().__init__(detail)


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "open-content").is_dir() and (candidate / "schemas").is_dir():
            return candidate
    raise FileNotFoundError("could not locate the ebbifold repository root")


def schema_path(name: str, repo_root: Path | None = None) -> Path:
    root = repo_root or find_repo_root()
    path = root / "schemas" / name
    if not path.is_file():
        raise FileNotFoundError(f"schema not found: {path}")
    return path


@lru_cache(maxsize=8)
def _validator_for(path_text: str) -> jsonschema.Draft202012Validator:
    path = Path(path_text)
    schema = json.loads(path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())


def validate_against_schema(
    payload: dict[str, Any],
    schema_name: str,
    *,
    repo_root: Path | None = None,
    source: str = "<memory>",
) -> None:
    path = schema_path(schema_name, repo_root)
    validator = _validator_for(str(path))
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.absolute_path))
    if errors:
        details = []
        for error in errors:
            location = ".".join(str(part) for part in error.absolute_path) or "(root)"
            details.append(f"{source}: {location}: {error.message}")
        raise CorpusSchemaError("\n".join(details))
