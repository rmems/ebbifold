"""SQLite persistence contract for local review sessions and attempts.

Runtime learner memory is stored outside version control by default. The schema
is intentionally domain-neutral: items are opaque engine item IDs, while corpus
adapters decide what those IDs mean.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

STORE_ENV = "EBBIFOLD_REVIEW_DB"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class AttemptInput:
    session_id: str
    item_id: str
    selected_label: str
    is_correct: bool
    answered_at: datetime
    time_to_answer_ms: int | None = None
    difficulty_at_attempt: str | None = None
    concept_at_attempt: str | None = None


def default_store_path() -> Path:
    override = os.environ.get(STORE_ENV)
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "ebbifold" / "review.sqlite3"
    return Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ebbifold" / "review.sqlite3"


def _validate_store_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.suffix not in {".db", ".sqlite", ".sqlite3"} or any(mark in candidate.name for mark in "?#"):
        raise ValueError("review database path must be a local .db/.sqlite/.sqlite3 file")
    return candidate


class SQLiteReviewStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = _validate_store_path(path if path is not None else default_store_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    content_root TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('active', 'completed'))
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES review_sessions(session_id),
                    item_id TEXT NOT NULL,
                    selected_label TEXT NOT NULL,
                    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
                    attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
                    answered_at TEXT NOT NULL,
                    time_to_answer_ms INTEGER CHECK(time_to_answer_ms IS NULL OR time_to_answer_ms >= 0),
                    difficulty_at_attempt TEXT,
                    concept_at_attempt TEXT
                );
                CREATE TABLE IF NOT EXISTS archived_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    selected_label TEXT NOT NULL,
                    is_correct INTEGER NOT NULL CHECK(is_correct IN (0, 1)),
                    attempt_number INTEGER NOT NULL CHECK(attempt_number >= 1),
                    answered_at TEXT NOT NULL,
                    time_to_answer_ms INTEGER CHECK(time_to_answer_ms IS NULL OR time_to_answer_ms >= 0),
                    difficulty_at_attempt TEXT,
                    concept_at_attempt TEXT,
                    archived_at TEXT NOT NULL,
                    archive_reason TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS attempts_session_idx ON attempts(session_id, answered_at);
                CREATE INDEX IF NOT EXISTS attempts_item_idx ON attempts(item_id, answered_at);
                """
            )
            conn.execute("DROP INDEX IF EXISTS attempts_session_item_attempt_unique")
            conn.execute("DROP INDEX IF EXISTS attempts_session_item_unique")
            conn.execute(
                """
                INSERT OR IGNORE INTO archived_attempts(
                    attempt_id, session_id, item_id, selected_label, is_correct, attempt_number,
                    answered_at, time_to_answer_ms, difficulty_at_attempt, concept_at_attempt,
                    archived_at, archive_reason
                )
                SELECT
                    attempt_id, session_id, item_id, selected_label, is_correct, attempt_number,
                    answered_at, time_to_answer_ms, difficulty_at_attempt, concept_at_attempt,
                    datetime('now'), 'duplicate-session-item'
                FROM attempts
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM attempts GROUP BY session_id, item_id
                )
                """
            )
            conn.execute(
                """
                DELETE FROM attempts
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) FROM attempts GROUP BY session_id, item_id
                )
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX attempts_session_item_unique
                ON attempts(session_id, item_id)
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def create_session(self, content_root: str) -> str:
        session_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO review_sessions(session_id, created_at, content_root, status) VALUES (?, ?, ?, 'active')",
                (session_id, datetime.now(UTC).isoformat(), content_root),
            )
        return session_id

    def session(self, session_id: str) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM review_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()

    def session_exists(self, session_id: str) -> bool:
        return self.session(session_id) is not None

    def record_attempt(self, attempt: AttemptInput) -> str:
        attempt_id = str(uuid.uuid4())
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(attempt_number), 0) + 1 AS n FROM attempts WHERE session_id = ? AND item_id = ?",
                (attempt.session_id, attempt.item_id),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO attempts(
                    attempt_id, session_id, item_id, selected_label, is_correct, attempt_number,
                    answered_at, time_to_answer_ms, difficulty_at_attempt, concept_at_attempt
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    attempt.session_id,
                    attempt.item_id,
                    attempt.selected_label,
                    1 if attempt.is_correct else 0,
                    row["n"],
                    attempt.answered_at.isoformat(),
                    attempt.time_to_answer_ms,
                    attempt.difficulty_at_attempt,
                    attempt.concept_at_attempt,
                ),
            )
        return attempt_id

    def complete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE review_sessions SET status = 'completed' WHERE session_id = ?",
                (session_id,),
            )

    def attempts(self, session_id: str | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM attempts"
        args: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            args = (session_id,)
        sql += " ORDER BY answered_at, rowid"
        with self._connect() as conn:
            return list(conn.execute(sql, args))
