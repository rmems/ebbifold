"""Local HTTP surface for the domain-neutral review loop.

The API is deliberately thin: scheduler decisions stay in ``ReviewEngine`` and
learner memory stays behind ``SQLiteReviewStore``. Runtime database files default
to the user data directory, outside version control.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from ebbifold.corpus import load_corpus
from ebbifold.corpus.validation import open_content_root

from .engine import ReviewEngine, ReviewQuestion
from .store import AttemptInput, SQLiteReviewStore, default_store_path


class AnswerIn(BaseModel):
    selected_label: str = Field(min_length=1)
    item_id: str | None = None
    time_to_answer_ms: int | None = Field(default=None, ge=0)


class SessionIn(BaseModel):
    pass


def _question_payload(question: ReviewQuestion | None) -> dict[str, Any] | None:
    if question is None:
        return None
    payload: dict[str, Any] = {
        "item_id": question.id,
        "stem": question.stem,
        "difficulty": question.difficulty,
        "concept_id": question.concept_id,
        "concept_priority": question.concept_priority,
        "options": [{"label": option.label, "text": option.text} for option in question.options],
    }
    if question.record.image_url:
        payload["media"] = {
            "url": question.record.image_url,
            "caption": question.record.image_caption or "image attachment",
        }
    return payload


def _attempt_payload(row: Any) -> dict[str, Any]:
    return {
        "attempt_id": row["attempt_id"],
        "item_id": row["item_id"],
        "selected_label": row["selected_label"],
        "is_correct": bool(row["is_correct"]),
        "attempt_number": row["attempt_number"],
        "answered_at": row["answered_at"],
        "time_to_answer_ms": row["time_to_answer_ms"],
        "difficulty_at_attempt": row["difficulty_at_attempt"],
        "concept_at_attempt": row["concept_at_attempt"],
    }


def _session_payload(row: Any, attempts: list[Any]) -> dict[str, Any]:
    return {
        "session_id": row["session_id"],
        "created_at": row["created_at"],
        "content_root": row["content_root"],
        "status": row["status"],
        "attempt_count": len(attempts),
        "attempts": [_attempt_payload(attempt) for attempt in attempts],
    }


def _valid_label(question: ReviewQuestion, answer: str) -> bool:
    wanted = answer.strip().upper()
    return any(option.label.upper() == wanted for option in question.options)


def create_app(content: str | Path | None = None, db: str | Path | None = None) -> FastAPI:
    content_root = open_content_root(Path(content).resolve() if content else Path.cwd())
    engine = ReviewEngine(load_corpus(content_root))
    store = SQLiteReviewStore(db)
    app = FastAPI(title="ebbifold local review API")

    def get_store() -> SQLiteReviewStore:
        return store

    def require_session(session_id: str, review_store: SQLiteReviewStore = Depends(get_store)) -> Any:
        row = review_store.session(session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown review session")
        return row

    def current_question(session_id: str) -> ReviewQuestion | None:
        return engine.next_question(store.attempts(session_id), store.attempts())

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/sessions", status_code=201)
    def create_session(_: SessionIn | None = None, review_store: SQLiteReviewStore = Depends(get_store)) -> dict[str, Any]:
        session_id = review_store.create_session(str(content_root))
        row = review_store.session(session_id)
        return {
            "session": _session_payload(row, []),
            "current_item": _question_payload(current_question(session_id)),
        }

    @app.get("/sessions/{session_id}")
    def inspect_session(row: Any = Depends(require_session), review_store: SQLiteReviewStore = Depends(get_store)) -> dict[str, Any]:
        return _session_payload(row, review_store.attempts(row["session_id"]))

    @app.get("/sessions/{session_id}/current")
    def inspect_current(row: Any = Depends(require_session)) -> dict[str, Any]:
        return {"item": _question_payload(current_question(row["session_id"]))}

    @app.post("/sessions/{session_id}/answers", status_code=201)
    def submit_answer(answer: AnswerIn, row: Any = Depends(require_session), review_store: SQLiteReviewStore = Depends(get_store)) -> dict[str, Any]:
        session_id = row["session_id"]
        question = current_question(session_id)
        if question is None:
            raise HTTPException(status_code=409, detail="no unanswered items remain")
        if answer.item_id is not None and answer.item_id != question.id:
            raise HTTPException(status_code=409, detail="answer item does not match the current item")
        if not _valid_label(question, answer.selected_label):
            raise HTTPException(status_code=422, detail="answer must be one of the displayed labels")
        correct = ReviewEngine.grade(question, answer.selected_label)
        attempt_id = review_store.record_attempt(
            AttemptInput(
                session_id=session_id,
                item_id=question.id,
                selected_label=answer.selected_label.strip().upper(),
                is_correct=correct,
                answered_at=datetime.now(UTC),
                time_to_answer_ms=answer.time_to_answer_ms,
                difficulty_at_attempt=question.difficulty,
                concept_at_attempt=question.concept_id,
            )
        )
        return {
            "attempt_id": attempt_id,
            "correct": correct,
            "next_item": _question_payload(current_question(session_id)),
        }

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the local domain-neutral review API")
    parser.add_argument("--content", help="repository root or open-content directory")
    parser.add_argument("--db", default=str(default_store_path()), help="SQLite review database path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    import uvicorn

    uvicorn.run(create_app(args.content, args.db), host=args.host, port=args.port)
    return 0
