"""Local HTTP surface for the domain-neutral review loop.

The API is deliberately thin: scheduler decisions stay in ``ReviewEngine`` and
learner memory stays behind ``SQLiteReviewStore``. Runtime database files default
to the user data directory, outside version control.
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ebbifold.corpus import load_corpus
from ebbifold.corpus.validation import open_content_root

from .engine import ReviewEngine, ReviewQuestion
from .store import AttemptInput, SQLiteReviewStore, default_store_path


class AnswerIn(BaseModel):
    selected_label: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
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


def _invalid_path_part(part: str) -> bool:
    separators = "/" in part or "\\" in part
    return separators or part in {"", ".", ".."}


def _figure_path(content_root: Path, domain: str, dataset: str, filename: str) -> Path | None:
    if any(map(_invalid_path_part, (domain, dataset, filename))):
        return None
    media_root = (content_root / domain / dataset / "media").resolve()
    path = (media_root / filename).resolve()
    if media_root not in path.parents or not path.is_file():
        return None
    return path


class ReviewApi:
    def __init__(self, engine: ReviewEngine, store: SQLiteReviewStore, content_root: Path) -> None:
        self.engine = engine
        self.store = store
        self.content_root = content_root

    def session_state(self, session_id: str) -> tuple[Any, ReviewQuestion | None]:
        row = self.store.session(session_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown review session")
        question = self.current_question(session_id)
        if row["status"] == "active" and question is None:
            self.store.complete_session(session_id)
            row = self.store.session(session_id)
        return row, question

    def fresh_engine(self) -> ReviewEngine:
        return ReviewEngine(self.engine.snapshot, days_to_exam=self.engine.days_to_exam)

    def current_question(self, session_id: str) -> ReviewQuestion | None:
        return self.fresh_engine().next_question(self.store.attempts(session_id), self.store.attempts())

    def health(self) -> dict[str, str]:
        return {"status": "ok"}

    def create_session(self, _: SessionIn | None = None) -> dict[str, Any]:
        session_id = self.store.create_session(str(self.content_root))
        return {
            "session": _session_payload(self.store.session(session_id), []),
            "current_item": _question_payload(self.current_question(session_id)),
        }

    def inspect_session(self, session_id: str) -> dict[str, Any]:
        row, _ = self.session_state(session_id)
        return _session_payload(row, self.store.attempts(session_id))

    def inspect_current(self, session_id: str) -> dict[str, Any]:
        _, question = self.session_state(session_id)
        return {"item": _question_payload(question)}

    def validate_answer_target(self, answer: AnswerIn, question: ReviewQuestion | None) -> ReviewQuestion:
        if question is None:
            raise HTTPException(status_code=409, detail="no unanswered items remain")
        if answer.item_id != question.id:
            raise HTTPException(status_code=409, detail="answer item does not match the current item")
        if not _valid_label(question, answer.selected_label):
            raise HTTPException(status_code=422, detail="answer must be one of the displayed labels")
        return question

    def submit_answer(self, session_id: str, answer: AnswerIn) -> dict[str, Any]:
        _, current = self.session_state(session_id)
        question = self.validate_answer_target(answer, current)
        correct = ReviewEngine.grade(question, answer.selected_label)
        try:
            attempt_id = self.store.record_attempt(
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
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail="current item was already answered") from exc
        next_question = self.current_question(session_id)
        if next_question is None:
            self.store.complete_session(session_id)
        return {
            "attempt_id": attempt_id,
            "correct": correct,
            "next_item": _question_payload(next_question),
        }

    def figure(self, domain: str, dataset: str, filename: str) -> FileResponse:
        path = _figure_path(self.content_root, domain, dataset, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="unknown figure")
        return FileResponse(path)


def create_app(content: str | Path | None = None, db: str | Path | None = None) -> FastAPI:
    content_root = open_content_root(Path(content).resolve() if content else Path.cwd())
    api = ReviewApi(ReviewEngine(load_corpus(content_root)), SQLiteReviewStore(db), content_root)
    app = FastAPI(title="ebbifold local review API")
    app.get("/health")(api.health)
    app.post("/sessions", status_code=201)(api.create_session)
    missing_session = {404: {"description": "Unknown review session"}}
    app.get("/sessions/{session_id}", responses=missing_session)(api.inspect_session)
    app.get("/sessions/{session_id}/current", responses=missing_session)(api.inspect_current)
    app.get(
        "/figures/{domain}/{dataset}/{filename}",
        responses={404: {"description": "Unknown corpus figure"}},
    )(api.figure)
    app.post(
        "/sessions/{session_id}/answers",
        status_code=201,
        responses={
            **missing_session,
            409: {"description": "No current item remains or the answer item is stale"},
            422: {"description": "Answer label is not one of the displayed labels"},
        },
    )(api.submit_answer)
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
