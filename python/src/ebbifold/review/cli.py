"""Command-line review loop.

Examples:
    python -m ebbifold.review.cli --size 5
    python -m ebbifold.review.cli --script A,B,C --db review.sqlite3
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ebbifold.corpus import load_corpus
from ebbifold.corpus.validation import open_content_root

from .engine import ReviewEngine, ReviewQuestion
from .store import AttemptInput, SQLiteReviewStore, default_store_path


@dataclass
class CliContext:
    engine: ReviewEngine
    store: SQLiteReviewStore
    session_id: str
    scripted: list[str]
    script_mode: bool


def _print_question(question: ReviewQuestion) -> None:
    print(f"\n{question.stem}")
    if question.record.image_url:
        caption = question.record.image_caption or "image attachment"
        print(f"[media: {caption}] {question.record.image_url}")
    for option in question.options:
        print(f"{option.label}. {option.text}")


def _valid_label(question: ReviewQuestion, answer: str) -> bool:
    wanted = answer.strip().upper()
    return any(option.label.upper() == wanted for option in question.options)


def _answer(ctx: CliContext, question: ReviewQuestion, answer: str, started: datetime) -> bool:
    correct = ReviewEngine.grade(question, answer)
    answered_at = datetime.now(UTC)
    ctx.store.record_attempt(
        AttemptInput(
            session_id=ctx.session_id,
            item_id=question.id,
            selected_label=answer.strip().upper(),
            is_correct=correct,
            answered_at=answered_at,
            time_to_answer_ms=int((answered_at - started).total_seconds() * 1000),
            difficulty_at_attempt=question.difficulty,
            concept_at_attempt=question.concept_id,
        )
    )
    print("correct" if correct else "incorrect")
    return correct


def _open_session(args: argparse.Namespace, store: SQLiteReviewStore, content_root: Path) -> str | None:
    if not args.session:
        return store.create_session(str(content_root))
    if store.session_exists(args.session):
        return args.session
    print(f"unknown review session: {args.session}", file=sys.stderr)
    return None


def _read_answer(ctx: CliContext, scripted_index: int) -> tuple[str | None, int]:
    if scripted_index < len(ctx.scripted):
        answer = ctx.scripted[scripted_index]
        print(f"> {answer}")
        return answer, scripted_index + 1
    if ctx.script_mode:
        return None, scripted_index
    try:
        return input("answer: "), scripted_index
    except EOFError:
        return None, scripted_index


def _submit_answer(ctx: CliContext, question: ReviewQuestion, answer: str, started: datetime) -> str:
    if answer.strip().lower() in {"q", "quit", "exit"}:
        return "stop"
    if _valid_label(question, answer):
        _answer(ctx, question, answer, started)
        return "answered"
    print("answer must be one of the displayed labels", file=sys.stderr)
    return "invalid-scripted" if ctx.script_mode else "retry"


def _review_loop(ctx: CliContext, size: int) -> int:
    scripted_index = 0
    answered = 0
    while answered < size:
        session_attempts = ctx.store.attempts(ctx.session_id)
        question = ctx.engine.next_question(session_attempts, ctx.store.attempts())
        if question is None:
            print("no unanswered items remain")
            break
        started = datetime.now(UTC)
        _print_question(question)
        answer, scripted_index = _read_answer(ctx, scripted_index)
        if answer is None:
            break
        outcome = _submit_answer(ctx, question, answer, started)
        if outcome == "invalid-scripted":
            return 2
        if outcome == "stop":
            break
        if outcome == "answered":
            answered += 1
    return 0


def run(args: argparse.Namespace) -> int:
    content_root = open_content_root(Path(args.content).resolve() if args.content else Path.cwd())
    snapshot = load_corpus(content_root)
    store = SQLiteReviewStore(args.db)
    session_id = _open_session(args, store, content_root)
    if session_id is None:
        return 2
    print(f"session {session_id}")
    ctx = CliContext(
        engine=ReviewEngine(snapshot),
        store=store,
        session_id=session_id,
        scripted=[part.strip() for part in args.script.split(",")] if args.script else [],
        script_mode=args.script is not None,
    )
    return _review_loop(ctx, args.size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local domain-neutral review session")
    parser.add_argument("--content", help="repository root or open-content directory")
    parser.add_argument("--db", default=str(default_store_path()), help="SQLite review database path")
    parser.add_argument("--session", help="resume an existing session id")
    parser.add_argument("--size", type=int, default=10, help="number of answers this invocation")
    parser.add_argument("--script", help="comma-separated answer labels for scripted/noninteractive runs")
    return parser


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
