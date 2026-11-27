"""Command-line review loop.

Examples:
    python -m ebbifold.review.cli --size 5
    python -m ebbifold.review.cli --script A,B,C --db review.sqlite3
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from ebbifold.corpus import load_corpus

from .engine import ReviewEngine, ReviewQuestion
from .store import AttemptInput, SQLiteReviewStore, default_store_path


def _print_question(question: ReviewQuestion) -> None:
    print(f"\n{question.stem}")
    for option in question.options:
        print(f"{option.label}. {option.text}")


def _valid_label(question: ReviewQuestion, answer: str) -> bool:
    wanted = answer.strip().upper()
    return any(option.label.upper() == wanted for option in question.options)


def _answer(question: ReviewQuestion, answer: str, store: SQLiteReviewStore, session_id: str, started: datetime) -> bool:
    correct = ReviewEngine.grade(question, answer)
    store.record_attempt(
        AttemptInput(
            session_id=session_id,
            item_id=question.id,
            selected_label=answer.strip().upper(),
            is_correct=correct,
            answered_at=datetime.now(UTC),
            time_to_answer_ms=int((datetime.now(UTC) - started).total_seconds() * 1000),
            difficulty_at_attempt=question.difficulty,
            concept_at_attempt=question.concept_id,
        )
    )
    print("correct" if correct else "incorrect")
    return correct


def run(args: argparse.Namespace) -> int:
    content_root = Path(args.content).resolve() if args.content else None
    snapshot = load_corpus(content_root)
    engine = ReviewEngine(snapshot)
    store = SQLiteReviewStore(args.db)
    if args.session:
        if not store.session_exists(args.session):
            print(f"unknown review session: {args.session}", file=sys.stderr)
            return 2
        session_id = args.session
    else:
        session_id = store.create_session(str(content_root or Path.cwd()))
    print(f"session {session_id}")

    scripted = [part.strip() for part in args.script.split(",")] if args.script else []
    scripted_index = 0
    answered = 0
    while answered < args.size:
        session_attempts = store.attempts(session_id)
        question = engine.next_question(session_attempts, store.attempts())
        if question is None:
            print("no unanswered items remain")
            break
        started = datetime.now(UTC)
        _print_question(question)
        if scripted_index < len(scripted):
            answer = scripted[scripted_index]
            scripted_index += 1
            print(f"> {answer}")
        elif args.script:
            break
        else:
            answer = input("answer: ")
        if answer.strip().lower() in {"q", "quit", "exit"}:
            break
        if not _valid_label(question, answer):
            print("answer must be one of the displayed labels", file=sys.stderr)
            if args.script:
                return 2
            continue
        _answer(question, answer, store, session_id, started)
        answered += 1
    return 0


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
