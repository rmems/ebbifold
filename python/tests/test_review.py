from datetime import UTC, datetime
from pathlib import Path
from typing import cast
import argparse
import sqlite3

from ebbifold.corpus import load_corpus
from ebbifold.review.cli import CliContext, _read_answer, run
import ebbifold.review.engine as engine_module
from ebbifold.review.engine import ReviewEngine
from ebbifold.review.store import AttemptInput, SQLiteReviewStore


def _attempt(store: SQLiteReviewStore, session_id: str, item_id: str, correct: bool) -> None:
    store.record_attempt(
        AttemptInput(
            session_id=session_id,
            item_id=item_id,
            selected_label="A",
            is_correct=correct,
            answered_at=datetime.now(UTC),
        )
    )


def test_review_store_persists_attempts_across_restarts(tmp_path):
    path = tmp_path / "review.sqlite3"
    first = SQLiteReviewStore(path)
    session_id = first.create_session("validation-corpus")
    _attempt(first, session_id, "item-1", True)

    reopened = SQLiteReviewStore(path)
    attempts = reopened.attempts(session_id)

    assert len(attempts) == 1
    assert attempts[0]["item_id"] == "item-1"
    assert attempts[0]["is_correct"] == 1


def test_review_engine_uses_persisted_attempts_for_next_item(tmp_path, monkeypatch):
    snapshot = load_corpus()
    engine = ReviewEngine(snapshot, now_ms=1_700_000_000_000)
    store = SQLiteReviewStore(tmp_path / "review.sqlite3")
    session_id = store.create_session("validation-corpus")
    rank_inputs = []
    real_rank_adaptive = engine_module.rank_adaptive

    def spy_rank_adaptive(questions, history, size):
        rank_inputs.append({question.id: question.preference_rank for question in questions})
        return real_rank_adaptive(questions, history, size)

    monkeypatch.setattr(engine_module, "rank_adaptive", spy_rank_adaptive)

    first = engine.next_question([], [])
    assert first is not None
    baseline_priority = engine.concept_priorities([])[first.concept_id]
    _attempt(store, session_id, first.id, False)

    session_attempts = store.attempts(session_id)
    persisted_attempts = store.attempts()
    persisted_priority = engine.concept_priorities(persisted_attempts)[first.concept_id]
    second = engine.next_question(session_attempts, persisted_attempts)

    assert persisted_priority != baseline_priority
    assert rank_inputs[1][first.id] != rank_inputs[0][first.id]
    assert second is not None
    assert second.id != first.id


def test_review_store_schema_is_domain_neutral(tmp_path):
    path = tmp_path / "review.sqlite3"
    SQLiteReviewStore(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
    assert "item_id" in columns
    assert "concept_at_attempt" in columns
    assert not any("usmle" in column or "medical" in column for column in columns)


def test_scripted_cli_session_persists_attempt_and_resumes(tmp_path, capsys):
    db = tmp_path / "review.sqlite3"
    repo_root = Path(__file__).resolve().parents[2]
    args = argparse.Namespace(
        content=str(repo_root),
        db=str(db),
        session=None,
        size=1,
        script="A",
    )

    assert run(args) == 0
    output = capsys.readouterr().out
    session_id = next(line.split()[-1] for line in output.splitlines() if line.startswith("session "))
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1

    resume_args = argparse.Namespace(
        content=str(repo_root),
        db=str(db),
        session=session_id,
        size=1,
        script="B",
    )
    assert run(resume_args) == 0
    store = SQLiteReviewStore(db)
    assert store.session_exists(session_id)
    assert len(store.attempts(session_id)) == 2
    with sqlite3.connect(db) as conn:
        stored_root = conn.execute("SELECT content_root FROM review_sessions WHERE session_id = ?", (session_id,)).fetchone()[0]
    assert stored_root.replace("\\", "/").endswith("/open-content")


def test_empty_script_and_eof_end_without_input(monkeypatch):
    ctx = CliContext(engine=cast(ReviewEngine, None), store=cast(SQLiteReviewStore, None), session_id="s", scripted=[], script_mode=True)
    assert _read_answer(ctx, 0) == (None, 0)

    interactive = CliContext(engine=cast(ReviewEngine, None), store=cast(SQLiteReviewStore, None), session_id="s", scripted=[], script_mode=False)
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(EOFError()))
    assert _read_answer(interactive, 0) == (None, 0)
