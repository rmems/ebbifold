from datetime import UTC, datetime
import sqlite3

from ebbifold.corpus import load_corpus
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


def test_review_engine_uses_persisted_attempts_for_next_item(tmp_path):
    snapshot = load_corpus()
    engine = ReviewEngine(snapshot, now_ms=1_700_000_000_000)
    store = SQLiteReviewStore(tmp_path / "review.sqlite3")
    session_id = store.create_session("validation-corpus")

    first = engine.next_question([], [])
    assert first is not None
    _attempt(store, session_id, first.id, False)

    session_attempts = store.attempts(session_id)
    second = engine.next_question(session_attempts, store.attempts())

    assert second is not None
    assert second.id != first.id
    assert second.concept_priority != 0 or first.concept_priority != 0


def test_review_store_schema_is_domain_neutral(tmp_path):
    path = tmp_path / "review.sqlite3"
    SQLiteReviewStore(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(attempts)")}
    assert "item_id" in columns
    assert "concept_at_attempt" in columns
    assert not any("usmle" in column or "medical" in column for column in columns)
