from pathlib import Path

from fastapi.testclient import TestClient

from ebbifold.corpus import load_corpus
from ebbifold.review.api import ReviewApi, build_parser, create_app
from ebbifold.review.engine import ReviewEngine
from ebbifold.review.store import SQLiteReviewStore, default_store_path


def _client(db_path):
    repo_root = Path(__file__).resolve().parents[2]
    return TestClient(create_app(repo_root, db_path))


def test_scripted_http_session_persists_across_restart(tmp_path):
    db = tmp_path / "review.sqlite3"
    client = _client(db)

    created = client.post("/sessions")
    assert created.status_code == 201
    body = created.json()
    session_id = body["session"]["session_id"]
    current = body["current_item"]
    assert current["item_id"]
    assert "is_correct" not in current["options"][0]

    answer = client.post(
        f"/sessions/{session_id}/answers",
        json={"item_id": current["item_id"], "selected_label": current["options"][0]["label"], "time_to_answer_ms": 25},
    )
    assert answer.status_code == 201
    assert answer.json()["correct"] in {True, False}

    restarted = _client(db)
    inspected = restarted.get(f"/sessions/{session_id}")
    assert inspected.status_code == 200
    state = inspected.json()
    assert state["attempt_count"] == 1
    assert state["attempts"][0]["item_id"] == current["item_id"]
    assert state["attempts"][0]["concept_at_attempt"]

    next_item = restarted.get(f"/sessions/{session_id}/current")
    assert next_item.status_code == 200
    assert next_item.json()["item"] is None or next_item.json()["item"]["item_id"] != current["item_id"]


def test_api_rejects_unknown_session_and_stale_item(tmp_path):
    client = _client(tmp_path / "review.sqlite3")
    assert client.get("/sessions/missing").status_code == 404

    session_id = client.post("/sessions").json()["session"]["session_id"]
    current = client.get(f"/sessions/{session_id}/current").json()["item"]
    stale = client.post(f"/sessions/{session_id}/answers", json={"item_id": "not-current", "selected_label": "A"})
    invalid = client.post(f"/sessions/{session_id}/answers", json={"item_id": current["item_id"], "selected_label": "not-a-label"})
    missing_item = client.post(f"/sessions/{session_id}/answers", json={"selected_label": "A"})
    assert stale.status_code == 409
    assert invalid.status_code == 422
    assert missing_item.status_code == 422


def test_api_serves_corpus_figures_and_rejects_traversal(tmp_path):
    client = _client(tmp_path / "review.sqlite3")
    figure = client.get("/figures/usmle/step1/ecg-case-6r3t9x.svg")
    assert figure.status_code == 200
    assert figure.headers["content-type"].startswith("image/svg+xml")
    assert client.get("/figures/usmle/step1/..").status_code in {404, 422}
    assert client.get("/figures/usmle/step1/missing.svg").status_code == 404


def test_session_state_repairs_exhausted_session(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    store = SQLiteReviewStore(tmp_path / "review.sqlite3")
    api = ReviewApi(ReviewEngine(load_corpus(repo_root)), store, repo_root)
    session_id = store.create_session(str(repo_root))
    monkeypatch.setattr(api, "current_question", lambda _: None)

    row, question = api.session_state(session_id)

    assert question is None
    assert row["status"] == "completed"
    assert store.session(session_id)["status"] == "completed"


def test_default_memory_path_is_outside_repository():
    repo_root = Path(__file__).resolve().parents[2].resolve()
    path = default_store_path().resolve()
    assert repo_root not in path.parents
    assert path.suffix in {".db", ".sqlite", ".sqlite3"}


def test_health_and_api_parser_defaults(tmp_path):
    client = _client(tmp_path / "review.sqlite3")
    assert client.get("/health").json() == {"status": "ok"}

    args = build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.db.endswith(("review.sqlite3", "review.sqlite", "review.db"))
