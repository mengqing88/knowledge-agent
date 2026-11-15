import app.api.main as main_module
from app.api.main import app
from app.core.db import get_db
from app.core.models import Corpus, QuestionRun
from fastapi.testclient import TestClient
from sqlalchemy import select


def test_api_scope_idempotency_cancel_and_event_resume(sessions, settings, corpus, monkeypatch):
    def db_override():
        with sessions() as db:
            yield db
    app.dependency_overrides[get_db] = db_override
    monkeypatch.setattr(main_module, "Session", sessions)
    monkeypatch.setattr(main_module, "settings", settings)
    try:
        with TestClient(app) as client:
            assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password"}).status_code == 200
            with sessions.begin() as db:
                db.add(Corpus(id="not-yours", owner_id="other-user", name="Private"))
            assert client.get("/api/corpora/not-yours/papers").status_code == 404
            assert client.post("/api/corpora/corpus1/papers", files={"file": ("fake.pdf", b"not a PDF")}).status_code == 400
            body = {"question": "M 的 F1 是多少", "snapshot_id": "snap1"}
            headers = {"Idempotency-Key": "same-key-123"}
            response = client.post("/api/question-runs", json=body, headers=headers)
            assert response.status_code == 202, response.text
            run_id = response.json()["id"]
            assert client.post("/api/question-runs", json=body, headers=headers).json()["id"] == run_id
            conflict = client.post("/api/question-runs", json={**body, "question": "另一个问题"}, headers=headers)
            assert conflict.status_code == 409
            result = client.post(f"/api/question-runs/{run_id}/cancel")
            assert result.json()["status"] == "cancelled"
            events = client.get(f"/api/question-runs/{run_id}/events?stream=false").json()
            assert [e["kind"] for e in events] == ["queued", "cancelled"]
            resumed = client.get(f"/api/question-runs/{run_id}/events?after_seq=1&stream=false").json()
            assert [e["seq"] for e in resumed] == [2]
            stream = client.get(f"/api/question-runs/{run_id}/events", headers={"Last-Event-ID": "1"})
            assert "id: 2" in stream.text
            assert "id: 1" not in stream.text
            assert client.get(f"/api/question-runs/{run_id}/events", headers={"Last-Event-ID": "oops"}).status_code == 400
            with sessions() as db:
                assert len(list(db.scalars(select(QuestionRun)))) == 1
    finally:
        app.dependency_overrides.clear()
