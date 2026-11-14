import asyncio
import json
from datetime import timedelta
from types import SimpleNamespace

import app.main as main_module
import pytest
from app.auth import create_session, token_hash
from app.db import get_db
from app.errors import InterruptedRun
from app.main import app
from app.models import AnswerRevision, LoginSession, Membership, QuestionRun, RunEvent, User, now
from app.progress import progress_event
from app.schemas import RunCreate
from app.services import claim_job, create_run, emit, run_data
from app.worker import process_job
from app.workflow import Workflow, initial_state
from conftest import FakeProvider
from fastapi.testclient import TestClient
from sqlalchemy import event, select


@pytest.fixture
def client(sessions, settings, corpus, monkeypatch):
    def override():
        with sessions() as db:
            yield db
    app.dependency_overrides[get_db] = override
    monkeypatch.setattr(main_module, "Session", sessions)
    monkeypatch.setattr(main_module, "settings", settings)
    try:
        with TestClient(app) as client:
            assert client.post("/api/auth/login", json={
                "username": "admin", "password": "test-password"}).status_code == 200
            yield client
    finally:
        app.dependency_overrides.clear()


def queue(sessions, settings):
    with sessions() as db:
        return create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                          "streaming-progress-key", "local-user", settings).id


def test_phase_is_committed_before_model_call_and_draft_stays_hidden(sessions, settings, corpus):
    run_id = queue(sessions, settings)

    class InspectingProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            with sessions() as db:
                events = list(db.scalars(select(RunEvent).where(
                    RunEvent.run_id == run_id).order_by(RunEvent.seq)))
                starts = [e.payload["node"] for e in events if e.kind == "node_started"]
                assert starts[-1] == {"Planner": "planner", "Reranker": "retriever",
                                      "EvidenceGate": "evidence_gate", "Analyst": "analyst", "Verifier": "verifier"}[role]
                public = run_data(db, db.get(QuestionRun, run_id))
                assert public["answer"] is None and public["state"] == {}
                if role == "Reranker":
                    counts = [e.payload for e in events if e.kind == "retrieval_progress"]
                    assert counts[-1]["queries_done"] == counts[-1]["queries_total"] > 0
                    assert counts[-1]["candidate_count"] > 0
            return super().structured(role, prompt, data, schema)

    provider = InspectingProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        events = list(db.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)))
        assert [e.payload["node"] for e in events if e.kind == "node_started"] == [
            "planner", "retriever", "evidence_gate", "analyst", "verifier", "route", "publish"]
        assert events[-1].kind == "completed"


def test_progress_feed_omits_model_text_but_retains_cursor_and_counts(client, sessions, settings):
    run_id = queue(sessions, settings)
    hidden = "THIS_IS_AN_UNVERIFIED_DRAFT"
    with sessions.begin() as db:
        emit(db, run_id, 0, "analyst", {"message": hidden, "claim_count": 2, "gaps": [hidden],
             "reference_issues": [{"message": hidden}], "draft": hidden, "plan": hidden})
    data = client.get(f"/api/question-runs/{run_id}/events?stream=false").json()
    assert hidden not in json.dumps(data)
    assert data[-1]["payload"]["claim_count"] == 2
    assert data[-1]["seq"] == 2
    public = client.get(f"/api/question-runs/{run_id}").json()
    assert public["answer"] is None and public["state"] == {}
    client.post(f"/api/question-runs/{run_id}/cancel")
    stream = client.get(f"/api/question-runs/{run_id}/events")
    assert hidden not in stream.text
    with sessions() as db:
        assert db.scalar(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.seq == 2)).payload["gaps"] == [hidden]


def test_sse_drains_terminal_pages_and_resumes_without_duplicates(client, sessions, settings):
    run_id = queue(sessions, settings)
    with sessions.begin() as db:
        db.add_all([RunEvent(run_id=run_id, seq=seq, attempt=0, kind="retrieval_progress",
                            payload={"queries_done": seq}) for seq in range(2, 407)])
        db.get(QuestionRun, run_id).status = "cancelled"
        db.flush()
        emit(db, run_id, 0, "cancelled", {})
    url = f"/api/question-runs/{run_id}/events"
    response = client.get(url + "?after_seq=5", headers={"Last-Event-ID": "6"})
    ids = [int(line[4:]) for line in response.text.splitlines() if line.startswith("id: ")]
    assert ids == list(range(7, 408))
    assert response.headers["x-accel-buffering"] == "no"
    assert response.text.startswith(": connected\nretry: 2000\n\n")
    assert client.get(url, headers={"Last-Event-ID": "-1"}).status_code == 400


def test_disconnected_stream_stops_before_database_poll(sessions, settings, corpus, monkeypatch):
    run_id = queue(sessions, settings)

    async def disconnected():
        return True

    def unexpected_poll(*args):
        raise AssertionError("a disconnected stream must not poll")

    monkeypatch.setattr(main_module, "run_in_threadpool", unexpected_poll)
    with sessions() as db:
        response = main_module.events(run_id, SimpleNamespace(is_disconnected=disconnected),
                                      SimpleNamespace(id="local-user"), db, after_seq=0, stream=True)

    async def consume():
        return [item async for item in response.body_iterator]

    assert asyncio.run(consume()) == [": connected\nretry: 2000\n\n"]


def test_sse_stops_when_membership_is_revoked_after_connect(client, sessions, settings, monkeypatch):
    with sessions.begin() as db:
        db.add(User(id="stream-reader", username="stream-reader", password_hash="unused"))
        db.flush()
        db.add(Membership(corpus_id="corpus1", user_id="stream-reader", role="reader"))
    with sessions.begin() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                            "stream-revocation", "stream-reader", settings).id
    with sessions() as db:
        token = create_session(db, db.get(User, "stream-reader"), 1)

    async def connected():
        return False

    async def revoke_before_batch(read_batch, cursor):
        with sessions.begin() as db:
            membership = db.scalar(select(Membership).where(
                Membership.corpus_id == "corpus1", Membership.user_id == "stream-reader"))
            db.delete(membership)
        return read_batch(cursor)

    monkeypatch.setattr(main_module, "run_in_threadpool", revoke_before_batch)
    with sessions() as db:
        response = main_module.events(
            run_id, SimpleNamespace(is_disconnected=connected, headers={}, cookies={"kb_session": token}),
            SimpleNamespace(id="stream-reader"), db, after_seq=0, stream=True)

    async def consume():
        return [item async for item in response.body_iterator]

    assert asyncio.run(consume()) == [": connected\nretry: 2000\n\n"]


@pytest.mark.parametrize("transition", ["logout", "expired"])
def test_sse_stops_after_login_session_ends(client, sessions, settings, monkeypatch, transition):
    run_id = queue(sessions, settings)
    token = client.cookies.get("kb_session")
    calls = 0

    async def connected():
        return False

    async def change_session_before_next_batch(read_batch, cursor):
        nonlocal calls
        calls += 1
        if calls == 2:
            with sessions.begin() as db:
                login = db.scalar(select(LoginSession).where(LoginSession.token_hash == token_hash(token)))
                if transition == "logout":
                    db.delete(login)
                else:
                    login.expires_at = now() - timedelta(seconds=1)
                emit(db, run_id, 0, "started", {})
        return read_batch(cursor)

    monkeypatch.setattr(main_module, "run_in_threadpool", change_session_before_next_batch)
    with sessions() as db:
        response = main_module.events(
            run_id, SimpleNamespace(is_disconnected=connected, headers={}, cookies={"kb_session": token}),
            SimpleNamespace(id="local-user"), db, after_seq=0, stream=True)

    async def consume():
        return [item async for item in response.body_iterator]

    messages = asyncio.run(consume())
    assert calls == 2
    assert len(messages) == 2 and '"kind": "queued"' in messages[1]


def test_cancelled_attempt_cannot_emit_a_new_phase_or_publish(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        state = initial_state(run)
        run.status, run.cancel_requested = "cancelled", True
    with pytest.raises(InterruptedRun):
        workflow.timed_node("planner", workflow.planner)(state)
    with pytest.raises(InterruptedRun):
        workflow.publish(state)
    with sessions() as db:
        assert not list(db.scalars(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "node_started")))
        assert db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id)) is None
    assert fake.roles == []


@pytest.mark.parametrize("transition", ["cancel", "retry"])
@pytest.mark.parametrize("raises", [False, True])
def test_finished_event_respects_attempt_transition(sessions, settings, corpus, fake, transition, raises):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))

    finished_check_locks = []
    after_transition = False

    def record_lock(execution):
        statement = execution.statement
        if after_transition and execution.is_select and getattr(statement, "_for_update_arg", None) is not None:
            finished_check_locks.append(statement)

    def node(state):
        nonlocal after_transition
        with sessions.begin() as db:
            run = db.get(QuestionRun, run_id)
            if transition == "cancel":
                run.status, run.cancel_requested = "cancelled", True
            else:
                run.attempt += 1
            emit(db, run_id, run.attempt, transition, {})
        after_transition = True
        if raises:
            raise ValueError("node failure")
        return state

    event.listen(sessions.class_, "do_orm_execute", record_lock)
    try:
        if raises:
            with pytest.raises(ValueError, match="node failure"):
                workflow.timed_node("planner", node)(state)
        else:
            assert workflow.timed_node("planner", node)(state) is state
    finally:
        event.remove(sessions.class_, "do_orm_execute", record_lock)

    assert finished_check_locks
    with sessions() as db:
        events = list(db.scalars(select(RunEvent).where(
            RunEvent.run_id == run_id).order_by(RunEvent.seq)))
        assert events[-1].kind == transition
        assert not any(item.kind == "node_finished" for item in events)


def test_progress_projection_only_accepts_numeric_counts_and_known_stages():
    from datetime import datetime
    event = SimpleNamespace(seq=4, attempt=1, kind="node_started", created_at=datetime(2026, 11, 12),
                            payload={"node": "draft text", "role": "Analyst", "round": "draft text",
                                     "queries_done": 2, "new_evidence": True, "answer": "draft text"})
    assert progress_event(event)["payload"] == {"role": "Analyst", "queries_done": 2}
