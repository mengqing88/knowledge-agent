from datetime import timedelta

import pytest
from app import main as main_module
from app.db import get_db
from app.errors import BudgetExhausted, ProviderError
from app.governance import digest, usage_summary
from app.main import app
from app.models import AnswerRevision, ModelCall, QuestionRun, ReviewRecord, RunEvent, now
from app.providers import ModelProvider
from app.schemas import Plan, RunCreate
from app.services import claim_job, create_run
from app.worker import process_job
from app.workflow import Workflow, initial_state
from fastapi.testclient import TestClient
from sqlalchemy import func, select


def queue(sessions, settings, key="part4-first"):
    with sessions() as db:
        return create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                          key, "local-user", settings).id


def test_role_output_limit_is_frozen_and_reserved_per_call(sessions, settings, corpus):
    settings.max_output_tokens = 4000
    settings.role_output_token_limits = {"Analyst": 8000}
    run_id = queue(sessions, settings, "role-output-limit")
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, object(), run_id, job[2])
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.config["role_output_token_limits"] == {"Analyst": 8000}
        state = initial_state(run)
    _, output_limit = workflow.begin_call(state, "Analyst", {"question": run.question}, 100)
    with sessions() as db:
        call = db.scalar(select(ModelCall).where(ModelCall.run_id == run_id))
        assert output_limit == call.request["output_limit"] == 8000
        assert call.reserved_tokens == 8100


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
            client.post("/api/auth/login", json={"username": "admin", "password": "test-password"})
            yield client
    finally:
        app.dependency_overrides.clear()


def test_feedback_preserves_original_category_and_idempotency_without_model_calls(client, sessions, settings, fake):
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    answer = client.get(f"/api/question-runs/{run_id}").json()["answer"]
    roles = list(fake.roles)
    url = f"/api/answers/{answer['id']}/reviews"
    body = {"text": "  不满意，请补充比较条件。  ", "category": "unclear"}
    headers = {"Idempotency-Key": "feedback-key-1"}
    first = client.post(url, json=body, headers=headers)
    assert first.status_code == 201
    assert first.json()["status"] == "needs_clarification" and first.json()["clarification"]
    assert first.json()["text"] == body["text"]
    assert first.json()["answer_sha256"] == digest(answer)
    assert client.post(url, json=body, headers=headers).json()["id"] == first.json()["id"]
    assert client.post(url, json={**body, "category": "factual_error"}, headers=headers).status_code == 409
    assert client.get(url).json()[0]["text"] == body["text"]
    assert client.post(url, json={"text": "    "}).status_code == 422
    assert fake.roles == roles
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(ReviewRecord)) == 1
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 1
    client.post("/api/corpora/corpus1/members", json={
        "username": "part4-reader", "password": "test-password", "role": "reader"})
    client.post("/api/auth/login", json={"username": "part4-reader", "password": "test-password"})
    assert client.get(url).status_code == 404
    assert client.post(url, json=body).status_code == 404
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(ReviewRecord)) == 1


def test_recording_reads_only_stored_data_and_marks_incomplete_calls(client, sessions, settings, fake):
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions.begin() as db:
        db.add(ModelCall(run_id=run_id, attempt=1, role="Planner", model="test-model", request={}, response={},
                         status="started", elapsed_seconds=0, reserved_tokens=100))
    before = list(fake.roles)
    response = client.get(f"/api/question-runs/{run_id}/recording")
    assert response.status_code == 200
    data = response.json()
    assert data["external_calls"] is False and len(data["incomplete_call_ids"]) == 1
    assert data["events"][-1]["kind"] == "completed"
    assert data["trace"]["accounting"]["usage_status"] == "unknown"
    assert any(c["input_tokens"] is None for c in data["trace"]["model_calls"])
    assert fake.roles == before


def test_reexecution_lineage_is_frozen_and_idempotency_cannot_repurpose_original(client, sessions, settings, fake):
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    url = f"/api/question-runs/{run_id}/replay"
    assert client.post(url, headers={"Idempotency-Key": "part4-first"}).status_code == 409
    headers = {"Idempotency-Key": "part4-reexecution"}
    first = client.post(url, headers=headers)
    assert first.status_code == 202
    assert first.json()["config"]["replay_of"] == run_id
    assert client.post(url, headers=headers).json()["id"] == first.json()["id"]
    config = first.json()["config"]
    assert digest({k: v for k, v in config.items() if k != "manifest_sha256"}) == config["manifest_sha256"]


def test_changed_input_is_rejected_before_model(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    with sessions.begin() as db:
        db.get(QuestionRun, run_id).question = "不同的问题"
    with pytest.raises(ProviderError, match="问题内容"):
        Workflow(sessions, settings, fake, run_id, job[2])
    assert not fake.roles


@pytest.mark.parametrize("change", ["model", "chat_base_url"])
def test_changed_provider_identity_fails_before_call(sessions, settings, corpus, fake, change):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    setattr(settings, change, "changed")
    with pytest.raises(ProviderError):
        Workflow(sessions, settings, fake, run_id, job[2])
    assert fake.roles == []


def test_changed_budget_uses_frozen_values_and_manifest_tampering_is_rejected(sessions, settings, corpus, fake):
    queue(sessions, settings)
    settings.max_model_calls = 1
    job = claim_job(sessions, settings)
    process_job(sessions, settings, fake, job)
    assert len(fake.roles) == 5
    second = queue(sessions, settings, "part4-second")
    job = claim_job(sessions, settings)
    with sessions.begin() as db:
        run = db.get(QuestionRun, second)
        run.config = {**run.config, "max_model_calls": 20}
    with pytest.raises(ProviderError, match="清单校验失败"):
        Workflow(sessions, settings, fake, second, job[2])


def test_request_budget_blocks_before_provider(sessions, settings, corpus, fake):
    settings.max_request_input_tokens = 1000
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    assert fake.roles == []
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed" and run.state["budget_category"] == "request_input_tokens"
        assert db.scalar(select(func.count()).select_from(ModelCall)) == 0
        answer = db.scalar(select(AnswerRevision))
        assert answer.evidence_status == "insufficient" and "单次输入" in answer.gaps[0]


def test_unknown_reservation_survives_new_attempt_and_prevents_reset(sessions, settings, corpus, fake):
    settings.max_model_calls = 1
    run_id = queue(sessions, settings)
    first = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, first[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    workflow.begin_call(state, "Planner", {}, 500)
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        run.lease_until = now()-timedelta(seconds=1)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    assert fake.roles == []
    with sessions() as db:
        summary = usage_summary(db, run_id, db.get(QuestionRun, run_id).config)
        assert summary["unknown_usage_calls"] == 1 and summary["reserved_or_actual_tokens"] == 6500
        assert db.get(QuestionRun, run_id).state["budget_category"] == "model_calls"


def test_retry_reserves_budget_and_cannot_hide_unknown_usage(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    fake.before_retry = lambda: None
    original = fake.structured

    def structured(*args):
        fake.before_retry()
        return original(*args)
    fake.structured = structured
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    workflow.call(state, "Planner", "", {"question": "M 的 F1 是多少？"}, Plan)
    with sessions() as db:
        summary = usage_summary(db, run_id, workflow.limits)
        assert summary["model_calls"] == 2 and summary["unknown_usage_calls"] == 1
        assert summary["estimated_cost"] is None
        assert {c.status for c in db.scalars(select(ModelCall))} == {"usage_unknown", "completed"}


def test_retry_blocked_at_same_call_limit(sessions, settings, corpus, fake):
    settings.max_model_calls = 1
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    fake.before_retry = lambda: None
    called = []

    def structured(*_):
        fake.before_retry()
        called.append("second_http_request")
        raise AssertionError("must not reach another HTTP request")
    fake.structured = structured
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    with pytest.raises(BudgetExhausted, match="model_calls"):
        workflow.call(state, "Planner", "", {"question": "M F1"}, Plan)
    assert not called


def test_smaller_retry_output_cannot_shrink_previous_unknown_reservation(sessions, settings, corpus, fake):
    from pydantic import BaseModel

    class TinyResponse(BaseModel):
        value: int

    settings.max_run_tokens = 10000
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    fake.before_retry = lambda: None
    fake.request_output_limit = settings.max_output_tokens
    first_reservation = []

    def structured(*_):
        with sessions() as db:
            first_reservation.append(db.scalar(select(ModelCall.reserved_tokens)))
        assert fake.before_retry() < settings.max_output_tokens
        return TinyResponse(value=7), {"input_tokens": 10, "output_tokens": 10, "usage_known": True,
                                      "provider_request_id": "retry-response"}

    fake.structured = structured
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    workflow.call(state, "Fixture", "", {}, TinyResponse)
    with sessions() as db:
        summary = usage_summary(db, run_id, workflow.limits)
        assert summary["reserved_or_actual_tokens"] == first_reservation[0] + 20
        calls = list(db.scalars(select(ModelCall).order_by(ModelCall.created_at)))
        assert calls[0].usage_known is False and calls[0].reserved_tokens == first_reservation[0]
        assert calls[1].provider_request_id == "retry-response"


def test_audit_commit_failure_prevents_provider_call(sessions, settings, corpus, fake, monkeypatch):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    monkeypatch.setattr("app.workflow.emit", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("disk")))
    with pytest.raises(RuntimeError, match="disk"):
        workflow.call(state, "Planner", "", {"question": "M F1"}, Plan)
    assert not fake.roles
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(ModelCall)) == 0


def test_cost_cap_requires_price_and_unknown_cost_is_not_zero(sessions, settings, corpus, fake):
    settings.max_run_cost = .01
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "failed"
        assert usage_summary(db, run_id, db.get(QuestionRun, run_id).config)["estimated_cost"] is None
    assert fake.roles == []


def test_cost_cap_rejects_expensive_input(sessions, settings, corpus, fake):
    settings.max_run_cost = .0001
    settings.chat_input_price_per_million = settings.chat_output_price_per_million = 10.
    settings.price_currency, settings.price_source = "TEST", "synthetic-test-price"
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        assert db.get(QuestionRun, run_id).state["budget_category"] == "cost"
    assert fake.roles == []


@pytest.mark.parametrize("usage,known", [({}, False), ({"input_tokens": 10}, False),
                                       ({"input_tokens": 10, "output_tokens": 0}, True)])
def test_provider_missing_usage_stays_unknown(settings, monkeypatch, usage, known):
    provider = ModelProvider(settings)
    monkeypatch.setattr(provider, "_post", lambda *_: {"status": "completed", "usage": usage,
        "output": [{"type": "message", "content": [{"type": "output_text", "text":
            '{"normalized_question":"M F1","needs_clarification":true,"clarification":"方法？"}'}]}]})
    _, result = provider.structured("Planner", "", {}, Plan)
    assert result["usage_known"] is known


def test_embedding_preflight_is_before_http(settings, monkeypatch):
    settings.max_embedding_batch_tokens = 1000
    provider = ModelProvider(settings)
    monkeypatch.setattr(provider, "_post", lambda *_: pytest.fail("must not send HTTP"))
    with pytest.raises(ProviderError, match="上限"):
        provider.embed(["a"*2000])


def test_recovery_keeps_original_wall_clock_deadline(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    with sessions.begin() as db:
        event = db.scalar(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "started"))
        event.created_at = now()-timedelta(seconds=100)
    process_job(sessions, settings, fake, job)
    assert not fake.roles
    with sessions() as db:
        assert db.get(QuestionRun, run_id).state["budget_category"] == "time"
