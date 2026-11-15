from datetime import timedelta

import pytest
from app.core.errors import DomainError
from app.core.models import QuestionRun, RunEvent, now
from app.core.schemas import RunCreate
from app.jobs.services import claim_job, create_run, heartbeat
from sqlalchemy import select


def enqueue(sessions, settings, key):
    with sessions() as db:
        return create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"), key,
                          "local-user", settings).id


def test_user_and_global_queue_limits_keep_idempotent_replay_available(sessions, settings, corpus):
    settings.max_pending_jobs = 2
    settings.max_pending_runs_per_user = 1
    first = enqueue(sessions, settings, "capacity-key-1")

    assert enqueue(sessions, settings, "capacity-key-1") == first
    with pytest.raises(DomainError) as user_limit:
        enqueue(sessions, settings, "capacity-key-2")
    assert user_limit.value.status == 429
    assert user_limit.value.code == "user_queue_full"

    settings.max_pending_runs_per_user = 10
    enqueue(sessions, settings, "capacity-key-2")
    with pytest.raises(DomainError) as global_limit:
        enqueue(sessions, settings, "capacity-key-3")
    assert global_limit.value.status == 429
    assert global_limit.value.code == "queue_full"


def test_job_claims_respect_running_capacity(sessions, settings, corpus):
    settings.max_running_jobs = 1
    enqueue(sessions, settings, "running-key-1")
    enqueue(sessions, settings, "running-key-2")

    assert claim_job(sessions, settings)[0] == "run"
    assert claim_job(sessions, settings) is None


def test_expired_job_attempts_stop_at_configured_recovery_limit(sessions, settings, corpus):
    settings.max_job_attempts = 1
    run_id = enqueue(sessions, settings, "attempt-key-1")
    assert claim_job(sessions, settings)[2] == 1
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        run.lease_until = now() - timedelta(seconds=1)

    assert claim_job(sessions, settings) is None
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "failed"
        assert run.error == "租约恢复次数已达上限，请检查后台进程后重新发起"
        event = db.scalar(select(RunEvent).where(RunEvent.run_id == run_id,
                                               RunEvent.kind == "failed"))
        assert event.payload["code"] == "recovery_limit"


def test_heartbeat_cannot_resurrect_a_cancelled_lease(sessions, settings, corpus):
    run_id = enqueue(sessions, settings, "heartbeat-key-1")
    job = claim_job(sessions, settings)
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        run.cancel_requested, run.status, run.lease_until = True, "cancelled", None

    heartbeat(sessions, settings, job)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).lease_until is None
