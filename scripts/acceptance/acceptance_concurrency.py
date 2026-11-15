"""Real PostgreSQL checks called only by the disposable acceptance runner."""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from hashlib import sha256
from threading import Barrier, Event
from uuid import uuid4

from app.agents.workflow import Workflow, initial_state
from app.api.main import decide_answer
from app.core.errors import DomainError, InterruptedRun
from app.core.models import (
    AnswerApprovalRecord,
    AnswerRevision,
    Membership,
    PaperVersion,
    QuestionRun,
    User,
    now,
)
from app.core.schemas import AnswerApprovalCreate, RunCreate
from app.jobs.services import answer_content_hash, answer_data, claim_job, create_run
from app.jobs.worker import process_job
from conftest import FakeProvider
from sqlalchemy import func, select, text


def concurrent_claim(sessions, settings, expected):
    barrier = Barrier(8)

    def claim(_):
        barrier.wait(timeout=10)
        return claim_job(sessions, settings)

    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [job for job in pool.map(claim, range(8)) if job]
    assert jobs == [expected], jobs


def check_concurrency(sessions, settings, snapshot_id, corpus_id, reader_id, reviewer_id, version_id):
    def new_run():
        with sessions() as db:
            return create_run(db, RunCreate(snapshot_id=snapshot_id, question="M 的 F1 是多少？"),
                              str(uuid4()), reader_id, settings).id

    run_id = new_run()
    concurrent_claim(sessions, settings, ("run", run_id, 1))
    old_provider = FakeProvider()
    old_workflow = Workflow(sessions, settings, old_provider, run_id, 1)
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        old_state = initial_state(run)
        run.lease_until = now() - timedelta(seconds=1)
    concurrent_claim(sessions, settings, ("run", run_id, 2))
    process_job(sessions, settings, old_provider, ("run", run_id, 1))
    assert old_provider.roles == []
    try:
        old_workflow.publish(old_state)
    except InterruptedRun:
        pass
    else:
        raise AssertionError("superseded attempt published")
    process_job(sessions, settings, FakeProvider(), ("run", run_id, 2))
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "review_pending"
        assert db.scalar(select(func.count()).select_from(AnswerRevision).where(
            AnswerRevision.run_id == run_id)) == 1
    print("PASS: 8 concurrent PG claimants, lease recovery, stale execution/publication fencing", flush=True)

    for mutation in ("membership", "version"):
        run_id = new_run()
        job = claim_job(sessions, settings)
        assert job == ("run", run_id, 1)
        process_job(sessions, settings, FakeProvider(), job)
        with sessions() as db:
            answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
            answer_id = answer.id
            answer_hash = answer_content_hash(answer_data(db, answer, reviewer_id))
        started, connection_pid = Event(), []

        def approve():
            with sessions() as db:
                connection_pid.append(db.scalar(text("SELECT pg_backend_pid()")))
                user = db.get(User, reviewer_id)
                started.set()
                try:
                    decide_answer(answer_id, AnswerApprovalCreate(decision="accept", answer_sha256=answer_hash),
                                  user, db)
                except DomainError as exc:
                    return exc.status
                raise AssertionError("approval accepted invalidated permissions/source")

        with ThreadPoolExecutor(max_workers=1) as pool:
            with sessions.begin() as blocker:
                if mutation == "membership":
                    member = blocker.scalar(select(Membership).where(
                        Membership.corpus_id == corpus_id, Membership.user_id == reader_id))
                    blocker.delete(member)
                else:
                    blocker.get(PaperVersion, version_id).is_active = False
                blocker.flush()
                future = pool.submit(approve)
                assert started.wait(timeout=10)
                # Observe an actual PG row-lock wait; a thread merely starting is insufficient evidence.
                deadline = time.monotonic() + 10
                blocked = False
                while time.monotonic() < deadline:
                    with sessions() as observer:
                        blocked = observer.scalar(text(
                            "SELECT wait_event_type = 'Lock' FROM pg_stat_activity WHERE pid = :pid"),
                            {"pid": connection_pid[0]})
                    if blocked:
                        break
                    assert not future.done(), "approval did not wait for the source/permission transaction"
                    time.sleep(.05)
                assert blocked, "no PostgreSQL lock wait observed"
            assert future.result(timeout=15) == (404 if mutation == "membership" else 409)
        with sessions.begin() as db:
            assert db.get(QuestionRun, run_id).status == "review_pending"
            assert db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run_id)) is None
            if mutation == "membership":
                db.add(Membership(corpus_id=corpus_id, user_id=reader_id, role="reader"))
            else:
                db.get(PaperVersion, version_id).is_active = True
    print("PASS: real PG lock waits serialize approval against revocation and version deactivation", flush=True)


def check_version_switch(sessions, settings):
    from app.core.auth import hash_password
    from app.core.errors import ProviderError
    from app.core.models import Corpus, Paper

    prefix = uuid4().hex
    with sessions.begin() as db:
        db.add(User(id=prefix, username=f"version-{prefix}", password_hash=hash_password("accept-pass")))
        db.flush()
        db.add(Corpus(id=prefix, owner_id=prefix, name="Concurrent version acceptance"))
        db.flush()
        db.add(Paper(id=prefix, corpus_id=prefix, title="Concurrent TXT"))
    settings.storage_dir.mkdir(parents=True, exist_ok=True)

    def version(number):
        content = f"On dataset X, M achieved F1 82 and baseline F1 80.\nRevision {number}.\n".encode()
        digest = sha256(content).hexdigest()
        filename = f"{digest}.txt"
        (settings.storage_dir / filename).write_bytes(content)
        with sessions.begin() as db:
            obj = PaperVersion(paper_id=prefix, filename="version.txt", sha256=digest,
                               byte_size=len(content), storage_key=filename, status="uploaded", is_active=False)
            db.add(obj)
            db.flush()
            return obj.id

    first, second = version(1), version(2)
    jobs = [claim_job(sessions, settings), claim_job(sessions, settings)]
    assert {job[1] for job in jobs} == {first, second}, jobs
    barrier = Barrier(2)

    class ConcurrentEmbedding(FakeProvider):
        def embed(self, texts, expected_dim=0):
            barrier.wait(timeout=15)
            return super().embed(texts, expected_dim)

    def ingest(job):
        process_job(sessions, settings, ConcurrentEmbedding(), job)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(ingest, jobs))
    with sessions() as db:
        versions = db.scalars(select(PaperVersion).where(PaperVersion.paper_id == prefix)).all()
        assert all(obj.status == "ready" for obj in versions), [(obj.status, obj.error) for obj in versions]
        assert [obj.id for obj in versions if obj.is_active] == [second]

    class FailedEmbedding(FakeProvider):
        def embed(self, texts, expected_dim=0):
            raise ProviderError("Acceptance-only embedding failure")

    failed = version(3)
    job = claim_job(sessions, settings)
    assert job == ("ingest", failed, 1)
    process_job(sessions, settings, FailedEmbedding(), job)
    with sessions() as db:
        assert db.get(PaperVersion, failed).status == "failed"
        assert db.get(PaperVersion, second).status == "ready" and db.get(PaperVersion, second).is_active
        assert db.scalar(select(func.count()).select_from(PaperVersion).where(
            PaperVersion.paper_id == prefix, PaperVersion.is_active.is_(True))) == 1
    print("PASS: simultaneous PG/Milvus indexing keeps newest version active; failed update preserves it", flush=True)
