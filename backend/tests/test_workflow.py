from datetime import timedelta

import pytest
from app.agents.workflow import Workflow, initial_state
from app.core.errors import DomainError, InterruptedRun
from app.core.models import (
    AnswerClaim,
    AnswerRevision,
    Chunk,
    IndexRevision,
    Paper,
    PaperVersion,
    ParseRevision,
    QuestionRun,
    RetrievalRecord,
    RunEvent,
    Snapshot,
    SourceBlock,
    now,
)
from app.core.schemas import RunCreate
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from conftest import FakeProvider
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError


def queue(sessions, settings, mode="full_agent", key="test-key-1"):
    with sessions() as db:
        return create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？", mode=mode),
                          key, "local-user", settings).id


def test_full_agent_saves_claim_and_exact_evidence(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    process_job(sessions, settings, fake, job)
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "supported"
        assert answer.usage["model_calls"] == 5
        claim = db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id))
        assert claim.semantic_verified
        events = list(db.scalars(select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.seq)))
        assert [e.seq for e in events] == list(range(1, len(events)+1))
        assert events[-1].kind == "completed"
    assert fake.roles == ["Planner", "Reranker", "EvidenceGate", "Analyst", "Verifier"]


@pytest.mark.parametrize("source_case", ["other_parse", "manifest_parse", "out_of_bounds",
                                       "boolean_offset", "valid_documents"])
def test_chunk_source_validation_before_reranking(sessions, settings, corpus, fake, source_case):
    with sessions.begin() as db:
        chunk = db.get(Chunk, "chunk1")
        if source_case in {"other_parse", "manifest_parse", "valid_documents"}:
            db.add(Paper(id="paper2", corpus_id="corpus1", title="Fictional paper B"))
            db.flush()
            db.add(PaperVersion(id="v2", paper_id="paper2", sha256="b" * 64, storage_key="b.pdf",
                                filename="b.pdf", byte_size=1, status="ready"))
            db.flush()
            db.add(ParseRevision(id="parse2", paper_version_id="v2", parser="test",
                                 config_version="test", status="ready"))
            db.flush()
            original_index = db.get(IndexRevision, "index1")
            db.add(IndexRevision(id="index2", parse_revision_id="parse2",
                                 config_version=original_index.config_version, embedding_model="test-embedding",
                                 embedding_dimensions=3, status="ready"))
            db.add(SourceBlock(id="block2", parse_revision_id="parse2", page_index=0, block_index=0,
                               bbox=[], text="M achieved F1 82 on dataset X."))
            snapshot = db.get(Snapshot, "snap1")
            snapshot.manifest = [*snapshot.manifest, {**snapshot.manifest[0],
                "paper_id": "paper2", "title": "Fictional paper B", "paper_version_id": "v2",
                "sha256": "b" * 64, "parse_revision_id": "parse2", "index_revision_id": "index2"}]
            if source_case == "other_parse":
                chunk.source_ranges = [{"block_id": "block2", "char_start": 0, "char_end": 28}]
            elif source_case == "manifest_parse":
                snapshot.manifest = [{**snapshot.manifest[0], "parse_revision_id": "parse2"},
                                     snapshot.manifest[1]]
            else:
                db.add(Chunk(id="chunk2", index_revision_id="index2", text="M achieved F1 82 on dataset X.",
                             source_ranges=[{"block_id": "block2", "char_start": 0, "char_end": 28}],
                             embedding=[1., 0., 0.]))
        else:
            chunk.source_ranges = [{"block_id": "block1", "char_start": False if source_case == "boolean_offset"
                                    else 0, "char_end": 99999 if source_case == "out_of_bounds" else 28}]
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        if source_case == "valid_documents":
            assert run.status == "completed", run.error
            assert {(s["block_id"], s["parse_revision_id"], s["paper_version_id"]) for s in run.state["sources"]} == {
                ("block1", "parse1", "v1"), ("block2", "parse2", "v2")}
            return
        assert run.status == "failed"
        expected_error = {"other_parse": "不属于其解析版本", "manifest_parse": "与快照的解析版本不一致",
                          "out_of_bounds": "字符范围", "boolean_offset": "字符范围"}[source_case]
        assert expected_error in run.error
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0
    assert fake.roles == ["Planner"]


@pytest.mark.parametrize("mode,roles", [("baseline_rag", ["Analyst"]),
                                         ("without_verifier", ["Planner", "Reranker", "Analyst"])])
def test_comparison_modes_keep_location_checks(sessions, settings, corpus, mode, roles):
    run_id = queue(sessions, settings, mode)
    fake = FakeProvider(invalid=True)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "insufficient"
        assert db.scalar(select(func.count()).select_from(AnswerClaim)) == 0
    assert fake.roles == roles


def test_false_but_existing_citation_is_removed_and_revision_bounded(sessions, settings, corpus):
    run_id = queue(sessions, settings)
    fake = FakeProvider(reject=True)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert run.state["revisions"] == 1
        assert len(fake.roles) == 7
        assert db.scalar(select(func.count()).select_from(AnswerClaim)) == 0


def test_revision_receives_the_rejected_draft_and_feedback(sessions, settings, corpus):
    import copy

    class CapturingProvider(FakeProvider):
        def __init__(self):
            super().__init__(reject=True)
            self.analyst_requests = []

        def structured(self, role, prompt, data, schema):
            if role == "Analyst":
                self.analyst_requests.append(copy.deepcopy(data))
            return super().structured(role, prompt, data, schema)

    run_id = queue(sessions, settings)
    provider = CapturingProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
    assert len(provider.analyst_requests) == 2
    first, revision = provider.analyst_requests
    assert not first.get("previous_draft")
    assert revision["previous_draft"]["claims"][0]["text"] == "数据集 X 上 M 的 F1 为 82。"
    assert revision["verification_feedback"][0]["claim_id"] == revision["previous_draft"]["claims"][0]["id"]
    assert revision["verification_feedback"][0]["message"] == "原文不支持"


def test_followup_retrieval_stops_without_new_evidence(sessions, settings, corpus):
    settings.max_model_calls = 10  # The gate adds one call per round; this case must permit a second round.
    run_id = queue(sessions, settings)
    fake = FakeProvider(gaps=True)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert run.state["round"] == 2
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "partial"
        assert answer.stop_reason == "no_new_evidence"


def test_clarification_does_not_retrieve(sessions, settings, corpus):
    run_id = queue(sessions, settings)
    fake = FakeProvider(clarify=True)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.evidence_status == "needs_clarification"
        assert answer.usage["retrieval_rounds"] == 0
    assert fake.roles == ["Planner"]


def test_timeout_is_technical_failure_not_refusal(sessions, settings, corpus):
    run_id = queue(sessions, settings)
    process_job(sessions, settings, FakeProvider(fail=True), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "failed"
        assert run.error == "模型服务超时"
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0


def test_reranker_failure_is_recorded_without_unverified_fallback(sessions, settings, corpus):
    from app.core.errors import ProviderError

    class FailingReranker(FakeProvider):
        def structured(self, role, *args):
            if role == "Reranker":
                raise ProviderError("重排模型不可用")
            return super().structured(role, *args)

    run_id = queue(sessions, settings)
    process_job(sessions, settings, FailingReranker(), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "failed"
        assert run.error == "重排模型不可用"
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0
        assert db.scalar(select(func.count()).select_from(RetrievalRecord)) > 0


def test_idempotency_and_snapshot_immutable(sessions, settings, corpus):
    first = queue(sessions, settings)
    assert queue(sessions, settings) == first
    with sessions() as db:
        with pytest.raises(DomainError, match="幂等键"):
            create_run(db, RunCreate(snapshot_id="snap1", question="另一问题"), "test-key-1", "local-user", settings)
        snap = db.get(Snapshot, "snap1")
        assert [m["paper_version_id"] for m in snap.manifest] == ["v1"]


def test_unrelated_integrity_error_does_not_retry_creation(sessions, settings, corpus, monkeypatch):
    import app.jobs.services as services

    calls = []

    def fail_emit(*args, **kwargs):
        calls.append(1)
        raise IntegrityError("insert event", {}, ValueError("unrelated constraint"))

    monkeypatch.setattr(services, "emit", fail_emit)
    with sessions() as db:
        with pytest.raises(IntegrityError, match="unrelated constraint"):
            create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                       "failed-create", "local-user", settings)
        assert db.scalar(select(QuestionRun).where(QuestionRun.idempotency_key == "failed-create")) is None
    assert len(calls) == 1


def test_cancel_and_old_attempt_cannot_publish(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    first_job = claim_job(sessions, settings)
    with sessions() as db:
        initial = initial_state(db.get(QuestionRun, run_id))
    old = Workflow(sessions, settings, fake, run_id, first_job[2])
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        run.lease_until = now() - timedelta(seconds=5)
    second_job = claim_job(sessions, settings)
    assert second_job[2] == 2
    with pytest.raises(InterruptedRun):
        old.publish(initial)
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        run.cancel_requested, run.status = True, "cancelled"
    process_job(sessions, settings, fake, second_job)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "cancelled"
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0


@pytest.mark.parametrize("change", ["deactivate", "delete"])
def test_publish_rechecks_sources_after_start(sessions, settings, corpus, fake, monkeypatch, change):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    original_start = workflow.start_node

    def change_after_start(name, state):
        original_start(name, state)
        with sessions.begin() as db:
            if change == "deactivate":
                db.get(PaperVersion, "v1").is_active = False
            else:
                db.get(Paper, "paper1").deleted = True

    monkeypatch.setattr(workflow, "start_node", change_after_start)
    with pytest.raises(DomainError, match="停用或删除"):
        workflow.publish(state)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "running"
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0


def test_server_budget_not_model_controlled(sessions, settings, corpus):
    settings.max_model_calls = 1
    run_id = queue(sessions, settings)
    fake = FakeProvider()
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert answer.stop_reason == "budget_exhausted"
        assert answer.evidence_status == "insufficient"
    assert fake.roles == ["Planner"]


def test_late_worker_failure_preserves_recovery_limit(sessions, settings, corpus, fake, monkeypatch):
    settings.max_job_attempts = 1
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)

    def expire_then_fail(self, state, checkpointer):
        with sessions.begin() as db:
            db.get(QuestionRun, run_id).lease_until = now() - timedelta(seconds=5)
        assert claim_job(sessions, settings) is None
        raise RuntimeError("late failure")

    monkeypatch.setattr(Workflow, "execute", expire_then_fail)
    process_job(sessions, settings, fake, job)
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "failed"
        assert "租约恢复次数已达上限" in run.error
        failures = list(db.scalars(select(RunEvent).where(
            RunEvent.run_id == run_id, RunEvent.kind == "failed").order_by(RunEvent.seq)))
        assert len(failures) == 1
        assert failures[0].payload["code"] == "recovery_limit"


def test_repeated_delivery_does_not_duplicate_answer(sessions, settings, corpus, fake):
    queue(sessions, settings)
    job = claim_job(sessions, settings)
    process_job(sessions, settings, fake, job)
    calls = list(fake.roles)
    process_job(sessions, settings, fake, job)
    assert fake.roles == calls
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 1


def test_cancellation_during_model_call_stops_following_nodes(sessions, settings, corpus):
    run_id = queue(sessions, settings)

    class CancellingProvider(FakeProvider):
        def structured(self, *args, **kwargs):
            result = super().structured(*args, **kwargs)
            with sessions.begin() as db:
                run = db.get(QuestionRun, run_id)
                run.status, run.cancel_requested = "cancelled", True
            return result

    provider = CancellingProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    assert provider.roles == ["Planner"]
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(AnswerRevision)) == 0
