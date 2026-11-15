import copy

import pytest
from app.agents.prompts import EVIDENCE_GATE
from app.agents.quality_gate import validate_readiness
from app.agents.workflow import Workflow, initial_state
from app.core.errors import InterruptedRun, ProviderError
from app.core.models import AnswerClaim, AnswerRevision, QuestionRun
from app.core.schemas import EvidenceReadiness, RunCreate, Subquestion
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from conftest import FakeProvider
from pydantic import ValidationError
from sqlalchemy import select


def queue(sessions, settings):
    with sessions() as db:
        return create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                          "evidence-gate-test", "local-user", settings).id


def review(status, qid="q1", ids=None, queries=None):
    return {"subquestion_id": qid, "status": status, "evidence_ids": ids or [],
            "reason": "原文缺少该指标", "followup_queries": queries or []}


@pytest.mark.parametrize("reviews", [
    [], [review("ready", ids=["unknown"])], [review("ready")],
    [review("ready", qid="invented", ids=["e1"])],
    [review("ready", ids=["e1"]), review("missing")],
])
def test_gate_rejects_missing_duplicate_unknown_or_unbound_reviews(reviews):
    with pytest.raises(ProviderError):
        validate_readiness({"subquestions": [{"id": "q1", "target": "F1"}]},
                           [{"id": "e1"}], {"reviews": reviews}, [])


def test_gate_deduplicates_queries_and_preserves_partial_and_uncertain_gaps():
    result = validate_readiness({"subquestions": [{"id": "q1", "target": "F1"}, {"id": "q2", "target": "延迟"}]},
        [{"id": "e1"}], {"reviews": [review("partial", ids=["e1"], queries=["OLD", "New"]),
                                     review("uncertain", qid="q2", queries=["new", "Latency"])]}, [" OLD "])
    assert result["pending_queries"] == [("q1", "New"), ("q2", "Latency")]
    assert result["blocked_ids"] == ["q2"] and len(result["gaps"]) == 2


def test_missing_gate_skips_generation_and_keeps_refusal_reason(sessions, settings, corpus):
    class MissingProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            if role == "EvidenceGate":
                self.roles.append(role)
                return schema.model_validate({"reviews": [review("missing")]}), {"input_tokens": 1, "output_tokens": 1}
            return super().structured(role, prompt, data, schema)
    run_id = queue(sessions, settings)
    provider = MissingProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert run.status == "completed", run.error
        assert answer.evidence_status == "insufficient" and answer.stop_reason == "evidence_gate_insufficient"
        assert any("生成前检查" in gap for gap in answer.gaps)
    assert provider.roles == ["Planner", "Reranker", "EvidenceGate"]


def test_gate_retry_is_bounded_and_budgeted_before_any_analyst(sessions, settings, corpus):
    settings.max_model_calls = 10
    class RetryProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            if role == "EvidenceGate":
                self.roles.append(role)
                return schema.model_validate({"reviews": [review("missing", queries=["new targeted query"])]}), {
                    "input_tokens": 1, "output_tokens": 1}
            return super().structured(role, prompt, data, schema)
    run_id = queue(sessions, settings)
    provider = RetryProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert run.state["round"] == 2
        assert run.state["stop_reason"] == "evidence_gate_insufficient"
    assert provider.roles == ["Planner", "Reranker", "EvidenceGate", "Reranker", "EvidenceGate"]


def test_shared_candidate_retains_all_retrieving_subquestions(sessions, settings, corpus):
    class SharedCandidateProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.gate_calls = 0
            self.rerank_calls = []

        def structured(self, role, prompt, data, schema):
            if role == "Reranker":
                self.rerank_calls.append(data["candidates"])
            if role == "Planner":
                result, usage = super().structured(role, prompt, data, schema)
                result.subquestions.append(Subquestion(id="q2", target="M 的 F1 分数",
                                                      queries=["F1 result for M on dataset X"]))
                return result, usage
            if role == "EvidenceGate":
                self.gate_calls += 1
                source_id = data["sources"][0]["id"]
                self.gate_sources = data["sources"]
                if self.gate_calls == 1:
                    return schema.model_validate({"reviews": [
                        review("missing", qid="q1", queries=["retry M F1 dataset X"]),
                        review("ready", qid="q2", ids=[source_id]),
                    ]}), {"input_tokens": 1, "output_tokens": 1}
                return schema.model_validate({"reviews": [
                    review("ready", qid="q1", ids=[source_id]),
                    review("ready", qid="q2", ids=[source_id]),
                ]}), {"input_tokens": 1, "output_tokens": 1}
            result, usage = super().structured(role, prompt, data, schema)
            return result, usage

    queue(sessions, settings)
    provider = SharedCandidateProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))

    assert provider.gate_calls == 2 and provider.gate_sources
    assert provider.rerank_calls[0][0]["retrieved_for_subquestion_ids"] == ["q1", "q2"]
    assert provider.rerank_calls[1][0]["retrieved_for_subquestion_ids"] == ["q1"]
    assert provider.gate_sources[0]["retrieved_for_subquestion_ids"] == ["q1", "q2"]


def test_gate_disabled_is_frozen_and_keeps_prior_four_call_pipeline(sessions, settings, corpus, fake):
    settings.evidence_gate_enabled = False
    run_id = queue(sessions, settings)
    settings.evidence_gate_enabled = True
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "completed"
    assert fake.roles == ["Planner", "Reranker", "Analyst", "Verifier"]


def test_gate_cannot_run_after_cancellation(sessions, settings, corpus, fake):
    run_id = queue(sessions, settings)
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, fake, run_id, job[2])
    with sessions.begin() as db:
        run = db.get(QuestionRun, run_id)
        state = copy.deepcopy(initial_state(run))
        run.status, run.cancel_requested = "cancelled", True
    with pytest.raises(InterruptedRun):
        workflow.evidence_gate(state)
    assert fake.roles == []


def test_budget_exhaustion_after_generation_cannot_publish_unverified_claim(sessions, settings, corpus, fake):
    settings.max_model_calls = 4
    run_id = queue(sessions, settings)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert run.status == "completed", run.error
        assert answer.evidence_status == "insufficient" and answer.stop_reason == "budget_exhausted"
        assert db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id)) is None
    assert fake.roles == ["Planner", "Reranker", "EvidenceGate", "Analyst"]


def test_no_evidence_skips_paid_gate_and_generation(sessions, settings, corpus):
    settings.max_retrieval_rounds = 1
    class EmptyProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            if role == "Reranker":
                self.roles.append(role)
                return schema.model_validate({"items": []}), {"input_tokens": 1, "output_tokens": 1}
            return super().structured(role, prompt, data, schema)
    run_id = queue(sessions, settings)
    provider = EmptyProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert run.status == "completed", run.error
        assert answer.evidence_status == "insufficient" and answer.stop_reason == "evidence_gate_insufficient"
    assert provider.roles == ["Planner", "Reranker"]


def test_blocked_subquestion_claim_is_removed_even_when_models_approve(sessions, settings, corpus):
    settings.max_revision_rounds = 0
    class IgnoringProvider(FakeProvider):
        def structured(self, role, prompt, data, schema):
            if role == "EvidenceGate":
                self.roles.append(role)
                return schema.model_validate({"reviews": [
                    review("ready", ids=[data["sources"][0]["id"]]), review("missing", qid="q2")]}), {
                        "input_tokens": 1, "output_tokens": 1}
            result, usage = super().structured(role, prompt, data, schema)
            if role == "Planner":
                result.subquestions.append(Subquestion(id="q2", target="延迟", queries=["M latency"]))
            if role == "Analyst":
                assert data["evidence_readiness"]["blocked_ids"] == ["q2"]
                result.claims[0].subquestion_ids = ["q2"]
            return result, usage
    run_id = queue(sessions, settings)
    provider = IgnoringProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
        assert run.status == "completed", run.error
        assert answer.evidence_status == "insufficient"
        assert any("生成前证据检查" in issue["message"] for issue in answer.issues)
        assert db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id)) is None
    assert provider.roles.count("Verifier") == 1


def test_gate_reason_cannot_be_whitespace():
    with pytest.raises(ValidationError):
        EvidenceReadiness.model_validate({**review("missing"), "reason": "   "})


def test_gate_prompt_treats_retrieval_provenance_as_a_hint():
    assert "只表示哪些子问题的检索查询返回了该来源" in EVIDENCE_GATE
    assert "不代表相关性或支持性" in EVIDENCE_GATE
