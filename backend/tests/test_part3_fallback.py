import pytest
from app.core.errors import ProviderError
from app.core.models import ModelCall, QuestionRun, RetrievalRecord, RunEvent
from app.core.schemas import RunCreate
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from conftest import FakeProvider
from sqlalchemy import select


def test_transient_embedding_failure_uses_scoped_keyword_search_and_records_degradation(
        sessions, settings, corpus):
    class UnavailableEmbedding(FakeProvider):
        def embed(self, _texts, expected_dim=0):
            raise ProviderError("向量服务暂时不可用", "provider_connect", True)

    with sessions() as db:
        run_id = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                            "keyword-fallback-1", "local-user", settings).id
    provider = UnavailableEmbedding()
    process_job(sessions, settings, provider, claim_job(sessions, settings))

    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        embedding_call = db.scalar(select(ModelCall).where(
            ModelCall.run_id == run_id, ModelCall.role == "Embedding"))
        assert embedding_call.status == "failed"
        assert embedding_call.response["code"] == "provider_connect"
        degradation = db.scalar(select(RunEvent).where(
            RunEvent.run_id == run_id, RunEvent.kind == "retrieval_degraded"))
        assert degradation.payload == {"subquestion_id": "q1", "round": 1,
                                       "code": "provider_connect", "route": "keyword_only"}
        retrieval = db.scalar(select(RetrievalRecord).where(RetrievalRecord.run_id == run_id))
        assert retrieval.rankings[0]["chunk_id"] == "chunk1"
        assert "关键词检索" not in str(run.error or "")


def test_permanent_embedding_error_does_not_silently_degrade(sessions, settings, corpus):
    class InvalidEmbedding(FakeProvider):
        def embed(self, _texts, expected_dim=0):
            raise ProviderError("向量维度不匹配", "embedding_invalid")

    with sessions() as db:
        from app.core.models import Snapshot
        from app.knowledge.retrieval import hybrid_search

        with pytest.raises(ProviderError) as exc:
            hybrid_search(db, db.get(Snapshot, "snap1"), "M F1 dataset X", InvalidEmbedding(),
                          "test-embedding")
    assert exc.value.code == "embedding_invalid"
