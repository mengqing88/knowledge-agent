import pytest
from app.agents import workflow as workflow_module
from app.core.errors import DomainError, InterruptedRun, ProviderError
from app.core.models import Chunk, IndexRevision, QuestionRun, Snapshot
from app.core.schemas import RunCreate
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from app.knowledge.lexical import scoped_bm25
from app.knowledge.retrieval import hybrid_search, retrieve_rankings


def add_chinese_sources(sessions):
    with sessions.begin() as db:
        db.add(IndexRevision(id="outside-index", parse_revision_id="parse1", config_version="outside",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add_all([
            Chunk(id="chinese", index_revision_id="index1", text="外部成员需要管理员批准后才有访问权限。",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="outside-secret", index_revision_id="outside-index", text="外部成员访问权限 " * 20,
                  source_ranges=[], embedding=[1., 0., 0.]),
        ])


def test_new_runtime_route_improves_chinese_matching_and_fallback_stays_scoped(sessions, corpus):
    add_chinese_sources(sessions)

    class Unavailable:
        def __init__(self):
            self.calls = 0

        def embed(self, *args):
            self.calls += 1
            raise ProviderError("offline", "embedding_unavailable", retryable=True)

    provider = Unavailable()
    with sessions() as db:
        snapshot = db.get(Snapshot, "snap1")
        legacy = hybrid_search(db, snapshot, "外部成员怎样申请访问权限？", provider, "test-embedding")
        assert legacy == []
        result = hybrid_search(db, snapshot, "外部成员怎样申请访问权限？", provider,
                               "test-embedding", lexical_backend="ngram_bm25")
        assert [hit["chunk_id"] for hit in result] == ["chinese"]
        assert result.degradation == {"code": "embedding_unavailable", "route": "keyword_only"}
    assert provider.calls == 2


@pytest.mark.parametrize("limits", [{"lexical_max_chunks": 1}, {"lexical_max_chars": 1}])
def test_capacity_errors_do_not_truncate_the_snapshot_or_request_embeddings(sessions, corpus, limits):
    add_chinese_sources(sessions)

    class NoCalls:
        def embed(self, *args):
            pytest.fail("Capacity must be checked before paid embeddings")

    cache = {}
    with sessions() as db:
        with pytest.raises(ProviderError) as error:
            retrieve_rankings(db, db.get(Snapshot, "snap1"), "访问权限", NoCalls(), "test-embedding",
                              lexical_backend="ngram_bm25", lexical_cache=cache, **limits)
        assert error.value.code == "lexical_capacity"
        assert not cache


def test_per_run_cache_respects_scope_and_still_rechecks_index_readiness(sessions, corpus, fake):
    add_chinese_sources(sessions)
    cache = {}
    with sessions() as db:
        assert scoped_bm25(db, ["index1"], "访问权限", cache=cache) == ["chinese"]
        assert scoped_bm25(db, ["outside-index"], "访问权限", cache=cache) == ["outside-secret"]
        assert len(cache) == 2
        result = hybrid_search(db, db.get(Snapshot, "snap1"), "访问权限", fake, "test-embedding",
                               lexical_backend="ngram_bm25", lexical_cache=cache)
        assert "outside-secret" not in [hit["chunk_id"] for hit in result]
    with sessions.begin() as db:
        db.get(IndexRevision, "index1").status = "failed"
    with sessions() as db:
        with pytest.raises(DomainError, match="不可用索引"):
            hybrid_search(db, db.get(Snapshot, "snap1"), "访问权限", fake, "test-embedding",
                          lexical_backend="ngram_bm25", lexical_cache=cache)


def test_cached_lexical_search_is_interruptible(sessions, corpus):
    cache = {}
    with sessions() as db:
        assert scoped_bm25(db, ["index1"], "F1", cache=cache) == ["chunk1"]

        def cancelled():
            raise InterruptedRun()

        with pytest.raises(InterruptedRun):
            scoped_bm25(db, ["index1"], "F1", cache=cache, guard=cancelled)


def test_workflow_uses_frozen_lexical_route_after_global_setting_changes(
        sessions, settings, corpus, fake, monkeypatch):
    settings.lexical_backend = "ngram_bm25"
    settings.lexical_max_chunks = 17
    settings.lexical_max_chars = 12345
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M F1 dataset X"),
                         "frozen-lexical", "local-user", settings)
        run_id = run.id
    settings.lexical_backend = "legacy"
    settings.lexical_max_chunks = 1
    received = []
    original = workflow_module.ScopedTools

    class ObserveTools(original):
        def search_evidence(self, query, limit=12):
            received.append(self.lexical_options.copy())
            return super().search_evidence(query, limit)

    monkeypatch.setattr(workflow_module, "ScopedTools", ObserveTools)
    process_job(sessions, settings, fake, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed"
        assert run.config["lexical_backend"] == "ngram_bm25"
    assert received and received[0]["lexical_backend"] == "ngram_bm25"
    assert received[0]["lexical_max_chunks"] == 17 and received[0]["lexical_max_chars"] == 12345
    assert len(received[0]["lexical_cache"]) == 1


def test_invalid_lexical_configuration_is_rejected_before_model(sessions, corpus):
    with sessions() as db:
        with pytest.raises(DomainError, match="词法检索配置"):
            hybrid_search(db, db.get(Snapshot, "snap1"), "F1", object(), "test-embedding",
                          lexical_backend="unknown")
