import pytest
from app.errors import DomainError, ProviderError
from app.evidence import get_or_create_span, validate_span
from app.models import Chunk, IndexRevision, Snapshot, SourceBlock
from app.retrieval import collapse_duplicate_hits, hybrid_search, rrf
from app.schemas import RunCreate
from app.services import create_run
from app.tools import ScopedTools
from sqlalchemy import select


def test_rrf_known_order_and_duplicate_not_double_counted():
    result = rrf([["A", "B", "C"], ["B", "C", "A"]])
    assert [r["chunk_id"] for r in result] == ["B", "A", "C"]
    assert result[0]["score"] == pytest.approx(1/62 + 1/61)
    assert rrf([["A", "A"]])[0]["score"] == pytest.approx(1/61)


def test_duplicate_hits_fold_only_exact_text_and_keep_audit_ids(sessions, corpus):
    similar_a = "Under light load, the service returns within 120 ms. " + "This is a long repeated performance note. " * 4
    similar_b = similar_a.replace("light load", "heavy load")
    with sessions.begin() as db:
        db.add_all([
            Chunk(id="duplicate-exact", index_revision_id="index1",
                  text="  M achieved F1 82 on dataset X.  ", source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="condition-light", index_revision_id="index1",
                  text=similar_a, source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="condition-heavy", index_revision_id="index1",
                  text=similar_b, source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="measure-small", index_revision_id="index1",
                  text="The device consumes 1.5 mW in standby.", source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="measure-large", index_revision_id="index1",
                  text="The device consumes 15 MW in standby.", source_ranges=[], embedding=[1., 0., 0.]),
        ])
    with sessions() as db:
        result = collapse_duplicate_hits(db, [
            {"chunk_id": "chunk1", "score": .4},
            {"chunk_id": "duplicate-exact", "score": .3},
            {"chunk_id": "condition-light", "score": .2},
            {"chunk_id": "condition-heavy", "score": .1},
            {"chunk_id": "measure-small", "score": .05},
            {"chunk_id": "measure-large", "score": .04},
        ])
    assert [item["chunk_id"] for item in result] == [
        "chunk1", "condition-light", "condition-heavy", "measure-small", "measure-large"]
    assert result[0]["duplicate_chunk_ids"] == ["duplicate-exact"]
    assert all(item["duplicate_chunk_ids"] == [] for item in result[1:])


def test_duplicate_hits_preserve_distinct_index_provenance(sessions, corpus):
    with sessions.begin() as db:
        db.add(IndexRevision(id="other-index", parse_revision_id="parse1", config_version="other",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add(Chunk(id="same-text-other-index", index_revision_id="other-index",
                     text="M achieved F1 82 on dataset X.", source_ranges=[], embedding=[1., 0., 0.]))
    with sessions() as db:
        result = collapse_duplicate_hits(db, [
            {"chunk_id": "chunk1", "score": .4},
            {"chunk_id": "same-text-other-index", "score": .3},
        ])
    assert [hit["chunk_id"] for hit in result] == ["chunk1", "same-text-other-index"]
    assert all(not hit["duplicate_chunk_ids"] for hit in result)


def test_duplicate_hits_preserve_compatibility_character_distinctions(sessions, corpus):
    with sessions.begin() as db:
        db.add_all([
            Chunk(id="superscript", index_revision_id="index1", text="Area is 10 m².",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="plain-digit", index_revision_id="index1", text="Area is 10 m2.",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="composed", index_revision_id="index1", text="Café result.",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="decomposed", index_revision_id="index1", text="Cafe\u0301 result.",
                  source_ranges=[], embedding=[1., 0., 0.]),
        ])
    with sessions() as db:
        result = collapse_duplicate_hits(db, [{"chunk_id": chunk_id, "score": 1.} for chunk_id in
                                             ["superscript", "plain-digit", "composed", "decomposed"]])
    assert [hit["chunk_id"] for hit in result] == ["superscript", "plain-digit", "composed"]
    assert result[-1]["duplicate_chunk_ids"] == ["decomposed"]


def test_evidence_rejects_wrong_parse_forged_quote_and_range(sessions, corpus):
    with sessions() as db:
        snap = db.get(Snapshot, "snap1")
        block = db.get(SourceBlock, "block1")
        assert validate_span(db, snap, block.id, 0, 2, "On").page_index == 3
        for start, end, quote in [(0, 2, "xx"), (-1, 2, "On"), (0, 99999, "On")]:
            with pytest.raises(DomainError):
                validate_span(db, snap, block.id, start, end, quote)
        snap.manifest = [{"parse_revision_id": "other-parse"}]
        with pytest.raises(DomainError, match="快照"):
            validate_span(db, snap, block.id, 0, 2, "On")


def test_emoji_python_character_offsets(sessions, corpus):
    with sessions() as db:
        block = db.get(SourceBlock, "block1")
        block.text = "甲🧪乙\n"
        db.flush()
        assert validate_span(db, db.get(Snapshot, "snap1"), block.id, 1, 2, "🧪")
        with pytest.raises(DomainError):
            validate_span(db, db.get(Snapshot, "snap1"), block.id, 1, 3, "🧪")


@pytest.mark.parametrize("query,limit", [(None, 12), ("M", True), ("M", 1.5), (" ", 12)])
def test_invalid_search_arguments_are_rejected_before_embedding(sessions, corpus, query, limit):
    class NoCallsProvider:
        def embed(self, *args):
            pytest.fail("Invalid input must not reach the embedding service")

    with sessions() as db:
        with pytest.raises(DomainError):
            hybrid_search(db, db.get(Snapshot, "snap1"), query, NoCallsProvider(), "test-embedding", limit)


@pytest.mark.parametrize("start,end", [(False, 2), (0, True), (0.5, 2), (0, "2")])
def test_invalid_span_types_are_rejected_before_slicing(sessions, settings, corpus, start, end):
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                         "span-type-test", "local-user", settings)
        with pytest.raises(DomainError):
            get_or_create_span(db, run.id, db.get(Snapshot, "snap1"), "block1", start, end)


@pytest.mark.parametrize("block_ids", [None, "block1", [True], [""]])
def test_source_tool_rejects_invalid_ids_before_database_access(block_ids):
    tools = ScopedTools(None, None, None, None, "test-embedding")
    with pytest.raises(DomainError, match="来源块参数"):
        tools.read_source_blocks(block_ids)


def test_changed_chunk_rule_requires_reindex_without_mutating_old_snapshot(sessions, settings, corpus, fake):
    from app.ingestion import CHUNK_CONFIG, index_config

    with sessions.begin() as db:
        index = db.get(IndexRevision, "index1")
        index.config_version = index_config(settings).replace(CHUNK_CONFIG, "section-page-3200chars-overlap-block-v2")
    with sessions() as db:
        snapshot = db.get(Snapshot, "snap1")
        manifest = list(snapshot.manifest)
        block = db.get(SourceBlock, "block1")
        with pytest.raises(ProviderError, match="索引配置已改变"):
            hybrid_search(db, snapshot, "M F1", fake, "test-embedding",
                          expected_index_config=index_config(settings))
        assert snapshot.manifest == manifest
        assert validate_span(db, snapshot, block.id, 0, 2, "On")


def test_both_retrieval_routes_are_snapshot_scoped(sessions, corpus, fake):
    with sessions.begin() as db:
        db.add(IndexRevision(id="outside", parse_revision_id="parse1", config_version="other",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add(Chunk(id="outside-chunk", index_revision_id="outside", text="M F1 dataset X "*30,
                     source_ranges=[], embedding=[1., 0., 0.]))
    with sessions() as db:
        result = hybrid_search(db, db.get(Snapshot, "snap1"), "M F1 dataset X", fake, "test-embedding")
        assert [r["chunk_id"] for r in result] == ["chunk1"]
        with pytest.raises(ProviderError, match="不兼容"):
            hybrid_search(db, db.get(Snapshot, "snap1"), "M", fake, "wrong-model")
        assert db.scalar(select(Chunk.id).where(Chunk.id == "outside-chunk"))


def test_keyword_fallback_remains_snapshot_scoped(sessions, corpus, fake, monkeypatch):
    with sessions.begin() as db:
        db.add(IndexRevision(id="outside", parse_revision_id="parse1", config_version="other",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add(Chunk(id="outside-match", index_revision_id="outside", text="M F1 dataset X " * 20,
                     source_ranges=[], embedding=[1., 0., 0.]))

    def fail_embedding(_self, _query):
        raise ProviderError("embedding is temporarily unavailable", "embedding_unavailable", retryable=True)

    monkeypatch.setattr("app.retrieval.ProviderEmbedding.get_query_embedding", fail_embedding)
    with sessions() as db:
        result = hybrid_search(db, db.get(Snapshot, "snap1"), "M F1 dataset X", fake, "test-embedding")
    assert result.degradation == {"code": "embedding_unavailable", "route": "keyword_only"}
    assert [item["chunk_id"] for item in result] == ["chunk1"]


def test_keyword_fallback_collapses_duplicates_before_limit(sessions, corpus, fake, monkeypatch):
    with sessions.begin() as db:
        db.add(IndexRevision(id="second-index", parse_revision_id="parse1", config_version="other",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready"))
        db.flush()
        db.add_all([
            Chunk(id="a-fallback", index_revision_id="index1", text="fallbackmarker same",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="b-fallback", index_revision_id="index1", text="fallbackmarker same",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="c-fallback", index_revision_id="second-index", text="fallbackmarker same",
                  source_ranges=[], embedding=[1., 0., 0.]),
            Chunk(id="d-fallback", index_revision_id="index1", text="fallbackmarker different",
                  source_ranges=[], embedding=[1., 0., 0.]),
        ])
        snapshot = db.get(Snapshot, "snap1")
        snapshot.manifest = [*snapshot.manifest, {**snapshot.manifest[0], "index_revision_id": "second-index"}]

    def fail_embedding(_self, _query):
        raise ProviderError("embedding is temporarily unavailable", "embedding_unavailable", retryable=True)

    monkeypatch.setattr("app.retrieval.ProviderEmbedding.get_query_embedding", fail_embedding)
    with sessions() as db:
        result = hybrid_search(db, db.get(Snapshot, "snap1"), "fallbackmarker", fake, "test-embedding", limit=3)
    assert result.degradation == {"code": "embedding_unavailable", "route": "keyword_only"}
    assert [hit["chunk_id"] for hit in result] == ["a-fallback", "c-fallback", "d-fallback"]
    assert result[0]["duplicate_chunk_ids"] == ["b-fallback"]
