from types import SimpleNamespace

import pytest
from app.core.errors import BudgetExhausted, ProviderError
from app.core.models import Chunk, IndexRevision
from app.knowledge.llama_rag import (
    ProviderEmbedding,
    RevisionVectorStore,
    chunk_node,
    embed_chunks,
    index_chunks,
    retrieve_revision,
)
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import MetadataMode, QueryBundle
from llama_index.core.settings import Settings as LlamaSettings
from llama_index.core.vector_stores.types import VectorStoreQuery


class MemoryVectors:
    def __init__(self):
        self.data, self.verified = {}, []

    def ensure_collection(self, name, dimension):
        self.data.setdefault(name, {})

    def insert(self, name, rows):
        self.data[name].update({row["chunk_id"]: row["embedding"] for row in rows})

    def verify_ids(self, name, ids):
        assert set(ids) <= self.data[name].keys()
        self.verified.extend(ids)

    def search(self, name, vector, limit):
        return [(key, sum(a * b for a, b in zip(value, vector)))
                for key, value in self.data[name].items()][:limit]


def test_llamaindex_batches_through_project_provider_and_pins_discovered_dimension():
    calls = []
    provider = SimpleNamespace(embed=lambda texts, dim=0: (
        calls.append((list(texts), dim)) or [[1., 0., 0.] for _ in texts]))
    chunks = [{"text": f"raw fragment {i}"} for i in range(35)]
    vectors = embed_chunks(chunks, provider, "configured-embedding")
    assert len(vectors) == 35
    assert [len(texts) for texts, _ in calls] == [8, 8, 8, 8, 3]
    assert [dim for _, dim in calls] == [0, 3, 3, 3, 3]
    assert [text for batch, _ in calls for text in batch] == [chunk["text"] for chunk in chunks]


def test_llamaindex_embedding_guard_stops_before_second_batch():
    calls, checks = [], []

    def guard():
        checks.append(True)
        if len(checks) == 2:
            raise BudgetExhausted("ingestion_time")

    provider = SimpleNamespace(embed=lambda texts, dim=0: (
        calls.append(texts) or [[1., 0.] for _ in texts]))
    with pytest.raises(BudgetExhausted):
        embed_chunks([{"text": "source"}] * 17, provider, "configured", guard)
    assert len(calls) == 1


def test_index_and_retriever_preserve_ids_sql_sources_and_never_use_default_models(
        sessions, corpus, monkeypatch):
    # A framework accidentally resolving a default model must fail this test,
    # even on machines with default API credentials installed.
    def no_default(_self):
        pytest.fail("The RAG bridge attempted to resolve a global default model")

    monkeypatch.setattr(type(LlamaSettings), "embed_model", property(no_default))
    monkeypatch.setattr(type(LlamaSettings), "llm", property(no_default))
    store = MemoryVectors()
    with sessions.begin() as db:
        index = db.get(IndexRevision, "index1")
        index.vector_collection = "revision1"
        chunk = db.get(Chunk, "chunk1")
        original_ranges = list(chunk.source_ranges)
        built = index_chunks(db, index, [chunk], [[1., 0., 0.]], store, None)
        assert isinstance(built, VectorStoreIndex)
        assert store.verified == ["chunk1"]
        direct_hits = built.as_retriever().retrieve(QueryBundle(query_str="question", embedding=[1., 0., 0.]))
        assert [hit.node.node_id for hit in direct_hits] == ["chunk1"]
        # Existing embeddings were passed into VectorStoreIndex, so no model
        # call is required while the business transaction holds the row lock.
        queries = []
        provider = SimpleNamespace(embed=lambda texts, dim=0: (
            queries.append((texts, dim)) or [[1., 0., 0.]]))
        hits = retrieve_revision(db, index, store, provider, "question")
        assert queries == [(["question"], 3)]
        assert hits[0].node.node_id == "chunk1"
        assert hits[0].node.text == chunk.text
        assert hits[0].node.metadata["source_ranges"] == original_ranges
        assert hits[0].node.get_content(metadata_mode=MetadataMode.EMBED) == chunk.text
        assert hits[0].node.get_content(metadata_mode=MetadataMode.LLM) == chunk.text


def test_external_forged_and_duplicate_ids_do_not_supply_nodes(sessions, corpus):
    store = SimpleNamespace(search=lambda *_: [("chunk1", .1), ("outside", 1.), ("chunk1", .9)])
    with sessions.begin() as db:
        index = db.get(IndexRevision, "index1")
        index.vector_collection = "revision1"
        hits = retrieve_revision(db, index, store, None, "", embedding=[1., 0., 0.])
        assert len(hits) == 1
        assert hits[0].node.node_id == "chunk1" and hits[0].score == .9


def test_llamaindex_cannot_write_other_revision_or_delete_sources(sessions, corpus):
    store = MemoryVectors()
    with sessions() as db:
        index = db.get(IndexRevision, "index1")
        adapter = RevisionVectorStore(db, index, store)
        other = Chunk(id="forged", index_revision_id="outside", text="untrusted", source_ranges=[])
        with pytest.raises(ProviderError, match="不属于"):
            adapter.add([chunk_node(other, [1., 0., 0.])])
        with pytest.raises(ProviderError, match="删除"):
            adapter.delete("parse1")
        with pytest.raises(ProviderError, match="服务端"):
            adapter.query(VectorStoreQuery(query_embedding=[1., 0., 0.], node_ids=["outside"]))
    assert not store.data


def test_embedding_adapter_serialization_excludes_provider_and_credentials():
    provider = SimpleNamespace(secret="must-not-leak", embed=lambda *_: [[1.]])
    adapter = ProviderEmbedding(provider, "configured-model")
    assert "must-not-leak" not in adapter.model_dump_json()
    assert "must-not-leak" not in str(adapter.to_payload())
