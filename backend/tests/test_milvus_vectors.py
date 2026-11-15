from types import SimpleNamespace

import pytest
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from app.core.config import ROOT
from app.core.errors import ProviderError
from app.core.models import Chunk, IndexRevision, Snapshot
from app.knowledge.retrieval import milvus_search
from app.knowledge.vector_migration import migrate_vectors
from app.knowledge.vector_store import MilvusVectorStore
from sqlalchemy import create_engine, text


class MemoryVectors:
    def __init__(self, fail_at=None):
        self.data, self.calls, self.fail_at = {}, 0, fail_at

    def ensure_collection(self, name, dimension):
        self.data.setdefault(name, {})

    def insert(self, name, rows):
        self.calls += 1
        if self.calls == self.fail_at:
            raise ProviderError("injected Milvus outage", "vector_store_write", retryable=True)
        self.data[name].update({row["chunk_id"]: row["embedding"] for row in rows})

    def verify_ids(self, name, ids):
        assert set(ids) <= self.data[name].keys()


def test_backfill_verifies_before_switch_and_preserves_snapshot(sessions, settings, corpus):
    store = MemoryVectors()
    with sessions() as db:
        manifest = db.get(Snapshot, "snap1").manifest
    result = migrate_vectors(sessions, settings, store, batch_size=1)
    assert result["migrated_indexes"] == 1
    assert result["migrated_chunks"] == 1
    with sessions() as db:
        index = db.get(IndexRevision, "index1")
        assert index.vector_backend == "milvus"
        assert index.vector_collection == "paperqa_index1"
        assert db.get(Chunk, "chunk1").embedding is None
        assert db.get(Snapshot, "snap1").manifest == manifest
    assert migrate_vectors(sessions, settings, store)["migrated_indexes"] == 0
    assert store.calls == 1


def test_partial_backfill_failure_preserves_all_sql_vectors_and_can_retry(sessions, settings, corpus):
    with sessions.begin() as db:
        db.add(Chunk(id="chunk2", index_revision_id="index1", text="second chunk",
                     source_ranges=[], embedding=[0., 1., 0.]))
    store = MemoryVectors(fail_at=2)
    with pytest.raises(ProviderError, match="outage"):
        migrate_vectors(sessions, settings, store, batch_size=1)
    with sessions() as db:
        assert db.get(Chunk, "chunk1").embedding == [1., 0., 0.]
        assert db.get(Chunk, "chunk2").embedding == [0., 1., 0.]
        assert db.get(IndexRevision, "index1").vector_collection is None
    store.fail_at = None
    result = migrate_vectors(sessions, settings, store, batch_size=1)
    assert result["migrated_chunks"] == 2
    assert set(store.data["paperqa_index1"]) == {"chunk1", "chunk2"}


def test_invalid_legacy_dimension_never_publishes(sessions, settings, corpus):
    with sessions.begin() as db:
        db.get(Chunk, "chunk1").embedding = [1.]
    store = MemoryVectors()
    with pytest.raises(ProviderError, match="维度"):
        migrate_vectors(sessions, settings, store)
    assert store.calls == 0
    with sessions() as db:
        assert db.get(Chunk, "chunk1").embedding == [1.]
        assert db.get(IndexRevision, "index1").vector_collection is None


def test_cosine_descending_and_collection_hits_cannot_cross_index(sessions, corpus):
    with sessions.begin() as db:
        db.get(IndexRevision, "index1").vector_collection = "collection1"
        db.add(IndexRevision(id="index2", parse_revision_id="parse1", config_version="another",
                             embedding_model="test-embedding", embedding_dimensions=3, status="ready",
                             vector_backend="milvus", vector_collection="collection2"))
        db.flush()
        db.add(Chunk(id="chunk2", index_revision_id="index2", text="other", source_ranges=[], embedding=None))
    store = SimpleNamespace(search=lambda name, *_: (
        [("chunk1", .2), ("chunk2", 1.), ("forged", 1.)] if name == "collection1"
        else [("chunk2", .9)]))
    with sessions() as db:
        indexes = [db.get(IndexRevision, key) for key in ("index1", "index2")]
        assert milvus_search(db, indexes, [1., 0., 0.], store) == ["chunk2", "chunk1"]
        assert milvus_search(db, indexes[:1], [1., 0., 0.], store) == ["chunk1"]
        indexes[0].vector_backend = "legacy_pgvector"
        with pytest.raises(ProviderError, match="migrate-vectors"):
            milvus_search(db, indexes, [1., 0., 0.], store)


def test_milvus_verification_rejects_missing_acknowledged_chunk():
    store = object.__new__(MilvusVectorStore)
    store.timeout = 5
    store.client = SimpleNamespace(get=lambda **_: [{"chunk_id": "a"}])
    with pytest.raises(ProviderError, match="核对失败"):
        store.verify_ids("index", ["a", "b"])


def test_milvus_existing_collection_rejects_dimension_mismatch(settings):
    store = object.__new__(MilvusVectorStore)
    store.timeout, store.settings = 5, settings
    store.client = SimpleNamespace(
        has_collection=lambda **_: True,
        describe_collection=lambda **_: {"fields": [{"name": "embedding", "params": {"dim": 3}}]},
    )
    with pytest.raises(ProviderError, match="维度不匹配"):
        store.ensure_collection("index", 4096)


def test_legacy_schema_migration_does_not_invent_milvus_collection(tmp_path):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))
    migration = ScriptDirectory.from_config(config).get_revision("0009").module
    engine = create_engine(f"sqlite:///{(tmp_path / 'legacy-vectors.db').as_posix()}")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE index_revisions (id TEXT PRIMARY KEY)"))
            db.execute(text("INSERT INTO index_revisions VALUES ('old')"))
            with Operations.context(MigrationContext.configure(db)):
                migration.upgrade()
            assert db.execute(text("SELECT vector_backend FROM index_revisions")).scalar() == "legacy_pgvector"
            assert db.execute(text("SELECT vector_collection FROM index_revisions")).scalar() is None
            with Operations.context(MigrationContext.configure(db)):
                with pytest.raises(RuntimeError, match="备份"):
                    migration.downgrade()
    finally:
        engine.dispose()
