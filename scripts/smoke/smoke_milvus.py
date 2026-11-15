"""Real PostgreSQL/Milvus storage check using synthetic vectors, never model APIs.

Creates and removes a uniquely named scratch database and collections. The
optional restart check restarts only this project's Milvus Compose service.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from alembic.config import Config  # noqa: E402
from alembic.migration import MigrationContext  # noqa: E402
from alembic.operations import Operations  # noqa: E402
from alembic.script import ScriptDirectory  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.db import Base, make_engine  # noqa: E402
from app.core.errors import ProviderError  # noqa: E402
from app.core.models import (  # noqa: E402
    AnswerClaim,
    AnswerRevision,
    Chunk,
    ClaimEvidence,
    Corpus,
    EvidenceSpan,
    IndexRevision,
    ModelCall,
    Paper,
    PaperVersion,
    ParseRevision,
    QuestionRun,
    RetrievalRecord,
    Snapshot,
)
from app.core.schemas import RunCreate  # noqa: E402
from app.jobs.services import claim_job, create_run, create_snapshot  # noqa: E402
from app.jobs.worker import process_job  # noqa: E402
from app.knowledge.ingestion import index_config, ingest_version  # noqa: E402
from app.knowledge.retrieval import hybrid_search  # noqa: E402
from app.knowledge.vector_migration import migrate_vectors  # noqa: E402
from app.knowledge.vector_store import MilvusVectorStore  # noqa: E402
from langgraph.checkpoint.postgres import PostgresSaver  # noqa: E402
from sqlalchemy import create_engine, func, select, text  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


class SyntheticProvider:
    def embed(self, texts, expected_dim=0):
        assert expected_dim in (0, 4096)
        return [[1.] + [0.] * 4095 for _ in texts]

    def structured(self, role, prompt, data, schema):
        if role == "Planner":
            output = {"normalized_question": data["question"], "needs_clarification": False,
                      "subquestions": [{"id": "q1", "target": "evidence policy",
                                        "queries": ["selected document version"]}]}
        elif role == "Reranker":
            output = {"items": [{"chunk_id": chunk["chunk_id"], "score": .9,
                                 "relevant_block_ids": [block["block_id"] for block in chunk["blocks"]]}
                                for chunk in data["candidates"]]}
        elif role == "Analyst":
            output = {"claims": [{"id": "c1", "text": "Use evidence from the selected document version.",
                                   "conditions": [], "evidence": [{"evidence_id": data["sources"][0]["id"],
                                                                    "relation": "supports"}]}],
                      "gaps": [], "has_conflict": False}
        else:
            output = {"approved_claim_ids": ["c1"], "issues": [], "followup_queries": []}
        return schema.model_validate(output), {"input_tokens": 10, "output_tokens": 10, "elapsed_seconds": .001}


class FailedWrite:
    def __init__(self, store):
        self.store = store

    def ensure_collection(self, *args):
        self.store.ensure_collection(*args)

    def insert(self, *args):
        raise ProviderError("Injected Milvus write failure", "vector_store_write", retryable=True)


def check(restart=False):
    base = get_settings()
    url = make_url(base.database_url)
    if url.get_backend_name() != "postgresql":
        raise RuntimeError("This storage check requires a PostgreSQL configuration")
    suffix = uuid4().hex[:12]
    scratch_name = f"paperqa_milvus_check_{suffix}"
    prefix = f"smoke_{suffix}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    engine, store = None, None
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), "models": "synthetic_4096d",
              "frameworks": {name: version(name) for name in ("llama-index-core", "langgraph", "pymilvus")},
              "scope": "real PostgreSQL/Milvus + LangGraph/LlamaIndex with synthetic models; no quality or Celery acceptance"}
    created = False
    try:
        with admin.connect() as db:
            db.execute(text(f'CREATE DATABASE "{scratch_name}"'))
        created = True
        engine = make_engine(url.set(database=scratch_name).render_as_string(hide_password=False))
        sessions = sessionmaker(engine, expire_on_commit=False)
        Base.metadata.create_all(engine)
        with tempfile.TemporaryDirectory(prefix="paperqa-milvus-") as storage:
            settings = base.model_copy(update={"database_url": engine.url.render_as_string(hide_password=False),
                                               "milvus_collection_prefix": prefix,
                                               "embedding_model": "synthetic-4096", "embedding_dimensions": 4096,
                                               "answer_review_required": False,
                                               "storage_dir": Path(storage)})
            store = MilvusVectorStore(settings)
            provider = SyntheticProvider()
            with sessions.begin() as db:
                db.add(Corpus(id="corpus", name="Synthetic storage check", owner_id="owner"))
                db.flush()
                db.add(Paper(id="paper", corpus_id="corpus", title="Synthetic source"))
                db.flush()
                db.add(PaperVersion(id="old", paper_id="paper", sha256="0" * 64, storage_key="old.md",
                                    filename="old.md", byte_size=1, status="ready", is_active=True))
                db.flush()
                db.add(ParseRevision(id="parse", paper_version_id="old", parser="synthetic",
                                     config_version="synthetic", status="ready"))
                db.flush()
                db.add(IndexRevision(id="legacy", parse_revision_id="parse", config_version=index_config(settings),
                                     embedding_model=settings.embedding_model, embedding_dimensions=4096,
                                     status="ready"))
                db.flush()
                db.add_all([
                    Chunk(id="exact", index_revision_id="legacy", text="exact synthetic text",
                          source_ranges=[], embedding=[1.] + [0.] * 4095),
                    Chunk(id="orthogonal", index_revision_id="legacy", text="orthogonal synthetic text",
                          source_ranges=[], embedding=[0., 1.] + [0.] * 4094),
                ])
                db.add(Snapshot(id="snapshot", owner_id="owner", corpus_id="corpus", manifest=[{
                    "index_revision_id": "legacy", "parse_revision_id": "parse"}]))

            # Reconstruct the relevant old pgvector schema and exercise 0009 on
            # real PostgreSQL, including conversion of an existing 4096d VECTOR.
            with engine.begin() as db:
                db.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                db.execute(text("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector USING embedding::text::vector"))
                db.execute(text("ALTER TABLE chunks ALTER COLUMN embedding SET NOT NULL"))
                db.execute(text("ALTER TABLE index_revisions DROP COLUMN vector_backend"))
                db.execute(text("ALTER TABLE index_revisions DROP COLUMN vector_collection"))
                config = Config(str(ROOT / "alembic.ini"))
                config.set_main_option("script_location", str(ROOT / "backend/migrations"))
                migration = ScriptDirectory.from_config(config).get_revision("0009").module
                with Operations.context(MigrationContext.configure(db)):
                    migration.upgrade()
                db.execute(text("UPDATE chunks SET search_vector = to_tsvector('simple', text)"))
            with sessions() as db:
                assert db.get(IndexRevision, "legacy").vector_backend == "legacy_pgvector"
                assert len(db.get(Chunk, "exact").embedding) == 4096
                with_error = False
                try:
                    hybrid_search(db, db.get(Snapshot, "snapshot"), "unmatchedquery", provider,
                                  settings.embedding_model, vector_store=store)
                except ProviderError as exc:
                    with_error = exc.code == "vector_store_config"
                assert with_error
            result["legacy_schema_migration"] = "passed"
            result["backfill"] = migrate_vectors(sessions, settings, store, batch_size=1)
            assert result["backfill"]["migrated_chunks"] == 2
            assert migrate_vectors(sessions, settings, store)["migrated_indexes"] == 0
            with sessions() as db:
                assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.embedding.is_not(None))) == 0
                ranking = hybrid_search(db, db.get(Snapshot, "snapshot"), "unmatchedquery", provider,
                                        settings.embedding_model, vector_store=store)
                assert [item["chunk_id"] for item in ranking] == ["exact", "orthogonal"]
            result["backfill_retry_and_cosine_order"] = "passed"

            def add_version(version_id, content):
                path = settings.storage_dir / f"{version_id}.md"
                path.write_text(content, encoding="utf-8")
                with sessions.begin() as db:
                    db.add(PaperVersion(id=version_id, paper_id="paper", sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        storage_key=path.name, filename=path.name, byte_size=path.stat().st_size,
                                        status="parsing", attempt=1, is_active=False))

            add_version("new", "# Synthetic handbook\n\nOnly use evidence from the selected document version.\n")
            ingest_version(sessions, settings, provider, "new", 1, vector_store=store)
            with sessions.begin() as db:
                assert db.get(PaperVersion, "new").is_active
                assert not db.get(PaperVersion, "old").is_active
                new_index = db.scalar(select(IndexRevision).where(IndexRevision.id != "legacy"))
                new_chunk = db.scalar(select(Chunk).where(Chunk.index_revision_id == new_index.id))
                assert new_index.vector_backend == "milvus" and new_chunk.embedding is None
                assert new_chunk.search_vector
                # Insert a forged/out-of-snapshot hit into the other collection.
                legacy_name = db.get(IndexRevision, "legacy").vector_collection
                store.insert(legacy_name, [{"chunk_id": new_chunk.id, "embedding": [1.] + [0.] * 4095}])
                ranking = hybrid_search(db, db.get(Snapshot, "snapshot"), "unmatchedquery", provider,
                                        settings.embedding_model, vector_store=store)
                assert new_chunk.id not in {item["chunk_id"] for item in ranking}
            result["production_ingest_fts_and_snapshot_filter"] = "passed"

            add_version("failed", "# Failing replacement\n\nThe new version must not displace a usable old version.\n")
            try:
                ingest_version(sessions, settings, provider, "failed", 1, vector_store=FailedWrite(store))
                raise AssertionError("Injected write failure unexpectedly succeeded")
            except ProviderError as exc:
                assert exc.code == "vector_store_write"
            with sessions() as db:
                assert db.get(PaperVersion, "new").is_active
                assert not db.get(PaperVersion, "failed").is_active
                assert db.scalar(select(func.count()).select_from(IndexRevision)) == 2
            result["failed_replacement_keeps_active_version"] = "passed"

            # Run the real graph, LlamaIndex retriever and PostgreSQL checkpointer
            # together. Model responses alone are deterministic test doubles.
            with sessions() as db:
                snapshot = create_snapshot(db, "corpus", ["new"], "owner")
                run_id = create_run(db, RunCreate(snapshot_id=snapshot.id,
                                                 question="Which document version supplies evidence?"),
                                    f"smoke-{suffix}", "owner", settings).id
            job = claim_job(sessions, settings)
            assert job == ("run", run_id, 1)
            dsn = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
            with PostgresSaver.from_conn_string(dsn) as saver:
                saver.setup()
                process_job(sessions, settings, provider, job, saver)
                assert saver.get_tuple({"configurable": {"thread_id": f"{run_id}:1"}}) is not None
            with sessions() as db:
                run = db.get(QuestionRun, run_id)
                assert run.status == "completed", run.error
                assert run.config["rag_framework"] == "llamaindex-vectorstoreindex-v1"
                assert db.scalar(select(RetrievalRecord).where(RetrievalRecord.run_id == run_id)).rankings
                calls = list(db.scalars(select(ModelCall).where(ModelCall.run_id == run_id)))
                assert sorted(call.role for call in calls) == ["Analyst", "Embedding", "Planner", "Reranker", "Verifier"]
                answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run_id))
                assert answer.evidence_status == "supported"
                claim = db.scalar(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id))
                assert claim.semantic_verified
                link = db.scalar(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id))
                assert db.get(EvidenceSpan, link.evidence_id).quote
            result["langgraph_llamaindex_workflow_and_postgres_checkpoint"] = "passed"
            result["workflow_embedding_calls"] = 1

            if restart:
                store.close()
                store = None
                subprocess.run(["docker", "compose", "restart", "milvus"], cwd=ROOT, check=True,
                               capture_output=True, timeout=60)
                deadline = time.monotonic() + 150
                import httpx

                while True:
                    try:
                        if httpx.get("http://127.0.0.1:19091/healthz", timeout=2, trust_env=False).is_success:
                            break
                    except httpx.HTTPError:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Milvus did not become healthy after restart")
                    time.sleep(2)
                store = MilvusVectorStore(settings)
                # The HTTP health endpoint may precede data-plane readiness:
                # embedded etcd can still hold an old node's lease after restart.
                recovery_retries = 0
                while True:
                    try:
                        with sessions() as db:
                            ranking = hybrid_search(db, db.get(Snapshot, "snapshot"), "unmatchedquery", provider,
                                                    settings.embedding_model, vector_store=store)
                            assert ranking[0]["chunk_id"] == "exact"
                        break
                    except ProviderError:
                        recovery_retries += 1
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(2)
                result["restart_data_ready_retries"] = recovery_retries
                result["restart_persistence"] = "passed"
            else:
                result["restart_persistence"] = "not_requested"
            result["status"] = "passed"
    finally:
        cleanup_errors = []
        try:
            if store is not None:
                for name in store.client.list_collections(timeout=20):
                    if name.startswith(prefix + "_"):
                        try:
                            store.client.drop_collection(name, timeout=20)
                        except Exception as exc:
                            cleanup_errors.append(type(exc).__name__)
                store.close()
        finally:
            if engine is not None:
                engine.dispose()
            if created:
                with admin.connect() as db:
                    db.execute(text(f'DROP DATABASE "{scratch_name}" WITH (FORCE)'))
            admin.dispose()
        if cleanup_errors:
            raise RuntimeError(f"Scratch collection cleanup failed ({prefix}): {cleanup_errors}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-milvus", action="store_true")
    parser.add_argument("--output", default="tmp/reports/milvus-storage-check.json")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    result = check(args.restart_milvus)
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
