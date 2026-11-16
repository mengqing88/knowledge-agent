"""Exercise both runtime lexical routes against PostgreSQL/Milvus using saved vectors.

Copies only the supplied immutable snapshot to a fresh scratch database and
collections. No original tables, files or model endpoints are modified/called.
Question-level results are allowed only under the ignored local tmp tree.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core.config import Settings  # noqa: E402
from app.core.models import (  # noqa: E402
    Chunk,
    Corpus,
    IndexRevision,
    Paper,
    PaperVersion,
    ParseRevision,
    Snapshot,
    SourceBlock,
)
from app.knowledge.ingestion import index_config  # noqa: E402
from app.knowledge.llama_rag import index_chunks  # noqa: E402
from app.knowledge.retrieval import hybrid_search  # noqa: E402
from app.knowledge.vector_store import collection_name  # noqa: E402
from sqlalchemy import create_engine, func, select, update  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from scripts.benchmarks.benchmark_chunking import scratch  # noqa: E402
from scripts.benchmarks.benchmark_lexical_recall import score, sha, validate_scope  # noqa: E402


class SavedVectors:
    def __init__(self, cache):
        self.cache = cache
        self.requests = 0

    def embed(self, texts, expected_dim=0):
        output = []
        for text in texts:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if key not in self.cache:
                raise ValueError("Missing saved vector; external fallback is forbidden")
            vector = self.cache[key]
            if expected_dim and len(vector) != expected_dim:
                raise ValueError("Saved query vector has wrong dimensions")
            output.append(vector)
        self.requests += 1
        return output


def save(path, result):
    temp = path.with_suffix(".writing")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def copy_snapshot(source, target_sessions, store, isolated, suite, provider):
    with Session(source) as db:
        snapshot = db.get(Snapshot, suite["snapshot_id"])
        index_ids, indexes = validate_scope(db, snapshot, suite["questions"])
        parse_ids = {index.parse_revision_id for index in indexes}
        versions = list(db.scalars(select(ParseRevision.paper_version_id).where(ParseRevision.id.in_(parse_ids))))
        papers = list(db.scalars(select(PaperVersion.paper_id).where(PaperVersion.id.in_(versions))))
        clauses = [
            (Corpus, Corpus.id == snapshot.corpus_id),
            (Paper, Paper.id.in_(papers)),
            (PaperVersion, PaperVersion.id.in_(versions)),
            (ParseRevision, ParseRevision.id.in_(parse_ids)),
            (SourceBlock, SourceBlock.parse_revision_id.in_(parse_ids)),
            (IndexRevision, IndexRevision.id.in_(index_ids)),
        ]
        with target_sessions.begin() as target:
            for model, condition in clauses:
                rows = db.execute(select(model.__table__).where(condition)).mappings().all()
                values = [dict(row) for row in rows]
                if model is IndexRevision:
                    for value in values:
                        if value["config_version"] != index_config(isolated):
                            raise ValueError("Saved index config differs; no implicit reindexing is allowed")
                        value.update(vector_backend="milvus", vector_collection=collection_name(
                            value["id"], isolated.milvus_collection_prefix))
                if values:
                    target.execute(model.__table__.insert(), values)
        for index in indexes:
            # One document at a time bounds Python vector memory. The LlamaIndex
            # nodes already have embeddings; SavedVectors rejects any extra call.
            chunk_rows = list(db.execute(select(Chunk.__table__).where(
                Chunk.index_revision_id == index.id)).mappings())
            with target_sessions.begin() as target:
                for row in chunk_rows:
                    value = dict(row)
                    value.update(embedding=None, search_vector=None)
                    target.execute(Chunk.__table__.insert(), value)
                copied = list(target.scalars(select(Chunk).where(Chunk.index_revision_id == index.id).order_by(Chunk.id)))
                embeddings = {row["id"]: row["embedding"] for row in chunk_rows}
                target_index = target.get(IndexRevision, index.id)
                index_chunks(target, target_index, copied, [embeddings[chunk.id] for chunk in copied], store, provider)
                target.execute(update(Chunk).where(Chunk.index_revision_id == index.id).values(
                    search_vector=func.to_tsvector("simple", Chunk.text)))
            print(f"copied index {index.id}; chunks={len(chunk_rows)}", flush=True)
        with target_sessions.begin() as target:
            target.add(Snapshot(id=snapshot.id, corpus_id=snapshot.corpus_id, owner_id=snapshot.owner_id,
                                manifest=snapshot.manifest))
    return index_ids


def run(args):
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Use a new result file inside the local tmp tree")
    if not args.database.is_file():
        raise ValueError("An existing read-only source database is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    query_cache = json.loads(args.query_vectors.read_text(encoding="utf-8"))
    settings = Settings(provider_max_retries=0)
    identity = hashlib.sha256(json.dumps({
        "model": settings.embedding_model, "url": settings.embedding_base_url,
        "revision": settings.embedding_revision, "dimension": settings.embedding_dimensions,
    }, sort_keys=True).encode()).hexdigest()
    if query_cache["identity"] != identity:
        raise ValueError("Saved query vectors do not match the configured provider identity")
    provider = SavedVectors(query_cache["vectors"])
    settings.database_url = make_url(settings.database_url).update_query_dict(
        {"connect_timeout": "5"}).render_as_string(hide_password=False)
    source = create_engine("sqlite:///file:" + args.database.resolve().as_posix() + "?mode=ro&uri=true")
    result = {"status": "preparing", "scope": "real PostgreSQL FTS / bounded runtime BM25 + Milvus with saved vectors",
              "human_reviewed": False, "suite_sha256": sha(args.suite), "query_vectors_sha256": sha(args.query_vectors),
              "new_external_model_calls": 0, "new_external_cost_cny": "0", "cases": [],
              "limitations": ["Development queries were already inspected; no unseen or independent human test set.",
                              "Retrieval only; no chat reranking, generated answers, browser or SLA validation.",
                              "Saved vectors; latency excludes model requests. Corpus lexical cache is per mode/run."]}
    started = time.monotonic()
    try:
        with scratch(settings) as (sessions, store, isolated):
            result["scratch_database"] = sessions.kw["bind"].url.database
            result["scratch_prefix"] = isolated.milvus_collection_prefix
            save(args.output, result)
            index_ids = copy_snapshot(source, sessions, store, isolated, suite, provider)
            if provider.requests:
                raise ValueError("Copying precomputed vectors unexpectedly requested embeddings")
            result["status"] = "running"
            cache = {}
            for question in suite["questions"]:
                case = {"id": question["id"], "modes": {}}
                for mode in ("legacy", "ngram_bm25"):
                    begin = time.monotonic()
                    with sessions() as db:
                        snapshot = db.get(Snapshot, suite["snapshot_id"])
                        hits = hybrid_search(db, snapshot, question["question"], provider,
                                             isolated.embedding_model, limit=30,
                                             expected_index_config=index_config(isolated), vector_store=store,
                                             lexical_backend=mode, lexical_max_chunks=isolated.lexical_max_chunks,
                                             lexical_max_chars=isolated.lexical_max_chars, lexical_cache=cache)
                        if hits.degradation:
                            raise ValueError("Unexpected retrieval degradation")
                        ranking = [hit["chunk_id"] for hit in hits]
                        chunks = {chunk.id: chunk for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(ranking)))}
                        if any(chunk.index_revision_id not in index_ids for chunk in chunks.values()):
                            raise ValueError("Candidate scope leak")
                        scores = {str(k):score(ranking[:k], chunks, question["gold"]) for k in (5,12,30)}
                    case["modes"][mode] = {"scores": scores, "top30": ranking,
                                           "seconds_excluding_external_models": round(time.monotonic()-begin, 4)}
                result["cases"].append(case)
                save(args.output, result)
                print(f"retrieved {question['id']}; completed={len(result['cases'])}/{len(suite['questions'])}", flush=True)
            result["summary"] = {mode: {str(k):{
                "covered":sum(case["modes"][mode]["scores"][str(k)]["covered"] for case in result["cases"]),
                "gold":sum(case["modes"][mode]["scores"][str(k)]["gold"] for case in result["cases"]),
                "complete_queries":sum(case["modes"][mode]["scores"][str(k)]["complete"] for case in result["cases"]),
            } for k in (5,12,30)} for mode in ("legacy","ngram_bm25")}
            result["cached_embedding_reads"] = provider.requests
        result.update(status="completed", cleanup="passed", elapsed_seconds=round(time.monotonic()-started,3))
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error_code=getattr(exc,"code",None),
                      cleanup="context_manager_finished", elapsed_seconds=round(time.monotonic()-started,3))
    finally:
        source.dispose()
        save(args.output, result)
    print(json.dumps({key:result.get(key) for key in ("status","summary","cleanup","elapsed_seconds","error_type","error_code")},
                     ensure_ascii=False))
    if result["status"] != "completed":
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database",type=Path,required=True)
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--query-vectors",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
