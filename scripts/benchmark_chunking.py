"""Compare real hybrid retrieval under source-preserving chunk policies.

Uses a frozen, explicitly fictional long-document corpus and the configured
Embedding service. PostgreSQL/Milvus resources are temporary; no chat/reranker
calls or changes to production documents. Missing/failed queries score zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.chunking import ChunkPolicy, token_count  # noqa: E402
from app.config import Settings  # noqa: E402
from app.db import Base, make_engine  # noqa: E402
from app.errors import BudgetExhausted, ProviderError  # noqa: E402
from app.ingestion import build_chunks, extract_document, index_config  # noqa: E402
from app.llama_rag import embed_chunks, index_chunks  # noqa: E402
from app.models import Chunk, Corpus, IndexRevision, Paper, PaperVersion, ParseRevision, Snapshot, SourceBlock  # noqa: E402
from app.providers import ModelProvider  # noqa: E402
from app.retrieval import hybrid_search  # noqa: E402
from app.vector_store import MilvusVectorStore, collection_name  # noqa: E402
from sqlalchemy import create_engine, func, select, text, update  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402


def policies():
    return [ChunkPolicy("legacy"), ChunkPolicy("fixed", 256, 0),
            ChunkPolicy("fixed", 512, 0), ChunkPolicy("fixed", 512, 64), ChunkPolicy("fixed", 1024, 128),
            ChunkPolicy("paragraph", 512, 0), ChunkPolicy("structure", 256, 32),
            ChunkPolicy("structure", 512, 0), ChunkPolicy("structure", 512, 64),
            ChunkPolicy("structure", 1024, 128), ChunkPolicy("semantic", 512, 64, .55, 512)]


def name(policy):
    return "legacy_chars3200" if policy.strategy == "legacy" else f"{policy.strategy}_{policy.size}_{policy.overlap}"


class CachedEmbedding:
    """Share exact text/query vectors across experiments, with explicit caps."""

    def __init__(self, settings, max_tokens=600000, max_texts=3000, cache_dir=None):
        self.provider = ModelProvider(settings)
        self.cache, self.calls = {}, []
        self.max_tokens, self.max_texts = max_tokens, max_texts
        self.sent_tokens, self.dimension = 0, settings.embedding_dimensions
        self.deadline = time.monotonic() + 1800
        self.cache_dir, self.disk_cache_hits = cache_dir, 0
        self.phase, self.phase_requested_tokens = "query_embeddings", {}
        self.identity = hashlib.sha256(json.dumps({"model": settings.embedding_model,
            "url": settings.embedding_base_url, "revision": settings.embedding_revision,
            "dimension": settings.embedding_dimensions}, sort_keys=True).encode()).hexdigest()
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def cache_path(self, item):
        return self.cache_dir / (hashlib.sha256((self.identity + item).encode()).hexdigest() + ".json")

    def guard(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExhausted("evaluation_time")

    def embed(self, texts, expected_dim=0):
        self.guard()
        self.phase_requested_tokens[self.phase] = self.phase_requested_tokens.get(self.phase, 0) + sum(
            token_count(item) for item in texts)
        for item in dict.fromkeys(texts):
            if item not in self.cache and self.cache_dir and self.cache_path(item).is_file():
                vector = json.loads(self.cache_path(item).read_text(encoding="utf-8"))
                if (not vector or not all(type(x) in (int, float) and math.isfinite(x) for x in vector) or
                        sum(x*x for x in vector) == 0 or
                        (self.dimension and len(vector) != self.dimension)):
                    raise ValueError("评测向量缓存损坏或维度不匹配")
                self.dimension = self.dimension or len(vector)
                self.cache[item] = vector
                self.disk_cache_hits += 1
        if len(self.cache) > self.max_texts:
            raise BudgetExhausted("evaluation_embedding_limit")
        missing = list(dict.fromkeys(item for item in texts if item not in self.cache))
        for start in range(0, len(missing), 8):
            batch = missing[start:start + 8]
            estimated = sum(token_count(item) for item in batch)
            if self.sent_tokens + estimated > self.max_tokens or len(self.cache) + len(batch) > self.max_texts:
                raise BudgetExhausted("evaluation_embedding_limit")
            self.sent_tokens += estimated
            call = {"texts": len(batch), "reference_input_tokens": estimated, "status": "started", "phase": self.phase}
            self.calls.append(call)
            started = time.monotonic()
            try:
                with self.provider.execution(self.deadline, self.guard, lambda _event: None):
                    vectors = self.provider.embed(batch, expected_dim or self.dimension)
                self.dimension = self.dimension or len(vectors[0])
                self.cache.update(zip(batch, vectors))
                call.update(status="completed", **self.provider.last_embedding_usage)
                if self.cache_dir:
                    for item, vector in zip(batch, vectors):
                        self.cache_path(item).write_text(json.dumps(vector), encoding="utf-8")
            except Exception:
                call.update(status="failed", **self.provider.last_embedding_usage)
                raise
            finally:
                call["seconds"] = round(time.monotonic() - started, 3)
        vectors = [self.cache[item] for item in texts]
        if expected_dim and any(len(vector) != expected_dim for vector in vectors):
            raise ProviderError("评测缓存向量维度不匹配", "vector_dimension")
        return vectors

    def summary(self):
        known = [call for call in self.calls if call.get("usage_known")]
        return {"http_calls": len(self.calls), "unique_texts": len(self.cache),
                "reference_input_tokens": self.sent_tokens, "max_reference_input_tokens": self.max_tokens,
                "max_unique_texts": self.max_texts, "dimensions": self.dimension,
                "known_provider_input_tokens": sum(call["input_tokens"] for call in known),
                "unknown_usage_calls": len(self.calls) - len(known), "disk_cache_hits": self.disk_cache_hits,
                "reference_tokens_requested_by_phase": self.phase_requested_tokens,
                "http_batch_size": 8, "calls": self.calls}


def load_corpus(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    documents, all_blocks = {}, []
    for entry in manifest["documents"]:
        path = manifest_path.parent / entry["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"语料哈希不匹配: {entry['id']}")
        raw, _, _ = extract_document(path, entry["filename"])
        blocks = [SourceBlock(id=hashlib.sha256(f"{entry['id']}:{i}".encode()).hexdigest()[:32],
                              parse_revision_id=entry["id"], **{k: v for k, v in item.items() if k != "page_height"})
                  for i, item in enumerate(raw)]
        documents[entry["id"]] = (entry, blocks, {item["page_index"]: item["page_height"] for item in raw})
        all_blocks.extend(blocks)
    questions = []
    for question in manifest["questions"]:
        gold = []
        for label in question["evidence"]:
            matches = []
            for block in documents[label["document"]][1]:
                if "section" in label and label["section"] != block.section:
                    continue
                start = block.text.find(label["quote"])
                if start >= 0:
                    if block.text.find(label["quote"], start + 1) >= 0:
                        raise ValueError("参考摘录在同一来源块中不唯一")
                    matches.append({"block_id": block.id, "char_start": start,
                                    "char_end": start + len(label["quote"])})
            if len(matches) != 1:
                raise ValueError(f"标注无法唯一定位: {question['id']}")
            gold.extend(matches)
        questions.append({**question, "gold": gold})
    return manifest, documents, questions, all_blocks


def covers(chunk, gold):
    return any(span["block_id"] == gold["block_id"] and span["char_start"] <= gold["char_start"] and
               span["char_end"] >= gold["char_end"] for span in chunk.source_ranges)


def score_query(chunks, gold):
    if not gold:
        raise ValueError("Recall 分母不可为空")
    # Gold is a set of source spans, independent of each strategy's chunk IDs.
    # A gold span must be wholly contained in one retrieved chunk.
    recalls = {str(k): sum(any(covers(chunk, span) for chunk in chunks[:k]) for span in gold) / len(gold)
               for k in (1, 5)}
    rank = next((i for i, chunk in enumerate(chunks[:5], 1) if any(covers(chunk, span) for span in gold)), None)
    return {"evidence_recall_at_1": recalls["1"], "evidence_recall_at_5": recalls["5"],
            "hit_at_5": int(rank is not None), "reciprocal_rank_at_5": 1 / rank if rank else 0.,
            "top5_context_tokens": sum(token_count(chunk.text) for chunk in chunks[:5])}


def aggregate(rows):
    numeric = ("evidence_recall_at_1", "evidence_recall_at_5", "hit_at_5", "reciprocal_rank_at_5",
               "top5_context_tokens")
    return {"questions": len(rows), "failed_questions": sum(row["status"] != "passed" for row in rows),
            **{key: sum(row[key] for row in rows) / len(rows) if rows else None for key in numeric}}


@contextmanager
def scratch(settings):
    url = make_url(settings.database_url)
    if url.get_backend_name() != "postgresql":
        raise ValueError("分块检索评测需要 PostgreSQL/Milvus")
    suffix = uuid4().hex[:12]
    database = f"paperqa_chunk_check_{suffix}"
    admin = create_engine(url, isolation_level="AUTOCOMMIT")
    engine, store, created = None, None, False
    try:
        with admin.connect() as db:
            db.execute(text(f'CREATE DATABASE "{database}"'))
        created = True
        engine = make_engine(url.set(database=database).render_as_string(hide_password=False))
        Base.metadata.create_all(engine)
        isolated = settings.model_copy(update={"milvus_collection_prefix": f"chunk_{suffix}"})
        store = MilvusVectorStore(isolated)
        yield sessionmaker(engine, expire_on_commit=False), store, isolated
    finally:
        try:
            if store:
                try:
                    for collection in store.client.list_collections():
                        if collection.startswith(f"chunk_{suffix}_"):
                            store.client.drop_collection(collection_name=collection)
                finally:
                    store.close()
        finally:
            if engine:
                engine.dispose()
            try:
                if created:
                    with admin.connect() as db:
                        db.execute(text(f'DROP DATABASE "{database}" WITH (FORCE)'))
            finally:
                admin.dispose()


def evaluate_strategy(sessions, store, settings, provider, policy, documents, questions):
    config = settings.model_copy(update={"chunk_strategy": policy.strategy, "chunk_size_tokens": policy.size,
                                        "chunk_overlap_tokens": policy.overlap,
                                        "chunk_semantic_threshold": policy.semantic_threshold,
                                        "chunk_semantic_max_units": policy.semantic_max_units})
    strategy_id = name(policy)
    manifest, lengths, all_spans = [], [], []
    phase_before = dict(provider.phase_requested_tokens)
    calls_before = len(provider.calls)
    started = time.monotonic()
    for doc_id, (_entry, blocks, heights) in documents.items():
        provider.phase = "semantic_boundaries" if policy.strategy == "semantic" else "chunking"
        chunks = build_chunks(blocks, heights, policy=policy, provider=provider,
                              model_name=settings.embedding_model, guard=provider.guard)
        provider.phase = "index_embeddings"
        vectors = embed_chunks(chunks, provider, settings.embedding_model, provider.guard)
        index_id = hashlib.sha256(f"{strategy_id}:{doc_id}".encode()).hexdigest()[:32]
        with sessions.begin() as db:
            index = IndexRevision(id=index_id, parse_revision_id=doc_id, config_version=index_config(config),
                                  embedding_model=settings.embedding_model, embedding_dimensions=len(vectors[0]),
                                  vector_backend="milvus", vector_collection=collection_name(index_id, settings.milvus_collection_prefix),
                                  status="ready")
            db.add(index)
            db.flush()
            rows = [Chunk(id=hashlib.sha256(f"{index_id}:{i}".encode()).hexdigest()[:32],
                          index_revision_id=index_id, **item) for i, item in enumerate(chunks)]
            db.add_all(rows)
            db.flush()
            index_chunks(db, index, rows, vectors, store, provider, provider.guard)
            db.execute(update(Chunk).where(Chunk.index_revision_id == index_id).values(
                search_vector=func.to_tsvector("simple", Chunk.text)))
            lengths.extend(token_count(chunk.text) for chunk in rows)
            all_spans.extend(span for chunk in rows for span in chunk.source_ranges)
        manifest.append({"index_revision_id": index_id, "parse_revision_id": doc_id})
    with sessions.begin() as db:
        db.add(Snapshot(id=strategy_id, corpus_id="benchmark", owner_id="benchmark", manifest=manifest))
    build_seconds = time.monotonic() - started
    results = []
    provider.phase = "query_embeddings"
    for question in questions:
        started = time.monotonic()
        try:
            with sessions() as db:
                ranked = hybrid_search(db, db.get(Snapshot, strategy_id), question["question"], provider,
                                       settings.embedding_model, limit=5,
                                       expected_index_config=index_config(config), vector_store=store)
                if ranked.degradation:
                    raise ProviderError("评测不允许向量失败降级后混入对比", "evaluation_degraded")
                by_id = {row.id: row for row in db.scalars(select(Chunk).where(
                    Chunk.id.in_([hit["chunk_id"] for hit in ranked])))}
                ordered = [by_id[hit["chunk_id"]] for hit in ranked]
                score = score_query(ordered, question["gold"])
                result = {"id": question["id"], "split": question["split"], "status": "passed", **score,
                          "rankings": [{"chunk_id": row.id, "source_ranges": row.source_ranges} for row in ordered]}
        except (ProviderError, BudgetExhausted) as exc:
            result = {"id": question["id"], "split": question["split"], "status": "failed",
                      "error_code": getattr(exc, "code", "budget"), **score_query([], question["gold"])}
        result["retrieval_seconds_with_cached_query_vector"] = round(time.monotonic() - started, 4)
        results.append(result)
    # Count interval unions, not a set of identical ranges: partial overlaps count too.
    unique_chars = 0
    by_block = {}
    for span in all_spans:
        by_block.setdefault(span["block_id"], []).append((span["char_start"], span["char_end"]))
    for intervals in by_block.values():
        end = -1
        for start, stop in sorted(intervals):
            unique_chars += max(0, stop - max(start, end))
            end = max(stop, end)
    source_chars = sum(span["char_end"] - span["char_start"] for span in all_spans)
    latencies = sorted(row["retrieval_seconds_with_cached_query_vector"] for row in results)
    return {"policy": policy.manifest(), "chunks": len(lengths), "embedded_reference_tokens": sum(lengths),
            "additional_boundary_reference_tokens": provider.phase_requested_tokens.get("semantic_boundaries", 0) -
                phase_before.get("semantic_boundaries", 0),
            "new_http_calls_this_run": len(provider.calls) - calls_before,
            "chunk_tokens_p95": sorted(lengths)[math.ceil(.95 * len(lengths)) - 1],
            "chunk_tokens_max": max(lengths), "source_overlap_ratio": source_chars / unique_chars - 1,
            "build_seconds": round(build_seconds, 3),
            "retrieval_p95_seconds_cached_query": latencies[math.ceil(.95 * len(latencies)) - 1],
            "overall": aggregate(results),
            "development": aggregate([row for row in results if row["split"] == "development"]),
            "held_out": aggregate([row for row in results if row["split"] == "held_out"]), "queries": results}


def run(args, output):
    settings = Settings(provider_max_retries=0)
    if not (settings.embedding_model and settings.embedding_base_url and settings.embedding_api_key):
        raise ValueError("Embedding 配置不完整，未创建评测外部资源")
    manifest, documents, questions, blocks = load_corpus(args.manifest)
    provider = CachedEmbedding(settings, args.max_embedding_tokens, args.max_embedding_texts, args.cache_dir)
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "real configured Embedding + PostgreSQL FTS/RRF + Milvus/LlamaIndex; fictional source-span labels",
              "corpus_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "fictional": manifest.get("fictional"), "human_reviewed": manifest.get("human_reviewed", False),
              "embedding_model": settings.embedding_model, "chat_calls": 0, "documents": len(documents),
              "embedding_identity": provider.identity,
              "questions": len(questions), "strategies": {},
              "packages": {package: version(package) for package in ("tiktoken", "llama-index-core", "pymilvus")},
              "application_sha256": {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in (
                  "backend/app/chunking.py", "backend/app/ingestion.py", "backend/app/retrieval.py",
                  "backend/app/llama_rag.py", "backend/app/config.py", "scripts/benchmark_chunking.py")},
              "metric_definition": "Macro source-span Recall@K: each labelled span must fit wholly within one retrieved chunk; labels exhaustive only for these authored facts.",
              "limitations": ["Synthetic corpus, authored labels, small held-out set; no business quality or answer accuracy claim.",
                              "cl100k_base is a reference tokenizer, not the provider billing tokenizer.",
                              "No chat/reranker run. ANN/FTS times use cached query vectors and do not represent end-to-end latency.",
                              "Shared text cache reduces experiment API cost; per-strategy embedded tokens still count all chunk text.",
                              "Build times depend on shared cache/order and are not uncached production-ingestion latency.",
                              "No automatic production parameter selection from the synthetic scores."],
              "cleanup": "pending"}
    if args.resume_from:
        previous = json.loads(args.resume_from.read_text(encoding="utf-8"))
        if (previous["corpus_manifest_sha256"] != report["corpus_manifest_sha256"] or
                previous["embedding_model"] != report["embedding_model"] or
                (previous.get("embedding_identity") and previous["embedding_identity"] != provider.identity)):
            raise ValueError("续跑语料或 Embedding 身份不匹配")
        for file, digest in previous["application_sha256"].items():
            if file != "scripts/benchmark_chunking.py" and report["application_sha256"].get(file) != digest:
                raise ValueError(f"续跑核心代码已变更: {file}")
        valid = {name(policy): policy.manifest() for policy in policies()}
        for key, item in previous["strategies"].items():
            if (valid.get(key) == item["policy"] and item["overall"]["failed_questions"] == 0 and
                    {row["id"] for row in item["queries"]} == {row["id"] for row in questions}):
                report["strategies"][key] = item
        report["resumed_from"] = {"sha256": hashlib.sha256(args.resume_from.read_bytes()).hexdigest(),
                                  "source_application_sha256": previous["application_sha256"],
                                  "source_embedding": previous["embedding"],
                                  "reused_strategies": list(report["strategies"]),
                                  "note": "Current embedding usage excludes earlier requests; source logs remain separate."}

    def save():
        report["embedding"] = provider.summary()
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        with scratch(settings) as (sessions, store, isolated):
            report["scratch_collection_prefix"] = isolated.milvus_collection_prefix
            report["scratch_database"] = sessions.kw["bind"].url.database
            save()
            with sessions.begin() as db:
                db.add(Corpus(id="benchmark", owner_id="benchmark", name="Fictional chunk experiment"))
                db.flush()
                for doc_id, (entry, _blocks, _heights) in documents.items():
                    db.add(Paper(id=doc_id, corpus_id="benchmark", title=doc_id))
                    db.flush()
                    db.add(PaperVersion(id=doc_id, paper_id=doc_id, filename=entry["filename"],
                                        storage_key=entry["filename"], sha256=entry["sha256"], byte_size=1,
                                        status="ready", is_active=True))
                    db.flush()
                    db.add(ParseRevision(id=doc_id, paper_version_id=doc_id, parser="frozen-fixture",
                                         config_version="benchmark-v1", status="ready"))
                db.flush()
                db.add_all(blocks)
            # Prewarm each query once so strategy order does not affect latency.
            embed_chunks([{"text": row["question"]} for row in questions], provider, settings.embedding_model)
            for policy in policies():
                if name(policy) in report["strategies"]:
                    continue
                print(f"Evaluating {name(policy)}", flush=True)
                report["strategies"][name(policy)] = evaluate_strategy(
                    sessions, store, isolated, provider, policy, documents, questions)
                save()
                # Avoid keeping every experimental collection loaded in memory.
                for collection in store.client.list_collections():
                    if collection.startswith(isolated.milvus_collection_prefix + "_"):
                        store.client.drop_collection(collection_name=collection)
        report["cleanup"] = "passed"
        report["status"] = ("passed" if all(item["overall"]["failed_questions"] == 0
                                             for item in report["strategies"].values()) else "query_failures")
    except Exception as exc:
        report["status"] = "failed"
        report["error_type"] = type(exc).__name__
        report["error_code"] = getattr(exc, "code", None)
        raise
    finally:
        save()
    for strategy, result in report["strategies"].items():
        print(strategy, json.dumps({"chunks": result["chunks"], **result["overall"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "fixtures/chunking/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/chunking-benchmark.json")
    parser.add_argument("--max-embedding-tokens", type=int, default=600000)
    parser.add_argument("--max-embedding-texts", type=int, default=3000)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "tmp/chunking-embedding-cache")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args, args.output)


if __name__ == "__main__":
    main()
