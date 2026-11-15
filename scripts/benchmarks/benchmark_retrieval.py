"""Freeze shared candidates and compare retrieval routes, chat reranking and local CrossEncoder."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.agents.prompts import RERANKER  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.models import Chunk, Corpus, Paper, PaperVersion, ParseRevision, Snapshot  # noqa: E402
from app.core.schemas import Reranking  # noqa: E402
from app.core.security import redact  # noqa: E402
from app.integrations.providers import ModelProvider  # noqa: E402
from app.knowledge.chunking import ChunkPolicy, token_count  # noqa: E402
from app.knowledge.ingestion import index_config  # noqa: E402
from app.knowledge.retrieval import collapse_duplicate_hits, retrieve_rankings  # noqa: E402

from evaluation.retrieval_profiles import PROFILES, query_route, rank_profile  # noqa: E402
from scripts.benchmarks.benchmark_chunking import (  # noqa: E402
    CachedEmbedding,
    aggregate,
    evaluate_strategy,
    load_corpus,
    score_query,
    scratch,
)


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def summarize(rows):
    return {"overall": aggregate(rows),
            "development": aggregate([r for r in rows if r["split"] == "development"]),
            "held_out": aggregate([r for r in rows if r["split"] == "held_out"])}


def prepare(args):
    settings = Settings(provider_max_retries=0)
    manifest, documents, questions, blocks = load_corpus(args.manifest)
    provider = CachedEmbedding(settings, max_tokens=100000, max_texts=200,
                               cache_dir=ROOT / "tmp/chunking-embedding-cache")
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "status": "preparing", "fictional": True,
              "human_reviewed": False, "scope": "retrieval only; no answer correctness score",
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "profiles": {}, "cases": [], "embedding_model": settings.embedding_model,
              "selection_policy": "Production RRF/chat reranker unchanged; experimental rules fixed before results",
              "policy": ChunkPolicy().manifest(), "limits": {"max_candidates": 12, "queries": len(questions)}}
    write(args.output, report)
    try:
        with scratch(settings) as (sessions, store, isolated):
            report["scratch_database"] = sessions.kw["bind"].url.database
            report["scratch_prefix"] = isolated.milvus_collection_prefix
            write(args.output, report)
            with sessions.begin() as db:
                db.add(Corpus(id="benchmark", owner_id="benchmark", name="Retrieval control experiment"))
                db.flush()
                for key, (entry, _, _) in documents.items():
                    db.add(Paper(id=key, corpus_id="benchmark", title=key))
                    db.flush()
                    db.add(PaperVersion(id=key, paper_id=key, filename=entry["filename"], storage_key=entry["filename"],
                                        sha256=entry["sha256"], byte_size=1, status="ready", is_active=True))
                    db.flush()
                    db.add(ParseRevision(id=key, paper_version_id=key, parser="frozen-fixture",
                                         config_version="retrieval-experiment-v1", status="ready"))
                db.flush()
                db.add_all(blocks)
            evaluate_strategy(sessions, store, isolated, provider, ChunkPolicy(), documents, questions)
            block_map = {b.id: b for b in blocks}
            rows_by_profile = {profile: [] for profile in PROFILES}
            for question in questions:
                with sessions() as db:
                    snapshot = db.get(Snapshot, "fixed_512_64")
                    keyword, vector, degradation = retrieve_rankings(db, snapshot, question["question"], provider,
                        isolated.embedding_model, index_config(isolated), store)
                    if degradation:
                        raise ValueError("Degraded retrieval cannot be compared as two-route recall")
                    chunks = {c.id: c for c in db.scalars(select_chunks(keyword, vector))}
                    rankings = {}
                    for profile in PROFILES:
                        ids = rank_profile(question["question"], keyword, vector, profile)
                        dedup = collapse_duplicate_hits(db, [{"chunk_id": key} for key in ids])
                        ids = [hit["chunk_id"] for hit in dedup]
                        rankings[profile] = ids
                        rows_by_profile[profile].append({"id": question["id"], "split": question["split"],
                            "status": "passed", **score_query([chunks[key] for key in ids[:5]], question["gold"])})
                    candidates = []
                    for key in rankings["rrf"][:12]:
                        chunk = chunks[key]
                        raw = [{**span, "text": block_map[span["block_id"]].text[span["char_start"]:span["char_end"]]}
                               for span in chunk.source_ranges]
                        candidates.append({"id": key, "text": "\n\n".join(b["text"] for b in raw),
                                           "blocks": raw, "source_ranges": chunk.source_ranges})
                    report["cases"].append({"id": question["id"], "split": question["split"],
                        "question": question["question"], "gold": question["gold"], "query_type": query_route(question["question"])[0],
                        "keyword": keyword, "vector": vector, "rankings": rankings, "candidates": candidates})
                    print("prepared", question["id"], flush=True)
            report["profiles"] = {key: {**summarize(rows), "queries": rows} for key, rows in rows_by_profile.items()}
        report["cleanup"], report["status"] = "passed", "candidates_ready"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error_code=getattr(exc, "code", None))
        raise
    finally:
        report["embedding"] = provider.summary()
        write(args.output, report)
    candidates = args.output.with_suffix(".candidates.jsonl")
    candidates.write_text("".join(json.dumps(case, ensure_ascii=False) + "\n" for case in report["cases"]), encoding="utf-8")
    report["candidates_sha256"] = hashlib.sha256(candidates.read_bytes()).hexdigest()
    write(args.output, report)


def select_chunks(keyword, vector):
    from sqlalchemy import select
    return select(Chunk).where(Chunk.id.in_(set(keyword + vector)))


def rank_rows(case, ids):
    chunks = {c["id"]: SimpleNamespace(text=c["text"], source_ranges=c["source_ranges"]) for c in case["candidates"]}
    if len(ids) != len(set(ids)) or not set(ids) <= set(chunks):
        raise ValueError("Reranker returned duplicate/unknown candidate IDs")
    return {"id": case["id"], "split": case["split"], "status": "passed",
            **score_query([chunks[key] for key in ids[:5]], case["gold"])}


def rerank(args):
    report = json.loads(args.output.read_text(encoding="utf-8"))
    candidate_path = args.output.with_suffix(".candidates.jsonl")
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest() != report["candidates_sha256"]:
        raise ValueError("Frozen candidates changed")
    if args.stage == "local":
        output = args.output.with_suffix(".local.jsonl")
        command = [str(ROOT / "tmp/rag-reranker-env/Scripts/python.exe"), str(ROOT / "scripts/models/local_rerank.py"),
                   str(candidate_path), str(output), "--model", str(ROOT / "tmp/rag-models/bge-reranker-v2-m3")]
        subprocess.run(command, check=True, timeout=3600)
        raw = {r["id"]: r for r in map(json.loads, output.read_text(encoding="utf-8").splitlines())}
        rows = [{**rank_rows(case, [s["id"] for s in raw[case["id"]]["scores"]]), **{
            "seconds": raw[case["id"]]["seconds"], "truncated_pairs": raw[case["id"]]["truncated_pairs"]}}
                for case in report["cases"]]
        model_manifest = ROOT / "tmp/rag-models/bge-reranker-v2-m3/download-manifest.json"
        report["reranker_model"] = json.loads(model_manifest.read_text(encoding="utf-8"))
        key = "local_bge_v2_m3"
    else:
        settings = Settings(provider_max_retries=0, model_timeout_seconds=60, max_output_tokens=2500)
        provider, rows, sent_tokens = ModelProvider(settings), [], 0
        for case in report["cases"]:
            data = {"question": case["question"], "plan": {"subquestions": [{"id": "q1", "target": case["question"]}]},
                    "candidates": [{"chunk_id": c["id"], "blocks": c["blocks"]} for c in case["candidates"]]}
            sent_tokens += token_count(RERANKER + json.dumps(data, ensure_ascii=False))
            if sent_tokens > 180000:
                raise ValueError("Chat experiment input cap reached")
            started = time.monotonic()
            try:
                result, usage = provider.structured("Reranker", RERANKER, data, Reranking)
                ids, seen = [], set()
                candidates = {c["id"]: c for c in case["candidates"]}
                for item in sorted(result.items, key=lambda x: -x.score):
                    if item.chunk_id in seen or item.chunk_id not in candidates:
                        raise ValueError("Unknown/repeated chat candidate")
                    seen.add(item.chunk_id)
                    valid = {b["block_id"] for b in candidates[item.chunk_id]["blocks"]}
                    if not set(item.relevant_block_ids) <= valid:
                        raise ValueError("Unknown chat block")
                    if item.score > 0 and item.relevant_block_ids:
                        ids.append(item.chunk_id)
                row = {**rank_rows(case, ids), "usage": usage, "selected": ids}
            except Exception as exc:
                row = {**rank_rows(case, []), "status": "failed", "error_type": type(exc).__name__,
                       "message": redact(str(exc), settings), "usage": provider.last_usage}
            row["seconds"] = round(time.monotonic() - started, 4)
            rows.append(row)
            report.setdefault("chat_partial", []).append(row)
            write(args.output, report)
            print("chat", case["id"], row["status"], row["seconds"], flush=True)
        report["chat_model"], report["chat_reference_input_tokens"] = settings.model, sent_tokens
        key = "chat_reranker"
    report["profiles"][key] = {**summarize(rows), "queries": rows,
        "latency_p50": sorted(r["seconds"] for r in rows)[math.ceil(.5 * len(rows)) - 1],
        "latency_p95": sorted(r["seconds"] for r in rows)[math.ceil(.95 * len(rows)) - 1]}
    complete = all(k in report["profiles"] for k in ("local_bge_v2_m3", "chat_reranker"))
    failures = sum(row["status"] != "passed" for profile in report["profiles"].values() for row in profile["queries"])
    report["failed_queries"] = failures
    report["status"] = ("completed_with_failures" if failures else "passed") if complete else "partial"
    report["limitations"] = ["15 authored synthetic queries, not independent business gold; no answer quality claims",
        "Same capped RRF pool for both rerankers; missing pool evidence cannot be recovered",
        "Chat relevance filtering vs CrossEncoder top5; no production replacement or calibrated score threshold",
        "Route comparisons share both recalls; no route-specific end-to-end latency claim",
        "CrossEncoder uses CPU batch2, maximum1024 tokens; truncation recorded; latency excludes model loading"]
    write(args.output, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "local", "chat"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "fixtures/chunking/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tmp/reports/retrieval-benchmark.json")
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        rerank(args)


if __name__ == "__main__":
    main()
