"""Compare identical source labels before/after terminology expansion.

Default: offline lexical control only. --live: real configured Embedding plus
temporary PostgreSQL/Milvus, no chat/reranker calls, at most 100 unique texts.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.chunking import token_count  # noqa: E402
from app.config import Settings  # noqa: E402
from app.ingestion import index_config  # noqa: E402
from app.llama_rag import embed_chunks, index_chunks  # noqa: E402
from app.models import Chunk, Corpus, IndexRevision, Paper, PaperVersion, ParseRevision, Snapshot, SourceBlock  # noqa: E402
from app.query_rewrite import rewrite_queries  # noqa: E402
from app.retrieval import hybrid_search, query_terms, rrf  # noqa: E402
from sqlalchemy import func, update  # noqa: E402
from sqlalchemy.engine import make_url  # noqa: E402

from scripts.benchmark_chunking import CachedEmbedding, scratch  # noqa: E402


def fixture():
    path = ROOT / "fixtures/query_rewrite/control.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data["fictional"] and not data["human_reviewed"]
    assert len(data["cases"]) <= 10 and len(data["distractors"]) <= 20
    texts = {c["id"]: c["text"] for c in data["cases"]}
    texts.update({f"distractor-{i}": t for i, t in enumerate(data["distractors"])})
    rules = [{"corpus_ids": [data["corpus_id"]], "term": c["term"], "aliases": [c["alias"]]} for c in data["cases"]]
    return data, texts, rules, hashlib.sha256(raw).hexdigest()


def compare(search, data, rules):
    rows = []
    for case in data["cases"]:
        variants, records = rewrite_queries(case["question"], [{"id": "q1", "target": case["question"],
            "queries": [case["question"]]}], data["corpus_id"], rules)
        row = {"case_id": case["id"], "question": case["question"], "rewrites": records}
        for mode, queries in (("original", [case["question"]]), ("expanded", variants[0]["queries"])):
            try:
                rankings = [search(query) for query in queries]
                top = [r["chunk_id"] for r in rrf(rankings)[:5]]
                row[mode] = {"status": "passed", "top5": top,
                             "evidence_recall_at_5": float(case["id"] in top)}
            except Exception as exc:
                row[mode] = {"status": "failed", "error_type": type(exc).__name__,
                             "top5": [], "evidence_recall_at_5": 0.}
        rows.append(row)
    return rows


def live_search(settings, texts, callback):
    # scratch() uses an administrative connection; bound it as well as normal
    # application connections so an unavailable local database cannot hang QA.
    settings = settings.model_copy(update={"database_url": make_url(settings.database_url).update_query_dict(
        {"connect_timeout": "5"}).render_as_string(hide_password=False)})
    provider = CachedEmbedding(settings, max_tokens=20000, max_texts=100)
    with scratch(settings) as (sessions, store, isolated):
        with sessions.begin() as db:
            db.add(Corpus(id="control", owner_id="control", name="Fictional rewrite control"))
            db.flush()
            db.add(Paper(id="control", corpus_id="control", title="Fictional rewrite control"))
            db.flush()
            db.add(PaperVersion(id="control", paper_id="control", sha256="a"*64, storage_key="control.md",
                filename="control.md", byte_size=sum(map(len, texts.values())), status="ready"))
            db.flush()
            db.add(ParseRevision(id="control", paper_version_id="control", parser="fictional-control",
                                 config_version="v1", status="ready"))
            db.flush()
            for i, (key, value) in enumerate(texts.items()):
                db.add(SourceBlock(id=key, parse_revision_id="control", page_index=0, block_index=i,
                                   bbox=[], text=value, section=key))
        vectors = embed_chunks([{"text": t} for t in texts.values()], provider, settings.embedding_model)
        from app.vector_store import collection_name
        with sessions.begin() as db:
            index = IndexRevision(id="control", parse_revision_id="control", config_version=index_config(settings),
                embedding_model=settings.embedding_model, embedding_dimensions=len(vectors[0]), vector_backend="milvus",
                vector_collection=collection_name("control", isolated.milvus_collection_prefix), status="ready")
            db.add(index)
            db.flush()
            chunks = [Chunk(id=key, index_revision_id="control", text=t, source_ranges=[{
                "block_id": key, "char_start": 0, "char_end": len(t)}]) for key, t in texts.items()]
            db.add_all(chunks)
            db.flush()
            index_chunks(db, index, chunks, vectors, store, provider)
            db.execute(update(Chunk).values(search_vector=func.to_tsvector("simple", Chunk.text)))
            db.add(Snapshot(id="control", owner_id="control", corpus_id="control", manifest=[{
                "index_revision_id": "control", "parse_revision_id": "control"}]))
        def search(query):
            with sessions() as db:
                return [r["chunk_id"] for r in hybrid_search(db, db.get(Snapshot, "control"), query, provider,
                    settings.embedding_model, limit=12, expected_index_config=index_config(settings), vector_store=store)]
        rows = callback(search)
        return rows, provider.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data, texts, rules, sha = fixture()
    report = {"scope": "fictional terminology control", "human_reviewed": False, "fixture_sha256": sha,
        "mode": "live_hybrid" if args.live else "offline_lexical_control", "status": "running",
        "limitations": ["Agent-authored fictional controls are not independent business-quality acceptance.",
                        "Only deterministic terminology expansion is compared, not Planner semantic quality.",
                        "Offline scores are lexical-control results, not Milvus/embedding Recall@5."]}
    output = args.output or ROOT / "docs" / ("query-rewrite-live-evaluation.json" if args.live else
                                              "query-rewrite-offline-evaluation.json")
    try:
        if args.live:
            rows, usage = live_search(Settings(provider_max_retries=0), texts,
                                     lambda search: compare(search, data, rules))
            report.update(embedding_usage=usage, cleanup="passed")
        else:
            def search(query):
                scores = [(key, sum(text.casefold().count(t.casefold()) for t in query_terms(query)))
                          for key, text in texts.items()]
                return [key for key, score in sorted(scores, key=lambda x: (-x[1], x[0])) if score > 0][:12]
            rows = compare(search, data, rules)
        report.update(status="completed", rows=rows, summary={mode: {
            "questions": len(rows), "failed": sum(r[mode]["status"] != "passed" for r in rows),
            "mean_evidence_recall_at_5": sum(r[mode]["evidence_recall_at_5"] for r in rows) / len(rows),
            "mean_top5_reference_tokens": sum(sum(token_count(texts[k]) for k in r[mode]["top5"]) for r in rows) / len(rows),
        } for mode in ("original", "expanded")})
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error_code=getattr(exc, "code", None))
        raise
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
