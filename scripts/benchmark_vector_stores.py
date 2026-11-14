"""Compare PostgreSQL/pgvector and Milvus on one isolated synthetic corpus.

This is an engineering benchmark, not a production quality claim. Both stores
receive identical normalized vectors, queries, tenant filter and exact Python
ground truth. Temporary table/collection names are removed in ``finally``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from uuid import uuid4

from sqlalchemy import create_engine, text


def normalized(values: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in values))
    return [value / length for value in values]


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def summary(samples: list[float]) -> dict:
    return {
        "n": len(samples),
        "p50_ms": percentile(samples, .50),
        "p95_ms": percentile(samples, .95),
        "max_ms": max(samples) if samples else None,
        "mean_ms": statistics.mean(samples) if samples else None,
    }


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.9f}" for value in vector) + "]"


def exact_top(rows: list[tuple[str, str, list[float]]], query: list[float], scope: str, k: int) -> list[str]:
    scored = [(chunk_id, sum(a * b for a, b in zip(vector, query)))
              for chunk_id, row_scope, vector in rows if row_scope == scope]
    return [chunk_id for chunk_id, _score in sorted(scored, key=lambda item: (-item[1], item[0]))[:k]]


def recall(actual: list[str], expected: list[str]) -> float:
    return len(set(actual) & set(expected)) / len(expected) if expected else 1.0


def run(args: argparse.Namespace) -> dict:
    from app.config import get_settings
    from app.vector_store import MilvusVectorStore

    settings = get_settings()
    rng = Random(args.seed)
    rows = [(f"chunk-{index:05d}", "tenant-a" if index % 2 else "tenant-b",
             normalized([rng.uniform(-1, 1) for _ in range(args.dimension)]))
            for index in range(args.vectors)]
    queries = [normalized([rng.uniform(-1, 1) for _ in range(args.dimension)])
               for _ in range(args.queries)]
    scope = "tenant-a"
    allowed_ids = {chunk_id for chunk_id, row_scope, _vector in rows if row_scope == scope}
    expected = [exact_top(rows, query, scope, args.k) for query in queries]
    suffix = uuid4().hex[:12]
    table = f"vector_bench_{suffix}"
    collection = f"vector_bench_{suffix}"
    engine = create_engine(settings.database_url, future=True)
    store = None
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scope": "isolated synthetic vectors; exact Python ground truth; no chat or embedding API",
        "parameters": {"vectors": args.vectors, "queries": args.queries, "dimension": args.dimension,
                       "k": args.k, "repeats": args.repeats, "seed": args.seed, "filter": scope},
        "services": {},
    }
    try:
        with engine.begin() as connection:
            if not connection.execute(text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'")).scalar_one_or_none():
                raise RuntimeError("pgvector extension must already be installed")
            connection.execute(text(
                f"CREATE TABLE {table} (chunk_id text PRIMARY KEY, scope text NOT NULL, "
                f"embedding vector({args.dimension}) NOT NULL)"))
            connection.execute(text(
                f"CREATE INDEX {table}_hnsw ON {table} USING hnsw (embedding vector_cosine_ops)"))
        pg_insert_started = time.perf_counter()
        with engine.begin() as connection:
            connection.exec_driver_sql(
                f"INSERT INTO {table} (chunk_id, scope, embedding) VALUES (%s, %s, %s)",
                [(chunk_id, row_scope, vector_literal(vector)) for chunk_id, row_scope, vector in rows],
            )
        pg_insert_ms = (time.perf_counter() - pg_insert_started) * 1000
        pg_samples, pg_hits, pg_filter_leaks = [], [], 0
        query_sql = text(f"SELECT chunk_id FROM {table} WHERE scope = :scope "
                         "ORDER BY embedding <=> CAST(:vector AS vector) LIMIT :limit")
        with engine.connect() as connection:
            for query_index, query in enumerate(queries):
                for repeat in range(args.repeats):
                    started = time.perf_counter()
                    result = connection.execute(query_sql, {"scope": scope, "vector": vector_literal(query),
                                                            "limit": args.k}).scalars().all()
                    pg_samples.append((time.perf_counter() - started) * 1000)
                    pg_filter_leaks += sum(chunk_id not in allowed_ids for chunk_id in result)
                    if repeat == args.repeats - 1:
                        pg_hits.append(list(result))
        updated = normalized([0.9] + [0.0] * (args.dimension - 1))
        pg_update_started = time.perf_counter()
        with engine.begin() as connection:
            connection.execute(text(f"UPDATE {table} SET embedding = CAST(:vector AS vector) WHERE chunk_id = :id"),
                               {"vector": vector_literal(updated), "id": rows[0][0]})
        pg_update_ms = (time.perf_counter() - pg_update_started) * 1000
        deleted = rows[1][0]
        pg_delete_started = time.perf_counter()
        with engine.begin() as connection:
            connection.execute(text(f"DELETE FROM {table} WHERE chunk_id = :id"), {"id": deleted})
        pg_delete_ms = (time.perf_counter() - pg_delete_started) * 1000
        with engine.connect() as connection:
            updated_distance = connection.execute(text(
                f"SELECT embedding <=> CAST(:vector AS vector) FROM {table} WHERE chunk_id = :id"),
                {"vector": vector_literal(updated), "id": rows[0][0]}).scalar_one_or_none()
            deleted_exists = connection.execute(text(
                f"SELECT EXISTS (SELECT 1 FROM {table} WHERE chunk_id = :id)"), {"id": deleted}).scalar_one()
        report["services"]["postgresql_pgvector"] = {
            "query": summary(pg_samples),
            "insert_and_commit_ms": pg_insert_ms,
            "update_and_commit_ms": pg_update_ms,
            "delete_and_commit_ms": pg_delete_ms,
            "recall_at_k": statistics.mean(recall(actual, wanted) for actual, wanted in zip(pg_hits, expected)),
            "filtered_count": len(rows) // 2,
            "filter_leak_count": pg_filter_leaks,
            "update_delete_check": updated_distance is not None and updated_distance < 1e-6 and not deleted_exists,
        }

        store = MilvusVectorStore(settings)
        client = store.client
        from pymilvus import DataType

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("scope", DataType.VARCHAR, max_length=32)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=args.dimension)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="HNSW", metric_type="COSINE",
                               params={"M": 16, "efConstruction": 200})
        client.create_collection(collection_name=collection, schema=schema, index_params=index_params)
        client.load_collection(collection_name=collection)
        milvus_insert_started = time.perf_counter()
        client.upsert(collection_name=collection,
                      data=[{"chunk_id": chunk_id, "scope": row_scope, "embedding": vector}
                            for chunk_id, row_scope, vector in rows])
        client.flush(collection_name=collection)
        milvus_insert_ms = (time.perf_counter() - milvus_insert_started) * 1000
        milvus_samples, milvus_hits, milvus_filter_leaks = [], [], 0
        for query_index, query in enumerate(queries):
            for repeat in range(args.repeats):
                started = time.perf_counter()
                result = client.search(collection_name=collection, data=[query], anns_field="embedding",
                                       limit=args.k, output_fields=["chunk_id"],
                                       filter=f'scope == "{scope}"',
                                       search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                                       consistency_level="Strong")
                milvus_samples.append((time.perf_counter() - started) * 1000)
                if repeat == args.repeats - 1:
                    hits = [hit.get("entity", {}).get("chunk_id") or hit.get("id")
                            for hit in (result[0] if result else [])]
                    milvus_hits.append(hits)
                milvus_filter_leaks += sum(
                    (hit.get("entity", {}).get("chunk_id") or hit.get("id")) not in allowed_ids
                    for hit in (result[0] if result else []))
        milvus_update_started = time.perf_counter()
        client.upsert(collection_name=collection,
                      data=[{"chunk_id": rows[0][0], "scope": rows[0][1], "embedding": updated}])
        client.flush(collection_name=collection)
        milvus_update_ms = (time.perf_counter() - milvus_update_started) * 1000
        milvus_delete_started = time.perf_counter()
        client.delete(collection_name=collection, ids=[rows[1][0]])
        client.flush(collection_name=collection)
        milvus_delete_ms = (time.perf_counter() - milvus_delete_started) * 1000
        check = client.query(collection_name=collection, filter=f'chunk_id in ["{rows[0][0]}", "{rows[1][0]}"]',
                             output_fields=["chunk_id"], consistency_level="Strong")
        check_ids = {row["chunk_id"] for row in check}
        report["services"]["milvus"] = {
            "query": summary(milvus_samples),
            "insert_and_flush_ms": milvus_insert_ms,
            "update_and_flush_ms": milvus_update_ms,
            "delete_and_flush_ms": milvus_delete_ms,
            "recall_at_k": statistics.mean(recall(actual, wanted) for actual, wanted in zip(milvus_hits, expected)),
            "filtered_count": len(rows) // 2,
            "filter_leak_count": milvus_filter_leaks,
            "update_delete_check": rows[0][0] in check_ids and rows[1][0] not in check_ids,
        }
        checks = [entry["filter_leak_count"] == 0 and entry["update_delete_check"]
                  for entry in report["services"].values()]
        report["status"] = "passed" if all(checks) else "failed"
        return report
    finally:
        try:
            if store is not None and store.client.has_collection(collection_name=collection):
                store.client.drop_collection(collection_name=collection)
        finally:
            if store is not None:
                store.close()
            with engine.begin() as connection:
                connection.execute(text(f"DROP TABLE IF EXISTS {table}"))
            engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=int, default=256)
    parser.add_argument("--queries", type=int, default=24)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--output", default="docs/vector-store-benchmark.json")
    args = parser.parse_args()
    if min(args.vectors, args.queries, args.dimension, args.k, args.repeats) <= 0:
        parser.error("benchmark sizes must be positive")
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
