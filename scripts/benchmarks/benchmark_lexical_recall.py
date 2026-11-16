"""Offline retrieval experiments on a frozen, local SQLite corpus.

Requires a local suite of exact source offsets and saved route rankings. Does
not load credentials, instantiate providers, or generate missing embeddings.
All source text and question-level outputs must remain in the ignored tmp/ tree.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.core.models import Chunk, IndexRevision, Snapshot, SourceBlock  # noqa: E402
from app.knowledge.chunking import token_count  # noqa: E402
from app.knowledge.lexical import LexicalBM25, lexical_windows  # noqa: E402
from app.knowledge.retrieval import collapse_duplicate_hits, rrf  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session, defer  # noqa: E402

from evaluation.lexical_recall import query_clauses, round_robin  # noqa: E402

CUTOFFS = (5, 12, 20, 30)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fuse(rankings):
    return [hit["chunk_id"] for hit in rrf(rankings)]


def score(ranking, chunks, gold):
    covered = [any(any(span["block_id"] == label["block_id"] and
                       span["char_start"] <= label["char_start"] and
                       span["char_end"] >= label["char_end"]
                       for span in chunks[key].source_ranges) for key in ranking) for label in gold]
    return {"gold": len(gold), "covered": sum(covered), "complete": all(covered), "found": covered}


def summarize(rows):
    total = sum(row["5"]["gold"] for row in rows)
    return {str(k): {"queries": len(rows), "gold": total,
                     "covered": sum(row[str(k)]["covered"] for row in rows),
                     "recall": sum(row[str(k)]["covered"] for row in rows) / total,
                     "complete_queries": sum(row[str(k)]["complete"] for row in rows),
                     "mean_context_reference_tokens": round(sum(row[str(k)]["context_reference_tokens"]
                                                                 for row in rows) / len(rows), 1)}
            for k in CUTOFFS}


def validate_scope(db, snapshot, questions):
    index_ids = {member["index_revision_id"] for member in snapshot.manifest}
    indexes = list(db.scalars(select(IndexRevision).where(IndexRevision.id.in_(index_ids))))
    if len(indexes) != len(index_ids) or any(index.status != "ready" for index in indexes):
        raise ValueError("Frozen snapshot has unavailable indexes")
    parse_ids = {index.parse_revision_id for index in indexes}
    block_ids = {label["block_id"] for question in questions for label in question["gold"]}
    blocks = {block.id: block for block in db.scalars(select(SourceBlock).where(SourceBlock.id.in_(block_ids)))}
    for question in questions:
        if not question["gold"] or not question["question"].strip() or len(question["question"]) > 500:
            raise ValueError("Nonempty bounded questions and gold labels are required")
        for label in question["gold"]:
            block = blocks.get(label["block_id"])
            start, end = label["char_start"], label["char_end"]
            if (block is None or block.parse_revision_id not in parse_ids or
                    type(start) is not int or type(end) is not int or not 0 <= start < end <= len(block.text)):
                raise ValueError("Evidence label is invalid or outside the frozen snapshot")
    return index_ids, indexes


def clause_rankings(db, index_ids, chunk_ids, questions, cache_path, dimensions):
    if cache_path is None:
        return None
    # numpy is already a LlamaIndex dependency. Local cosine only; never infer a
    # missing vector, nor silently compare different dimensions or zero vectors.
    import numpy as np

    cache = read_json(cache_path)
    if len(dimensions) != 1:
        raise ValueError("Frozen index dimensions differ")
    dimension = next(iter(dimensions))
    matrix = np.zeros((len(chunk_ids), dimension), dtype=np.float64)
    positions = {key: i for i, key in enumerate(chunk_ids)}
    query = select(Chunk.id, Chunk.embedding).where(Chunk.index_revision_id.in_(index_ids))
    for key, vector in db.execute(query):
        if len(vector) != dimension:
            raise ValueError("Stored vector dimensions differ")
        matrix[positions[key]] = vector
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Stored vectors must be finite and nonzero")
    matrix /= norms
    output = {}
    for question in questions:
        rankings = []
        for clause in query_clauses(question["question"]):
            key = hashlib.sha256(clause.encode("utf-8")).hexdigest()
            if key not in cache:
                raise ValueError("Missing cached clause embedding; paid fallback is forbidden")
            vector = np.asarray(cache[key], dtype=np.float64)
            if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.linalg.norm(vector) > 0:
                raise ValueError("Invalid cached clause embedding")
            similarities = matrix @ (vector / np.linalg.norm(vector))
            order = sorted(range(len(chunk_ids)), key=lambda i: (-similarities[i], chunk_ids[i]))[:30]
            rankings.append([chunk_ids[i] for i in order])
        output[question["id"]] = rankings
    return output


def run(args):
    if args.output.exists():
        raise ValueError("Use a new output path; prior runs are immutable")
    if not args.output.resolve().is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Question-level reports must remain in the local tmp tree")
    suite, saved = read_json(args.suite), read_json(args.saved_rankings)
    questions = suite["questions"]
    ids = [question["id"] for question in questions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Suite question IDs must be unique and nonempty")
    if (saved["status"] != "completed" or suite["snapshot_id"] != saved["snapshot_id"] or
            suite["original_suite_sha256"] != saved["question_set_sha256"]):
        raise ValueError("Saved baseline and local suite identities differ")
    baseline = {case["id"]: case for case in saved["cases"]}
    if set(baseline) != set(ids):
        raise ValueError("Every baseline question must be evaluated, including failures")
    if not args.database.is_file():
        raise ValueError("Existing local database is required")
    url = "sqlite:///file:" + args.database.resolve().as_posix() + "?mode=ro&uri=true"
    engine = create_engine(url)
    started = time.monotonic()
    with Session(engine) as db:
        snapshot = db.get(Snapshot, suite["snapshot_id"])
        if snapshot is None:
            raise ValueError("Frozen snapshot is missing")
        index_ids, indexes = validate_scope(db, snapshot, questions)
        chunks = {chunk.id: chunk for chunk in db.scalars(select(Chunk).where(
            Chunk.index_revision_id.in_(index_ids)).options(defer(Chunk.embedding)))}
        bm25 = LexicalBM25({key: chunk.text for key, chunk in chunks.items()})
        children, parents = {}, {}
        for key, chunk in chunks.items():
            for number, text in enumerate(lexical_windows(chunk.text)):
                child_id = f"{key}:{number}"
                children[child_id], parents[child_id] = text, key
        small_bm25 = LexicalBM25(children)
        dense_clauses = clause_rankings(db, index_ids, list(chunks), questions, args.clause_vectors,
                                       {index.embedding_dimensions for index in indexes})
        metrics, cases = defaultdict(list), []
        token_counts = {key: token_count(chunk.text) for key, chunk in chunks.items()}
        oracle_total = 0
        for question in questions:
            qid, query, gold = question["id"], question["question"], question["gold"]
            old = baseline[qid]["profiles"]
            dense, original = old["vector"]["top_30"], old["rrf"]["top_30"]
            if (not set(dense + original) <= set(chunks) or
                    len(dense) != len(set(dense)) or len(original) != len(set(original))):
                raise ValueError("Saved candidates are duplicated or outside the frozen snapshot")
            lexical = bm25.search(query)
            hybrid = fuse([lexical, dense])
            lexical_clauses = [bm25.search(clause) for clause in query_clauses(query)]
            small = list(dict.fromkeys(parents[key] for key in small_bm25.search(query, len(children))))[:30]
            rankings = {"baseline_rrf": original, "ngram_bm25": lexical, "ngram_hybrid": hybrid,
                        "balanced_clauses": round_robin([hybrid, *lexical_clauses]),
                        "small_to_parent_hybrid": fuse([small, dense])}
            if dense_clauses is not None:
                rankings["multiquery_hybrid"] = fuse([lexical, dense, *lexical_clauses, *dense_clauses[qid]])
            oracle = score(list(chunks), chunks, gold)
            oracle_total += oracle["covered"]
            pool = list(dict.fromkeys(lexical + dense))
            case = {"id": qid, "oracle": oracle, "bm25_vector_pool": {
                        "candidates": len(pool), **score(pool, chunks, gold)}, "methods": {}}
            for method, ranking in rankings.items():
                deduped = collapse_duplicate_hits(db, [{"chunk_id": key} for key in ranking])
                ranking = [hit["chunk_id"] for hit in deduped]
                scores = {str(k): {**score(ranking[:k], chunks, gold),
                                   "context_reference_tokens": sum(token_counts[key] for key in ranking[:k])}
                          for k in CUTOFFS}
                metrics[method].append(scores)
                case["methods"][method] = {"top30": ranking[:30], "scores": scores}
            cases.append(case)
    comparisons = {}
    for name, rows in metrics.items():
        comparisons[name] = {}
        for k in CUTOFFS:
            key = str(k)
            pairs = [(row[key], base[key]) for row, base in zip(rows, metrics["baseline_rrf"])]
            comparisons[name][key] = {
                "improved_queries": sum(row["covered"] > base["covered"] for row, base in pairs),
                "regressed_queries": sum(row["covered"] < base["covered"] for row, base in pairs),
                "unchanged_queries": sum(row["covered"] == base["covered"] for row, base in pairs),
                "newly_covered_labels": sum(sum(a and not b for a, b in zip(row["found"], base["found"]))
                                            for row, base in pairs),
                "lost_labels": sum(sum(b and not a for a, b in zip(row["found"], base["found"]))
                                   for row, base in pairs),
            }
    result = {"status": "completed", "human_reviewed": False, "scope": "offline development diagnostic",
              "inputs": {"suite_sha256": sha(args.suite), "saved_rankings_sha256": sha(args.saved_rankings),
                         "clause_vectors_sha256": sha(args.clause_vectors) if args.clause_vectors else None,
                         "snapshot_id": suite["snapshot_id"]},
              "parameters": {"bm25_k1": 1.2, "bm25_b": .75, "han_ngram_sizes": [1, 2],
                             "window_target_chars": 160, "long_unit_threshold_chars": 200,
                             "long_unit_stride_chars": 130,
                             "route_candidates": 30, "rrf_k": 60},
              "oracle_coverable_labels": oracle_total, "chunk_count": len(chunks),
              "bm25_vector_pool_ceiling": {
                  "covered_labels": sum(case["bm25_vector_pool"]["covered"] for case in cases),
                  "complete_queries": sum(case["bm25_vector_pool"]["complete"] for case in cases),
                  "maximum_candidates": max(case["bm25_vector_pool"]["candidates"] for case in cases),
                  "interpretation": "Oracle bound for this candidate pool, not a reranker result or answer accuracy"},
              "summary": {name: summarize(rows) for name, rows in metrics.items()},
              "paired_comparisons": comparisons, "cases": cases,
              "new_model_calls": 0, "new_model_cost_cny": "0",
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "limitations": ["Development queries were already inspected; there is no unseen test set.",
                              "SQLite snapshot and cached vectors, not PostgreSQL/Milvus or answer accuracy.",
                              "Clause embeddings were prepared separately under the corpus budget; this replay is offline.",
                              "No resolver for pronouns, synonyms or cross-document discourse; all regressions retained.",
                              "BM25 and window indexes are in-memory experimental indexes, not production features."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "summary": result["summary"],
                      "paired_comparisons": comparisons}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--saved-rankings", type=Path, required=True)
    parser.add_argument("--clause-vectors", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
