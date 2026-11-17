"""Prepare and score a local wide-recall/BGE/forward-context experiment.

Only frozen SQLite and cached query vectors are accepted. Private request text
and per-question reports are confined to the ignored tmp tree. Inference runs
separately with scripts/models/local_rerank.py and never receives gold labels.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

import numpy as np  # noqa: E402
from app.core.models import Chunk, Snapshot, SourceBlock  # noqa: E402
from app.knowledge.chunking import token_count  # noqa: E402
from app.knowledge.retrieval import collapse_duplicate_hits, rrf  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from evaluation.context_recall import (  # noqa: E402
    WideCandidates,
    source_successors,
    validated_order,
    with_forward_context,
)
from scripts.benchmarks.benchmark_lexical_recall import score, validate_scope  # noqa: E402
from scripts.models.local_rerank import request_digest  # noqa: E402

CUTOFFS = (5, 12, 30)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def private_output(path):
    if not path.resolve().is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("Private results must stay inside the ignored tmp tree")


def corpus(database, suite):
    if not database.is_file():
        raise ValueError("Existing local database is required")
    ids = [q["id"] for q in suite["questions"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be nonempty and unique")
    engine = create_engine("sqlite:///file:" + database.resolve().as_posix() + "?mode=ro&uri=true")
    with Session(engine) as db:
        snapshot = db.get(Snapshot, suite["snapshot_id"])
        if snapshot is None:
            raise ValueError("Missing frozen snapshot")
        scope, indexes = validate_scope(db, snapshot, suite["questions"])
        chunks = {c.id: c for c in db.scalars(select(Chunk).where(
            Chunk.index_revision_id.in_(scope)).order_by(Chunk.id))}
        if not chunks or len(chunks) > 5000 or sum(len(c.text) for c in chunks.values()) > 2_000_000:
            raise ValueError("Experiment snapshot exceeds bounded corpus size")
        block_ids = {span["block_id"] for c in chunks.values() for span in c.source_ranges}
        blocks = {b.id: b for b in db.scalars(select(SourceBlock).where(SourceBlock.id.in_(block_ids)))}
        successors = source_successors(chunks, blocks, {i.id: i.parse_revision_id for i in indexes})
        dimensions = {i.embedding_dimensions for i in indexes}
        if len(dimensions) != 1:
            raise ValueError("Frozen index dimensions differ")
    return engine, chunks, successors, dimensions.pop()


def check_model(model):
    manifest = read(model / "download-manifest.json")
    for filename, expected in manifest["files"].items():
        path = model / filename
        if (not path.resolve().is_relative_to(model.resolve()) or not path.is_file() or
                path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]):
            raise ValueError("Local model files differ from the frozen manifest")
    return {"manifest_sha256": sha(model / "download-manifest.json"),
            "name": manifest["model"], "revision": manifest["revision"]}


def prepare(args):
    private_output(args.output)
    if args.output.exists():
        raise ValueError("Use a new output directory; prepared runs are immutable")
    started = time.monotonic()
    suite, cache = read(args.suite), read(args.query_vectors)
    model = check_model(args.model)
    engine, chunks, _successors, dimension = corpus(args.database, suite)
    keys = list(chunks)
    matrix = np.asarray([chunks[key].embedding for key in keys], dtype=np.float32)
    if matrix.shape != (len(keys), dimension) or not np.isfinite(matrix).all():
        raise ValueError("Stored embeddings differ in dimension or contain nonfinite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Stored embeddings must be finite and nonzero")
    matrix /= norms
    ranker = WideCandidates({key: chunk.text for key, chunk in chunks.items()})
    requests, selections = [], []
    for question in suite["questions"]:
        query = question["question"]
        vector = cache["vectors"].get(hashlib.sha256(query.encode("utf-8")).hexdigest())
        if vector is None:
            raise ValueError("Missing cached query; paid fallback is forbidden")
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (dimension,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Cached query vector must be finite, nonzero and dimension-compatible")
        sims = matrix @ (vector / norm)
        dense = [keys[i] for i in sorted(range(len(keys)), key=lambda i: (-float(sims[i]), keys[i]))]
        routes, combined = ranker.rankings(query, dense, limit=100)
        request = {"id": question["id"], "question": query,
                   "candidates": [{"id": key, "text": chunks[key].text} for key in combined]}
        baseline = [hit["chunk_id"] for hit in rrf([routes["window"][:30], routes["dense"][:30]])]
        requests.append(request)
        selections.append({"id": question["id"], "request_sha256": request_digest(request),
                           "baseline_ranking": baseline, "routes": routes})
    args.output.mkdir(parents=True)
    with (args.output / "requests.jsonl").open("x", encoding="utf-8") as stream:
        for request in requests:
            stream.write(json.dumps(request, ensure_ascii=False) + "\n")
    manifest = {"schema": 1, "database": str(args.database.resolve()), "database_sha256": sha(args.database),
                "suite": str(args.suite.resolve()), "suite_sha256": sha(args.suite),
                "snapshot_id": suite["snapshot_id"], "query_vectors_sha256": sha(args.query_vectors),
                "query_vector_identity": cache["identity"], "model": model,
                "requests_sha256": sha(args.output / "requests.jsonl"),
                "parameters": {"route_limit": 100, "final_limit": 12, "forward_seed_count": 6,
                               "han_ngram_sizes": [1, 2], "window_target_chars": 160,
                               "rrf_k": 60}, "questions": selections,
                "prepare_seconds": round(time.monotonic() - started, 3), "paid_calls": 0}
    write(args.output / "manifest.json", manifest)
    engine.dispose()
    print(json.dumps({"prepared": str(args.output), "questions": len(requests),
                      "pairs": sum(len(r["candidates"]) for r in requests),
                      "seconds": manifest["prepare_seconds"]}, ensure_ascii=False))


def summarize(cases, method):
    return {str(k): {"covered": sum(c["methods"][method][str(k)]["covered"] for c in cases),
                     "gold": sum(c["gold"] for c in cases),
                     "recall": sum(c["methods"][method][str(k)]["covered"] for c in cases) /
                     sum(c["gold"] for c in cases),
                     "complete_questions": sum(c["methods"][method][str(k)]["complete"] for c in cases),
                     "mean_reference_tokens": round(sum(c["methods"][method][str(k)]["tokens"]
                                                        for c in cases) / len(cases), 2)} for k in CUTOFFS}


def score_run(args):
    private_output(args.run)
    private_output(args.scores)
    if (args.run / "report.json").exists():
        raise ValueError("Existing reports are immutable")
    manifest = read(args.run / "manifest.json")
    database, suite_path = Path(manifest["database"]), Path(manifest["suite"])
    if (sha(database) != manifest["database_sha256"] or sha(suite_path) != manifest["suite_sha256"] or
            sha(args.run / "requests.jsonl") != manifest["requests_sha256"]):
        raise ValueError("Frozen source inputs have drifted")
    suite = read(suite_path)
    engine, chunks, successors, _dimension = corpus(database, suite)
    requests = [json.loads(line) for line in (args.run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    results = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in results}
    expected = [q["id"] for q in suite["questions"]]
    if (len(by_id) != len(results) or set(by_id) != set(expected) or
            [r["id"] for r in requests] != expected or [r["id"] for r in manifest["questions"]] != expected):
        raise ValueError("Every question, including failures, must have exactly one inference result")
    cases = []
    tokens = {key: token_count(c.text) for key, c in chunks.items()}
    with Session(engine) as db:
        for q, request, saved in zip(suite["questions"], requests, manifest["questions"]):
            result = by_id[q["id"]]
            if result.get("model_manifest_sha256") != manifest["model"]["manifest_sha256"]:
                raise ValueError("Inference model manifest differs")
            for candidate in request["candidates"]:
                if candidate["id"] not in chunks or candidate["text"] != chunks[candidate["id"]].text:
                    raise ValueError("Candidate text or scope differs from frozen source")
            ranking = validated_order(result, request, saved["request_sha256"])
            rankings = {"window_rrf": saved["baseline_ranking"], "wide_bge": ranking,
                        "wide_bge_forward": with_forward_context(ranking, successors, chunks, seed_count=6)}
            row = {"id": q["id"], "gold": len(q["gold"]), "candidates": len(ranking),
                   "candidate_pool_ceiling": score(ranking, chunks, q["gold"])["covered"],
                   "inference_seconds": result["seconds"], "truncated_pairs": result["truncated_pairs"],
                   "methods": {}, "rankings": {}}
            for name, order in rankings.items():
                if not set(order) <= set(chunks):
                    raise ValueError("Ranking is outside the frozen snapshot")
                order = [hit["chunk_id"] for hit in collapse_duplicate_hits(db, [{"chunk_id": k} for k in order])]
                row["methods"][name] = {str(k): {**score(order[:k], chunks, q["gold"]),
                                               "tokens": sum(tokens[key] for key in order[:k])} for k in CUTOFFS}
                row["rankings"][name] = order[:30]
            cases.append(row)
    comparisons = {}
    for name in ("wide_bge", "wide_bge_forward"):
        pairs = [(row["methods"][name]["12"], row["methods"]["window_rrf"]["12"]) for row in cases]
        comparisons[name] = {"improved_questions": sum(a["covered"] > b["covered"] for a, b in pairs),
                             "regressed_questions": sum(a["covered"] < b["covered"] for a, b in pairs),
                             "new_labels": sum(sum(x and not y for x, y in zip(a["found"], b["found"]))
                                               for a, b in pairs),
                             "lost_labels": sum(sum(y and not x for x, y in zip(a["found"], b["found"]))
                                                for a, b in pairs)}
    total = sum(c["gold"] for c in cases)
    covered = sum(c["methods"]["wide_bge_forward"]["12"]["covered"] for c in cases)
    report = {"status": "completed", "scope": "frozen SQLite development diagnostic",
              "human_reviewed": False, "question_count": len(cases), "gold_count": total,
              "suite_sha256": manifest["suite_sha256"], "database_sha256": manifest["database_sha256"],
              "requests_sha256": manifest["requests_sha256"], "scores_sha256": sha(args.scores),
              "model": manifest["model"], "parameters": manifest["parameters"],
              "inference": {key: results[0].get(key) for key in (
                  "device", "precision", "torch", "transformers", "max_length", "load_seconds")},
              "summary": {name: summarize(cases, name) for name in ("window_rrf", "wide_bge", "wide_bge_forward")},
              "comparison_at_12": comparisons, "candidate_pool_covered": sum(c["candidate_pool_ceiling"] for c in cases),
              "inference_seconds": round(sum(row["seconds"] for row in results), 3),
              "truncated_pairs": sum(row["truncated_pairs"] for row in results),
              "threshold": {"target": .8, "covered": covered, "total": total, "met": covered / total >= .8},
              "new_paid_model_requests": 0, "new_model_cost_cny": "0", "cases": cases,
              "limitations": ["Parameters were explored on these development questions; no independent test set.",
                              "Exact source-range coverage is not answer accuracy or independent human review.",
                              "One successor consumes one of the same final 12 original chunk slots.",
                              "Runtime web defaults and PostgreSQL/Milvus have not been changed or revalidated.",
                              "A forward neighbor can be irrelevant or cross a section boundary within the same document."]}
    write(args.run / "report.json", report)
    engine.dispose()
    print(json.dumps({"report": str(args.run / "report.json"), "summary": report["summary"],
                      "comparison_at_12": comparisons, "threshold": report["threshold"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("database", "suite", "query-vectors", "model", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--run", type=Path, required=True)
    scoring.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else score_run(args)


if __name__ == "__main__":
    main()
