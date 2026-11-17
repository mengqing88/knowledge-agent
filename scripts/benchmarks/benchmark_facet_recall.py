"""Frozen, local facet reranking/context-budget comparison; private tmp only."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.knowledge.chunking import token_count  # noqa: E402

from evaluation.context_recall import query_facets, validated_order, with_facet_context  # noqa: E402
from scripts.benchmarks import benchmark_context_recall as base  # noqa: E402
from scripts.models.local_rerank import request_digest  # noqa: E402


def source_run(path):
    base.private_output(path)
    manifest = base.read(path / "manifest.json")
    report = base.read(path / "report.json")
    for filename, expected in (("requests.jsonl", manifest["requests_sha256"]),
                               ("scores.jsonl", report["scores_sha256"])):
        if base.sha(path / filename) != expected:
            raise ValueError("Frozen base inference has drifted")
    for field in ("database", "suite"):
        if base.sha(Path(manifest[field])) != manifest[field + "_sha256"]:
            raise ValueError("Frozen base source has drifted")
    requests = [json.loads(line) for line in (path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    scores = [json.loads(line) for line in (path / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = [q["id"] for q in base.read(Path(manifest["suite"]))["questions"]]
    if ([r["id"] for r in requests] != ids or len(scores) != len(ids) or
            {r["id"] for r in scores} != set(ids) or [r["id"] for r in manifest["questions"]] != ids or
            [r["id"] for r in report["cases"]] != ids):
        raise ValueError("Frozen base question membership differs")
    by_id = {r["id"]: r for r in scores}
    rankings = {}
    for request, saved in zip(requests, manifest["questions"]):
        result = by_id[request["id"]]
        if result.get("model_manifest_sha256") != manifest["model"]["manifest_sha256"]:
            raise ValueError("Base model identity differs")
        rankings[request["id"]] = validated_order(result, request, saved["request_sha256"])
    return manifest, report, requests, rankings


def prepare(args):
    base.private_output(args.output)
    if args.output.exists():
        raise ValueError("Use a new output directory; prepared runs are immutable")
    started = time.monotonic()
    manifest, _report, originals, _rankings = source_run(args.base_run)
    model = base.check_model(args.model)
    if model != manifest["model"]:
        raise ValueError("Facet model must match the frozen full-query model")
    requests, groups = [], []
    for original in originals:
        facets = []
        for number, query in enumerate(query_facets(original["question"])):
            row = {"id": f"{original['id']}/facet{number}", "question": query,
                   "candidates": original["candidates"]}
            requests.append(row)
            facets.append({"id": row["id"], "request_sha256": request_digest(row)})
        groups.append({"id": original["id"], "facets": facets})
    if not requests:
        raise ValueError("This experiment requires at least one multi-facet question")
    args.output.mkdir(parents=True)
    with (args.output / "requests.jsonl").open("x", encoding="utf-8") as output:
        for row in requests:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    base.write(args.output / "manifest.json", {
        "schema": 1, "base_run": str(args.base_run.resolve()),
        "base_manifest_sha256": base.sha(args.base_run / "manifest.json"),
        "base_report_sha256": base.sha(args.base_run / "report.json"),
        "requests_sha256": base.sha(args.output / "requests.jsonl"), "model": model,
        "implementation_sha256": {name: base.sha(ROOT / name) for name in (
            "evaluation/context_recall.py", "scripts/benchmarks/benchmark_facet_recall.py",
            "scripts/models/local_rerank.py")},
        "parameters": {"full_query_anchors": 5, "max_facets": 2, "final_limit": 12,
                       "extra_unseen_per_facet": 2, "max_forward_steps": 5},
        "questions": groups, "prepare_seconds": round(time.monotonic() - started, 3),
        "new_paid_model_requests": 0})
    print(json.dumps({"questions": len(groups), "facets": len(requests),
                      "pairs": sum(len(r["candidates"]) for r in requests)}, ensure_ascii=False))


def score_run(args):
    base.private_output(args.run)
    base.private_output(args.scores)
    if (args.run / "report.json").exists():
        raise ValueError("Existing reports are immutable")
    frozen = base.read(args.run / "manifest.json")
    source = Path(frozen["base_run"])
    if (base.sha(source / "manifest.json") != frozen["base_manifest_sha256"] or
            base.sha(source / "report.json") != frozen["base_report_sha256"] or
            base.sha(args.run / "requests.jsonl") != frozen["requests_sha256"] or
            any(base.sha(ROOT / name) != digest for name, digest in frozen["implementation_sha256"].items())):
        raise ValueError("Frozen facet inputs or implementation have drifted")
    manifest, previous, originals, full = source_run(source)
    if frozen["model"] != manifest["model"]:
        raise ValueError("Facet model differs from the base model")
    suite = base.read(Path(manifest["suite"]))
    engine, chunks, successors, _ = base.corpus(Path(manifest["database"]), suite)
    for original in originals:
        for candidate in original["candidates"]:
            if candidate["id"] not in chunks or candidate["text"] != chunks[candidate["id"]].text:
                raise ValueError("Candidate text or scope differs from the frozen source")
    requests = [json.loads(line) for line in (args.run / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    results = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines()]
    by_id = {r["id"]: r for r in results}
    expected = [facet["id"] for group in frozen["questions"] for facet in group["facets"]]
    if (len(by_id) != len(results) or set(by_id) != set(expected) or
            [r["id"] for r in requests] != expected or
            [g["id"] for g in frozen["questions"]] != [q["id"] for q in suite["questions"]]):
        raise ValueError("Every facet and original question must remain present exactly once")
    requests_by_id = {r["id"]: r for r in requests}
    tokens = {key: token_count(c.text) for key, c in chunks.items()}
    cases = []
    for question, original, group, old in zip(suite["questions"], originals, frozen["questions"], previous["cases"]):
        parts = []
        facets = query_facets(question["question"])
        if len(facets) != len(group["facets"]):
            raise ValueError("Frozen facet count differs")
        for query, facet in zip(facets, group["facets"]):
            request = requests_by_id[facet["id"]]
            result = by_id[facet["id"]]
            if (request["question"] != query or request["candidates"] != original["candidates"] or
                    request_digest(request) != facet["request_sha256"] or
                    result.get("model_manifest_sha256") != frozen["model"]["manifest_sha256"]):
                raise ValueError("Facet query, candidates or model identity differs")
            parts.append(validated_order(result, request, facet["request_sha256"]))
        ranking = with_facet_context(full[question["id"]], parts, successors, chunks)
        if len(ranking) != min(12, len(chunks)) or len(ranking) != len(set(ranking)):
            raise ValueError("Final selection changed the fixed result budget")
        scored = base.score(ranking, chunks, question["gold"])
        cases.append({"id": question["id"], "ranking": ranking, "metrics": scored,
                      "baseline": old["methods"]["wide_bge_forward"]["12"],
                      "reference_tokens": sum(tokens[key] for key in ranking)})
    covered = sum(c["metrics"]["covered"] for c in cases)
    total = sum(c["metrics"]["gold"] for c in cases)
    report = {"status": "completed", "scope": "frozen SQLite development diagnostic",
              "human_reviewed": False, "question_count": len(cases), "gold_count": total,
              "suite_sha256": manifest["suite_sha256"], "database_sha256": manifest["database_sha256"],
              "manifest_sha256": base.sha(args.run / "manifest.json"),
              "requests_sha256": frozen["requests_sha256"], "scores_sha256": base.sha(args.scores),
              "base_report_sha256": frozen["base_report_sha256"], "model": frozen["model"],
              "parameters": frozen["parameters"],
              "inference": {key: results[0].get(key) for key in (
                  "device", "precision", "torch", "transformers", "max_length", "load_seconds")},
              "summary_at_12": {"covered": covered, "gold": total, "recall": covered / total,
                                "complete_questions": sum(c["metrics"]["complete"] for c in cases),
                                "mean_reference_tokens": round(sum(c["reference_tokens"] for c in cases)/len(cases), 2)},
              "baseline_at_12": previous["summary"]["wide_bge_forward"]["12"],
              "comparison_at_12": {
                  "improved_questions": sum(c["metrics"]["covered"] > c["baseline"]["covered"] for c in cases),
                  "regressed_questions": sum(c["metrics"]["covered"] < c["baseline"]["covered"] for c in cases),
                  "new_labels": sum(sum(x and not y for x, y in zip(c["metrics"]["found"], c["baseline"]["found"]))
                                    for c in cases),
                  "lost_labels": sum(sum(y and not x for x, y in zip(c["metrics"]["found"], c["baseline"]["found"]))
                                     for c in cases)},
              "threshold": {"target": .9, "covered": covered, "total": total, "met": covered/total >= .9},
              "local_inference_pairs": sum(len(r["candidates"]) for r in requests),
              "facet_count": len(requests), "prepare_seconds": frozen["prepare_seconds"],
              "inference_seconds": round(sum(r["seconds"] for r in results), 3),
              "truncated_pairs": sum(r["truncated_pairs"] for r in results),
              "new_paid_model_requests": 0, "new_model_cost_cny": "0", "cases": cases,
              "limitations": ["Syntax and context rules were selected on this same development set.",
                              "Exact source coverage is not answer accuracy or independent human review.",
                              "Facet queries may lose pronouns; successors may cross same-document sections.",
                              "Runtime web defaults and PostgreSQL/Milvus were not changed or revalidated.",
                              "Local facet inference adds latency; no production performance claim."]}
    base.write(args.run / "report.json", report)
    engine.dispose()
    print(json.dumps({k: report[k] for k in ("summary_at_12", "comparison_at_12", "threshold",
                                           "local_inference_pairs", "inference_seconds")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for name in ("base-run", "model", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    scoring = commands.add_parser("score")
    for name in ("run", "scores"):
        scoring.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args) if args.command == "prepare" else score_run(args)


if __name__ == "__main__":
    main()
