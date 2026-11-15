"""Compare local BGE against saved LLM reranker outputs without paid requests."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_cases(results, suite):
    result_rows = {}
    for path in sorted((results / "full_agent").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["question_id"] in result_rows:
            raise ValueError("Duplicate saved question result")
        row["source_result_sha256"] = sha256(path)
        result_rows[row["question_id"]] = row
    cases, excluded = [], []
    for question in suite["questions"]:
        question_id = question["question_id"]
        row = result_rows.get(question_id)
        if row is None:
            excluded.append({"question_id": question_id, "reason": "missing_saved_result"})
            continue
        trace = row.get("trace", {})
        rerank_calls = [call for call in trace.get("model_calls", []) if call.get("role") == "Reranker"]
        if not rerank_calls:
            excluded.append({"question_id": question_id, "reason": "reranker_not_called"})
            continue
        if len(rerank_calls) != 1:
            raise ValueError(f"Expected one saved reranker request for {question_id}")
        call = rerank_calls[0]
        request = call.get("request", {}).get("data", {})
        candidates = request.get("candidates", [])
        if not candidates:
            raise ValueError(f"Saved reranker candidates are missing for {question_id}")
        chunk_documents = {}
        for retrieval in trace.get("retrievals", []):
            for ranking in retrieval.get("rankings", []):
                if ranking.get("paper_version_id"):
                    chunk_documents[ranking["chunk_id"]] = ranking["paper_version_id"]
        candidate_ids, candidate_docs, model_candidates = [], {}, []
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            document_id = chunk_documents.get(chunk_id)
            if document_id is None:
                raise ValueError(f"No saved document mapping for candidate {chunk_id}")
            candidate_ids.append(chunk_id)
            candidate_docs[chunk_id] = document_id
            model_candidates.append({"id": chunk_id, "text": "\n".join(
                block["text"] for block in candidate.get("blocks", []))})
        llm_order = [item["chunk_id"] for item in call.get("response", {}).get("items", [])]
        gold_documents = sorted({evidence["document"] for claim in question.get("required_claims", [])
                                 for evidence in claim.get("evidence", [])})
        query = request.get("question", "")
        targets = list(dict.fromkeys(item["target"].strip() for item in request.get("plan", {}).get(
            "subquestions", []) if isinstance(item.get("target"), str) and item["target"].strip()))
        if targets:
            query += "\n\n问题要点：\n" + "\n".join(f"- {target}" for target in targets)
        cases.append({"id": question_id, "question": query, "candidates": model_candidates,
                      "candidate_documents": candidate_docs, "retrieval_order": candidate_ids,
                      "llm_order": llm_order, "gold_documents": gold_documents,
                      "source_result_sha256": row["source_result_sha256"]})
    return cases, excluded


def ranking_metrics(order, candidate_documents, gold_documents, k=3):
    relevant = set(gold_documents)
    if not relevant:
        return None
    ranked_documents = []
    for chunk_id in order:
        document_id = candidate_documents.get(chunk_id)
        if document_id and document_id not in ranked_documents:
            ranked_documents.append(document_id)
    hits = [index + 1 for index, document_id in enumerate(ranked_documents) if document_id in relevant]
    recall = len(set(ranked_documents[:k]) & relevant) / len(relevant)
    reciprocal_rank = 1 / hits[0] if hits else 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank in hits if rank <= k)
    ideal_count = min(len(relevant), k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {"recall_at_3": recall, "mrr": reciprocal_rank, "ndcg_at_3": dcg / ideal if ideal else 0.0}


def summarize(rows, method):
    metrics = [row[method] for row in rows if row[method] is not None]
    if not metrics:
        return {"questions": 0, "recall_at_3": None, "mrr": None, "ndcg_at_3": None}
    return {"questions": len(metrics), **{key: round(sum(item[key] for item in metrics) / len(metrics), 6)
                                         for key in ("recall_at_3", "mrr", "ndcg_at_3")}}


def run(args):
    results = args.results.resolve()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    cases, excluded = saved_cases(results, suite)
    if not cases:
        raise ValueError("No saved reranker requests to compare")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_path, score_path = output / "bge-input.jsonl", output / "bge-scores.jsonl"
    with input_path.open("x", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps({key: case[key] for key in ("id", "question", "candidates")},
                                    ensure_ascii=False) + "\n")
    command = [str(args.python.resolve()), str(ROOT / "scripts/local_rerank.py"), str(input_path),
               str(score_path), "--model", str(args.model.resolve()), "--device", args.device,
               "--batch-size", str(args.batch_size)]
    subprocess.run(command, cwd=ROOT, check=True)
    score_rows = [json.loads(line) for line in score_path.read_text(encoding="utf-8").splitlines() if line]
    scores = {row["id"]: row for row in score_rows}
    if set(scores) != {case["id"] for case in cases}:
        raise ValueError("BGE output does not match the saved request set")
    rows = []
    for case in cases:
        bge_order = [item["id"] for item in scores[case["id"]]["scores"]]
        rows.append({"question_id": case["id"], "gold_documents": case["gold_documents"],
                     "source_result_sha256": case["source_result_sha256"],
                     "candidate_count": len(case["candidates"]),
                     "retrieval": ranking_metrics(case["retrieval_order"], case["candidate_documents"],
                                                   case["gold_documents"]),
                     "llm_reranker": ranking_metrics(case["llm_order"], case["candidate_documents"],
                                                      case["gold_documents"]),
                     "bge_reranker": ranking_metrics(bge_order, case["candidate_documents"],
                                                      case["gold_documents"]),
                     "bge": {key: scores[case["id"]][key] for key in
                             ("seconds", "truncated_pairs", "device", "torch", "transformers")}})
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": "offline replay of saved candidates",
              "new_paid_requests": 0, "suite_sha256": sha256(args.suite),
              "source_manifest_sha256": sha256(results / "manifest.json"),
              "model_manifest_sha256": sha256(args.model / "download-manifest.json"),
              "model": "BAAI/bge-reranker-v2-m3", "model_revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
              "ranking_unit": "unique source document, first occurrence in ranked chunks",
              "metrics": {name: summarize(rows, name) for name in ("retrieval", "llm_reranker", "bge_reranker")},
              "excluded": excluded, "per_question": rows,
              "limitations": ["Small fixed candidate set; offline ranking quality only.",
                              "Does not measure end-to-end answer quality or production latency."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "evaluation/results/public-3yuan-v1")
    parser.add_argument("--suite", type=Path, default=ROOT / "evaluation/public_docs_v1/suite.json")
    parser.add_argument("--model", type=Path, default=ROOT / "tmp/rag-models/bge-reranker-v2-m3")
    parser.add_argument("--python", type=Path, required=True, help="Python executable in the separate inference environment")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
