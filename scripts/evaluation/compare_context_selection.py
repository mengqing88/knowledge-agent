"""Replay three context strategies on frozen candidates, without model or database calls."""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "backend")]

from app.knowledge.chunking import token_count  # noqa: E402

METHODS = ("selected_blocks", "whole_chunks", "neighbor_blocks")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def load_inputs(results, corpus):
    manifest = read_json(results / "manifest.json")
    suite_path, corpus_path = corpus / "suite.json", corpus / "corpus.json"
    if digest(suite_path) != manifest["question_set_sha256"]:
        raise ValueError("Frozen suite hash mismatch")
    if digest(corpus_path) != manifest["corpus_manifest_sha256"]:
        raise ValueError("Frozen corpus hash mismatch")
    suite = read_json(suite_path)
    if [q["question_id"] for q in suite["questions"]] != [q["question_id"] for q in manifest["questions"]]:
        raise ValueError("Frozen question identities mismatch")
    documents = {}
    for doc in read_json(corpus_path)["documents"]:
        path = corpus / doc["filename"]
        if not path.resolve().is_relative_to(corpus.resolve()) or digest(path) != doc["sha256"]:
            raise ValueError("Frozen document hash/path mismatch")
        documents[doc["id"]] = path.read_bytes().decode("utf-8")
    questions = {q["question_id"]: q for q in suite["questions"]}
    records = {}
    for path in sorted((results / "full_agent").glob("*.json")):
        row = read_json(path)
        qid = row["question_id"]
        if qid not in questions or qid in records:
            raise ValueError("Unknown or duplicate saved question")
        records[qid] = (row, digest(path))
    for question in questions.values():
        for point in question.get("required_claims", []):
            for evidence in point["evidence"]:
                if normalize(evidence["quote"]) not in normalize(documents[evidence["document"]]):
                    raise ValueError("Gold quotation absent from frozen document")
    return manifest, suite, documents, records


def saved_selection(row, documents):
    calls = row.get("trace", {}).get("model_calls", [])
    rerank = [call for call in calls if call.get("role") == "Reranker"]
    if not rerank:
        return None
    if len(rerank) != 1:
        raise ValueError("Replay requires exactly one saved reranking round per case")
    call = rerank[0]
    request = call.get("request", {}).get("data", {})
    candidates = request.get("candidates")
    response = call.get("response", {})
    if not isinstance(candidates, list) or not candidates or not isinstance(response.get("items"), list):
        raise ValueError("Saved reranking request/response is missing")
    mapping = {}
    for retrieval in row["trace"].get("retrievals", []):
        for item in retrieval["rankings"]:
            if item.get("paper_version_id"):
                previous = mapping.setdefault(item["chunk_id"], item["paper_version_id"])
                if previous != item["paper_version_id"]:
                    raise ValueError("Ambiguous saved chunk/document mapping")
    by_id = {}
    for candidate in candidates:
        cid = candidate["chunk_id"]
        if cid in by_id or cid not in mapping:
            raise ValueError("Duplicate or unmapped candidate")
        document = mapping[cid]
        # Every alternative uses only existing, scope-bound source ranges from the same request.
        for block in candidate["blocks"]:
            if (type(block["char_start"]) is not int or type(block["char_end"]) is not int
                    or not 0 <= block["char_start"] < block["char_end"]
                    or block["char_end"] - block["char_start"] != len(block["text"])
                    or block["text"] not in documents[document]):
                raise ValueError("Saved candidate is not an exact frozen source range")
        by_id[cid] = {**candidate, "document": document}
    selected, seen = [], set()
    for item in sorted(response["items"], key=lambda r: -r["score"]):
        cid = item["chunk_id"]
        if cid not in by_id or cid in seen:
            raise ValueError("Unknown or duplicate selected chunk")
        seen.add(cid)
        candidate = by_id[cid]
        ids = set(item["relevant_block_ids"])
        if not ids <= {b["block_id"] for b in candidate["blocks"]}:
            raise ValueError("Selected block does not belong to candidate")
        if item["score"] > 0 and ids:
            selected.append({**candidate, "selected_block_ids": ids})
    return selected


def select_context(candidates, method, max_chars, max_sources=32, per_document=16):
    if method not in METHODS:
        raise ValueError("Unknown context strategy")
    spans, counts, size, seen = [], {}, 0, set()
    for candidate in candidates:
        blocks = candidate["blocks"]
        chosen = {i for i, block in enumerate(blocks) if block["block_id"] in candidate["selected_block_ids"]}
        if method == "whole_chunks":
            chosen = set(range(len(blocks)))
        elif method == "neighbor_blocks":
            chosen = {j for i in chosen for j in (i - 1, i, i + 1) if 0 <= j < len(blocks)}
        for i, block in enumerate(blocks):
            if i not in chosen:
                continue
            document = candidate["document"]
            identity = (document, block["block_id"], block["char_start"], block["char_end"])
            if identity in seen:
                continue
            if (counts.get(document, 0) >= per_document or len(spans) >= max_sources
                    or size + len(block["text"]) > max_chars):
                continue
            spans.append({**block, "document": document})
            seen.add(identity)
            counts[document] = counts.get(document, 0) + 1
            size += len(block["text"])
    return spans


def compare_saved_gate(row, spans):
    calls = [call for call in row.get("trace", {}).get("model_calls", []) if call.get("role") == "EvidenceGate"]
    if not calls and not spans:
        answer = row.get("run", {}).get("answer") or {}
        if (answer.get("evidence_status") == "insufficient" and not answer.get("evidence")
                and answer.get("stop_reason") == "evidence_gate_insufficient"):
            return "empty_context_gate_short_circuit"
    if len(calls) != 1:
        raise ValueError("Current replay cannot be verified against a single saved EvidenceGate request")
    sources = calls[0]["request"]["data"]["sources"]
    actual = [(s["document"], s["block_id"], s["char_start"], s["char_end"], s["text"]) for s in spans]
    saved = [(s["paper_version_id"], s["block_id"], s["char_start"], s["char_end"], s["quote"]) for s in sources]
    if actual != saved:
        raise ValueError("Current-method replay differs from the historical production context")
    return "saved_gate_request_equal"


def evidence_metrics(question, spans):
    normalized = [(s["document"], normalize(s["text"])) for s in spans]
    gold = []
    for point in question.get("required_claims", []):
        covered = []
        for citation in point["evidence"]:
            hit = any(document == citation["document"] and normalize(citation["quote"]) in text
                      for document, text in normalized)
            covered.append(hit)
            gold.append({"required_key": point["key"], "document": citation["document"],
                         "quote": citation["quote"], "covered": hit})
        if not covered:
            raise ValueError("An evidence group cannot have no quotations")
    groups = {point["key"]: all(g["covered"] for g in gold if g["required_key"] == point["key"])
              for point in question.get("required_claims", [])}
    return {"gold_quotes": len(gold), "covered_gold_quotes": sum(g["covered"] for g in gold),
            "gold_quote_recall": sum(g["covered"] for g in gold) / len(gold) if gold else None,
            "all_gold_quotes_present": all(g["covered"] for g in gold) if gold else None,
            "gold_groups": len(groups), "covered_gold_groups": sum(groups.values()),
            "gold_group_recall": sum(groups.values()) / len(groups) if groups else None,
            "quotation_checks": gold,
            "source_count": len(spans), "quote_chars": sum(len(s["text"]) for s in spans),
            "quote_reference_tokens": sum(token_count(s["text"]) for s in spans)}


def aggregate(rows, method):
    replayed = [row["methods"][method] for row in rows if row["status"] == "replayed"]
    positive = [m for m in replayed if m["gold_quotes"]]
    no_gold = [m for m in replayed if not m["gold_quotes"]]
    return {"replayed_questions": len(replayed), "positive_label_questions": len(positive),
            "macro_gold_quote_recall": sum(m["gold_quote_recall"] for m in positive) / len(positive) if positive else None,
            "all_gold_quotes_present_questions": sum(m["all_gold_quotes_present"] for m in positive),
            "gold_quotes": sum(m["gold_quotes"] for m in positive),
            "covered_gold_quotes": sum(m["covered_gold_quotes"] for m in positive),
            "macro_gold_group_recall": sum(m["gold_group_recall"] for m in positive) / len(positive) if positive else None,
            "mean_positive_quote_reference_tokens": sum(m["quote_reference_tokens"] for m in positive) / len(positive)
                if positive else None,
            "mean_positive_source_count": sum(m["source_count"] for m in positive) / len(positive) if positive else None,
            "no_gold_label_questions": len(no_gold),
            "no_gold_nonempty_contexts": sum(bool(m["source_count"]) for m in no_gold)}


def run(results, corpus, protocol_path, output):
    protocol = read_json(protocol_path)
    if tuple(protocol["methods"]) != METHODS or protocol["version"] != 1:
        raise ValueError("Unsupported protocol; freeze changes under a new version")
    _, suite, documents, records = load_inputs(results, corpus)
    # Warm the already-installed reference tokenizer before measuring local assembly.
    token_count("local context replay")
    rows = []
    for question in suite["questions"]:
        qid = question["question_id"]
        row, result_hash = records.get(qid, ({}, None))
        selected = saved_selection(row, documents)
        result = {"question_id": qid, "category": question["category"],
                  "saved_answer_status": row.get("run", {}).get("status", "missing"),
                  "source_result_sha256": result_hash, "methods": {}}
        if selected is None:
            result.update(status="not_replayed", reason="no_saved_reranker_call")
            rows.append(result)
            continue
        limit = row["execution_manifest"]["max_context_chars"]
        result.update(status="replayed", max_context_chars=limit, selected_chunk_count=len(selected))
        for method in METHODS:
            started = time.perf_counter()
            spans = select_context(selected, method, limit, max_sources=protocol["limits"]["max_sources"],
                                   per_document=protocol["limits"]["max_sources_per_document"])
            elapsed = time.perf_counter() - started
            if method == "selected_blocks":
                result["current_context_validation"] = compare_saved_gate(row, spans)
            result["methods"][method] = {**evidence_metrics(question, spans),
                "local_selection_seconds": elapsed,
                "selected_ranges": [{k: span[k] for k in ("document", "block_id", "char_start", "char_end")}
                                    for span in spans]}
        rows.append(result)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "scope": protocol["scope"],
              "new_model_requests": 0, "new_paid_cost_cny": 0, "protocol_sha256": digest(protocol_path),
              "script_sha256": digest(Path(__file__)), "suite_sha256": digest(corpus / "suite.json"),
              "corpus_manifest_sha256": digest(corpus / "corpus.json"),
              "source_manifest_sha256": digest(results / "manifest.json"),
              "questions_in_suite": len(rows),
              "not_replayed": [r["question_id"] for r in rows if r["status"] != "replayed"],
              "summary": {method: aggregate(rows, method) for method in METHODS}, "per_question": rows,
              "production_default_changed": False,
              "limitations": protocol["constraints"] + [
                  "Gold quotation presence does not judge semantics, all required conditions, answer correctness or hallucinations.",
                  "Saved model calls are reused. No total latency or cost improvement is measured.",
                  "Current-method contexts are checked byte-for-byte against saved EvidenceGate inputs."]}
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(protocol_path.read_bytes())
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(output.resolve() / "report.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "evaluation/results/public-3yuan-v1")
    parser.add_argument("--corpus", type=Path, default=ROOT / "evaluation/public_docs_v1")
    parser.add_argument("--protocol", type=Path, default=ROOT / "evaluation/context_selection_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.results, args.corpus, args.protocol, args.output)


if __name__ == "__main__":
    main()
