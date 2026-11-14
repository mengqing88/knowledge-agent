"""Report objective counts. Semantic support and coverage require independent manual annotation."""
import argparse
import json
import math
from pathlib import Path

if __package__:
    from .review import modes_for, summarize_manual
else:
    from review import modes_for, summarize_manual

EXPECTED_ACTIONS = ("supported", "partial", "conflicted", "insufficient", "needs_clarification")


def summarize(directory: Path):
    counts = {"questions": 0, "completed": 0, "technical_failures": 0, "cancelled": 0,
              "claims": 0, "citations": 0, "empty_answers": 0, "uncited_factual_claims": 0,
              "model_calls": 0, "input_tokens": 0, "output_tokens": 0, "retrieval_requests": 0,
              "wall_seconds": 0., "expected_action_matches": 0, "expected_action_labeled_questions": 0,
              "refusal_metrics_note": "拒答统计仅比较预期与返回状态；内容正确性须经独立人工审核。",
              "answerable_questions": 0,
              "false_refusals": 0, "unanswerable_questions": 0, "correct_refusals": 0,
              "missing_records": 0, "missing_traces": 0, "missing_call_traces": 0,
              "unknown_usage_calls": 0, "unconfirmed_final_usage_runs": 0, "embedding_calls": 0,
              "top5_queries": 0, "top5_hits": 0, "top5_missing_queries": 0,
              "top5_scope": "first_query_first_round_first_attempt_per_labeled_question",
              "cost_known_runs": 0, "cost_unknown_runs": 0}
    manifest = json.loads((directory.parent / "manifest.json").read_text(encoding="utf-8"))
    gold = {q["question_id"]: q for q in manifest["questions"]}
    records = {}
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["question_id"]
        if key not in gold or key in records:
            raise ValueError("评测结果包含未知或重复 question_id")
        records[key] = row
    durations, currencies, price_sources = [], set(), set()
    for key, question in gold.items():
        row = records.get(key, {"technical_error": "missing_record", "wall_seconds": 0})
        counts["missing_records"] += key not in records
        counts["questions"] += 1
        counts["wall_seconds"] += row["wall_seconds"]
        run = row.get("run", {})
        final_usage_observed = run.get("status") == "completed" and not row.get("technical_error")
        counts["unconfirmed_final_usage_runs"] += not final_usage_observed
        if key in records:
            durations.append(row["wall_seconds"])
        expected = question.get("expected_action")
        counts["expected_action_labeled_questions"] += expected in EXPECTED_ACTIONS
        if expected == "insufficient":
            counts["unanswerable_questions"] += 1
        elif expected in ("supported", "partial", "conflicted"):
            counts["answerable_questions"] += 1
        if row.get("technical_error") or run.get("status") not in ("completed", "cancelled"):
            counts["technical_failures"] += 1
        elif run.get("status") == "cancelled":
            counts["cancelled"] += 1
        else:
            counts["completed"] += 1
            answer = run.get("answer") or {}
            claims = answer.get("claims", [])
            counts["claims"] += len(claims)
            counts["citations"] += sum(len(c["evidence"]) for c in claims)
            counts["empty_answers"] += not bool(claims)
            counts["uncited_factual_claims"] += sum(not any(e["relation"] == "supports" for e in c["evidence"])
                                                    for c in claims)
            actual = answer.get("evidence_status")
            counts["expected_action_matches"] += expected in EXPECTED_ACTIONS and actual == expected
            counts["correct_refusals"] += actual == expected == "insufficient"
            counts["false_refusals"] += expected in ("supported", "partial", "conflicted") and actual == "insufficient"
        trace = row.get("trace", {})
        retrievals = trace.get("retrievals", [])
        relevant_chunks = set(question.get("relevant_chunk_ids", []))
        # Fixed question denominator: do not reward more subqueries or retries.
        # This primary-query Hit@5 is deliberately separate from Recall@5.
        primary = next((item for item in retrievals if item.get("attempt", 1) == 1
                        and item.get("round", 1) == 1), None)
        if relevant_chunks:
            counts["top5_queries"] += 1
            counts["top5_missing_queries"] += primary is None
            top5 = {rank.get("chunk_id") for rank in (primary or {}).get("rankings", [])[:5]}
            counts["top5_hits"] += bool(top5 & relevant_chunks)
        calls_present = isinstance(trace.get("model_calls"), list)
        calls = trace.get("model_calls") if calls_present else []
        counts["missing_traces"] += "trace" not in row
        counts["missing_call_traces"] += not calls_present
        counts["model_calls"] += sum(isinstance(c, dict) and c.get("role") != "Embedding" for c in calls)
        counts["embedding_calls"] += sum(isinstance(c, dict) and c.get("role") == "Embedding" for c in calls)
        known = [c for c in calls if isinstance(c, dict) and c.get("usage_known") is True
                 and all(type(c.get(k)) is int and c[k] >= 0 for k in ("input_tokens", "output_tokens"))]
        counts["unknown_usage_calls"] += len(calls) - len(known)
        counts["input_tokens"] += sum(c.get("input_tokens") or 0 for c in known)
        counts["output_tokens"] += sum(c.get("output_tokens") or 0 for c in known)
        counts["retrieval_requests"] += len(retrievals)
        accounting = trace.get("accounting", {})
        cost = accounting.get("estimated_cost")
        if (final_usage_observed and calls_present and len(known) == len(calls)
                and accounting.get("cost_status") == "estimated" and type(cost) in (int, float)
                and math.isfinite(cost) and cost >= 0 and accounting.get("currency") and accounting.get("price_source")):
            counts["cost_known_runs"] += 1
            currencies.add(accounting["currency"])
            price_sources.add(accounting["price_source"])
            counts.setdefault("estimated_cost", 0.)
            counts["estimated_cost"] += cost
        else:
            counts["cost_unknown_runs"] += 1
    durations.sort()
    counts["wall_seconds_p50"] = durations[math.ceil(.50*len(durations))-1] if durations else None
    counts["wall_seconds_p95"] = durations[math.ceil(.95*len(durations))-1] if durations else None
    counts["wall_seconds_max"] = max(durations, default=None)
    counts["latency_observations"] = len(durations)
    counts["top5_hit_rate"] = (counts["top5_hits"] / counts["top5_queries"]
                               if counts["top5_queries"] else None)
    counts["token_totals_complete"] = (counts["unknown_usage_calls"] == 0 and counts["missing_records"] == 0
                                       and counts["missing_traces"] == 0 and counts["missing_call_traces"] == 0
                                       and counts["unconfirmed_final_usage_runs"] == 0)
    counts["cost_currency"] = next(iter(currencies)) if len(currencies) == 1 else None
    counts["cost_price_sources"] = sorted(price_sources)
    if counts.get("cost_unknown_runs") or not counts["cost_known_runs"] or len(currencies) != 1:
        counts["estimated_cost"] = None
    else:
        counts["estimated_cost"] = round(counts["estimated_cost"], 8)
    counts["technical_failure_rate"] = counts["technical_failures"] / counts["questions"] if counts["questions"] else None
    counts["false_refusal_rate"] = (counts["false_refusals"] / counts["answerable_questions"]
                                     if counts["answerable_questions"] else None)
    counts["correct_refusal_rate"] = (counts["correct_refusals"] / counts["unanswerable_questions"]
                                       if counts["unanswerable_questions"] else None)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--reviews", type=Path, help="Independent human annotations exported by review.py")
    parser.add_argument("--agent-reviews", type=Path, help="Agent annotations; never counted as human review")
    args = parser.parse_args()
    manifest = json.loads((args.run_directory / "manifest.json").read_text(encoding="utf-8"))
    modes = modes_for(manifest)
    results = {mode: summarize(args.run_directory / mode) for mode in modes}
    annotations = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else None
    results["manual_metrics"] = {mode: summarize_manual(args.run_directory, mode, annotations)
                                 for mode in modes}
    results["manual_metrics"]["note"] = "Only independent human annotations score semantics; pending metrics remain null"
    from evaluation.review import summarize_agent
    agent_annotations = json.loads(args.agent_reviews.read_text(encoding="utf-8")) if args.agent_reviews else None
    results["agent_metrics"] = {mode: summarize_agent(args.run_directory, mode, agent_annotations) for mode in modes}
    path = args.run_directory / "summary.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
