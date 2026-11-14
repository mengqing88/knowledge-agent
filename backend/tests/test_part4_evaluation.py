import json

import httpx
import pytest

from evaluation.report import summarize
from evaluation.run import evaluate_case


def test_report_missing_cases_unknown_usage_and_cancelled_stay_in_denominator(tmp_path):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [
        {"question_id": "q1", "expected_action": "supported"},
        {"question_id": "q2", "expected_action": "insufficient"},
        {"question_id": "q3", "expected_action": "supported"}]}), encoding="utf-8")
    (directory / "0001.json").write_text(json.dumps({"question_id": "q1", "wall_seconds": 10,
        "run": {"status": "completed", "answer": {"evidence_status": "insufficient", "claims": []}},
        "trace": {"model_calls": [
            {"role": "Planner", "input_tokens": None, "output_tokens": None, "usage_known": False},
            {"role": "Embedding", "input_tokens": 10, "output_tokens": 0, "usage_known": True}]}}), encoding="utf-8")
    (directory / "0002.json").write_text(json.dumps({"question_id": "q2", "wall_seconds": 30,
        "run": {"status": "cancelled"}}), encoding="utf-8")
    result = summarize(directory)
    assert result["questions"] == 3 and result["technical_failures"] == 1
    assert result["cancelled"] == 1 and result["missing_records"] == 1
    assert result["false_refusal_rate"] == .5 and result["correct_refusal_rate"] == 0
    assert result["unknown_usage_calls"] == 1 and not result["token_totals_complete"]
    assert result["model_calls"] == result["embedding_calls"] == 1
    assert result["input_tokens"] == 10 and result["wall_seconds_p95"] == 30
    assert result["latency_observations"] == 2 and result["estimated_cost"] is None


def test_report_cannot_count_nonterminal_or_duplicate_cases_as_completed(tmp_path):
    directory = tmp_path / "baseline_rag"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [{"question_id": "q1"}]}), encoding="utf-8")
    row = json.dumps({"question_id": "q1", "wall_seconds": 1, "run": {"status": "running"}})
    (directory / "0001.json").write_text(row, encoding="utf-8")
    assert summarize(directory)["technical_failures"] == 1
    (directory / "duplicate.json").write_text(row, encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        summarize(directory)


def test_top5_uses_one_primary_query_and_keeps_missing_labeled_questions(tmp_path):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [
        {"question_id": "q1", "relevant_chunk_ids": ["gold"]},
        {"question_id": "q2", "relevant_chunk_ids": ["other-gold"]}]}), encoding="utf-8")
    rankings = [{"chunk_id": f"unrelated-{i}"} for i in range(5)] + [{"chunk_id": "gold"}]
    (directory / "0001.json").write_text(json.dumps({"question_id": "q1", "wall_seconds": 1,
        "run": {"status": "failed"}, "trace": {"retrievals": [
            {"attempt": 1, "round": 1, "rankings": rankings},
            {"attempt": 1, "round": 1, "rankings": [{"chunk_id": "gold"}]},
            {"attempt": 1, "round": 2, "rankings": [{"chunk_id": "gold"}]},
            {"attempt": 2, "round": 1, "rankings": [{"chunk_id": "gold"}]}]}}), encoding="utf-8")
    result = summarize(directory)
    assert result["top5_queries"] == 2 and result["top5_hits"] == 0
    assert result["top5_missing_queries"] == 1 and result["top5_hit_rate"] == 0


def test_cost_summary_cannot_add_different_currencies(tmp_path):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [
        {"question_id": "q1"}, {"question_id": "q2"}]}), encoding="utf-8")
    for index, currency in enumerate(("CNY", "USD"), 1):
        (directory / f"{index:04d}.json").write_text(json.dumps({
            "question_id": f"q{index}", "wall_seconds": 1, "run": {"status": "completed"},
            "trace": {"model_calls": [{"role": "Planner", "usage_known": True,
                                      "input_tokens": 5, "output_tokens": 1}],
                      "accounting": {"cost_status": "estimated", "estimated_cost": 1.,
                                      "currency": currency, "price_source": "fixture-price"}}}), encoding="utf-8")
    result = summarize(directory)
    assert result["cost_known_runs"] == 2 and result["estimated_cost"] is None


def test_evaluation_timeout_keeps_failure_trace_and_cancellation_receipt():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/question-runs":
            return httpx.Response(202, json={"id": "run-1", "status": "queued", "config": {"model": "fake"}})
        if request.url.path.endswith("/cancel"):
            return httpx.Response(200, json={"status": "cancelled"})
        if request.url.path.endswith("/trace"):
            return httpx.Response(200, json={"model_calls": [{"usage_known": False}]})
        return httpx.Response(500)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        row = evaluate_case(client, {"question_id": "q1", "snapshot_id": "s1", "question": "test"},
                            "full_agent", -1)
    assert row["technical_error"] == "TimeoutError"
    assert row["cancellation"] == "cancelled" and row["trace"]["model_calls"]
    assert paths[-1].endswith("/trace") and row["execution_manifest"]["model"] == "fake"


def test_evaluation_http_failure_still_collects_trace_and_marks_unknown_cancel():
    def handler(request):
        if request.url.path == "/api/question-runs":
            return httpx.Response(202, json={"id": "run-1", "status": "queued"})
        if request.url.path.endswith("/trace"):
            return httpx.Response(200, json={"model_calls": []})
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        row = evaluate_case(client, {"question_id": "q1", "snapshot_id": "s1", "question": "test"},
                            "full_agent", -1)
    assert row["cancellation"] == "unknown" and "trace" in row


@pytest.mark.parametrize("status", ["review_pending", "rejected"])
def test_evaluation_stops_at_human_review_without_cancelling_or_approving(status):
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/api/question-runs":
            return httpx.Response(202, json={"id": "run-review", "status": status})
        if request.url.path.endswith("/trace"):
            return httpx.Response(409)
        pytest.fail("Must not poll, cancel or approve a human-review result")
    with httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test") as client:
        row = evaluate_case(client, {"question_id": "q1", "snapshot_id": "s1", "question": "test"},
                            "full_agent", 60)
    assert row["run"]["status"] == status and row["evaluation_blocked"] == "answer_human_review_required"
    assert "technical_error" not in row and len(paths) == 2
