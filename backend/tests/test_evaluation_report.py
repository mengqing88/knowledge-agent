import json

import pytest

from evaluation.report import summarize


def test_report_keeps_unknown_usage_unknown_and_reports_labeled_top5(tmp_path):
    run_dir = tmp_path / "full_agent"
    run_dir.mkdir()
    manifest = {
        "questions": [
            {"question_id": "q1", "expected_action": "supported", "relevant_chunk_ids": ["gold"]},
            {"question_id": "q2", "expected_action": "insufficient"},
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    first = {
        "question_id": "q1", "wall_seconds": 1.0,
        "run": {"status": "completed", "answer": {"evidence_status": "supported", "claims": []}},
        "trace": {
            "retrievals": [
                {"attempt": 1, "rankings": [{"chunk_id": "gold"}]},
                {"attempt": 1, "rankings": [{"chunk_id": "other"}]},
            ],
            "model_calls": [{"role": "Analyst", "usage_known": False,
                             "input_tokens": None, "output_tokens": None}],
            "accounting": {"cost_status": "unknown", "estimated_cost": None},
        },
    }
    second = {
        "question_id": "q2", "wall_seconds": 3.0,
        "run": {"status": "completed", "answer": {"evidence_status": "insufficient", "claims": []}},
        "trace": {
            "retrievals": [], "model_calls": [],
            "accounting": {"cost_status": "estimated", "estimated_cost": 0.01},
        },
    }
    for index, row in enumerate((first, second), 1):
        (run_dir / f"{index:04d}.json").write_text(json.dumps(row), encoding="utf-8")

    counts = summarize(run_dir)

    assert counts["top5_queries"] == 1
    assert counts["top5_hits"] == 1
    assert counts["top5_hit_rate"] == 1.0
    assert counts["wall_seconds_p50"] == 1.0
    assert counts["wall_seconds_p95"] == 3.0
    assert counts["unknown_usage_calls"] == 1
    assert counts["token_totals_complete"] is False
    assert counts["estimated_cost"] is None


def test_missing_expected_and_actual_actions_do_not_count_as_match(tmp_path):
    run_dir = tmp_path / "full_agent"
    run_dir.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [
        {"question_id": "missing"}, {"question_id": "labeled", "expected_action": "insufficient"},
    ]}), encoding="utf-8")
    for index, row in enumerate([
        {"question_id": "missing", "wall_seconds": 1, "run": {"status": "completed", "answer": {}}},
        {"question_id": "labeled", "wall_seconds": 1, "run": {"status": "completed",
         "answer": {"evidence_status": "insufficient", "claims": []}}},
    ]):
        (run_dir / f"{index}.json").write_text(json.dumps(row), encoding="utf-8")

    counts = summarize(run_dir)
    assert counts["expected_action_matches"] == 1
    assert counts["expected_action_labeled_questions"] == 1
    assert "状态" in counts["refusal_metrics_note"]


@pytest.mark.parametrize("calls", [None, [{"role": "Analyst", "usage_known": False}],
                                   [{"role": "Analyst", "usage_known": True,
                                     "input_tokens": None, "output_tokens": 2}],
                                   [{"role": "Analyst", "usage_known": True,
                                     "input_tokens": -1, "output_tokens": 2}]])
def test_incomplete_call_usage_cannot_claim_complete_tokens_or_cost(tmp_path, calls):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [{"question_id": "q1"}]}), encoding="utf-8")
    trace = {"accounting": {"cost_status": "estimated", "estimated_cost": 1,
                            "currency": "CNY", "price_source": "fixture"}}
    if calls is not None:
        trace["model_calls"] = calls
    (directory / "0001.json").write_text(json.dumps({"question_id": "q1", "wall_seconds": 1,
        "run": {"status": "completed"}, "trace": trace}), encoding="utf-8")
    result = summarize(directory)
    assert result["token_totals_complete"] is False
    assert result["estimated_cost"] is None


def test_complete_call_usage_and_cost_can_be_summarized(tmp_path):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [{"question_id": "q1"}]}), encoding="utf-8")
    trace = {"model_calls": [{"role": "Embedding", "usage_known": True, "input_tokens": 12, "output_tokens": 0}],
             "accounting": {"cost_status": "estimated", "estimated_cost": 0.01,
                            "currency": "CNY", "price_source": "fixture"}}
    (directory / "0001.json").write_text(json.dumps({"question_id": "q1", "wall_seconds": 1,
        "run": {"status": "completed"}, "trace": trace}), encoding="utf-8")
    result = summarize(directory)
    assert result["token_totals_complete"] is True
    assert result["input_tokens"] == 12 and result["estimated_cost"] == 0.01


@pytest.mark.parametrize("run,technical_error,cancellation", [
    ({"status": "running"}, None, None),
    ({}, "TimeoutError", "unknown"),
    ({}, "TimeoutError", "cancelled"),
    ({"status": "cancelled"}, None, None),
])
def test_unfinished_or_cancelled_run_does_not_claim_final_usage(tmp_path, run, technical_error, cancellation):
    directory = tmp_path / "full_agent"
    directory.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"questions": [{"question_id": "q1"}]}), encoding="utf-8")
    trace = {"model_calls": [{"role": "Planner", "usage_known": True, "input_tokens": 10, "output_tokens": 2}],
             "accounting": {"cost_status": "estimated", "estimated_cost": 0.01,
                            "currency": "CNY", "price_source": "fixture"}}
    row = {"question_id": "q1", "wall_seconds": 1, "run": run, "trace": trace}
    if technical_error:
        row["technical_error"] = technical_error
    if cancellation:
        row["cancellation"] = cancellation
    (directory / "0001.json").write_text(json.dumps(row), encoding="utf-8")
    result = summarize(directory)
    assert result["input_tokens"] == 10
    assert result["unconfirmed_final_usage_runs"] == 1
    assert result["token_totals_complete"] is False and result["estimated_cost"] is None
