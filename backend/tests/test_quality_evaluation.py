import json
from pathlib import Path

import pytest

from evaluation.report import summarize
from evaluation.review import make_template, modes_for, summarize_manual
from scripts.benchmarks.benchmark_quality import load_suite

ROOT = Path(__file__).resolve().parents[2]


def test_gate_ablation_preserves_missing_cases_and_pending_human_scores(tmp_path):
    modes = ["without_evidence_gate", "full_agent"]
    manifest = {"mode_order": modes, "dev_only": True, "questions": [
        {"question_id": "q1", "fictional": True, "required_claims": [{"key": "fact"}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for mode in modes:
        (tmp_path / mode).mkdir()
    # No result is still one failed question, never an empty successful arm.
    report = summarize(tmp_path / "without_evidence_gate")
    assert report["questions"] == report["technical_failures"] == report["missing_records"] == 1
    template = make_template(tmp_path)
    assert [r["mode"] for r in template["reviews"]] == modes
    assert summarize_manual(tmp_path, modes[0], template)["required_point_coverage"] is None
    for row in template["reviews"]:
        row.update(reviewer="simulated-test-reviewer", independent_human_review=True,
                   covered_required_keys=[], forbidden_conclusions_present=False)
    assert summarize_manual(tmp_path, modes[0], template)["required_point_coverage"] == 0


@pytest.mark.parametrize("modes", [[], ["../outside"], ["full_agent", "full_agent"], [["nested"]], [None]])
def test_profile_directory_names_reject_invalid_or_duplicate_values(modes):
    with pytest.raises(ValueError):
        modes_for({"mode_order": modes})


def test_fixed_development_suite_binds_unchanged_sources_and_declares_label_limits(tmp_path):
    suite, _, _, _ = load_suite(ROOT / "evaluation/quality_suite.json")
    assert suite["fictional"] and not suite["human_reviewed"]
    assert all(q["fictional"] and not q["human_reviewed"] for q in suite["questions"])
    assert set(suite["smoke_question_ids"]) <= {q["question_id"] for q in suite["questions"]}
    assert {q["expected_action"] for q in suite["questions"]} == {"supported", "partial", "insufficient"}
    suite["corpus_manifest_sha256"] = "changed"
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest changed"):
        load_suite(path)
