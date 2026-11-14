import json
import sys
from pathlib import Path

import pytest

from evaluation import run as evaluation_runner
from evaluation.review import make_template, summarize_manual


@pytest.mark.parametrize("reviewed,fictional", [("false", False), (1, False), (True, True), (True, "false"), (True, None)])
def test_formal_evaluation_rejects_invalid_review_or_fictional_flags(tmp_path, monkeypatch, reviewed, fictional):
    questions = [{"question_id": f"q{i}", "question": "test", "snapshot_id": "snap",
                  "human_reviewed": reviewed, "fictional": fictional} for i in range(30)]
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    output_path = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", ["run", str(input_path), "--output", str(output_path)])
    monkeypatch.setattr(evaluation_runner.httpx, "Client", lambda **kwargs: pytest.fail("HTTP must not start"))
    with pytest.raises(SystemExit):
        evaluation_runner.main()
    assert not output_path.exists()


@pytest.mark.parametrize("include_fictional", [True, False])
def test_formal_evaluation_accepts_declared_reviewed_nonfiction(tmp_path, monkeypatch, include_fictional):
    questions = [{"question_id": f"q{i}", "question": "test", "snapshot_id": "snap",
                  "human_reviewed": True, **({"fictional": False} if include_fictional else {})}
                 for i in range(30)]
    input_path = tmp_path / "questions.jsonl"
    input_path.write_text("\n".join(json.dumps(q) for q in questions), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run", str(input_path), "--output", str(tmp_path / "results")])

    class ClientReached(Exception):
        pass

    def stop_before_network(**kwargs):
        raise ClientReached

    monkeypatch.setattr(evaluation_runner.httpx, "Client", stop_before_network)
    with pytest.raises(ClientReached):
        evaluation_runner.main()


def test_development_set_is_unreviewed_fiction_and_uses_real_fixture_quotes():
    root = Path(__file__).resolve().parents[2]
    questions = [json.loads(line) for line in (root / "evaluation/hallucination_dev.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    papers = json.loads((root / "fixtures/manifest.json").read_text(encoding="utf-8"))["papers"]
    assert len(questions) == len({q["question_id"] for q in questions}) == 12
    for question in questions:
        assert question["fictional"] is True and question["human_reviewed"] is False
        assert question["snapshot_id"] == "BIND_TEACHING_SNAPSHOT"
        for point in question["required_claims"]:
            for evidence in point["evidence"]:
                paper = next(p for p in papers if p["title"].startswith(evidence["source_label"]))
                assert evidence["quote"] in paper["verbatim_lines"]
                assert evidence["pdf_page_number"] == paper["evidence_pdf_page"]


@pytest.fixture
def evaluation_run(tmp_path):
    (tmp_path / "full_agent").mkdir()
    manifest = {"dev_only": True, "questions": [
        {"question_id": "q1", "fictional": True, "required_claims": [{"key": "f1"}, {"key": "latency"}]},
        {"question_id": "q2", "fictional": True, "required_claims": [{"key": "memory"}]}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    row = {"question_id": "q1", "run": {"status": "completed", "answer": {"claims": [
        {"id": "c1", "evidence": [{"evidence_id": "e1", "relation": "supports"}]}]}}}
    (tmp_path / "full_agent/0001.json").write_text(json.dumps(row), encoding="utf-8")
    return tmp_path


def annotated(root):
    # Simulated annotation for validation tests, never a real quality result.
    template = make_template(root)
    for review in template["reviews"]:
        if review["mode"] != "full_agent":
            continue
        review.update(independent_human_review=True, reviewer="simulated-test-reviewer",
                      covered_required_keys=["f1"] if review["question_id"] == "q1" else [],
                      forbidden_conclusions_present=False)
        for link in review["links"]:
            link.update(supports_claim=True, relation_correct=True)
    return template


def test_export_stays_pending_without_human_annotation(evaluation_run):
    template = make_template(evaluation_run)
    result = summarize_manual(evaluation_run, "full_agent", template)
    assert len(template["reviews"]) == 6
    assert result["reviewed_questions"] == 0 and result["citation_semantic_support_rate"] is None
    assert result["required_point_coverage"] is None and result["fictional"] and result["dev_only"]


def test_review_scoring_includes_missing_question_required_points(evaluation_run):
    result = summarize_manual(evaluation_run, "full_agent", annotated(evaluation_run))
    assert result["reviewed_questions"] == 2 and result["required_points"] == 3
    assert result["required_point_coverage"] == pytest.approx(1 / 3)
    assert result["support_links"] == 1 and result["citation_semantic_support_rate"] == 1


@pytest.mark.parametrize("mutation", ["duplicate", "unknown_link", "missing_link", "unknown_point",
                                     "null_verdict", "failed_coverage", "no_reviewer"])
def test_invalid_manual_labels_are_rejected(evaluation_run, mutation):
    template = annotated(evaluation_run)
    first = next(r for r in template["reviews"] if r["mode"] == "full_agent")
    if mutation == "duplicate":
        template["reviews"].append(first)
    elif mutation == "unknown_link":
        first["links"][0]["evidence_id"] = "invented"
    elif mutation == "missing_link":
        first["links"] = []
    elif mutation == "unknown_point":
        first["covered_required_keys"] = ["invented"]
    elif mutation == "null_verdict":
        first["links"][0]["supports_claim"] = None
    elif mutation == "failed_coverage":
        second = next(r for r in template["reviews"] if r["mode"] == "full_agent" and r["question_id"] == "q2")
        second["covered_required_keys"] = ["memory"]
    else:
        first["reviewer"] = ""
    with pytest.raises(ValueError):
        summarize_manual(evaluation_run, "full_agent", template)


def test_partial_human_review_cannot_present_quality_rate(evaluation_run):
    template = annotated(evaluation_run)
    template["reviews"] = [r for r in template["reviews"] if r["question_id"] == "q1"]
    result = summarize_manual(evaluation_run, "full_agent", template)
    assert result["reviewed_questions"] == 1
    assert result["citation_semantic_support_rate"] is result["required_point_coverage"] is None


@pytest.mark.parametrize("target", ["result", "manifest", "new_record"])
def test_changed_records_or_manifest_invalidate_stale_review(evaluation_run, target):
    template = annotated(evaluation_run)
    path = evaluation_run / ("manifest.json" if target == "manifest" else "full_agent/0001.json")
    if target == "new_record":
        (evaluation_run / "full_agent/0002.json").write_text(
            json.dumps({"question_id": "q2", "run": {"status": "failed"}}), encoding="utf-8")
    else:
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        summarize_manual(evaluation_run, "full_agent", template)
