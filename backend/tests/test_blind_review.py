import copy
import zipfile

import pytest

from evaluation.blind_review import export_packet, import_scorecard, json_text, read_json
from evaluation.review import sha256


@pytest.fixture
def packet(tmp_path):
    corpus = tmp_path / "corpus"
    (corpus / "originals").mkdir(parents=True)
    for name, text in [("doc.txt", "The default is Read Committed."),
                       ("originals/doc.rst", "The default is Read Committed."),
                       ("originals/LICENSE", "Test fixture, not a real evaluation corpus.")]:
        (corpus / name).write_text(text, encoding="utf-8")
    questions = [{"question_id": f"q{i}", "question": "Default?", "expected_action": "supported",
                  "required_claims": [{"key": "default", "meaning": "Read Committed",
                                       "evidence": [{"document": "doc", "quote": "The default is Read Committed."}]}],
                  "forbidden_conclusions": ["Serializable"], "fictional": True} for i in (1, 2)]
    (corpus / "suite.json").write_text(json_text({"questions": questions}), encoding="utf-8")
    (corpus / "corpus.json").write_text(json_text({"documents": [
        {"id": "doc", "filename": "doc.txt", "sha256": sha256(corpus / "doc.txt"),
         "source": {"file": "doc.rst", "sha256": sha256(corpus / "originals/doc.rst")},
         "license_file": "originals/LICENSE"}]}), encoding="utf-8")
    root = tmp_path / "run"
    root.mkdir()
    (root / "manifest.json").write_text(json_text({"dev_only": True, "questions": questions,
        "mode_order": ["baseline_rag", "full_agent"], "model": "DO_NOT_LEAK_MODEL",
        "corpus_manifest_sha256": sha256(corpus / "corpus.json"),
        "question_set_sha256": sha256(corpus / "suite.json")}), encoding="utf-8")
    for mode in ("baseline_rag", "full_agent"):
        (root / mode).mkdir()
        row = {"question_id": "q1", "wall_seconds": 1., "trace": {"secret": "DO_NOT_LEAK_TRACE"},
               "run": {"status": "completed", "answer": {
                   "stop_reason": "DO_NOT_LEAK_VERIFIER", "usage": {"model_calls": 1},
                   "claims": [{"id": "internal-claim", "text": "Read Committed", "conditions": ["PostgreSQL"],
                               "semantic_verified": mode == "full_agent",
                               "evidence": [{"evidence_id": "internal-evidence", "relation": "supports"}]}],
                   "evidence": [{"id": "internal-evidence", "quote": "The default is Read Committed.",
                                 "paper_id": "doc", "file_url": "DO_NOT_LEAK_URL"}],
                   "gaps": ["Published gap text"], "clarification": None}}}
        (root / mode / "0001.json").write_text(json_text(row), encoding="utf-8")
    failed = {"question_id": "q2", "wall_seconds": 1., "trace": {"draft": "DO_NOT_LEAK_DRAFT"},
              "run": {"status": "failed", "error": "DO_NOT_LEAK_ERROR", "answer": None}}
    (root / "full_agent/0002.json").write_text(json_text(failed), encoding="utf-8")
    output = tmp_path / "packet"
    export_packet(root, corpus, output)
    return root, corpus, output


def filled_card(output):
    card = read_json(output / "reviewer/scorecard.json")
    signature = {"reviewer": "simulated-test-only", "reviewed_at": "2026-11-15T18:00:00+08:00",
                 "notes": "Simulated control assertion; not a human evaluation."}
    for question in card["question_reviews"]:
        question.update(signature, rubric_correct=True)
    for answer in card["reviews"]:
        answer.update(signature, independent_human_review=True,
                      covered_required_keys=["default"] if answer["question_id"] == "q1" else [],
                      forbidden_conclusions_present=False, action_correct=answer["question_id"] == "q1",
                      conditions_correct=True)
        for link in answer["links"]:
            link.update(supports_claim=True, relation_correct=True)
    return card


def import_card(packet, card):
    root, _, output = packet
    path = output / "filled.json"
    path.write_text(json_text(card), encoding="utf-8")
    return import_scorecard(root, output / "operator-key.json", path)


def test_reviewer_zip_hides_identity_and_self_scores_but_keeps_published_content(packet):
    _, _, output = packet
    with zipfile.ZipFile(output / "reviewer.zip") as archive:
        assert "operator-key.json" not in archive.namelist()
        text = "\n".join(archive.read(name).decode("utf-8") for name in archive.namelist())
    for secret in ("DO_NOT_LEAK", "semantic_verified", "baseline_rag", "full_agent",
                   "internal-claim", "internal-evidence", "agent_review"):
        assert secret not in text
    book = (output / "reviewer/review-book.md").read_text(encoding="utf-8")
    assert "Read Committed" in book and "PostgreSQL" in book and "Published gap text" in book
    assert '"status": "failed"' in book and '"status": "missing"' in book
    assert '"claim_id"' not in book and '"id": "C1"' in book and '"id": "E1"' in book
    assert len(read_json(output / "reviewer/scorecard.json")["reviews"]) == 4


def test_blank_review_stays_pending_and_complete_review_restores_source_ids(packet):
    root, _, output = packet
    source_hash = sha256(root / "full_agent/0001.json")
    empty = read_json(output / "reviewer/scorecard.json")
    annotation, report = import_card(packet, empty)
    assert all(r["independent_human_review"] is False for r in annotation["reviews"])
    assert all(m["required_point_coverage"] is None for m in report["manual_metrics"].values())
    annotation, report = import_card(packet, filled_card(output))
    for review in annotation["reviews"]:
        if review["question_id"] == "q1":
            assert review["links"][0]["claim_id"] == "internal-claim"
            assert review["links"][0]["evidence_id"] == "internal-evidence"
    for mode in ("baseline_rag", "full_agent"):
        assert report["manual_metrics"][mode]["required_point_coverage"] == .5
        assert report["objective_counts"][mode]["technical_failures"] == 1
    assert sha256(root / "full_agent/0001.json") == source_hash
    assert not (root / "summary.json").exists()


def test_partial_review_does_not_publish_whole_group_quality(packet):
    _, _, output = packet
    card = filled_card(output)
    for row in card["reviews"]:
        if row["question_id"] == "q2":
            row["independent_human_review"] = False
    _, report = import_card(packet, card)
    for mode in ("baseline_rag", "full_agent"):
        assert report["manual_metrics"][mode]["reviewed_questions"] == 1
        assert report["manual_metrics"][mode]["required_point_coverage"] is None


@pytest.mark.parametrize("target", ["manifest", "result", "new_record", "excerpt", "original", "book"])
def test_source_or_packet_changes_invalidate_review(packet, target):
    root, corpus, output = packet
    paths = {"manifest": root / "manifest.json", "result": root / "full_agent/0001.json",
             "excerpt": corpus / "doc.txt", "original": corpus / "originals/doc.rst",
             "book": output / "reviewer/review-book.md"}
    if target == "new_record":
        (root / "baseline_rag/0002.json").write_text(json_text(
            {"question_id": "q2", "wall_seconds": 0., "run": {"status": "failed"}}), encoding="utf-8")
    else:
        path = paths[target]
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA"):
        import_card(packet, filled_card(output))


@pytest.mark.parametrize("mutation", ["agent", "string_flag", "no_signature", "no_timezone", "rubric_disputed",
                                     "deleted", "duplicate", "changed_link", "failed_credit"])
def test_import_rejects_unconfirmed_identity_or_modified_scorecard(packet, mutation):
    _, _, output = packet
    card = filled_card(output)
    answer = next(row for row in card["reviews"] if row["question_id"] == "q1")
    if mutation == "agent":
        card["review_source"] = "agent"
    elif mutation == "string_flag":
        answer["independent_human_review"] = "true"
    elif mutation == "no_signature":
        answer["reviewer"] = ""
    elif mutation == "no_timezone":
        answer["reviewed_at"] = "2026-11-15T18:00:00"
    elif mutation == "rubric_disputed":
        card["question_reviews"][0]["rubric_correct"] = False
    elif mutation == "deleted":
        card["reviews"].pop()
    elif mutation == "duplicate":
        card["reviews"].append(copy.deepcopy(answer))
    elif mutation == "changed_link":
        answer["links"][0]["evidence_id"] = "invented"
    else:
        next(row for row in card["reviews"] if row["question_id"] == "q2")["action_correct"] = True
    with pytest.raises(ValueError):
        import_card(packet, card)


def test_blind_labels_follow_each_question_shuffle(packet, monkeypatch, tmp_path):
    root, corpus, _ = packet

    class Alternating:
        calls = 0

        def shuffle(self, values):
            self.calls += 1
            if self.calls % 2:
                values.reverse()

    shuffler = Alternating()
    monkeypatch.setattr("evaluation.blind_review.random.SystemRandom", lambda: shuffler)
    second = tmp_path / "second"
    export_packet(root, corpus, second)
    bindings = read_json(second / "operator-key.json")["bindings"]
    label_a = [b["mode"] for b in bindings if b["blind_id"].endswith("_A")]
    assert label_a == ["full_agent", "baseline_rag"]
    assert shuffler.calls == 2


def test_export_refuses_existing_destination_and_modified_corpus(packet):
    root, corpus, output = packet
    with pytest.raises(FileExistsError):
        export_packet(root, corpus, output)
    (corpus / "doc.txt").write_text("altered", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA"):
        export_packet(root, corpus, output / "bad")
    assert not (output / "bad").exists()
