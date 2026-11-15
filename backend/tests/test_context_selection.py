import copy

import pytest

from scripts.evaluation.compare_context_selection import (
    compare_saved_gate,
    evidence_metrics,
    select_context,
)


def candidate(document="doc"):
    texts = ["background", "fact", "condition", "unrelated"]
    return {"document": document, "selected_block_ids": {"b1"}, "blocks": [
        {"block_id": f"b{i}", "char_start": 0, "char_end": len(text), "text": text}
        for i, text in enumerate(texts)]}


def test_fixed_alternatives_keep_core_selection_and_bound_neighbor_expansion():
    source = candidate()
    before = copy.deepcopy(source)
    selected = select_context([source], "selected_blocks", 1000)
    whole = select_context([source], "whole_chunks", 1000)
    neighbors = select_context([source], "neighbor_blocks", 1000)
    assert [r["text"] for r in selected] == ["fact"]
    assert [r["text"] for r in whole] == ["background", "fact", "condition", "unrelated"]
    assert [r["text"] for r in neighbors] == ["background", "fact", "condition"]
    assert source == before


def test_expansion_can_spend_the_same_budget_before_reaching_the_fact():
    source = candidate()
    assert [r["text"] for r in select_context([source], "selected_blocks", 10)] == ["fact"]
    assert [r["text"] for r in select_context([source], "whole_chunks", 10)] == ["background"]


def test_duplicate_ranges_and_document_source_limits_are_respected():
    first, second = candidate("a"), candidate("b")
    rows = select_context([first, first, second], "whole_chunks", 1000, max_sources=3, per_document=2)
    assert [(r["document"], r["block_id"]) for r in rows] == [("a", "b0"), ("a", "b1"), ("b", "b0")]


def test_gold_requires_whole_quote_in_correct_document_not_separated_fragments():
    question = {"required_claims": [{"key": "rule", "evidence": [
        {"document": "a", "quote": "fact condition"}]}]}
    separate = select_context([candidate("a")], "whole_chunks", 1000)
    assert evidence_metrics(question, separate)["covered_gold_quotes"] == 0
    wrong_doc = [{"document": "b", "text": "fact condition"}]
    assert evidence_metrics(question, wrong_doc)["covered_gold_quotes"] == 0
    matching = [{"document": "a", "text": "fact\r\n  condition"}]
    metrics = evidence_metrics(question, matching)
    assert metrics["gold_quote_recall"] == metrics["gold_group_recall"] == 1


def test_no_positive_labels_stay_not_applicable_instead_of_perfect():
    metrics = evidence_metrics({"required_claims": []}, [])
    assert metrics["gold_quote_recall"] is None
    assert metrics["all_gold_quotes_present"] is None
    assert metrics["gold_group_recall"] is None


def test_replay_must_match_real_saved_gate_context_including_order_and_text():
    spans = select_context([candidate()], "selected_blocks", 1000)
    row = {"trace": {"model_calls": [{"role": "EvidenceGate", "request": {"data": {"sources": [
        {"paper_version_id": "doc", "block_id": "b1", "char_start": 0, "char_end": 4, "quote": "fact"}
    ]}}}]}}
    compare_saved_gate(row, spans)
    row["trace"]["model_calls"][0]["request"]["data"]["sources"][0]["quote"] = "fake"
    with pytest.raises(ValueError, match="historical production"):
        compare_saved_gate(row, spans)


def test_missing_gate_call_only_accepts_the_saved_empty_context_insufficient_branch():
    row = {"trace": {"model_calls": []}, "run": {"answer": {
        "evidence_status": "insufficient", "evidence": [], "stop_reason": "evidence_gate_insufficient"}}}
    assert compare_saved_gate(row, []) == "empty_context_gate_short_circuit"
    row["run"]["answer"]["stop_reason"] = "failed"
    with pytest.raises(ValueError, match="single saved"):
        compare_saved_gate(row, [])
