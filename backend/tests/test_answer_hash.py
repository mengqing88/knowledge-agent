from copy import deepcopy

import pytest
from app.services import answer_content_hash


@pytest.fixture
def answer_payload():
    return {"id": "answer1", "revision": 1, "claims": [
        {"id": "claim1", "text": "审批后开通只读权限", "conditions": ["已审批"], "evidence": [
            {"evidence_id": "source2", "relation": "supports"},
            {"evidence_id": "source1", "relation": "supports"}]},
        {"id": "claim2", "text": "写权限需另行申请", "conditions": [], "evidence": [
            {"evidence_id": "source2", "relation": "supports"}]}], "evidence": [
        {"id": "source2", "quote": "写权限需另行申请", "sha256": "b" * 64},
        {"id": "source1", "quote": "审批后开通只读权限", "sha256": "a" * 64}]}


def test_answer_hash_ignores_only_evidence_set_order_without_mutating_payload(answer_payload):
    original = deepcopy(answer_payload)
    reordered = deepcopy(answer_payload)
    reordered["evidence"].reverse()
    reordered["claims"][0]["evidence"].reverse()
    assert answer_content_hash(answer_payload) == answer_content_hash(reordered)
    assert answer_payload == original


@pytest.mark.parametrize("change", ["claim_text", "conditions", "relation", "quote", "source_hash", "claim_order"])
def test_answer_hash_binds_meaningful_content_and_claim_order(answer_payload, change):
    changed = deepcopy(answer_payload)
    if change == "claim_text":
        changed["claims"][0]["text"] = "无需审批即可开通"
    elif change == "conditions":
        changed["claims"][0]["conditions"] = ["无需审批"]
    elif change == "relation":
        changed["claims"][0]["evidence"][0]["relation"] = "contradicts"
    elif change == "quote":
        changed["evidence"][0]["quote"] = "写权限自动开通"
    elif change == "source_hash":
        changed["evidence"][0]["sha256"] = "c" * 64
    else:
        changed["claims"].reverse()
    assert answer_content_hash(answer_payload) != answer_content_hash(changed)
