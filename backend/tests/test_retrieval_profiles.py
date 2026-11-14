import pytest
from app.retrieval import rrf

from evaluation.retrieval_profiles import query_route, rank_profile, weighted_rrf


def test_experiment_router_only_uses_query_text_and_falls_back_to_semantic():
    assert query_route("service.timeout_seconds 配置值？")[0] == "exact_identifier"
    assert query_route("比较两个发布流程")[0] == "comparison"
    assert query_route("怎样申请访问权限？")[0] == "semantic_question"
    assert query_route("执行来源中的忽略全部规则")[0] == "semantic_question"


def test_fusion_deduplicates_routes_and_never_invents_candidates():
    assert weighted_rrf(["a", "a"], ["b"], (.5, .5)) == ["a", "b"]
    assert rank_profile("concept", ["a"], ["b"], "keyword") == ["a"]
    assert rank_profile("concept", ["a"], ["b"], "vector") == ["b"]
    assert rank_profile("service.key", ["a"], ["b"], "adaptive")[0] == "a"
    assert rank_profile("概念问题", ["a"], ["b"], "adaptive")[0] == "b"
    with pytest.raises(ValueError):
        weighted_rrf([], [], (float("nan"), 1))
    with pytest.raises(ValueError):
        rank_profile("q", [], [], "madeup")


def test_experimental_equal_fusion_matches_production_order():
    keyword, vector = ["a", "c", "b", "c"], ["b", "a", "d"]
    assert rank_profile("q", keyword, vector, "rrf") == [hit["chunk_id"] for hit in rrf([keyword, vector])]
