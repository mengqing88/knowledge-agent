"""Small real-service smoke for explicit role routes and a dedicated reranker."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.agents.model_routing import ModelRoute  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.core.schemas import Plan, Reranking, ToolDecision  # noqa: E402
from app.integrations.providers import ModelProvider  # noqa: E402


def main():
    settings = Settings(max_output_tokens=800, provider_max_retries=0, model_timeout_seconds=45)
    settings.role_model_routes = {
        "Planner": ModelRoute(model="qwen3.8-flash"),
        "ToolRouter": ModelRoute(model="qwen3.8-flash"),
        "Reranker": ModelRoute(model="Qwen/Qwen3-Reranker-0.6B", base_url=settings.embedding_base_url,
                               api_key_env="EMBEDDING_API_KEY"),
    }
    settings.reranker_backend = "rerank_api"
    provider = ModelProvider(settings)
    report = {"scope": "configured live services; bounded three-call smoke, not a quality benchmark", "calls": []}
    plan, usage = provider.structured("Planner", "将问题拆成一个子问题，保持原始实体和条件。", {
        "question": "Atlas 服务超时上限为多少秒？"}, Plan)
    report["calls"].append({"role": "Planner", "model": "qwen3.8-flash", "valid_schema": bool(plan.normalized_question),
                             "usage": usage})
    decision, usage = provider.structured("ToolRouter", "没有可用工具，返回 tool_identity=null 并说明原因。", {
        "question": "Atlas 服务超时上限为多少秒？", "tools": []}, ToolDecision)
    assert decision.tool_identity is None
    report["calls"].append({"role": "ToolRouter", "model": "qwen3.8-flash", "valid_schema": True, "usage": usage})
    ranking, usage = provider.structured("Reranker", "", {
        "question": "Atlas 服务的请求超时是多少秒？", "candidates": [
            {"chunk_id": "irrelevant", "blocks": [{"block_id": "a", "text": "办公室每周五开展读书活动。"}]},
            {"chunk_id": "relevant", "blocks": [{"block_id": "b", "text": "Atlas 服务的请求超时上限是三十秒。"}]}]}, Reranking)
    assert ranking.items and ranking.items[0].chunk_id == "relevant"
    report["calls"].append({"role": "Reranker", "model": "Qwen/Qwen3-Reranker-0.6B",
                             "expected_top_candidate": True, "usage": usage})
    path = ROOT / "tmp/reports/role-model-live-smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "calls": len(report["calls"])}))


if __name__ == "__main__":
    main()
