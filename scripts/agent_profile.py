"""Print a credential-free process environment for the verified upgrade profile."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings  # noqa: E402


def profile(settings):
    return {
        "QUERY_REWRITE_ENABLED": "true", "WORKFLOW_MCP_ENABLED": "true", "PROJECT_LOGS_MCP_ENABLED": "true",
        "PROVIDER_QUOTA_ENABLED": "true", "PROVIDER_QUOTA_RPM": "30", "PROVIDER_QUOTA_TPM": "200000",
        "ROLE_MODEL_ROUTES": json.dumps({
            "Planner": {"model": "qwen3.8-flash"}, "ToolRouter": {"model": "qwen3.8-flash"},
            "Reranker": {"model": "Qwen/Qwen3-Reranker-0.6B", "base_url": settings.embedding_base_url,
                         "api_key_env": "EMBEDDING_API_KEY"}}, ensure_ascii=False),
        "RERANKER_BACKEND": "rerank_api",
    }


if __name__ == "__main__":
    print(json.dumps(profile(Settings()), ensure_ascii=False))
