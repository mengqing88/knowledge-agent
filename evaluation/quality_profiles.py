"""Predeclared answer-pipeline ablations; never changes the running application's settings."""

PROFILES = {
    "baseline_rag": {"mode": "baseline_rag", "evidence_gate_enabled": False},
    "without_verifier": {"mode": "without_verifier", "evidence_gate_enabled": False},
    "without_evidence_gate": {"mode": "full_agent", "evidence_gate_enabled": False},
    "full_agent": {"mode": "full_agent", "evidence_gate_enabled": True},
}

# Equal limits isolate components. One retrieval round avoids tuning retries per arm.
LIMITS = {"max_model_calls": 6, "max_retrieval_rounds": 1, "max_revision_rounds": 0,
          "max_run_seconds": 240, "max_run_tokens": 150000, "max_output_tokens": 4000,
          "model_timeout_seconds": 60, "provider_max_retries": 0, "answer_review_required": False,
          "workflow_mcp_enabled": False, "project_logs_mcp_enabled": False, "mcp_enabled": False,
          "mcp_writes_enabled": False, "reranker_backend": "llm", "role_model_routes": {}}
