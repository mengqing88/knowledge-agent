"""Small, explicit records for reproducibility and resource accounting."""
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version

from sqlalchemy import select

from .config import ROOT
from .model_routing import role_rates, route_manifest
from .models import ModelCall
from .prompts import ANALYST, EVIDENCE_GATE, PLANNER, RERANKER, TOOL_ROUTER, VERIFIER
from .query_rewrite import load_rules
from .schemas import Analysis, EvidenceGate, Plan, Reranking, ToolDecision, Verification

ROLES = {"Planner": (PLANNER, Plan), "Reranker": (RERANKER, Reranking),
         "Analyst": (ANALYST, Analysis), "Verifier": (VERIFIER, Verification),
         "EvidenceGate": (EVIDENCE_GATE, EvidenceGate), "ToolRouter": (TOOL_ROUTER, ToolDecision)}
PUBLIC_SETTINGS = (
    "model", "embedding_model", "chat_api_style", "embedding_dimensions", "embedding_revision",
    "chunk_strategy", "chunk_size_tokens", "chunk_overlap_tokens", "chunk_semantic_threshold",
    "chunk_semantic_max_units",
    "max_model_calls", "max_run_seconds", "max_retrieval_rounds", "max_revision_rounds", "max_context_chars",
    "max_request_input_tokens", "max_run_tokens", "max_run_cost", "max_output_tokens", "max_embedding_batch_tokens",
    "model_timeout_seconds", "provider_max_retries", "provider_backoff_seconds", "provider_backoff_jitter",
    "circuit_failure_threshold", "circuit_cooldown_seconds",
    "chat_input_price_per_million", "chat_output_price_per_million",
    "embedding_price_per_million", "price_currency", "price_source", "answer_review_required", "evidence_gate_enabled",
    "vector_store", "milvus_index_type", "milvus_timeout_seconds",
    "mcp_approval_ttl_seconds",
    "query_rewrite_enabled", "query_max_per_subquestion", "workflow_mcp_enabled", "project_logs_mcp_enabled",
    "max_workflow_tool_calls", "max_external_evidence_chars", "reranker_backend",
    "provider_quota_enabled", "provider_quota_namespace", "provider_quota_rpm", "provider_quota_tpm",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def estimate_tokens(value):
    """UTF-8 bytes plus framing allowance: a conservative heuristic, not a tokenizer."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(text.encode("utf-8")) + 256


def provider_identity(settings):
    # A base URL may contain embedded credentials. Store only its identity hash.
    return digest({name: getattr(settings, name) for name in (
        "chat_base_url", "embedding_base_url", "model", "embedding_model", "chat_api_style",
        "embedding_dimensions", "embedding_revision", "vector_store", "milvus_uri",
        "milvus_db_name", "milvus_collection_prefix", "milvus_index_type", "provider_quota_redis_url")} |
        {"role_model_routes": route_manifest(settings)})


def code_identity():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "backend" / "app").glob("*.py"))}


def environment_identity():
    packages = {}
    for name in ("fastapi", "sqlalchemy", "langgraph", "llama-index-core", "httpx", "pymilvus", "celery"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unavailable"
    locks = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
             for name in ("uv.lock", "frontend/package-lock.json") if (ROOT / name).is_file()}
    return packages, locks


def freeze_execution(settings, snapshot, question, context, memory_context=None, replay_of=None):
    from .workflow_tools import scope_identity
    packages, locks = environment_identity()
    code = code_identity()
    frozen = {"version": "workflow-v5-evidence-gate", **{k: getattr(settings, k) for k in PUBLIC_SETTINGS},
              "provider_identity": provider_identity(settings),
              "rag_framework": "llamaindex-vectorstoreindex-v1",
              "credentials": {"chat": "CHAT_API_KEY", "embedding": "EMBEDDING_API_KEY"},
              "prompts": {role: {"text": prompt, "sha256": digest(prompt),
                                  "schema": schema.model_json_schema()}
                          for role, (prompt, schema) in ROLES.items()},
              "application_files": code, "lock_files": locks, "runtime_packages": packages,
              "snapshot_sha256": digest(snapshot.manifest), "question_sha256": digest(question),
              "previous_questions": context, "memory_context": memory_context or [],
              "role_model_routes": route_manifest(settings),
              "query_term_rules": [r for r in load_rules(settings) if snapshot.corpus_id in r["corpus_ids"]],
              "workflow_tools_identity": scope_identity(settings) if settings.workflow_mcp_enabled else None,
              "replay_of": replay_of,
              "model_version": "provider_alias_only",
              "rebuild_status": "hashes_only_no_source_or_image_archive"}
    return {**frozen, "manifest_sha256": digest(frozen)}


def usage_summary(db, run_id, config):
    calls = list(db.scalars(select(ModelCall).where(ModelCall.run_id == run_id)))
    known = [c for c in calls if c.usage_known]
    unknown = [c for c in calls if not c.usage_known]
    charged = sum((c.input_tokens + c.output_tokens) if c.usage_known else
                  max(c.reserved_tokens, c.input_tokens + c.output_tokens) for c in calls)
    price_known = bool(config.get("price_currency") and config.get("price_source"))
    cost, guard_cost = 0., 0.
    for c in calls:
        rates = role_rates(config, c.role)
        if None in rates:
            price_known = False
        else:
            if c.usage_known:
                cost += (c.input_tokens * rates[0] + c.output_tokens * rates[1]) / 1000000
                guard_cost += (c.input_tokens * rates[0] + c.output_tokens * rates[1]) / 1000000
            else:
                guard_cost += (max(c.estimated_input_tokens, c.input_tokens) * rates[0] +
                               max(c.reserved_tokens-c.estimated_input_tokens, c.output_tokens) * rates[1]) / 1000000
    return {"model_calls": sum(c.role != "Embedding" for c in calls),
            "embedding_calls": sum(c.role == "Embedding" for c in calls),
            "known_input_tokens": sum(c.input_tokens for c in known),
            "known_output_tokens": sum(c.output_tokens for c in known),
            "unknown_usage_calls": len(unknown), "reserved_or_actual_tokens": charged,
            "usage_status": "unknown" if unknown else "known", "all_attempts": True,
            "estimated_cost": round(cost, 8) if price_known and not unknown and calls else None,
            "reserved_or_actual_cost": guard_cost if price_known else None,
            "currency": config.get("price_currency") or None,
            "price_source": config.get("price_source") or None,
            "cost_status": "estimated" if price_known and not unknown and calls else "unknown"}
