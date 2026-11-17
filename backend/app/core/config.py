import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictInt, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.agents.model_routing import ROLES, ModelRoute

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_file_encoding="utf-8-sig", extra="ignore")

    database_url: str = "postgresql+psycopg://paperqa:paperqa@localhost:55432/paperqa"
    storage_dir: Path = ROOT / "data" / "pdfs"
    local_user_id: str = "local-user"
    chat_api_key: str = ""
    chat_base_url: str = ""
    model: str = ""
    chat_api_style: str = "responses"
    role_model_routes: dict[str, ModelRoute] = Field(default_factory=dict)
    role_output_token_limits: dict[str, StrictInt] = Field(default_factory=dict)
    reranker_backend: Literal["llm", "rerank_api"] = "llm"
    retrieval_strategy: Literal["legacy", "facet_context"] = "facet_context"
    local_reranker_python: str = str(ROOT / "tmp" / "rag-reranker-env" /
                                     ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    local_reranker_model: str = str(ROOT / "tmp" / "rag-models" / "bge-reranker-v2-m3")
    local_reranker_device: Literal["auto", "cpu", "cuda"] = "auto"
    local_reranker_timeout_seconds: int = Field(120, ge=5, le=600)
    query_rewrite_enabled: bool = True
    query_terms_file: Path | None = None
    query_max_per_subquestion: int = Field(5, ge=1, le=6)
    lexical_backend: Literal["legacy", "ngram_bm25"] = "legacy"
    lexical_max_chunks: int = Field(5000, ge=1, le=20000)
    lexical_max_chars: int = Field(2_000_000, ge=1, le=8_000_000)
    workflow_mcp_enabled: bool = False
    project_logs_mcp_enabled: bool = False
    max_workflow_tool_calls: int = Field(2, ge=1, le=5)
    max_external_evidence_chars: int = Field(10000, ge=500, le=20000)
    provider_quota_enabled: bool = False
    provider_quota_redis_url: str = "redis://127.0.0.1:56379/0"
    provider_quota_namespace: str = Field("paperqa_model_quota", pattern=r"^[A-Za-z0-9_-]{1,60}$")
    provider_quota_rpm: int = Field(60, ge=1, le=10000)
    provider_quota_tpm: int = Field(200000, ge=100, le=10000000)
    embedding_api_key: str = ""
    embedding_base_url: str = ""
    embedding_model: str = ""
    embedding_dimensions: int = 0
    embedding_revision: str = "default"
    chunk_strategy: Literal["legacy", "fixed", "paragraph", "structure", "semantic"] = "fixed"
    chunk_size_tokens: int = Field(512, ge=32, le=4096)
    chunk_overlap_tokens: int = Field(64, ge=0, le=2048)
    chunk_semantic_threshold: float = Field(.55, ge=0, le=1, allow_inf_nan=False)
    chunk_semantic_max_units: int = Field(128, ge=1, le=1024)
    vector_store: Literal["milvus"] = "milvus"
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_db_name: str = "default"
    milvus_collection_prefix: str = Field("paperqa", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,39}$")
    milvus_index_type: Literal["HNSW"] = "HNSW"
    milvus_timeout_seconds: int = Field(20, ge=1, le=120)
    max_upload_mb: int = 40
    max_retrieval_rounds: int = Field(2, ge=1, le=4)
    max_revision_rounds: int = Field(1, ge=0, le=3)
    max_model_calls: int = Field(8, ge=1, le=20)
    max_run_seconds: int = Field(240, ge=15, le=900)
    max_context_chars: int = Field(28000, ge=1000, le=100000)
    max_request_input_tokens: int = Field(120000, ge=1000, le=1000000)
    max_run_tokens: int = Field(1000000, ge=1000, le=10000000)
    max_run_cost: float | None = Field(None, gt=0, allow_inf_nan=False)
    max_output_tokens: int = Field(6000, ge=1, le=32000)
    max_embedding_batch_tokens: int = Field(100000, ge=1000, le=1000000)
    chat_input_price_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)
    chat_output_price_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)
    embedding_price_per_million: float | None = Field(None, ge=0, allow_inf_nan=False)
    price_currency: str = ""
    price_source: str = ""
    model_timeout_seconds: int = Field(45, ge=5, le=120)
    provider_max_retries: int = Field(1, ge=0, le=2)
    provider_backoff_seconds: float = Field(0.5, ge=0, le=5)
    provider_backoff_jitter: float = Field(.25, ge=0, le=1, allow_inf_nan=False)
    circuit_failure_threshold: int = Field(3, ge=1, le=20)
    circuit_cooldown_seconds: float = Field(30, ge=1, le=300)
    max_pending_jobs: int = Field(40, ge=1, le=1000)
    max_pending_runs_per_user: int = Field(3, ge=1, le=100)
    max_running_jobs: int = Field(2, ge=1, le=32)
    max_job_attempts: int = Field(3, ge=1, le=10)
    max_ingest_seconds: int = Field(600, ge=15, le=3600)
    worker_lease_seconds: int = Field(90, ge=30)
    answer_review_required: bool = True
    evidence_gate_enabled: bool = True
    code_repository_root: Path | None = None
    code_task_workspace_dir: Path = ROOT / "data" / "code_tasks"
    code_test_docker_image: str = "code-repair-sandbox:py311"
    task_backend: str = "celery"
    celery_broker_url: str = "redis://127.0.0.1:56379/0"
    session_hours: int = Field(24, ge=1, le=168)
    mcp_enabled: bool = False
    mcp_writes_enabled: bool = False
    mcp_approval_ttl_seconds: int = Field(900, ge=30, le=86400)
    mcp_servers_file: Path | None = None

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("CHUNK_OVERLAP_TOKENS 必须小于 CHUNK_SIZE_TOKENS")
        if set(self.role_model_routes) - ROLES:
            raise ValueError("ROLE_MODEL_ROUTES 包含未知工作流角色")
        if set(self.role_output_token_limits) - ROLES:
            raise ValueError("ROLE_OUTPUT_TOKEN_LIMITS 包含未知工作流角色")
        if any(not 1 <= value <= 32000 for value in self.role_output_token_limits.values()):
            raise ValueError("ROLE_OUTPUT_TOKEN_LIMITS 必须在 1 到 32000 之间")
        if self.reranker_backend == "rerank_api" and "Reranker" not in self.role_model_routes:
            raise ValueError("专用重排接口需要显式的 Reranker 模型路由")
        if self.provider_quota_enabled and not self.provider_quota_redis_url.startswith(("redis://", "rediss://")):
            raise ValueError("模型共享配额需要 Redis 地址")
        return self


@lru_cache
def get_settings() -> Settings:
    # Pydantic reads process environment variables first and then this project-local
    # .env. Runtime overrides therefore work for Docker/CI/demo processes without
    # changing the copied project-1 credentials on disk.
    return Settings()
