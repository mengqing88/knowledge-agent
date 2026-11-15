from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


def uid() -> str:
    return uuid4().hex


def now() -> datetime:
    return datetime.now(timezone.utc)


class Identity:
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=uid)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class WorkloadGate(Base):
    __tablename__ = "workload_gate"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)


@event.listens_for(WorkloadGate.__table__, "after_create")
def seed_workload_gate(table, connection, **_kwargs):
    connection.execute(table.insert().values(id=1))


class Corpus(Identity, Base):
    __tablename__ = "corpora"
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    name: Mapped[str] = mapped_column(String(160))


class User(Identity, Base):
    __tablename__ = "users"
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(Text)


class LoginSession(Identity, Base):
    __tablename__ = "login_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Membership(Identity, Base):
    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("corpus_id", "user_id"),)
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))


class Paper(Identity, Base):
    __tablename__ = "papers"
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    title: Mapped[str] = mapped_column(String(500))
    folder: Mapped[str] = mapped_column(String(160), default="", server_default="")
    tags: Mapped[list] = mapped_column(JSON, default=list, server_default="[]")
    deleted: Mapped[bool] = mapped_column(default=False, server_default="false")


class PaperVersion(Identity, Base):
    __tablename__ = "paper_versions"
    __table_args__ = (UniqueConstraint("paper_id", "sha256"),)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(String(100))
    filename: Mapped[str] = mapped_column(String(255))
    byte_size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="uploaded", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")


class ParseRevision(Identity, Base):
    __tablename__ = "parse_revisions"
    __table_args__ = (UniqueConstraint("paper_version_id", "config_version"),)
    paper_version_id: Mapped[str] = mapped_column(ForeignKey("paper_versions.id"), index=True)
    parser: Mapped[str] = mapped_column(String(100))
    config_version: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(24))


class SourceBlock(Identity, Base):
    __tablename__ = "source_blocks"
    __table_args__ = (UniqueConstraint("parse_revision_id", "page_index", "block_index"),)
    parse_revision_id: Mapped[str] = mapped_column(ForeignKey("parse_revisions.id"), index=True)
    page_index: Mapped[int] = mapped_column(Integer)
    block_index: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)
    printed_page_label: Mapped[str | None] = mapped_column(String(40))
    section: Mapped[str] = mapped_column(String(500), default="", server_default="")


class IndexRevision(Identity, Base):
    __tablename__ = "index_revisions"
    __table_args__ = (UniqueConstraint("parse_revision_id", "config_version", "embedding_model"),)
    parse_revision_id: Mapped[str] = mapped_column(ForeignKey("parse_revisions.id"), index=True)
    config_version: Mapped[str] = mapped_column(String(80))
    embedding_model: Mapped[str] = mapped_column(String(200))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    vector_backend: Mapped[str] = mapped_column(String(24), default="milvus", server_default="milvus")
    vector_collection: Mapped[str | None] = mapped_column(String(120), nullable=True)
    status: Mapped[str] = mapped_column(String(24))


class Chunk(Identity, Base):
    __tablename__ = "chunks"
    index_revision_id: Mapped[str] = mapped_column(ForeignKey("index_revisions.id"), index=True)
    text: Mapped[str] = mapped_column(Text)
    source_ranges: Mapped[list] = mapped_column(JSON)
    # Kept as nullable JSON for SQLite fixtures and one-time migration tooling.
    # Production PostgreSQL retrieval never reads this field; vectors live in Milvus.
    embedding: Mapped[list | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    search_vector: Mapped[str | None] = mapped_column(Text().with_variant(TSVECTOR(), "postgresql"))


Index("ix_chunks_fts", Chunk.search_vector, postgresql_using="gin")


class Snapshot(Identity, Base):
    __tablename__ = "snapshots"
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id"), index=True)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    manifest: Mapped[list] = mapped_column(JSON)


class QuestionRun(Identity, Base):
    __tablename__ = "question_runs"
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key"),)
    owner_id: Mapped[str] = mapped_column(String(100), index=True)
    snapshot_id: Mapped[str] = mapped_column(ForeignKey("snapshots.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    parent_run_id: Mapped[str | None] = mapped_column(ForeignKey("question_runs.id"))
    mode: Mapped[str] = mapped_column(String(30), default="full_agent")
    idempotency_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    attempt: Mapped[int] = mapped_column(Integer, default=0)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str | None] = mapped_column(Text)
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)


class RunEvent(Identity, Base):
    __tablename__ = "run_events"
    __table_args__ = (UniqueConstraint("run_id", "seq"),)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    seq: Mapped[int] = mapped_column(Integer)
    attempt: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(80))
    payload: Mapped[dict] = mapped_column(JSON)


class RetrievalRecord(Identity, Base):
    __tablename__ = "retrieval_records"
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    round: Mapped[int] = mapped_column(Integer)
    subquestion_id: Mapped[str] = mapped_column(String(40))
    query: Mapped[str] = mapped_column(Text)
    rankings: Mapped[list] = mapped_column(JSON)


class EvidenceSpan(Identity, Base):
    __tablename__ = "evidence_spans"
    __table_args__ = (UniqueConstraint("run_id", "block_id", "char_start", "char_end"),)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    block_id: Mapped[str | None] = mapped_column(ForeignKey("source_blocks.id"), index=True, nullable=True)
    invocation_id: Mapped[str | None] = mapped_column(ForeignKey("mcp_invocations.id"), nullable=True)
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)


class AnswerRevision(Identity, Base):
    __tablename__ = "answer_revisions"
    # One immutable published revision per run. A correction creates a new child run.
    __table_args__ = (UniqueConstraint("run_id", "revision"),)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    revision: Mapped[int] = mapped_column(Integer, default=1)
    evidence_status: Mapped[str] = mapped_column(String(30))
    gaps: Mapped[list] = mapped_column(JSON)
    issues: Mapped[list] = mapped_column(JSON)
    stop_reason: Mapped[str] = mapped_column(String(120))
    usage: Mapped[dict] = mapped_column(JSON)
    clarification: Mapped[str | None] = mapped_column(Text)


class AnswerClaim(Identity, Base):
    __tablename__ = "answer_claims"
    answer_id: Mapped[str] = mapped_column(ForeignKey("answer_revisions.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    conditions: Mapped[list] = mapped_column(JSON)
    semantic_verified: Mapped[bool] = mapped_column(default=False)


class ClaimEvidence(Identity, Base):
    __tablename__ = "claim_evidence"
    __table_args__ = (UniqueConstraint("claim_id", "evidence_id", "relation"),)
    claim_id: Mapped[str] = mapped_column(ForeignKey("answer_claims.id"), index=True)
    evidence_id: Mapped[str] = mapped_column(ForeignKey("evidence_spans.id"))
    relation: Mapped[str] = mapped_column(String(24))


class ReviewRecord(Identity, Base):
    __tablename__ = "review_records"
    answer_id: Mapped[str] = mapped_column(ForeignKey("answer_revisions.id"), index=True)
    owner_id: Mapped[str] = mapped_column(String(100))
    text: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(32), default="unclear", server_default="unclear")
    answer_sha256: Mapped[str | None] = mapped_column(String(64))
    idempotency_key: Mapped[str | None] = mapped_column(String(128))
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key"),)


class AnswerApprovalRecord(Identity, Base):
    __tablename__ = "answer_approval_records"
    __table_args__ = (UniqueConstraint("run_id"), UniqueConstraint("answer_id"))
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id", ondelete="CASCADE"), index=True)
    answer_id: Mapped[str] = mapped_column(ForeignKey("answer_revisions.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    decision: Mapped[str] = mapped_column(String(12))
    answer_sha256: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)


class AcceptedAnswerExperience(Identity, Base):
    __tablename__ = "accepted_answer_experiences"
    __table_args__ = (UniqueConstraint("run_id"),)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id", ondelete="CASCADE"), index=True)
    answer_id: Mapped[str] = mapped_column(ForeignKey("answer_revisions.id", ondelete="CASCADE"), index=True)
    corpus_id: Mapped[str] = mapped_column(ForeignKey("corpora.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    answer_sha256: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)


class CodeRepairTask(Identity, Base):
    __tablename__ = "code_repair_tasks"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    repository_path: Mapped[str] = mapped_column(Text)
    workspace_path: Mapped[str] = mapped_column(Text)
    objective: Mapped[str] = mapped_column(Text)
    base_ref: Mapped[str] = mapped_column(String(255))
    base_commit: Mapped[str] = mapped_column(String(64), index=True)
    selected_files: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(24), default="ready", index=True)
    proposed_summary: Mapped[str | None] = mapped_column(Text)
    patch_text: Mapped[str | None] = mapped_column(Text)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64))
    test_exit_code: Mapped[int | None] = mapped_column(Integer)
    test_output: Mapped[str | None] = mapped_column(Text)
    test_environment: Mapped[dict | None] = mapped_column(JSON)
    verification_sha256: Mapped[str | None] = mapped_column(String(64))
    workspace_state: Mapped[str] = mapped_column(String(24), default="active", server_default="active")
    error: Mapped[str | None] = mapped_column(Text)


class CodeRepairReview(Identity, Base):
    __tablename__ = "code_repair_reviews"
    __table_args__ = (UniqueConstraint("task_id"),)
    task_id: Mapped[str] = mapped_column(ForeignKey("code_repair_tasks.id", ondelete="CASCADE"), index=True)
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    decision: Mapped[str] = mapped_column(String(12))
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    bundle_sha256: Mapped[str | None] = mapped_column(String(64))
    reason: Mapped[str | None] = mapped_column(Text)
    feedback_category: Mapped[str] = mapped_column(String(32), default="unclear", server_default="unclear")
    root_cause_layer: Mapped[str] = mapped_column(String(32), default="unknown", server_default="unknown")
    uncertainty: Mapped[str | None] = mapped_column(Text)


class RepairExperience(Identity, Base):
    __tablename__ = "repair_experiences"
    __table_args__ = (UniqueConstraint("task_id"),)
    task_id: Mapped[str] = mapped_column(ForeignKey("code_repair_tasks.id", ondelete="CASCADE"), index=True)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    artifact_sha256: Mapped[str] = mapped_column(String(64))
    objective: Mapped[str] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    verified_command: Mapped[str] = mapped_column(String(100))
    verified_result: Mapped[str] = mapped_column(Text)


class ModelCall(Identity, Base):
    __tablename__ = "model_calls"
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    attempt: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(40))
    model: Mapped[str] = mapped_column(String(200))
    request: Mapped[dict] = mapped_column(JSON)
    response: Mapped[dict] = mapped_column(JSON)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    elapsed_seconds: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(24), default="completed", server_default="completed")
    usage_known: Mapped[bool] = mapped_column(default=False, server_default="false")
    estimated_input_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    reserved_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    provider_request_id: Mapped[str | None] = mapped_column(String(200))


class ExternalOperation(Identity, Base):
    __tablename__ = "external_operations"
    __table_args__ = (UniqueConstraint("owner_id", "idempotency_key"),)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    run_attempt: Mapped[int] = mapped_column(Integer)
    server_id: Mapped[str] = mapped_column(String(80))
    tool_name: Mapped[str] = mapped_column(String(200))
    resource: Mapped[str] = mapped_column(String(1000))
    arguments: Mapped[dict] = mapped_column(JSON)
    binding: Mapped[dict] = mapped_column(JSON)
    digest: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(24), default="proposed", index=True)
    approval_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_token: Mapped[str | None] = mapped_column(String(40))
    execution_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    receipt: Mapped[dict] = mapped_column(JSON, default=dict)


class ExternalOperationEvent(Identity, Base):
    __tablename__ = "external_operation_events"
    operation_id: Mapped[str] = mapped_column(ForeignKey("external_operations.id"), index=True)
    actor_id: Mapped[str] = mapped_column(String(100))
    kind: Mapped[str] = mapped_column(String(40))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)


class MCPInvocation(Identity, Base):
    __tablename__ = "mcp_invocations"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("question_runs.id"), index=True)
    server_id: Mapped[str] = mapped_column(String(80))
    tool_name: Mapped[str] = mapped_column(String(200))
    arguments: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(24), default="started")
    result: Mapped[dict] = mapped_column(JSON, default=dict)


class MemoryEntry(Identity, Base):
    __tablename__ = "memory_entries"
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    corpus_id: Mapped[str | None] = mapped_column(ForeignKey("corpora.id"), index=True)
    kind: Mapped[str] = mapped_column(String(24), index=True)
    key: Mapped[str] = mapped_column(String(160))
    content: Mapped[str] = mapped_column(Text)
    source_run_id: Mapped[str | None] = mapped_column(ForeignKey("question_runs.id"))
    source_question: Mapped[str] = mapped_column(Text, default="")
    scope_sha256: Mapped[str | None] = mapped_column(String(64))
    version: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(24), default="candidate", index=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class MemoryEvent(Identity, Base):
    __tablename__ = "memory_events"
    memory_id: Mapped[str] = mapped_column(ForeignKey("memory_entries.id"), index=True)
    owner_id: Mapped[str] = mapped_column(String(100))
    version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(24))
    # Audit records contain metadata, never a second copy of deleted memory text.
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
