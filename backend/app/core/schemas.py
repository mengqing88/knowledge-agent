from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusCreate(StrictModel):
    name: str = Field(min_length=1, max_length=160)


class Login(StrictModel):
    username: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class Register(Login):
    pass


class MemberCreate(StrictModel):
    username: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str | None = Field(None, min_length=8, max_length=128)
    role: Literal["admin", "editor", "reader"]


class RoleUpdate(StrictModel):
    role: Literal["admin", "editor", "reader"]


class DocumentUpdate(StrictModel):
    title: str = Field(min_length=1, max_length=500)
    folder: str = Field(default="", max_length=160)
    tags: list[str] = Field(default_factory=list, max_length=12)

    @field_validator("title", "folder")
    @classmethod
    def trimmed(cls, value):
        return value.strip()

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values):
        if any(not x.strip() or len(x) > 40 for x in values):
            raise ValueError("标签需要为 1–40 个字符")
        return list(dict.fromkeys(x.strip() for x in values))


class RankedChunk(StrictModel):
    chunk_id: str
    score: float = Field(ge=0, le=1)
    relevant_block_ids: list[str] = Field(default_factory=list, max_length=16)


class Reranking(StrictModel):
    items: list[RankedChunk] = Field(default_factory=list, max_length=24)


class SnapshotCreate(StrictModel):
    paper_version_ids: list[str] = Field(min_length=1, max_length=100)


class RunCreate(StrictModel):
    snapshot_id: str
    question: str = Field(min_length=2, max_length=4000)
    parent_run_id: str | None = None
    mode: Literal["baseline_rag", "full_agent", "without_verifier"] = "full_agent"

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("问题至少需要两个非空白字符")
        return value.strip()


class Subquestion(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    target: str = Field(min_length=1, max_length=400)
    queries: list[str] = Field(min_length=1, max_length=3)

    @field_validator("queries")
    @classmethod
    def bounded_queries(cls, values):
        if any(not q.strip() or len(q) > 500 for q in values):
            raise ValueError("检索表达必须为 1–500 字符")
        return values


class Plan(StrictModel):
    normalized_question: str = Field(min_length=1, max_length=4000)
    needs_clarification: bool = False
    clarification: str | None = Field(None, max_length=1000)
    subquestions: list[Subquestion] = Field(default_factory=list, max_length=5)


class ToolDecision(StrictModel):
    tool_identity: str | None = Field(None, max_length=300)
    arguments: dict = Field(default_factory=dict)
    reason: str = Field(min_length=1, max_length=500)


class EvidenceReadiness(StrictModel):
    subquestion_id: str = Field(min_length=1, max_length=40)
    status: Literal["ready", "partial", "missing", "uncertain"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    reason: str = Field(min_length=2, max_length=500)
    followup_queries: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("reason")
    @classmethod
    def nonblank_reason(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("证据检查需要明确理由")
        return value.strip()

    @field_validator("followup_queries")
    @classmethod
    def bounded_queries(cls, values):
        if any(not q.strip() or len(q) > 500 for q in values):
            raise ValueError("补检表达必须为 1–500 字符")
        return values


class EvidenceGate(StrictModel):
    reviews: list[EvidenceReadiness] = Field(min_length=1, max_length=5)


Relation = Literal["supports", "qualifies", "contradicts", "background"]


class EvidenceLink(StrictModel):
    evidence_id: str = Field(max_length=40)
    relation: Relation


class NumericFact(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=40)
    quote: str = Field(min_length=1, max_length=2000)
    subject: str = Field(min_length=1, max_length=120)
    metric: str = Field(min_length=1, max_length=120)
    value: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    unit: str = Field(min_length=1, max_length=40)
    conditions: list[str] = Field(default_factory=list, max_length=12)


class NumericCheck(StrictModel):
    statement: str = Field(min_length=1, max_length=2000)
    operation: Literal["reported", "difference", "relative_change"] = "reported"
    value: str = Field(pattern=r"^-?\d+(?:\.\d+)?$", max_length=40)
    unit: str = Field(min_length=1, max_length=40)
    facts: list[NumericFact] = Field(min_length=1, max_length=2)


class DraftClaim(StrictModel):
    id: str = Field(min_length=1, max_length=40)
    text: str = Field(min_length=1, max_length=2000)
    conditions: list[str] = Field(default_factory=list, max_length=12)
    evidence: list[EvidenceLink] = Field(min_length=1, max_length=12)
    subquestion_ids: list[str] = Field(default_factory=list, max_length=5)
    numeric_checks: list[NumericCheck] = Field(default_factory=list, max_length=12)


class SubquestionGap(StrictModel):
    subquestion_id: str = Field(min_length=1, max_length=40)
    reason: str = Field(min_length=1, max_length=1000)


class Analysis(StrictModel):
    claims: list[DraftClaim] = Field(default_factory=list, max_length=20)
    gaps: list[str] = Field(default_factory=list, max_length=12)
    subquestion_gaps: list[SubquestionGap] = Field(default_factory=list, max_length=5)
    has_conflict: bool = False


class VerificationIssue(StrictModel):
    claim_id: str | None = None
    type: Literal["invalid_reference", "unsupported_claim", "overgeneralization", "numeric_mismatch",
                  "comparison_mismatch", "missing_evidence"]
    message: str = Field(min_length=1, max_length=1000)


class ClaimReview(StrictModel):
    claim_id: str = Field(min_length=1, max_length=40)
    verdict: Literal["approved", "rejected", "uncertain"]
    evidence_ids: list[str] = Field(default_factory=list, max_length=12)
    rationale: str = Field(min_length=1, max_length=1000)


class Verification(StrictModel):
    issues: list[VerificationIssue] = Field(default_factory=list, max_length=30)
    approved_claim_ids: list[str] = Field(default_factory=list, max_length=20)
    followup_queries: list[str] = Field(default_factory=list, max_length=3)
    claim_reviews: list[ClaimReview] = Field(default_factory=list, max_length=20)


class ReviewCreate(StrictModel):
    text: str = Field(min_length=2, max_length=4000)
    category: Literal["preference", "factual_error", "execution_failure", "missing_scenario",
                      "requirement_change", "unable_to_complete", "unclear"] = "unclear"

    @field_validator("text")
    @classmethod
    def meaningful_feedback(cls, value):
        if len(value.strip()) < 2:
            raise ValueError("请补充具体反馈")
        return value


class AnswerApprovalCreate(StrictModel):
    decision: Literal["accept", "reject"]
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reason: str | None = Field(default=None, max_length=4000)

    @model_validator(mode="after")
    def rejection_requires_reason(self):
        if self.decision == "reject" and (not self.reason or len(self.reason.strip()) < 3):
            raise ValueError("拒绝答案时请说明原因")
        return self


class CodeTaskCreate(StrictModel):
    repository_path: str = Field(min_length=1, max_length=500)
    base_ref: str = Field(default="HEAD", min_length=1, max_length=255)
    objective: str = Field(min_length=5, max_length=4000)


class CodeTaskPropose(StrictModel):
    files: list[str] = Field(min_length=1, max_length=8)


class CodeRepairPatch(StrictModel):
    summary: str = Field(min_length=5, max_length=1000)
    diff: str = Field(min_length=1, max_length=30000)


class CodeTaskReview(StrictModel):
    decision: Literal["accept", "reject"]
    reason: str | None = Field(default=None, max_length=4000)
    feedback_category: Literal[
        "preference", "factual_error", "execution_failure", "missing_scenario",
        "requirement_change", "unable_to_complete", "unclear",
    ] = "unclear"
    root_cause_layer: Literal[
        "prompt", "knowledge", "tool_contract", "authorization", "workflow",
        "code", "environment", "unknown",
    ] = "unknown"
    uncertainty: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def rejection_requires_reason(self):
        if self.decision == "reject" and (not self.reason or len(self.reason.strip()) < 3):
            raise ValueError("拒绝结果时请说明原因")
        return self
