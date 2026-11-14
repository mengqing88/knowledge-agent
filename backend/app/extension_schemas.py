from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StrictInt, StringConstraints

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolCall(StrictBody):
    tool_name: Name
    arguments: dict = Field(default_factory=dict)


class ReadToolCall(ToolCall):
    run_id: Name


class OperationCreate(ToolCall):
    server_id: Name
    run_id: Name
    run_attempt: StrictInt = Field(ge=0)
    resource: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class OperationDecision(StrictBody):
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["approve", "reject"]


class OperationDigest(StrictBody):
    digest: str = Field(pattern=r"^[a-f0-9]{64}$")


class Resolution(StrictBody):
    outcome: Literal["succeeded", "failed"]
    evidence: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=2000)]


class MemoryCreate(StrictBody):
    kind: Literal["preference", "success", "failure"]
    key: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    source_run_id: Name | None = None
    expires_at: AwareDatetime | None = None


class MemoryVersion(StrictBody):
    version: StrictInt = Field(ge=1)


class MemoryEdit(MemoryVersion):
    content: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    expires_at: AwareDatetime | None = None
