from datetime import timedelta, timezone
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header, Query, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents import memory
from app.core.access import require_role
from app.core.auth import CurrentUser
from app.core.config import get_settings
from app.core.db import get_db
from app.core.errors import DomainError
from app.core.extension_schemas import (
    MemoryCreate,
    MemoryEdit,
    MemoryVersion,
    OperationCreate,
    OperationDecision,
    OperationDigest,
    ReadToolCall,
    Resolution,
)
from app.core.extension_security import protect
from app.core.models import (
    ExternalOperation,
    ExternalOperationEvent,
    MCPInvocation,
    MemoryEntry,
    MemoryEvent,
    QuestionRun,
    Snapshot,
    now,
    uid,
)
from app.integrations import external_operations as operations
from app.integrations.mcp_client import MCPClient, configured_server, load_servers
from app.jobs.services import owned


def same_origin(request: Request):
    origin = request.headers.get("origin")
    if origin and (urlsplit(origin).scheme, urlsplit(origin).netloc) != (
            request.url.scheme, request.url.netloc):
        raise DomainError("不接受其他站点发起的操作请求", 403)


router = APIRouter(dependencies=[Depends(same_origin)])
DB = Annotated[Session, Depends(get_db)]
Settings = Annotated[object, Depends(get_settings)]
Key = Annotated[str, Header(min_length=8, max_length=128)]


def owned_operation(db, operation_id, owner_id):
    row = db.get(ExternalOperation, operation_id)
    if not row or row.owner_id != owner_id:
        raise DomainError("操作不存在或不可访问", 404)
    return row


@router.get("/api/mcp/servers")
def servers(user: CurrentUser, settings: Settings):
    if not settings.mcp_enabled:
        return {"enabled": False, "writes_enabled": False, "servers": []}
    return {"enabled": True, "writes_enabled": settings.mcp_writes_enabled,
            "servers": [{"server_id": s.server_id, "transport": s.transport,
                         "corpus_ids": s.corpus_ids} for s in load_servers(settings).values()
                        if user.id in s.user_ids]}


@router.post("/api/mcp/servers/{server_id}/discover")
async def discover(server_id: str, corpus_id: str, user: CurrentUser, db: DB, settings: Settings):
    require_role(db, corpus_id, user.id)
    server = configured_server(settings, server_id, user.id, corpus_id)
    return {"server_id": server_id, "tools": await MCPClient(settings, server).discover()}


@router.post("/api/mcp/servers/{server_id}/call")
async def call_read(server_id: str, body: ReadToolCall, user: CurrentUser, db: DB, settings: Settings):
    run = owned(db, QuestionRun, body.run_id, user.id)
    snapshot = db.get(Snapshot, run.snapshot_id)
    server = configured_server(settings, server_id, user.id, snapshot.corpus_id)
    if body.tool_name not in server.read_tools:
        raise DomainError("只读接口不允许调用该工具", 403)
    if protect(body.arguments, settings, server.credentials().values()) != body.arguments:
        raise DomainError("工具参数不应包含凭据，请通过环境变量配置认证")
    row = MCPInvocation(id=uid(), owner_id=user.id, run_id=run.id, server_id=server_id,
                        tool_name=body.tool_name, arguments=body.arguments, status="started", result={})
    db.add(row)
    db.commit()
    def before_call():
        db.expire_all()
        latest_run = owned(db, QuestionRun, body.run_id, user.id)
        latest_snapshot = db.get(Snapshot, latest_run.snapshot_id)
        configured_server(settings, server_id, user.id, latest_snapshot.corpus_id)
        db.commit()

    try:
        result = await MCPClient(settings, server).invoke(body.tool_name, body.arguments, before_call=before_call)
        row.status, row.result = "succeeded", result
    except DomainError as exc:
        row.status, row.result = "failed", {"code": exc.code}
        db.commit()
        raise
    db.commit()
    return {"id": row.id, "status": row.status, "result": row.result,
            "knowledge_status": "external_unverified_not_indexed"}


@router.get("/api/mcp/invocations")
def invocations(user: CurrentUser, db: DB):
    return [{"id": row.id, "run_id": row.run_id, "server_id": row.server_id,
             "tool_name": row.tool_name, "arguments": row.arguments, "status": row.status, "result": row.result}
            for row in db.scalars(select(MCPInvocation).where(MCPInvocation.owner_id == user.id).order_by(
                MCPInvocation.created_at.desc()).limit(100))]


@router.post("/api/external-operations", status_code=201)
async def propose(body: OperationCreate, user: CurrentUser, db: DB, settings: Settings, idempotency_key: Key):
    run, corpus_id = operations.bound_run(db, body.run_id, user.id, body.run_attempt)
    server = configured_server(settings, body.server_id, user.id, corpus_id)
    if protect(body.arguments, settings, server.credentials().values()) != body.arguments:
        raise DomainError("操作参数不应包含凭据，请通过环境变量配置认证")
    try:
        row = await operations.propose(db, settings, user.id, body, idempotency_key)
    except IntegrityError:
        db.rollback()
        row = await operations.propose(db, settings, user.id, body, idempotency_key)
    return operations.operation_data(row)


@router.get("/api/external-operations")
def list_operations(user: CurrentUser, db: DB):
    return [operations.operation_data(row) for row in operations.list_operations(db, user.id)]


@router.get("/api/external-operations/{operation_id}")
def read_operation(operation_id: str, user: CurrentUser, db: DB):
    row = owned_operation(db, operation_id, user.id)
    # A lost process cannot resume execution. Conservatively preserve an unknown result.
    if row.status == "executing" and row.execution_started_at and row.execution_started_at.replace(
            tzinfo=timezone.utc) < now() - timedelta(seconds=150):
        row.status, row.receipt = "unknown", {"code": "execution_receipt_missing"}
        operations.event(db, row, user.id, "unknown", automatic_retry=False)
        db.commit()
    return {**operations.operation_data(row), "events": [
        {"kind": e.kind, "actor_id": e.actor_id, "detail": e.detail, "created_at": e.created_at.isoformat()}
        for e in db.scalars(select(ExternalOperationEvent).where(
            ExternalOperationEvent.operation_id == row.id).order_by(ExternalOperationEvent.created_at))]}


@router.post("/api/external-operations/{operation_id}/decision")
def decide(operation_id: str, body: OperationDecision, user: CurrentUser, db: DB, settings: Settings):
    return operations.operation_data(operations.decide(
        db, owned_operation(db, operation_id, user.id), user.id, body,
        settings.mcp_approval_ttl_seconds))


@router.post("/api/external-operations/{operation_id}/execute")
async def execute(operation_id: str, body: OperationDigest, user: CurrentUser, db: DB, settings: Settings):
    row = await operations.execute(db, settings, owned_operation(db, operation_id, user.id), user.id, body.digest)
    return operations.operation_data(row)


@router.post("/api/external-operations/{operation_id}/resolve")
def resolve(operation_id: str, body: Resolution, user: CurrentUser, db: DB, settings: Settings):
    if protect(body.evidence, settings) != body.evidence:
        raise DomainError("核对说明不能包含凭据")
    row = operations.resolve(db, owned_operation(db, operation_id, user.id), user.id, body)
    return operations.operation_data(row)


@router.post("/api/memories", status_code=201)
def create_memory(body: MemoryCreate, user: CurrentUser, db: DB, settings: Settings):
    if protect(body.model_dump(mode="json"), settings) != body.model_dump(mode="json"):
        raise DomainError("记忆不能保存已识别的凭据，请先移除")
    return memory.memory_data(memory.create_candidate(db, user.id, body))


@router.get("/api/memories")
def list_memories(user: CurrentUser, db: DB):
    return [memory.memory_data(row) for row in memory.list_memories(db, user.id)]


@router.post("/api/memories/cleanup")
def cleanup_memories(user: CurrentUser, db: DB):
    rows = list(db.scalars(select(MemoryEntry).where(
        MemoryEntry.owner_id == user.id, MemoryEntry.status != "deleted", MemoryEntry.expires_at <= now())))
    for row in rows:
        memory.delete_memory(db, row, user.id, row.version)
    return {"deleted": len(rows), "historical_run_records_retained": True}


@router.get("/api/memories/{memory_id}")
def read_memory(memory_id: str, user: CurrentUser, db: DB):
    return memory.memory_data(memory.get_owned(db, memory_id, user.id))


@router.post("/api/memories/{memory_id}/confirm")
def confirm_memory(memory_id: str, body: MemoryVersion, user: CurrentUser, db: DB):
    return memory.memory_data(memory.confirm_memory(db, memory.get_owned(db, memory_id, user.id), user.id, body.version))


@router.patch("/api/memories/{memory_id}")
def edit_memory(memory_id: str, body: MemoryEdit, user: CurrentUser, db: DB, settings: Settings):
    if protect(body.model_dump(mode="json"), settings) != body.model_dump(mode="json"):
        raise DomainError("记忆不能保存已识别的凭据，请先移除")
    return memory.memory_data(memory.update_memory(db, memory.get_owned(db, memory_id, user.id), user.id, body))


@router.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str, user: CurrentUser, db: DB, version: int = Query(ge=1)):
    row = db.get(MemoryEntry, memory_id)
    if not row or row.owner_id != user.id or row.status == "deleted":
        raise DomainError("记忆不存在或不可访问", 404)
    memory.delete_memory(db, row, user.id, version)
    return {"deleted": True, "historical_run_records_retained": True}


@router.get("/api/memories/{memory_id}/events")
def memory_events(memory_id: str, user: CurrentUser, db: DB):
    memory.get_owned(db, memory_id, user.id)
    return [{"kind": e.kind, "version": e.version, "detail": e.detail, "created_at": e.created_at.isoformat()}
            for e in db.scalars(select(MemoryEvent).where(
                MemoryEvent.memory_id == memory_id, MemoryEvent.owner_id == user.id).order_by(MemoryEvent.created_at))]
