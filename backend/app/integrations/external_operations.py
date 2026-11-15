from datetime import timedelta, timezone

from sqlalchemy import select, update

from app.agents.governance import digest
from app.core.access import require_role
from app.core.errors import DomainError
from app.core.extension_security import protect
from app.core.models import ExternalOperation, ExternalOperationEvent, QuestionRun, Snapshot, now, uid
from app.integrations.mcp_client import MCPClient, MCPFailure, configured_server, validate_arguments


def bound_run(db, run_id, owner_id, attempt):
    run = db.get(QuestionRun, run_id)
    if not run or run.owner_id != owner_id:
        raise DomainError("任务不存在或不可访问", 404)
    if run.attempt != attempt or run.status in ("cancelled", "failed"):
        raise DomainError("任务 attempt 已改变或已经取消，需重新提出操作", 409)
    snapshot = db.get(Snapshot, run.snapshot_id)
    if not snapshot:
        raise DomainError("任务快照不存在", 409)
    require_role(db, snapshot.corpus_id, owner_id, "editor")
    return run, snapshot.corpus_id


def event(db, operation, actor_id, kind, **detail):
    db.add(ExternalOperationEvent(operation_id=operation.id, actor_id=actor_id,
                                  kind=kind, detail=detail))


def approval_expired(expires_at):
    if not expires_at:
        return False
    # SQLite drops timezone information for DateTime(timezone=True), while
    # PostgreSQL returns aware UTC values. Interpret stored naive values as UTC.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= now()


def operation_data(row):
    return {"id": row.id, "run_id": row.run_id, "run_attempt": row.run_attempt,
            "server_id": row.server_id, "tool_name": row.tool_name, "resource": row.resource,
            "arguments": row.arguments, "digest": row.digest, "idempotency_key": row.idempotency_key,
            "status": row.status, "receipt": row.receipt, "binding": row.binding,
            "approval_expires_at": row.approval_expires_at.isoformat() if row.approval_expires_at else None,
            "created_at": row.created_at.isoformat()}


def verify_binding(row):
    binding = row.binding
    fields = {"owner_id": row.owner_id, "run_id": row.run_id, "run_attempt": row.run_attempt,
              "server_id": row.server_id, "tool_name": row.tool_name, "resource": row.resource,
              "idempotency_key": row.idempotency_key, "arguments_sha256": digest(row.arguments)}
    if digest(binding) != row.digest or any(binding.get(key) != value for key, value in fields.items()):
        raise DomainError("审批绑定数据已变化，拒绝执行", 409)


async def propose(db, settings, owner_id, body, idempotency_key):
    run, corpus_id = bound_run(db, body.run_id, owner_id, body.run_attempt)
    server = configured_server(settings, body.server_id, owner_id, corpus_id)
    policy = server.write_tools.get(body.tool_name)
    if not policy:
        raise DomainError("该工具未配置为需审批的写工具", 403)
    client = MCPClient(settings, server)
    if protect(body.arguments, settings, server.credentials().values()) != body.arguments:
        raise DomainError("操作参数不能包含凭据，请通过环境变量配置认证")
    schema = await client.schema(body.tool_name)
    validate_arguments(schema, body.arguments)
    if (body.arguments.get(policy.resource_argument) != body.resource or
            body.arguments.get(policy.idempotency_argument) != idempotency_key):
        raise DomainError("目标资源与幂等键必须和完整工具参数一致")
    binding = {"owner_id": owner_id, "run_id": run.id, "run_attempt": run.attempt,
               "run_config_sha256": digest(run.config),
               "server_id": server.server_id, "server_fingerprint": server.fingerprint(),
               "endpoint": {"transport": server.transport, "url": server.url,
                            "command": server.command, "args_sha256": digest(server.args)},
               "tool_name": body.tool_name, "schema_sha256": digest(schema),
               "arguments_sha256": digest(body.arguments), "resource": body.resource,
               "idempotency_key": idempotency_key}
    operation_digest = digest(binding)
    existing = db.scalar(select(ExternalOperation).where(
        ExternalOperation.owner_id == owner_id, ExternalOperation.idempotency_key == idempotency_key))
    if existing:
        if existing.digest != operation_digest:
            raise DomainError("幂等键已用于其他操作", 409)
        return existing
    expires_at = now() + timedelta(seconds=settings.mcp_approval_ttl_seconds)
    row = ExternalOperation(id=uid(), owner_id=owner_id, run_id=run.id, run_attempt=run.attempt,
                            server_id=server.server_id, tool_name=body.tool_name, resource=body.resource,
                            arguments=body.arguments, binding=binding, digest=operation_digest,
                            idempotency_key=idempotency_key, status="proposed", receipt={},
                            approval_expires_at=expires_at)
    db.add(row)
    db.flush()
    event(db, row, owner_id, "proposed", digest=operation_digest)
    db.commit()
    return row


def decide(db, row, owner_id, body, approval_ttl_seconds=900):
    if row.owner_id != owner_id:
        raise DomainError("操作不存在或不可访问", 404)
    verify_binding(row)
    if row.digest != body.digest:
        raise DomainError("操作内容与审批摘要不一致", 409)
    if row.status != "proposed":
        raise DomainError("该操作不再等待审批", 409)
    if approval_expired(row.approval_expires_at):
        row.status = "expired"
        event(db, row, owner_id, "expired", digest=row.digest)
        db.commit()
        raise DomainError("审批已过期，请重新提出操作", 409, "approval_expired")
    run, _ = bound_run(db, row.run_id, owner_id, row.run_attempt)
    status = "approved" if body.decision == "approve" else "rejected"
    # Keep the approval window configurable and apply it only after an
    # approval decision.  The proposal window prevents abandoned proposals
    # from living forever; a newly approved action receives a fresh window.
    if status == "approved":
        if approval_ttl_seconds is None or approval_ttl_seconds < 1:
            raise DomainError("审批有效期配置无效", 500, "invalid_approval_ttl")
        expires_at = now() + timedelta(seconds=approval_ttl_seconds)
    else:
        expires_at = None
    result = db.execute(update(ExternalOperation).where(
        ExternalOperation.id == row.id, ExternalOperation.status == "proposed",
        ExternalOperation.digest == body.digest).values(status=status,
            approval_expires_at=expires_at))
    if not result.rowcount:
        db.rollback()
        raise DomainError("操作状态已改变，请刷新后重试", 409)
    event(db, row, owner_id, status, digest=body.digest, run_attempt=run.attempt)
    db.commit()
    db.refresh(row)
    return row


async def execute(db, settings, row, owner_id, supplied_digest):
    verify_binding(row)
    if row.owner_id != owner_id or row.digest != supplied_digest:
        raise DomainError("操作不存在、不可访问或摘要不匹配", 404)
    if row.status != "approved":
        raise DomainError("只有已批准的操作可以执行", 409)
    if not row.approval_expires_at or approval_expired(row.approval_expires_at):
        row.status = "expired"
        event(db, row, owner_id, "expired", digest=row.digest)
        db.commit()
        raise DomainError("审批已过期，请重新提出操作", 409, "approval_expired")
    if not settings.mcp_writes_enabled:
        raise DomainError("外部写操作未在服务端启用", 503, "mcp_writes_disabled")
    run, corpus_id = bound_run(db, row.run_id, owner_id, row.run_attempt)
    if digest(run.config) != row.binding["run_config_sha256"]:
        raise DomainError("任务配置已变化，请重新提出并审批操作", 409)
    server = configured_server(settings, row.server_id, owner_id, corpus_id)
    if server.fingerprint() != row.binding["server_fingerprint"]:
        raise DomainError("服务端点、权限或凭据已变化，请重新提出并审批操作", 409)
    policy = server.write_tools.get(row.tool_name)
    if not policy or row.arguments.get(policy.resource_argument) != row.resource or row.arguments.get(
            policy.idempotency_argument) != row.idempotency_key:
        raise DomainError("工具策略或操作参数已变化", 409)
    validate_arguments(await MCPClient(settings, server).schema(row.tool_name), row.arguments)
    token = uid()
    result = db.execute(update(ExternalOperation).where(
        ExternalOperation.id == row.id, ExternalOperation.status == "approved",
        ExternalOperation.digest == supplied_digest).values(
            status="executing", execution_token=token, execution_started_at=now()))
    if not result.rowcount:
        db.rollback()
        raise DomainError("操作状态已改变，拒绝重复执行", 409)
    event(db, row, owner_id, "execution_started", digest=supplied_digest)
    db.commit()

    def before_call():
        db.expire_all()
        current = db.get(ExternalOperation, row.id)
        current_run = db.get(QuestionRun, row.run_id)
        if (not current or current.status != "executing" or current.execution_token != token or
                not current_run or current_run.owner_id != owner_id or current_run.attempt != row.run_attempt or
                current_run.status in ("cancelled", "failed")):
            db.rollback()
            raise DomainError("执行前操作或任务状态已改变", 409)
        verify_binding(current)
        _, latest_corpus_id = bound_run(db, row.run_id, owner_id, row.run_attempt)
        latest_server = configured_server(settings, row.server_id, owner_id, latest_corpus_id)
        if (latest_server.fingerprint() != row.binding["server_fingerprint"] or
                digest(current_run.config) != row.binding["run_config_sha256"]):
            raise DomainError("执行前权限或配置已变化", 409)
        db.commit()

    try:
        receipt = await MCPClient(settings, server).invoke(
            row.tool_name, row.arguments, write=True,
            expected_schema=row.binding["schema_sha256"], before_call=before_call)
    except MCPFailure as exc:
        db.expire_all()
        current = db.get(ExternalOperation, row.id)
        if current and current.status == "executing" and current.execution_token == token:
            current.status = "unknown" if exc.dispatched else "failed"
            current.receipt = {"code": exc.code}
            event(db, current, owner_id, current.status, code=exc.code,
                  dispatched=exc.dispatched, automatic_retry=False)
            db.commit()
        raise DomainError(str(exc), exc.status, exc.code) from None
    except DomainError:
        db.expire_all()
        current = db.get(ExternalOperation, row.id)
        if current and current.status == "executing" and current.execution_token == token:
            current.status = "failed"
            current.receipt = {"code": "pre_dispatch_validation_failed"}
            event(db, current, owner_id, "failed", dispatched=False)
            db.commit()
        raise
    db.expire_all()
    current = db.get(ExternalOperation, row.id)
    if not current or current.status != "executing" or current.execution_token != token:
        raise DomainError("操作已发送，回执写入状态未知；请人工核对，不要重试", 503, "operation_receipt_unknown")
    current.status, current.receipt = "succeeded", receipt
    event(db, current, owner_id, "succeeded", digest=supplied_digest)
    db.commit()
    db.refresh(current)
    return current


def resolve(db, row, owner_id, body):
    db.refresh(row, with_for_update=True)
    if row.owner_id != owner_id or row.status != "unknown":
        raise DomainError("只有待核对或结果未知的本人操作可登记核对结果", 409)
    row.status = body.outcome
    row.receipt = {**(row.receipt or {}), "manual_resolution": body.evidence}
    event(db, row, owner_id, "manually_resolved", outcome=body.outcome)
    db.commit()
    db.refresh(row)
    return row


def list_operations(db, owner_id, run_id=None):
    stmt = select(ExternalOperation).where(ExternalOperation.owner_id == owner_id)
    if run_id:
        stmt = stmt.where(ExternalOperation.run_id == run_id)
    return list(db.scalars(stmt.order_by(ExternalOperation.created_at.desc()).limit(100)))
