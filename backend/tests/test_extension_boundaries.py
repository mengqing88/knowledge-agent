import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
from app import external_operations, memory
from app.auth import hash_password
from app.config import get_settings
from app.db import get_db
from app.errors import DomainError
from app.extension_schemas import MemoryCreate, OperationDecision
from app.main import app
from app.mcp_client import validate_arguments
from app.models import ExternalOperation, Membership, QuestionRun, User, now
from fastapi.testclient import TestClient
from test_extensions import make_run, mcp_settings, proposal  # noqa: F401


@pytest.mark.parametrize("schema,args", [
    ({"type": "object", "properties": {"count": {"type": "integer"}}, "additionalProperties": False}, {"count": True}),
    ({"type": "object", "additionalProperties": False}, {"unexpected": 1}),
    ({"$ref": "https://example.invalid/schema"}, {}),
    ({"type": "object"}, {"too_large": "x"*33000}),
])
def test_invalid_schema_arguments_and_network_references_rejected(schema, args):
    with pytest.raises(DomainError, match="参数或 Schema"):
        validate_arguments(schema, args)


def test_memories_reject_other_user_source_scope_and_expired_confirmation(sessions, settings, corpus):
    run_id = make_run(sessions, settings)
    with sessions() as db:
        with pytest.raises(DomainError, match="不可访问"):
            memory.create_candidate(db, "another", MemoryCreate(kind="failure", key="F1", content="lesson", source_run_id=run_id))
        pref = memory.create_candidate(db, "local-user", MemoryCreate(kind="preference", key="style", content="简短中文"))
        with pytest.raises(DomainError, match="不可访问"):
            memory.get_owned(db, pref.id, "another")
        pref.expires_at = now() - timedelta(seconds=1)
        db.commit()
        with pytest.raises(DomainError, match="过期"):
            memory.confirm_memory(db, pref, "local-user", 1)
        assert memory.context_for_run(db, db.get(QuestionRun, run_id)) == []


def test_approval_api_identity_digest_rejection_and_receipts(sessions, corpus, mcp_settings):  # noqa: F811
    run_id = make_run(sessions, mcp_settings)
    with sessions.begin() as db:
        db.add(User(id="reader", username="reader", password_hash=hash_password("test-password")))
        db.flush()
        db.add(Membership(corpus_id="corpus1", user_id="reader", role="reader"))

    def override_db():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: mcp_settings
    try:
        with TestClient(app) as client:
            assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password"}).status_code == 200
            discover = client.post("/api/mcp/servers/fixture/discover?corpus_id=corpus1")
            assert discover.status_code == 200, discover.text
            read = client.post("/api/mcp/servers/fixture/call", json={"run_id": run_id, "tool_name": "echo", "arguments": {"text": "local"}})
            assert read.status_code == 200, read.text
            assert read.json()["knowledge_status"] == "external_unverified_not_indexed"
            row = client.post("/api/external-operations", json=proposal(run_id).model_dump(),
                              headers={"Idempotency-Key": "operation-key-123"})
            assert row.status_code == 201, row.text
            op = row.json()
            assert op["binding"]["endpoint"]["transport"] == "stdio"
            assert "fixture-secret-value" not in json.dumps(op)
            assert client.post(f'/api/external-operations/{op["id"]}/execute', json={"digest": op["digest"]}).status_code == 409
            assert client.post(f'/api/external-operations/{op["id"]}/decision', json={"digest": "0"*64, "decision": "approve"}).status_code == 409
            assert client.post(f'/api/external-operations/{op["id"]}/decision', json={"digest": op["digest"], "decision": "reject"}).json()["status"] == "rejected"
            assert client.post(f'/api/external-operations/{op["id"]}/execute', json={"digest": op["digest"]}).status_code == 409
            assert [e["kind"] for e in client.get(f'/api/external-operations/{op["id"]}').json()["events"]] == ["proposed", "rejected"]
            mcp_settings.mcp_approval_ttl_seconds = 60
            approval_body = proposal(run_id, name="approval-api.json", key="approval-api-key")
            approval_response = client.post("/api/external-operations",
                                            json=approval_body.model_dump(),
                                            headers={"Idempotency-Key": "approval-api-key"})
            assert approval_response.status_code == 201, approval_response.text
            approval = approval_response.json()
            approved = client.post(f'/api/external-operations/{approval["id"]}/decision', json={
                "digest": approval["digest"], "decision": "approve"})
            assert approved.status_code == 200, approved.text
            expires_at = datetime.fromisoformat(approved.json()["approval_expires_at"])
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            assert 55 <= (expires_at - now()).total_seconds() <= 60
            client.post("/api/auth/logout")
            client.post("/api/auth/login", json={"username": "reader", "password": "test-password"})
            assert client.get(f'/api/external-operations/{op["id"]}').status_code == 404
            assert client.post(f'/api/external-operations/{op["id"]}/decision', json={"digest": op["digest"], "decision": "approve"}).status_code == 404
            assert client.get("/api/external-operations").json() == []
            assert client.get("/api/mcp/invocations").json() == []
    finally:
        app.dependency_overrides.clear()


def test_write_schema_change_does_not_dispatch(sessions, corpus, mcp_settings):  # noqa: F811
    run_id = make_run(sessions, mcp_settings)
    with sessions() as db:
        row = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", proposal(run_id), "operation-key-123"))
        external_operations.decide(db, row, "local-user", OperationDecision(digest=row.digest, decision="approve"))
        row.binding = {**row.binding, "schema_sha256": "0"*64}
        row.digest = __import__("app.governance", fromlist=["digest"]).digest(row.binding)
        db.commit()
        with pytest.raises(DomainError):
            asyncio.run(external_operations.execute(db, mcp_settings, row, "local-user", row.digest))
        db.refresh(row)
        assert row.status == "failed"
    assert not (mcp_settings.mcp_servers_file.parent / "saved.json").exists()


def test_approval_reader_cannot_approve_and_revocation_prevents_memory_reuse(sessions, settings, corpus):
    run_id = make_run(sessions, settings)
    with sessions.begin() as db:
        db.add(User(id="editor", username="editor", password_hash="unused"))
        db.flush()
        db.add(Membership(id="editable-membership", corpus_id="corpus1", user_id="editor", role="editor"))
        db.get(QuestionRun, run_id).owner_id = "editor"
    with sessions() as db:
        row = memory.create_candidate(db, "editor", MemoryCreate(kind="success", key="F1", content="M F1 experience", source_run_id=run_id))
        memory.confirm_memory(db, row, "editor", 1)
        context = memory.context_for_run(db, db.get(QuestionRun, run_id))
        db.get(Membership, "editable-membership").role = "reader"
        db.commit()
        with pytest.raises(DomainError, match="权限"):
            external_operations.bound_run(db, run_id, "editor", 0)
        db.delete(db.get(Membership, "editable-membership"))
        db.commit()
        with pytest.raises(DomainError):
            memory.active_frozen_context(db, "editor", context)
        assert memory.list_memories(db, "editor") == []


def test_lost_receipt_stays_non_replayable_and_stale_execution_becomes_unknown(sessions, corpus, mcp_settings):  # noqa: F811
    run_id = make_run(sessions, mcp_settings)
    with sessions() as db:
        row = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", proposal(run_id), "operation-key-123"))
        external_operations.decide(db, row, "local-user", OperationDecision(digest=row.digest, decision="approve"))
        operation_id, bound_digest = row.id, row.digest
        actual_commit = db.commit
        count = 0

        def fail_receipt_commit():
            nonlocal count
            count += 1
            if count == 3:  # executing audit, pre-dispatch guard, then result receipt
                raise RuntimeError("receipt storage unavailable")
            actual_commit()

        db.commit = fail_receipt_commit
        with pytest.raises(RuntimeError, match="receipt storage"):
            asyncio.run(external_operations.execute(db, mcp_settings, row, "local-user", row.digest))
        db.rollback()
    with sessions.begin() as db:
        row = db.get(ExternalOperation, operation_id)
        assert row.status == "executing"
        row.execution_started_at = now() - timedelta(seconds=160)

    def override_db():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: mcp_settings
    try:
        with TestClient(app) as client:
            client.post("/api/auth/login", json={"username": "admin", "password": "test-password"})
            row = client.get(f"/api/external-operations/{operation_id}").json()
            assert row["status"] == "unknown"
            assert client.post(f"/api/external-operations/{operation_id}/execute", json={"digest": bound_digest}).status_code == 409
            assert client.post(f"/api/external-operations/{operation_id}/resolve", json={
                "outcome": "succeeded", "evidence": "人工核对临时服务中的资源、内容和幂等键一致"}).json()["status"] == "succeeded"
    finally:
        app.dependency_overrides.clear()
