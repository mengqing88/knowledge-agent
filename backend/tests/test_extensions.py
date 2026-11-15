import asyncio
import copy
import json
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path

import httpx
import pytest
from app.agents import memory
from app.api.main import app
from app.core.db import get_db
from app.core.errors import DomainError
from app.core.extension_schemas import (
    MemoryCreate,
    MemoryEdit,
    OperationCreate,
    OperationDecision,
    Resolution,
)
from app.core.models import MemoryEntry, QuestionRun, now
from app.core.schemas import RunCreate
from app.integrations import external_operations
from app.integrations.mcp_client import MCPClient, ServerConfig, configured_server, load_servers
from app.jobs.services import claim_job, create_run
from app.jobs.worker import process_job
from conftest import FakeProvider
from fastapi.testclient import TestClient

FIXTURE = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


@pytest.fixture
def mcp_settings(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("MCP_FIXTURE_TOKEN", "fixture-secret-value")
    monkeypatch.setenv("MCP_FIXTURE_STORE", str(tmp_path))
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-be-inherited")
    settings.mcp_enabled, settings.mcp_writes_enabled = True, True
    settings.mcp_servers_file = tmp_path / "servers.json"
    config = {"servers": [{"server_id": "fixture", "transport": "stdio", "command": sys.executable,
                           "args": [str(FIXTURE)], "env": {"MCP_FIXTURE_TOKEN": "MCP_FIXTURE_TOKEN",
                                                               "MCP_FIXTURE_STORE": "MCP_FIXTURE_STORE"},
                           "user_ids": ["local-user"], "corpus_ids": ["corpus1"],
                           "read_tools": ["echo"], "write_tools": {"save_note": {
                               "resource_argument": "resource", "idempotency_argument": "idempotency_key"}},
                           "timeout_seconds": 10}]}
    settings.mcp_servers_file.write_text(json.dumps(config), encoding="utf-8")
    return settings


def make_run(sessions, settings, key="extensions-run-1", status="completed"):
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M F1 dataset X"),
                         key, "local-user", settings)
        run.status = status
        db.commit()
        return run.id


def test_mcp_default_closed_and_strict_configuration(settings, tmp_path):
    with pytest.raises(DomainError, match="未启用"):
        load_servers(settings)
    with pytest.raises(ValueError):
        ServerConfig(server_id="unsafe", transport="streamable_http", url="http://example.com/mcp",
                     user_ids=["u"], corpus_ids=["c"])
    with pytest.raises(ValueError):
        ServerConfig(server_id="unsafe", transport="streamable_http", url="https://user:key@example.com/mcp",
                     user_ids=["u"], corpus_ids=["c"])


def test_mcp_stdio_discovery_allowlist_schema_credentials_and_no_secret_inheritance(mcp_settings):
    server = configured_server(mcp_settings, "fixture", "local-user", "corpus1")
    client = MCPClient(mcp_settings, server)
    names = {row["name"] for row in asyncio.run(client.discover())}
    assert names == {"echo", "save_note"}
    with pytest.raises(DomainError, match="授权"):
        asyncio.run(client.invoke("unlisted", {}))
    with pytest.raises(DomainError):
        asyncio.run(client.invoke("echo", {"text": False}))
    response = asyncio.run(client.invoke("echo", {"text": "hello"}))
    assert "hello" in json.dumps(response)
    assert "fixture-secret-value" not in json.dumps(response)
    assert json.loads(response["content"][0]["text"])["unexpected_environment"] is False
    with pytest.raises(DomainError, match="未授权"):
        configured_server(mcp_settings, "fixture", "other", "corpus1")


def test_mcp_streamable_http_real_local_protocol(mcp_settings):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(FIXTURE), str(port)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, env={**os.environ, "UNRELATED_SECRET": ""},
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            try:
                httpx.get(url, timeout=.2, trust_env=False)
                break
            except httpx.TransportError:
                time.sleep(.1)
        server = ServerConfig(server_id="httpfixture", transport="streamable_http", url=url,
                              read_tools=["echo"], user_ids=["local-user"], corpus_ids=["corpus1"],
                              timeout_seconds=10)
        client = MCPClient(mcp_settings, server)
        response = asyncio.run(client.invoke("echo", {"text": "http protocol"}))
        assert json.loads(response["content"][0]["text"])["text"] == "http protocol"
    finally:
        process.terminate()
        process.wait(timeout=8)


def proposal(run_id, *, name="saved.json", key="operation-key-123", delay=False):
    return OperationCreate(server_id="fixture", tool_name="save_note", run_id=run_id,
                           run_attempt=0, resource=name, arguments={"resource": name,
                           "idempotency_key": key, "text": "temporary fixture", "delay": delay})


def test_write_approval_real_dispatch_idempotency_and_tamper_protection(sessions, corpus, mcp_settings, tmp_path):
    run_id = make_run(sessions, mcp_settings)
    body = proposal(run_id)
    with sessions() as db:
        op = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", body, "operation-key-123"))
        assert not (tmp_path / "saved.json").exists()
        with pytest.raises(DomainError, match="已批准"):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        with pytest.raises(DomainError, match="摘要"):
            external_operations.decide(db, op, "local-user", OperationDecision(digest="0"*64, decision="approve"))
        external_operations.decide(db, op, "local-user", OperationDecision(digest=op.digest, decision="approve"))
        outcome = asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        assert outcome.status == "succeeded"
        original = (tmp_path / "saved.json").read_bytes()
        with pytest.raises(DomainError, match="已批准"):
            asyncio.run(external_operations.execute(db, mcp_settings, outcome, "local-user", outcome.digest))
        assert (tmp_path / "saved.json").read_bytes() == original
        duplicate = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", body, "operation-key-123"))
        assert duplicate.id == op.id
        duplicate.arguments = {**duplicate.arguments, "text": "tampered"}
        db.commit()
        with pytest.raises(DomainError, match="绑定"):
            asyncio.run(external_operations.execute(db, mcp_settings, duplicate, "local-user", duplicate.digest))


def test_expired_approval_is_persisted_and_never_dispatches(sessions, corpus, mcp_settings, tmp_path):
    run_id = make_run(sessions, mcp_settings)
    mcp_settings.mcp_approval_ttl_seconds = 75
    with sessions() as db:
        op = asyncio.run(external_operations.propose(
            db, mcp_settings, "local-user", proposal(run_id), "operation-key-123"))
        external_operations.decide(db, op, "local-user", OperationDecision(
            digest=op.digest, decision="approve"), mcp_settings.mcp_approval_ttl_seconds)
        db.refresh(op)
        expires_at = op.approval_expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        assert 70 <= (expires_at - now()).total_seconds() <= 75

        op.approval_expires_at = now().replace(tzinfo=timezone.utc) - timedelta(seconds=1)
        db.commit()
        db.expire(op)
        with pytest.raises(DomainError) as exc:
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        assert exc.value.code == "approval_expired"
        db.refresh(op)
        assert op.status == "expired"
    assert not (tmp_path / "saved.json").exists()


def test_write_timeout_unknown_no_replay_and_manual_receipt(sessions, corpus, mcp_settings, tmp_path):
    run_id = make_run(sessions, mcp_settings)
    document = json.loads(mcp_settings.mcp_servers_file.read_text())
    document["servers"][0]["timeout_seconds"] = 3
    mcp_settings.mcp_servers_file.write_text(json.dumps(document))
    with sessions() as db:
        op = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", proposal(
            run_id, name="unknown.json", delay=True), "operation-key-123"))
        external_operations.decide(db, op, "local-user", OperationDecision(digest=op.digest, decision="approve"))
        with pytest.raises(DomainError):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        db.refresh(op)
        assert op.status == "unknown"
        assert (tmp_path / "unknown.json").exists()
        with pytest.raises(DomainError):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        result = external_operations.resolve(db, op, "local-user", Resolution(
            outcome="succeeded", evidence="已核对临时服务保存文件与同一幂等键一致"))
        assert result.status == "succeeded"


def test_write_rechecks_config_attempt_and_audit_before_dispatch(sessions, corpus, mcp_settings, monkeypatch):
    run_id = make_run(sessions, mcp_settings)
    with sessions() as db:
        op = asyncio.run(external_operations.propose(db, mcp_settings, "local-user", proposal(run_id), "operation-key-123"))
        external_operations.decide(db, op, "local-user", OperationDecision(digest=op.digest, decision="approve"))
        document = json.loads(mcp_settings.mcp_servers_file.read_text())
        document["servers"][0]["read_tools"] = []
        mcp_settings.mcp_servers_file.write_text(json.dumps(document))
        with pytest.raises(DomainError, match="变化"):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        document["servers"][0]["read_tools"] = ["echo"]
        mcp_settings.mcp_servers_file.write_text(json.dumps(document))
        db.get(QuestionRun, run_id).attempt = 1
        db.commit()
        with pytest.raises(DomainError, match="attempt"):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
        db.get(QuestionRun, run_id).attempt = 0
        db.commit()
        monkeypatch.setattr(db, "commit", lambda: (_ for _ in ()).throw(RuntimeError("audit unavailable")))
        with pytest.raises(RuntimeError, match="audit"):
            asyncio.run(external_operations.execute(db, mcp_settings, op, "local-user", op.digest))
    assert not (mcp_settings.mcp_servers_file.parent / "saved.json").exists()


def test_memory_confirmation_context_separate_quotas_versions_deletion_and_scope(sessions, settings, corpus):
    run_id = make_run(sessions, settings)
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        pref = memory.create_candidate(db, "local-user", MemoryCreate(kind="preference", key="style", content="请用中文简短解释"))
        assert memory.context_for_run(db, run) == []
        memory.confirm_memory(db, pref, "local-user", pref.version)
        for kind in ("success", "failure"):
            for index in range(3):
                row = memory.create_candidate(db, "local-user", MemoryCreate(
                    kind=kind, key=f"F1-{kind}-{index}", content="M F1 dataset X 原文核对经验", source_run_id=run_id))
                memory.confirm_memory(db, row, "local-user", row.version)
        context = memory.context_for_run(db, run)
        assert len([r for r in context if r["memory_kind"] == "success"]) == 2
        assert len([r for r in context if r["memory_kind"] == "failure"]) == 2
        assert len([r for r in context if r["memory_kind"] == "preference"]) == 1
        duplicate = memory.create_candidate(db, "local-user", MemoryCreate(kind="preference", key="style", content="其他样式"))
        with pytest.raises(DomainError, match="同名"):
            memory.confirm_memory(db, duplicate, "local-user", duplicate.version)
        memory.update_memory(db, pref, "local-user", MemoryEdit(version=1, content="用例子解释"))
        with pytest.raises(DomainError, match="更新"):
            memory.confirm_memory(db, pref, "local-user", 1)
        assert not [r for r in memory.active_frozen_context(db, "local-user", context) if r["id"] == pref.id]
        memory.confirm_memory(db, pref, "local-user", 2)
        memory.delete_memory(db, pref, "local-user", 2)
        assert db.get(MemoryEntry, pref.id).content == ""
        assert memory.active_frozen_context(db, "other-user", context) == []
        snapshot = db.get(__import__("app.core.models", fromlist=["Snapshot"]).Snapshot, "snap1")
        snapshot.manifest = [{**snapshot.manifest[0], "sha256": "b"*64}]
        db.commit()
        assert not [r for r in memory.context_for_run(db, run) if r["memory_kind"] in ("success", "failure")]


def test_confirmed_memory_reaches_next_model_request_but_candidates_do_not(sessions, settings, corpus):
    with sessions() as db:
        candidate = memory.create_candidate(db, "local-user", MemoryCreate(kind="preference", key="candidate", content="待确认偏好"))
        preference = memory.create_candidate(db, "local-user", MemoryCreate(kind="preference", key="style", content="中文简短解释"))
        memory.confirm_memory(db, preference, "local-user", preference.version)
    run_id = make_run(sessions, settings, status="queued")

    class Capture(FakeProvider):
        def __init__(self):
            super().__init__()
            self.inputs = {}

        def structured(self, role, prompt, data, schema):
            self.inputs[role] = copy.deepcopy(data)
            return super().structured(role, prompt, data, schema)

    provider = Capture()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    assert provider.inputs["Planner"]["memory_hints"][0]["id"] == preference.id
    assert provider.inputs["Analyst"]["memory_hints"][0]["id"] == preference.id
    assert candidate.id not in json.dumps(provider.inputs)
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "completed"


def test_extension_api_auth_confirmation_scope_cleanup_and_default_closed(sessions, settings, corpus):
    from app.core.config import get_settings

    def override_db():
        with sessions() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(app) as client:
            assert client.get("/api/memories").status_code == 401
            assert client.post("/api/auth/login", json={"username": "admin", "password": "test-password"}).status_code == 200
            assert client.get("/api/mcp/servers").json()["enabled"] is False
            response = client.post("/api/memories", json={"kind": "preference", "key": "style", "content": "简短中文"})
            assert response.status_code == 201, response.text
            row = response.json()
            assert row["status"] == "candidate"
            assert client.post(f'/api/memories/{row["id"]}/confirm', json={"version": 1}).json()["status"] == "confirmed"
            assert client.patch(f'/api/memories/{row["id"]}', json={"version": 1, "content": "更多例子"}).json()["status"] == "candidate"
            assert client.post("/api/memories", json={"kind": "preference", "key": "secret", "content": "api_key=abcdefghi"}).status_code == 400
            assert client.post("/api/memories", json={"kind": "preference", "key": "bad", "content": "text"},
                               headers={"Origin": "https://other.example"}).status_code == 403
            with sessions.begin() as db:
                db.get(MemoryEntry, row["id"]).expires_at = now() - timedelta(seconds=1)
            assert client.post("/api/memories/cleanup").json()["deleted"] == 1
            assert client.get(f'/api/memories/{row["id"]}').status_code == 404
    finally:
        app.dependency_overrides.clear()
