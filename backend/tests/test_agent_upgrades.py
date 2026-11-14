import json
from unittest.mock import Mock

import pytest
from app import provider_quota
from app.errors import DomainError, ProviderError
from app.evidence import evidence_data, validate_evidence
from app.governance import role_rates
from app.model_routing import ModelRoute, resolve_route
from app.models import EvidenceSpan, MCPInvocation, ModelCall, QuestionRun, RunEvent, Snapshot
from app.providers import ModelProvider
from app.query_rewrite import rewrite_queries
from app.schemas import Plan, Reranking, RunCreate
from app.security import redact
from app.services import claim_job, create_run, run_data
from app.worker import process_job
from app.workflow_tools import project_server
from conftest import FakeProvider
from pydantic import ValidationError
from sqlalchemy import select


def test_rewrite_keeps_entities_numbers_conditions_and_scopes_terms():
    question = "Nova 新人不超过30天可以转正吗？"
    subs = [{"id": "q1", "target": "转正条件", "queries": ["Nova 转正", "Nova 60天转正", question]}]
    rules = [{"corpus_ids": ["c1"], "term": "试用期考核", "aliases": ["转正"]},
             {"corpus_ids": ["other"], "term": "秘密术语", "aliases": ["转正"]}]
    result, records = rewrite_queries(question, subs, "c1", rules)
    queries = result[0]["queries"]
    assert queries[0] == question
    assert any("试用期考核" in q for q in queries)
    assert all(question in q for q in queries)
    assert not any("秘密术语" in q or "60" in q for q in queries)
    assert sum(r["kind"] == "rejected_planner" for r in records) == 2


def test_term_rules_are_frozen_and_rewrites_reach_retrieval(sessions, settings, corpus, tmp_path):
    settings.query_rewrite_enabled = True
    settings.query_terms_file = tmp_path / "terms.json"
    settings.query_terms_file.write_text(json.dumps({"rules": [
        {"corpus_ids": ["corpus1"], "term": "F1 score", "aliases": ["F1"]}]}), encoding="utf-8")
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                         "rewrite-frozen-run", "local-user", settings)
        run_id = run.id
    settings.query_terms_file.write_text('{"rules": []}', encoding="utf-8")
    process_job(sessions, settings, FakeProvider(), claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert run.config["query_term_rules"][0]["term"] == "F1 score"
        record = db.scalar(select(RunEvent).where(RunEvent.run_id == run_id, RunEvent.kind == "query_rewritten"))
        assert any("F1 score" in r["query"] for r in record.payload["records"])
        assert run.state["executed_queries"][0]


def test_long_question_skips_expansion_without_replacing_it_with_short_target():
    question = "Nova 不超过30天 " + "具体条件。" * 110
    subs = [{"id": "q1", "target": "转正条件", "queries": ["Nova 不超过30天转正"]}]
    result, records = rewrite_queries(question, subs, "c1", [])
    assert result == subs
    assert records == [{"kind": "skipped", "reason": "original_question_over_search_limit",
                        "original_question": question}]


def test_role_routes_select_actual_model_endpoint_and_keep_secrets_out(settings, monkeypatch):
    monkeypatch.setenv("TEST_ROLE_KEY", "private-role-key-123")
    settings.role_model_routes = {"Planner": ModelRoute(model="small-model", base_url="https://route.invalid/v1",
        api_key_env="TEST_ROLE_KEY", api_style="chat_completions")}
    provider = ModelProvider(settings)
    calls = []
    def post(base, key, endpoint, payload):
        calls.append((base, key, endpoint, payload["model"]))
        return {"choices": [{"finish_reason": "stop", "message": {"content": '{"normalized_question":"test"}'}}]}
    monkeypatch.setattr(provider, "_post", post)
    provider.structured("Planner", "test", {}, Plan)
    assert calls == [("https://route.invalid/v1", "private-role-key-123", "chat/completions", "small-model")]
    assert resolve_route(settings, "Analyst").model == settings.model
    assert redact("private-role-key-123", settings) == "[REDACTED]"
    assert role_rates({"role_model_routes": {"Planner": {"model": "small"}},
                       "chat_input_price_per_million": 10}, "Planner") == (None, None)


@pytest.mark.parametrize("route", [{"model": "x", "base_url": "https://user:password@example.org"},
                                  {"model": "x", "api_key_env": "not-a-variable"}])
def test_role_routes_reject_unsafe_config(route):
    with pytest.raises(ValidationError):
        ModelRoute(**route)


def test_dedicated_rerank_orders_known_candidates_and_rejects_forged_ids(settings, monkeypatch):
    settings.reranker_backend = "rerank_api"
    settings.role_model_routes = {"Reranker": ModelRoute(model="ranker")}
    provider = ModelProvider(settings)
    data = {"question": "综合查询", "plan": {"subquestions": [
        {"id": "q1", "target": "离职账号何时停用"},
        {"id": "q2", "target": "审计日志保留多久"},
        {"id": "q3", "target": "审计日志保留多久"},
    ]}, "candidates": [
        {"chunk_id": "a", "blocks": [{"block_id": "b1", "text": "first"}]},
        {"chunk_id": "b", "blocks": [{"block_id": "b2", "text": "second"}]}]}
    post = Mock(return_value={"results": [{"index": 1, "relevance_score": .9}, {"index": 0, "relevance_score": .1}],
                              "usage": {"total_tokens": 50}})
    monkeypatch.setattr(provider, "_post", post)
    result, usage = provider.structured("Reranker", "", data, Reranking)
    assert [i.chunk_id for i in result.items] == ["b", "a"]
    assert usage["usage_known"] and usage["input_tokens"] == 50
    assert post.call_args.args[2] == "rerank"
    assert post.call_args.args[3]["query"] == "综合查询\n\n问题要点：\n- 离职账号何时停用\n- 审计日志保留多久"
    post.return_value = {"results": [{"index": 999, "relevance_score": .5}]}
    with pytest.raises(ProviderError):
        provider.structured("Reranker", "", data, Reranking)


@pytest.mark.parametrize("response", [{"results": [None]}, {"results": [], "usage": [1]}])
def test_dedicated_rerank_malformed_response_is_a_provider_failure(settings, monkeypatch, response):
    settings.reranker_backend = "rerank_api"
    settings.role_model_routes = {"Reranker": ModelRoute(model="ranker")}
    provider = ModelProvider(settings)
    monkeypatch.setattr(provider, "_post", Mock(return_value=response))
    with pytest.raises(ProviderError):
        provider.structured("Reranker", "", {"question": "test", "candidates": [
            {"chunk_id": "a", "blocks": [{"block_id": "b1", "text": "first"}]}]}, Reranking)


def test_role_model_name_is_saved_in_call_ledger(sessions, settings, corpus):
    settings.role_model_routes = {"Planner": ModelRoute(model="planner-alias")}
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="M 的 F1 是多少？"),
                         "role-ledger-test", "local-user", settings)
        run_id = run.id
    process_job(sessions, settings, FakeProvider(), claim_job(sessions, settings))
    with sessions() as db:
        assert db.get(QuestionRun, run_id).status == "completed"
        assert db.scalar(select(ModelCall).where(ModelCall.run_id == run_id, ModelCall.role == "Planner")).model == "planner-alias"


def test_quota_denial_and_redis_failure_send_zero_http_requests(settings, monkeypatch):
    import redis
    settings.provider_quota_enabled = True
    settings.chat_api_key, settings.chat_base_url = "test-key", "https://quota.invalid"
    client = Mock()
    client.eval.return_value = [0, 60, 1000]
    monkeypatch.setattr(provider_quota, "client_for", lambda _: client)
    http = Mock()
    monkeypatch.setattr("app.providers.httpx.Client", http)
    provider = ModelProvider(settings)
    with pytest.raises(ProviderError, match="配额已满"):
        provider._post(settings.chat_base_url, settings.chat_api_key, "responses", {"model": "test"})
    client.eval.side_effect = redis.ConnectionError("unavailable")
    with pytest.raises(ProviderError, match="配额服务不可用"):
        provider._post(settings.chat_base_url, settings.chat_api_key, "responses", {"model": "test"})
    http.assert_not_called()


def test_quota_rejects_oversized_request_before_redis(settings, monkeypatch):
    settings.provider_quota_enabled, settings.provider_quota_tpm = True, 100
    client = Mock()
    monkeypatch.setattr(provider_quota, "client_for", client)
    with pytest.raises(ProviderError, match="超过共享配额"):
        provider_quota.reserve(settings, "https://quota.invalid", "key", 101)
    client.assert_not_called()


class LogsProvider(FakeProvider):
    def structured(self, role, prompt, data, schema):
        if role == "ToolRouter":
            self.roles.append(role)
            return schema.model_validate({"tool_identity": None if data["previous_calls"] else
                "project_runtime_logs:query_run_logs", "arguments": {"hours": 24, "limit": 5},
                "reason": "查看本人近期故障记录"}), {"input_tokens": 10, "output_tokens": 10}
        if role == "Analyst":
            self.roles.append(role)
            source = next(s for s in data["sources"] if s.get("source_kind") == "mcp_observation")
            return schema.model_validate({"claims": [{"id": "log1", "text": "工具记录显示先前问答执行失败。",
                "conditions": ["本项目本人运行记录"], "subquestion_ids": ["q1"],
                "evidence": [{"evidence_id": source["id"], "relation": "supports"}]}]}), {
                    "input_tokens": 10, "output_tokens": 10}
        if role == "Verifier":
            self.roles.append(role)
            evidence = data["draft"]["claims"][0]["evidence"][0]["evidence_id"]
            return schema.model_validate({"approved_claim_ids": ["log1"], "claim_reviews": [{
                "claim_id": "log1", "verdict": "approved", "evidence_ids": [evidence],
                "rationale": "该实际运行回执保存了失败状态；没有据此推断根因。"}]}), {"input_tokens": 10, "output_tokens": 10}
        return super().structured(role, prompt, data, schema)


def test_real_stdio_project_logs_complete_workflow_and_tamper_protection(sessions, settings, corpus):
    # This is actual persisted execution failure, not a fabricated log response.
    with sessions() as db:
        settings.database_url = db.bind.url.render_as_string(hide_password=False)
        create_run(db, RunCreate(snapshot_id="snap1", question="控制失败运行"),
                   "actual-failed-run", "local-user", settings)
    process_job(sessions, settings, FakeProvider(fail=True), claim_job(sessions, settings))
    settings.workflow_mcp_enabled, settings.project_logs_mcp_enabled = True, True
    settings.evidence_gate_enabled = False
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="最近问答为什么失败？"),
                         "actual-logs-run", "local-user", settings)
        run_id = run.id
    provider = LogsProvider()
    process_job(sessions, settings, provider, claim_job(sessions, settings))
    with sessions() as db:
        run = db.get(QuestionRun, run_id)
        assert run.status == "completed", run.error
        assert provider.roles.count("ToolRouter") == 2
        row = db.scalar(select(MCPInvocation).where(MCPInvocation.run_id == run_id))
        assert row.status == "succeeded"
        assert "failed" in json.dumps(row.result)
        span = db.scalar(select(EvidenceSpan).where(EvidenceSpan.invocation_id == row.id))
        assert span.block_id is None
        info = evidence_data(db, span, "local-user")
        assert info["source_kind"] == "mcp_observation" and info["pdf_page_number"] is None
        assert run_data(db, run, "local-user")["answer"]["claims"][0]["text"].startswith("工具记录")
        from app.models import Membership, User
        db.add(User(id="log-reader", username="log-reader", password_hash="unused"))
        db.flush()
        member = Membership(corpus_id="corpus1", user_id="log-reader", role="reader")
        db.add(member)
        db.flush()
        with pytest.raises(DomainError, match="私人运行日志"):
            evidence_data(db, span, "log-reader")
        member.role = "editor"
        db.flush()
        assert evidence_data(db, span, "log-reader")["invocation_id"] == row.id
        row.result = {"tampered": True}
        with pytest.raises(DomainError, match="校验失败"):
            validate_evidence(db, db.get(Snapshot, "snap1"), span)


def test_project_logs_actor_scope_is_server_bound(sessions, settings, corpus, monkeypatch):
    from app.project_logs_mcp import query_run_logs
    with sessions() as db:
        monkeypatch.setenv("PROJECT_LOGS_DATABASE_URL", db.bind.url.render_as_string(hide_password=False))
    monkeypatch.setenv("PROJECT_LOGS_ACTOR", "outsider")
    monkeypatch.setenv("PROJECT_LOGS_CORPUS", "corpus1")
    with pytest.raises(DomainError):
        query_run_logs()


def test_reserved_builtin_server_is_not_user_configurable(settings, tmp_path):
    from app.mcp_client import load_servers
    settings.mcp_enabled = True
    settings.mcp_servers_file = tmp_path / "servers.json"
    settings.mcp_servers_file.write_text(json.dumps({"servers": [project_server().model_dump()]}), encoding="utf-8")
    with pytest.raises(DomainError, match="配置缺失或格式不合法"):
        load_servers(settings)


@pytest.mark.parametrize("identity", ["unlisted:read", "project_runtime_logs:delete_logs"])
def test_tool_router_cannot_select_unknown_or_write_tools(sessions, settings, corpus, identity, monkeypatch):
    from app.workflow import Workflow, initial_state
    from app.workflow_tools import WorkflowTools
    settings.workflow_mcp_enabled, settings.project_logs_mcp_enabled = True, True
    with sessions() as db:
        run = create_run(db, RunCreate(snapshot_id="snap1", question="查看最近故障记录"),
                         "unsafe-route-" + identity, "local-user", settings)
        run_id = run.id
    job = claim_job(sessions, settings)
    workflow = Workflow(sessions, settings, FakeProvider(), run_id, job[2])
    with sessions() as db:
        state = initial_state(db.get(QuestionRun, run_id))
    server = project_server()
    async def catalog(_corpus):
        return {"project_runtime_logs:query_run_logs": (server, {"name": "query_run_logs"})}
    tools = WorkflowTools(workflow)
    monkeypatch.setattr(tools, "catalog", catalog)
    from app.schemas import ToolDecision
    monkeypatch.setattr(workflow, "call", lambda *_args: ToolDecision(tool_identity=identity, arguments={}, reason="test"))
    with pytest.raises(ProviderError, match="未授权"):
        tools.run(state)
    with sessions() as db:
        assert list(db.scalars(select(MCPInvocation))) == []


def test_external_evidence_migration_preserves_document_offsets(tmp_path):
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory
    from app.config import ROOT
    from sqlalchemy import create_engine, inspect, text
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "backend/migrations"))
    migration = ScriptDirectory.from_config(config).get_revision("0011").module
    engine = create_engine(f"sqlite:///{(tmp_path / 'old-evidence.db').as_posix()}")
    try:
        with engine.begin() as db:
            db.execute(text("CREATE TABLE mcp_invocations (id VARCHAR(40) PRIMARY KEY)"))
            db.execute(text("CREATE TABLE evidence_spans (id VARCHAR(40) PRIMARY KEY, block_id VARCHAR(40) NOT NULL, char_start INTEGER, char_end INTEGER, quote TEXT)"))
            db.execute(text("INSERT INTO evidence_spans VALUES ('e1', 'b1', 2, 6, '原文摘录')"))
            with Operations.context(MigrationContext.configure(db)):
                migration.upgrade()
            row = db.execute(text("SELECT block_id,char_start,char_end,quote FROM evidence_spans WHERE id='e1'")).one()
            assert tuple(row) == ('b1', 2, 6, '原文摘录')
            columns = {c["name"]: c for c in inspect(db).get_columns("evidence_spans")}
            assert columns["block_id"]["nullable"] and "invocation_id" in columns
    finally:
        engine.dispose()
