"""Bounded read-only MCP loop between document retrieval and answer analysis."""
import asyncio
import sys

from app.agents.governance import digest
from app.agents.prompts import TOOL_ROUTER
from app.core.access import require_role
from app.core.config import ROOT
from app.core.errors import DomainError, ProviderError
from app.core.extension_security import protect
from app.core.models import EvidenceSpan, MCPInvocation, Snapshot, uid
from app.core.schemas import ToolDecision
from app.integrations.mcp_client import MCPClient, ServerConfig, configured_server, load_servers
from app.jobs.services import check_attempt
from app.knowledge.evidence import evidence_data, observation_text


def scope_identity(settings):
    servers = load_servers(settings) if settings.mcp_enabled else {}
    return digest({name: server.fingerprint() for name, server in servers.items()})


def project_server():
    return ServerConfig(server_id="project_runtime_logs", transport="stdio", command=sys.executable,
        args=["-m", "app.integrations.project_logs_mcp"], user_ids=["server-injected"], corpus_ids=["server-injected"],
        read_tools=["query_run_logs"], timeout_seconds=10)


class WorkflowTools:
    def __init__(self, workflow):
        self.w = workflow

    def _client(self, server, corpus_id):
        bound = ({"PROJECT_LOGS_ACTOR": self.w.owner_id, "PROJECT_LOGS_CORPUS": corpus_id,
                  "PROJECT_LOGS_DATABASE_URL": self.w.settings.database_url,
                  "PYTHONPATH": str(ROOT / "backend")}
                 if server.server_id == "project_runtime_logs" else {})
        return MCPClient(self.w.settings, server, bound_environment=bound)

    def _authorize(self, server, corpus_id):
        self.w.guard()
        with self.w.sessions() as db:
            require_role(db, corpus_id, self.w.owner_id)
        if scope_identity(self.w.settings) != self.w.limits.get("workflow_tools_identity"):
            raise DomainError("工具配置已改变，请创建新问题", 409)
        if server.server_id == "project_runtime_logs":
            if not self.w.settings.project_logs_mcp_enabled:
                raise DomainError("项目日志工具未启用", 403)
        else:
            current = configured_server(self.w.settings, server.server_id, self.w.owner_id, corpus_id)
            if current.fingerprint() != server.fingerprint():
                raise DomainError("工具配置已改变", 409)

    async def catalog(self, corpus_id):
        servers = load_servers(self.w.settings) if self.w.settings.mcp_enabled else {}
        eligible = [s for s in servers.values() if self.w.owner_id in s.user_ids and
                    corpus_id in s.corpus_ids and s.read_tools]
        if self.w.settings.project_logs_mcp_enabled:
            eligible.insert(0, project_server())
        output = {}
        for server in eligible[:4]:
            self._authorize(server, corpus_id)
            client = self._client(server, corpus_id)
            for tool in await client.discover():
                if tool["mode"] == "read" and len(output) < 12:
                    output[tool["identity"]] = (server, tool)
        return output

    def run(self, state):
        w = self.w
        if not w.limits.get("workflow_mcp_enabled") or state["mode"] == "baseline_rag":
            return state
        history = state.setdefault("tool_history", [])
        if len(history) >= w.limits["max_workflow_tool_calls"]:
            return state
        with w.sessions() as db:
            corpus_id = db.get(Snapshot, state["snapshot_id"]).corpus_id
        catalog = asyncio.run(self.catalog(corpus_id))
        if not catalog:
            return state
        while len(history) < w.limits["max_workflow_tool_calls"]:
            w.guard()
            decision = w.call(state, "ToolRouter", TOOL_ROUTER, {
                "question": state["question"], "sources": state["sources"], "previous_calls": history,
                "tools": [tool for _, tool in catalog.values()]}, ToolDecision)
            if decision.tool_identity is None:
                break
            if decision.tool_identity not in catalog:
                raise ProviderError("工具路由选择了未授权工具", "tool_route_invalid")
            server, tool = catalog[decision.tool_identity]
            if protect(decision.arguments, w.settings, server.credentials().values()) != decision.arguments:
                raise DomainError("工具参数包含凭据，未执行", 422)
            signature = digest([decision.tool_identity, decision.arguments])
            if any(item["signature"] == signature for item in history):
                w.save(state, "tool_stopped", {"reason": "duplicate_call"})
                break
            self._authorize(server, corpus_id)
            with w.sessions.begin() as db:
                check_attempt(db, w.run_id, w.attempt, lock=True)
                row = MCPInvocation(id=uid(), owner_id=w.owner_id, run_id=w.run_id, server_id=server.server_id,
                    tool_name=tool["name"], arguments=decision.arguments, status="started", result={})
                db.add(row)
                invocation_id = row.id
            history.append({"identity": decision.tool_identity, "signature": signature, "invocation_id": invocation_id})
            w.save(state, "tool_started", {"identity": decision.tool_identity, "reason": decision.reason})
            try:
                result = asyncio.run(self._client(server, corpus_id).invoke(tool["name"], decision.arguments,
                    expected_schema=digest(tool["input_schema"]), before_call=lambda: self._authorize(server, corpus_id)))
                w.guard()
                text = observation_text(result)
                allowed = min(w.limits["max_external_evidence_chars"],
                    w.limits["max_context_chars"] - sum(len(s["quote"]) for s in state["sources"]))
                if not text.strip() or len(text) > allowed:
                    raise DomainError("外部结果为空或超过上下文上限，未用于答案", 422, "tool_result_limit")
                with w.sessions.begin() as db:
                    check_attempt(db, w.run_id, w.attempt, lock=True)
                    row = db.get(MCPInvocation, invocation_id)
                    row.status, row.result = "succeeded", result
                    span = EvidenceSpan(id=uid(), run_id=w.run_id, invocation_id=row.id, block_id=None,
                        source_sha256=digest(result), char_start=0, char_end=len(text), quote=text)
                    db.add(span)
                    db.flush()
                    info = evidence_data(db, span, w.owner_id)
                if len(state["sources"]) < 32:
                    state["sources"].append(info)
                history[-1]["status"] = "succeeded"
            except DomainError as exc:
                with w.sessions.begin() as db:
                    row = db.get(MCPInvocation, invocation_id)
                    row.status, row.result = "failed", {"code": exc.code}
                history[-1].update(status="failed", code=exc.code)
                # A failed observation is visible; it cannot be cited as a result.
            w.save(state, "tool_finished", {"invocation_id": invocation_id, "status": history[-1]["status"]})
        return w.save(state, "tools_completed", {"tool_calls": len(history)})
