"""Configured MCP endpoints only; protocol annotations never grant permissions."""
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import anyio
import httpx
from jsonschema import Draft202012Validator, FormatChecker
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .errors import DomainError
from .extension_security import protect
from .governance import digest


class WritePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resource_argument: str = Field(min_length=1, max_length=160)
    idempotency_argument: str = Field(min_length=1, max_length=160)


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    server_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,79}$")
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: list[str] = Field(default_factory=list, max_length=40)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(default_factory=dict)
    user_ids: list[str] = Field(min_length=1)
    corpus_ids: list[str] = Field(min_length=1)
    read_tools: list[str] = Field(default_factory=list)
    write_tools: dict[str, WritePolicy] = Field(default_factory=dict)
    timeout_seconds: float = Field(15, ge=0.1, le=120, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_endpoint(self):
        if self.transport == "stdio":
            if not self.command or not Path(self.command).is_absolute() or self.url or self.headers_env:
                raise ValueError("stdio 需要绝对可执行文件路径，不能混用 HTTP 配置")
        else:
            url = urlsplit(self.url or "")
            if (url.scheme not in ("https", "http") or not url.hostname or url.username or
                    url.password or url.query or url.fragment or self.command or self.args or self.env):
                raise ValueError("HTTP 需要无内嵌凭据的端点，不能混用进程配置")
            if url.scheme == "http" and url.hostname not in ("127.0.0.1", "localhost", "::1"):
                raise ValueError("非本机端点必须使用 HTTPS")
        if set(self.read_tools) & self.write_tools.keys():
            raise ValueError("同一工具不能同时声明为只读和写入")
        if any(not name.strip() for name in [*self.read_tools, *self.write_tools]):
            raise ValueError("工具白名单不能包含空名称")
        if any(not name.isidentifier() for name in set(self.env.values()) | set(self.headers_env.values())):
            raise ValueError("凭据只能引用环境变量名")
        if any(not key or "\r" in key or "\n" in key for key in self.headers_env):
            raise ValueError("HTTP 请求头名称不合法")
        return self

    def credentials(self):
        names = {*self.env.values(), *self.headers_env.values()}
        if any(not os.environ.get(name) for name in names):
            raise DomainError("MCP 配置缺少所需环境变量", 503, "mcp_credentials_missing")
        return {name: os.environ[name] for name in names}

    def fingerprint(self):
        return digest({"config": self.model_dump(), "credentials_sha256": digest(self.credentials())})


class MCPFailure(DomainError):
    def __init__(self, *, dispatched=False, code="mcp_unavailable"):
        super().__init__("MCP 调用未取得可确认回执，请查看操作记录", 503, code)
        self.dispatched = dispatched


def load_servers(settings) -> dict[str, ServerConfig]:
    if not settings.mcp_enabled:
        raise DomainError("MCP 客户端未启用", 503, "mcp_disabled")
    try:
        path = settings.mcp_servers_file
        if not path or path.stat().st_size > 128000:
            raise ValueError()
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(document, dict) or set(document) != {"servers"}:
            raise ValueError()
        servers = [ServerConfig.model_validate(row) for row in document["servers"]]
        if any(s.server_id == "project_runtime_logs" for s in servers):
            raise ValueError("project_runtime_logs 是保留的内置服务名")
        if len(servers) > 30 or len({s.server_id for s in servers}) != len(servers):
            raise ValueError()
        return {s.server_id: s for s in servers}
    except (OSError, ValueError, TypeError, ValidationError):
        raise DomainError("MCP 服务配置缺失或格式不合法", 503, "mcp_config_invalid") from None


def configured_server(settings, server_id, user_id, corpus_id):
    server = load_servers(settings).get(server_id)
    if not server or user_id not in server.user_ids or corpus_id not in server.corpus_ids:
        raise DomainError("MCP 服务不存在或未授权给当前用户及知识库", 404)
    return server


def validate_arguments(schema, arguments):
    try:
        if len(json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode()) > 32000:
            raise ValueError()
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, format_checker=FormatChecker())
        # Network references are never resolved by a remote service.
        def check_refs(value):
            if isinstance(value, dict):
                for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if key in value and not str(value[key]).startswith("#"):
                        raise ValueError()
                for item in value.values():
                    check_refs(item)
            elif isinstance(value, list):
                for item in value:
                    check_refs(item)
        check_refs(schema)
        if not validator.is_valid(arguments):
            raise ValueError()
    except Exception:
        raise DomainError("MCP 工具参数或 Schema 不合法，未调用工具", 422, "mcp_arguments_invalid") from None


class MCPClient:
    def __init__(self, settings, server, *, bound_environment=None):
        self.settings, self.server = settings, server
        self.bound_environment = bound_environment or {}

    @asynccontextmanager
    async def session(self):
        server = self.server
        credentials = server.credentials()
        with anyio.fail_after(server.timeout_seconds):
            async with AsyncExitStack() as stack:
                if server.transport == "stdio":
                    errlog = stack.enter_context(open(os.devnull, "w"))
                    environment = get_default_environment()
                    environment.update({k: credentials[v] for k, v in server.env.items()})
                    environment.update(self.bound_environment)
                    params = StdioServerParameters(command=server.command, args=server.args, env=environment)
                    streams = await stack.enter_async_context(stdio_client(params, errlog=errlog))
                else:
                    client = await stack.enter_async_context(httpx.AsyncClient(
                        headers={k: credentials[v] for k, v in server.headers_env.items()},
                        timeout=server.timeout_seconds, follow_redirects=False, trust_env=False))
                    streams = await stack.enter_async_context(streamable_http_client(server.url, http_client=client))
                session = await stack.enter_async_context(ClientSession(
                    *streams[:2], read_timeout_seconds=timedelta(seconds=server.timeout_seconds)))
                await session.initialize()
                yield session

    async def tools(self, session):
        tools, cursor, seen = {}, None, set()
        for _ in range(20):
            page = await session.list_tools(cursor=cursor)
            for tool in page.tools:
                if tool.name in tools:
                    raise DomainError("MCP 工具身份重复", 502)
                tools[tool.name] = tool
            if len(tools) > 500:
                break
            cursor = page.nextCursor
            if not cursor:
                return tools
            if cursor in seen:
                break
            seen.add(cursor)
        raise DomainError("MCP 工具发现超出分页上限", 502)

    async def discover(self):
        try:
            async with self.session() as session:
                tools = await self.tools(session)
                allowed = set(self.server.read_tools)
                if self.settings.mcp_writes_enabled:
                    allowed.update(self.server.write_tools)
                return [{"identity": f"{self.server.server_id}:{name}", "name": name,
                         "mode": "read" if name in self.server.read_tools else "write_requires_approval",
                         "description": protect(tool.description or "", self.settings,
                                                self.server.credentials().values())[:2000],
                         "input_schema": tool.inputSchema}
                        for name, tool in tools.items() if name in allowed]
        except DomainError:
            raise
        except Exception:
            raise MCPFailure() from None

    async def schema(self, tool_name):
        rows = await self.discover()
        row = next((r for r in rows if r["name"] == tool_name), None)
        if not row:
            raise DomainError("工具不在已发现的显式白名单内", 403)
        return row["input_schema"]

    async def invoke(self, tool_name, arguments, *, write=False, expected_schema=None, before_call=None):
        allowed = self.server.write_tools if write and self.settings.mcp_writes_enabled else self.server.read_tools
        if tool_name not in allowed or (write and not self.settings.mcp_writes_enabled):
            raise DomainError("工具未获当前调用模式授权", 403)
        dispatched = False
        try:
            async with self.session() as session:
                tools = await self.tools(session)
                if tool_name not in tools:
                    raise DomainError("配置工具未被服务发现", 409)
                schema = tools[tool_name].inputSchema
                if expected_schema and digest(schema) != expected_schema:
                    raise DomainError("工具 Schema 已改变，请重新提出并审批操作", 409)
                validate_arguments(schema, arguments)
                if before_call:
                    before_call()
                dispatched = True
                result = await session.call_tool(tool_name, arguments)
                raw = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                if len(json.dumps(raw).encode()) > 128000:
                    raise MCPFailure(dispatched=True, code="mcp_result_too_large")
                # isError does not prove a remote write had no partial effects.
                if result.isError:
                    raise MCPFailure(dispatched=True, code="mcp_tool_error")
                return protect(raw, self.settings, self.server.credentials().values())
        except DomainError:
            raise
        except Exception:
            raise MCPFailure(dispatched=dispatched) from None
