# 后端代码导航

启动仍从项目根目录执行 `scripts/start.ps1`，或用 `scripts/manage.py`；不直接运行包内文件。

| 目录 | 职责 | 主要文件 |
| --- | --- | --- |
| `app/api/` | HTTP API、团队/文档与扩展接口 | `main.py`、`team_api.py`、`extensions_api.py` |
| `app/core/` | 配置、数据库、模型、Schema、认证与权限 | `config.py`、`db.py`、`models.py`、`auth.py`、`access.py` |
| `app/knowledge/` | 解析、分块、检索、来源引用和向量迁移 | `ingestion.py`、`chunking.py`、`retrieval.py`、`evidence.py`、`vector_store.py` |
| `app/agents/` | LangGraph、提示词、记忆、门控及执行清单 | `workflow.py`、`prompts.py`、`memory.py`、`governance.py` |
| `app/integrations/` | 模型调用、MCP、配额、熔断与外部操作 | `providers.py`、`mcp_client.py`、`external_operations.py` |
| `app/jobs/` | 持久任务、领取、租约与 Celery | `services.py`、`worker.py`、`tasks.py`、`capacity.py` |
| `app/repair/` | 独立代码修复 API 与隔离执行 | `api.py`、`service.py` |
| `app/bootstrap.py`、`app/cli.py` | 账号初始化与维护命令 | 由 `scripts/manage.py` 调用 |
| `migrations/` | Alembic 迁移 | 数据库 revision 保持原编号 |
| `tests/` | 后端回归与离线评测控制测试 | pytest 从项目根运行 |

ASGI 目标为 `app.api.main:app`；Celery 对象为 `app.jobs.tasks:celery_app`。内置只读日志 MCP 子进程为 `python -m app.integrations.project_logs_mcp`。对外 HTTP 路径和 Celery 任务名保持原有约定。

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check backend evaluation scripts fixtures
```

测试与真实模型效果评测分别看待，详见[评测导航](../evaluation/README.md)。
