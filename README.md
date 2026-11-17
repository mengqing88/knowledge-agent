# knowledge-agent · 团队知识库智能问答

在本机运行的团队知识库：上传资料、按权限检索、生成带原文引用的答案，经核验与审核后发布，保留运行进度、追问和审阅记录。

[快速开始](docs/guides/启动与日常使用.md) · [文档导航](docs/README.md) · [架构](docs/architecture/architecture.md) · [评测](evaluation/README.md) · [当前待办](TODO.md)

## 功能与流程

- 本机账号注册/登录，知识库管理员、编辑者、阅读者权限。
- PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML/HTM、XLSX 按版本入库，冻结问答资料快照。
- 向量与全文混合检索、重排、逐条论点与原文引用；等待时显示处理进度。
- LangGraph 编排规划、取证、分析、核验和有限补查；默认审核通过后发布答案。
- 默认接入术语扩展、本人项目日志 MCP 和 Redis 模型共享配额；外部工具、操作审批与确认记忆按授权配置，另有本机受限代码修复任务扩展。

```text
注册/登录 → 选择知识库 → 上传与索引 → 固定资料快照
        → 规划/检索/重排 → 证据检查 → 分析/核验 → 审核发布 → 引用/追问
```

后端：Python、FastAPI、SQLAlchemy/Alembic、LangGraph、LlamaIndex。
存储与调度：PostgreSQL、Milvus、Redis、Celery。前端：Vue 3、TypeScript、Vite。

## 本机启动

需要 Python 3.11–3.13、uv、Node.js、Docker Desktop（Linux containers）。新环境根据 `.env.example` 配置本机 `.env`；已有配置保留。

```powershell
uv sync --frozen
npm.cmd --prefix frontend ci
npm.cmd --prefix frontend run build
scripts\start.ps1
```

打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/)。使用 `data/admin-login.txt` 中的本机初始化账号，或在页面注册新账号，再创建知识库、上传材料并提问。停止：

```powershell
scripts\stop.ps1
```

新建完整 Agent 问答默认使用三路宽召回、本地 BGE 和子问句原文邻块选择。首次克隆还需准备独立的 BGE 环境与约 2.3 GB 模型，步骤及 `RETRIEVAL_STRATEGY=legacy` 回退方式见[检索配置](docs/guides/开发与运维.md#本地-bge-检索)。环境缺失会明确报错，不自动下载或改用收费重排。

只启动 API 不会处理入库/问答任务。启动脚本同时管理 API、Celery worker 和 beat；详细步骤、前端开发和隔离运行见[使用手册](docs/guides/启动与日常使用.md)与[开发运维](docs/guides/开发与运维.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| [backend/](backend/README.md) | API、配置/数据模型、知识检索、Agent、模型工具、任务、代码修复及测试 |
| [frontend/](frontend/README.md) | 组件、API 客户端、运行逻辑、类型和样式 |
| [scripts/](scripts/README.md) | 启停、维护、评测、实验、冒烟、隔离验收、模型与教学工具 |
| [docs/](docs/README.md) | 操作指南、架构、验证记录、历史资料和配置示例 |
| [evaluation/](evaluation/README.md) | 评测代码、冻结资料、原始结果及人工评审入口 |
| `fixtures/` | 明确标记为虚构的教学与控制测试材料 |
| `deploy/`、`docker/` | 依赖与隔离验收配置、代码修复沙箱镜像 |

本机 `.env`、`data/`、`tmp/`、模型、依赖和构建缓存不提交到 GitHub。评审包仍在原来的 `tmp/independent-review-20261115/`，详见[本地工作区](docs/guides/本地工作区.md)。

## 验证与边界

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check backend evaluation scripts fixtures
npm.cmd --prefix frontend test
npm.cmd --prefix frontend run build
```

已有隔离存储/队列及指定页面流程的实测记录；两组在线实验各 12 题，包含失败，Agent 判分与真人判分分开。BGE 已完成冻结候选离线排序对照。真人评分、角色路由/在线专用重排独立效果等仍见 [TODO](TODO.md)，不把测试数或小样本结果当业务准确率。

当前面向本机团队演示；扫描件 OCR、无边框 PDF 表格恢复、全网搜索及公网多租户部署不在范围。取消不能撤回已发送的模型请求，任务恢复可能再次计费。完整边界见[范围说明](docs/architecture/scope.md)。

开发协作约定和累计讲解见[项目共享说明](项目共享说明.md)；[简历 Agent 档案](项目档案_供简历Agent使用.md)保留原文，历史路径通过[迁移表](docs/records/2026-11/项目目录整理_2026-11-15.md)查找。
