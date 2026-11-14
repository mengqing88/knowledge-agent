# 团队知识库智能问答系统

面试准备与同步讲解约定见 [项目共享说明](项目共享说明.md)。本说明对应已确认的团队知识库最简版。

供简历 Agent 阅读的当前功能、技术细节、变更历史、评测证据与措辞边界见 [项目档案](项目档案_供简历Agent使用.md)。

当前未完成事项、依赖和验收条件见 [TODO](TODO.md)。

查询术语扩展、问答内 MCP 只读工具循环、角色模型路由/专用重排和 Redis 跨 worker 配额已接入。可用 `scripts\start.ps1 -AgentUpgrades` 为新进程启用配置；真实 `.env` 不变。Docker 已恢复，隔离 PG/Milvus/Redis、独立 Celery/beat 和页面流程有实测证据；改写收益、BGE 和角色路由的独立效果仍待验证。历史记录见 [四项升级记录](docs/查询改写_MCP编排_角色模型与共享配额.md)，本次范围见 [3 元收尾](docs/3元精简收尾_2026-11-14.md)。

面向本机团队成员的知识问答：PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML/HTM 和 XLSX 按版本解析入库；PDF 中可检测的有边框表格按表头和行结构保存，原文块保留页码与行坐标。在固定快照内混合检索，由 Planner / Retriever / Analyst / Verifier 生成逐论点引用答案，并保留任务状态和运行记录。输入需要匹配已实现的类型解析器，不代表任意二进制文件或在线数据源均可直接检索。

**当前存储：Milvus 存向量，PostgreSQL 保留业务数据、原文和全文检索；Docker 使用 Milvus Standalone。** 迁移、回填和检索实现见 [Milvus 迁移说明](docs/Milvus迁移与验证.md)。隔离 Celery/存储/浏览器链路已通过；在线模型质量另用同步隔离 worker 做两组小样本对照，不等于真实模型与 Celery 全部组合或正式业务效果验收。

另提供独立的本机代码修复任务 API：通过固定 Git worktree 隔离补丁，在受限 Docker 中运行 pytest，并由人工按具体补丁与测试摘要审核。该扩展不等于通用代码执行器；真实用户仓库兼容性和代码修复效果尚无正式评测。详情见[第一部分实施核对](docs/第一部分_协作输入与工作流基础_实施核对.md)。

问答答案在模型核验后默认等待管理员/编辑者人工审核，通过后才向知识库成员发布；拒绝理由可用于同一快照的新一轮修订。自动评测须在隔离进程配置 `ANSWER_REVIEW_REQUIRED=false`。

审核接口绑定审核者实际查看的内容：`GET /api/question-runs/{run_id}` 及待审队列的 `run` 同时返回 `answer` 和 `answer_sha256`。提交 `POST /api/answers/{answer_id}/approval` 时，接受或拒绝均须携带所见的 `answer_sha256`（64 位小写十六进制）、`decision`，拒绝另须提供 `reason`。缺少或非法哈希返回 422；阅读后内容变化返回 409（`answer_content_changed`），不会写入审批或经验记录，须重新查看正文后再决定。前端随显示内容保存哈希，不自动取新哈希重试。旧客户端须同步更新。

新审批哈希对证据链接和来源列表排序后计算，保留论点顺序与全部内容；发布读取兼容既有原始序列哈希，不改写历史审批。旧记录仍可能受原列表顺序影响，PostgreSQL 并发和整栈运行仍待验收。

运行详情与待审队列中的 `run.is_owner` 表示当前访问者是否为问题发起人。审核其他成员的答案时，页面只加载待审正文和快照；审核结束返回待审队列。发起人仍可读取自己的事件、结果与历史；反馈、运行记录和追问入口仅在本人答案发布后显示。服务端继续校验每个接口的访问权限。

为已有文档上传新版本时，提交前会重新确认并锁定目标文档；若文档在上传期间已被删除，返回 404，不为该文档创建新的入库任务。

**框架组合：LlamaIndex 负责向量索引与 Retriever，LangGraph 编排规划、重排、分析和核验。** 解析/分块保留原文范围规则，混合检索继续结合 PostgreSQL 全文。代码和验证见 [组合实现](docs/LlamaIndex与LangGraph组合实现.md)。

分块支持固定窗口、段落、结构分句和语义策略；依据本轮控制集对比，代码默认采用页/章节内固定窗口，512 参考 token / 最多 64 token 重叠。参数进入索引版本；改变后须重建索引和快照。配置、长文对比方法及实际结果见 [分块策略与对比评测](docs/分块策略与对比评测.md)。

## 环境

- Python 3.11–3.13、uv、Docker Desktop；前端接入后需要 Node.js。
- `uv.lock` 固定 Python 依赖。
- 模型配置已按用户要求从 `F:\项目1\.env` 复制到本项目 `.env`，未输出密钥。新环境参考 `.env.example`。
- 数据库默认 `127.0.0.1:55432`；API 只监听 `127.0.0.1:8000`。

推荐使用启动脚本。它会启动 PostgreSQL、Redis、Milvus，执行数据库迁移及旧向量回填，初始化管理员账号，并启动 API、Celery worker 和 beat：

```powershell
uv sync --frozen
scripts\start.ps1
```

管理员首次登录信息保存在 `data/admin-login.txt`。也可在前端登录页点击“没有账号？注册新账号”，设置账号及 8–128 位密码，注册后自动登录。账号支持字母、数字、下划线、点和连字符，最长 80 个字符。新账号可创建自己的知识库，加入其他知识库需由该库管理员共享；重复注册不会覆盖已有密码。

停止本项目进程：

```powershell
scripts\stop.ps1
```

如需手动启动，先准备依赖和数据库：

```powershell
docker compose up -d --wait db redis milvus
.venv\Scripts\python.exe -X utf8 scripts\manage.py migrate
.venv\Scripts\python.exe -X utf8 scripts\manage.py migrate-vectors
.venv\Scripts\python.exe -X utf8 scripts\manage.py init-admin
```

分别在三个终端运行：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\manage.py api
```

```powershell
.venv\Scripts\python.exe -X utf8 scripts\manage.py celery
```

```powershell
.venv\Scripts\python.exe -X utf8 scripts\manage.py beat
```

任务由 Celery 调度，业务状态和租约保存在 PostgreSQL；只启动 API 不会处理入库或问答任务。

接口文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)。本机演示也可运行 `scripts\start.ps1 -Demo`；它使用 SQLite 和 Celery SQLAlchemy broker，不使用 Milvus，不代表正式存储实机验收。

前端工作台：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

打开 [http://127.0.0.1:5174/](http://127.0.0.1:5174/)。Vite 会将 `/api` 代理到本机 FastAPI；没有可用数据库时页面仍能打开，但文档列表和问答会显示服务不可用。

启动脚本以隐藏窗口启动 API、Celery worker 和 beat；日志和 PID 默认在 `data/`。进程记录绑定 PID、创建时间及完整命令行，防止 PID 复用误停其他程序；失效记录经核对后归档，仍活跃或无法确认身份的记录继续阻止重复启动。旧 PID-only 记录若仍对应本项目活跃进程，需先核对身份，不自动强杀。启动/停止共用排他锁，不删除数据库或 Docker 卷。

隔离验收可设置进程环境变量指向临时依赖，并用 `scripts\start.ps1 -SkipDatabase -RuntimeDir 临时目录 -Port 18082`；停止时 `scripts\stop.ps1 -RuntimeDir 同一临时目录`。账号文件、日志、beat 状态和进程记录均保存于该目录；`-Demo` 也使用该目录中的 SQLite 文件。无需改真实 `.env`。

Python 3.11 的 Windows `.pth` 读取可能不能正确处理中文目录，因此统一通过 `scripts/manage.py` 入口加载本项目，不依赖 editable install 的路径编码。

## 模型设置

沿用项目1字段：`CHAT_API_KEY`、`CHAT_BASE_URL`、`MODEL`、`EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL`、`EMBEDDING_MODEL`。默认聊天协议为 Responses；兼容 Chat Completions 时设置 `CHAT_API_STYLE=chat_completions`。所有调用从后端发送，浏览器不持有模型密钥。

`EMBEDDING_DIMENSIONS=0` 让入库时发现维度，此后查询强制校验相同维度；本机已验证的模型返回 4096 维。更换服务、模型、维度或隐藏权重版本后，修改 `EMBEDDING_REVISION` 并重新索引、创建新快照，旧快照不被覆盖。

真实连通性检查（会发送少量固定测试文字，不发送用户论文）：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\manage.py doctor --output docs\connection-check.json
```

## 使用 API

1. 登录后调用 `POST /api/corpora`：`{"name":"团队知识库"}`。
2. `POST /api/corpora/{id}/papers`：multipart `file`，支持 PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML/HTM、XLSX；更新同一文档时传 `paper_id`，不按标题自动合并。
3. 查询版本状态，等待 `ready`；失败可 `POST /api/paper-versions/{id}/retry`。更改模型配置后可 `/reindex` 建新索引。
4. `POST /api/corpora/{id}/snapshots`：`{"paper_version_ids":["实际版本ID"]}`。
5. `POST /api/question-runs`，必须附加 `Idempotency-Key`（8–128 字符）：

```json
{
  "snapshot_id": "实际快照ID",
  "question": "方法相比基线有什么收益和代价？",
  "mode": "full_agent"
}
```

6. 查询运行，或连接 `/api/question-runs/{id}/events`；`?after_seq=3` / `Last-Event-ID` 可续读，`?stream=false` 返回事件数组。
7. 答案内的 evidence 来自服务端固定版本、原文范围和 PDF 页序。`file_url#page=n` 仅跳页，没有实现精准高亮。
8. 追问附带 `parent_run_id` 并沿用相同 snapshot；更新材料时创建新快照并开始新问题。答案 feedback 通过 `/api/answers/{id}/reviews` 保存，不成为文档事实。

技术状态与证据状态分开。正常完成可以是 supported / partial / conflicted / insufficient / needs_clarification；模型超时或数据库异常是 failed。取消只停止后续步骤，不能撤回已发送的网络请求。

## 教学演示

`fixtures/teaching-a.pdf`、`teaching-b.pdf`、`teaching-c.pdf` 是根据需求文档 A/B/C 例子生成的**虚构教学材料**，每页都有醒目标识，不是真实论文或正式评测依据。`fixtures/manifest.json` 保存哈希和逐字证据。

教学文件和五道虚构开发问题见 `fixtures/`。`scripts/load_teaching_demo.py` 支持账号登录后保留会话 Cookie，或读取 `KB_EVALUATION_TOKEN` 环境变量中的已有会话令牌；可用 `--token-env` 指定其他变量名。密码通过终端隐藏输入，不作为命令行参数。

```powershell
.venv\Scripts\python.exe -X utf8 scripts\load_teaching_demo.py --username admin
```

脚本通过已认证 API 上传教学材料、等待后台入库并创建快照，在 `data/teaching_dev_questions.jsonl` 生成绑定该快照的五道虚构开发问题；标记保持 `fictional=true`、`human_reviewed=false`。认证失败时停止，重复导入仍受 API 的重复内容约束。登录/令牌流程的本地模拟接口测试通过，独立 Celery、正式存储和真实模型全流程仍待验收。

没有 PostgreSQL 时，下面的诊断仅使用临时 SQLite 保存测试数据，模型和 Embedding 均为真实调用：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\smoke_models.py
```

其输出明确标记为教学材料上的模型集成检查，不等同于 PostgreSQL/Milvus 联调、人工正确性评测或完整产品验收。正常应用不会退化为假模型、假向量或 SQLite。

若希望在 Docker 环境不可用时查看可交互的本地演示，可运行 `scripts\start.ps1 -Demo` 并按提示登录。该模式使用 SQLite 与 Celery SQLAlchemy broker，不使用 Milvus，也不提供 PostgreSQL 或 PostgresSaver 检查点保证，只适合本地演示。

## 验证与评测

```powershell
.venv\Scripts\python.exe -X utf8 -m pytest -q
.venv\Scripts\ruff.exe check backend evaluation scripts fixtures
```

确定性测试使用假模型和临时 SQLite，覆盖 PDF/页序/来源映射、RRF、两路范围、伪造引用、Unicode 字符范围、核验退回、有限补查、预算、模型故障、幂等、取消、旧 attempt、重复投递和 SSE 续读。PostgreSQL schema 另有编译检查；Milvus 回填、失败回滚、余弦排序与返回 ID 的快照范围另有回归。真实存储检查使用 scripts/smoke_milvus.py，与模型质量和独立 Celery 验收分别记录。

`evaluation/README.md` 说明正式独立人审与 Agent 审核两种入口、可选配置、逐题交错执行、哈希绑定评分及失败分母。正式人审仍要求至少 30 题；另有六份官方技术摘录、12 道 Agent 审核题的两组精简实验和 3 元总预算保护。小样本 Agent 分数不代表独立人审或正式业务准确率。

## 持久任务与限制

数据库 worker 使用租约和 attempt，进程退出后租约到期会从头新建一次执行。检查点保存图状态，但不声称外部模型恰好调用一次；重试可能再次计费。旧 attempt 的迟到结果无法发布。已发布答案不覆盖；修订用新运行建立新答案，原答案保留。

本版提供本机账号登录和知识库级管理员/编辑者/阅读者权限，不是公开部署的多租户云服务；不支持扫描件 OCR、无边框 PDF 表格和公式结构恢复、全网搜索、综述写作或原文物理删除。可检测的有边框 PDF 表格会按表头和数据行结构化入库，仍需对照原文件核验。中文全文分词仍有限，跨语言效果需要真实题集验证。语义 Verifier 会出错，不能取代独立人审。

详细结构见 `docs/architecture.md`；首版边界见 `docs/scope.md`；当前界面提案见 `docs/design.md`。
