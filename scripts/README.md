# 脚本导航

以下路径均从项目根目录执行。`start.ps1`、`stop.ps1`、`manage.py`、`agent_profile.py` 和 `process-state.ps1` 保留在本层，日常启动命令不变。

| 用途 | 目录与脚本 | 外部依赖/调用 |
| --- | --- | --- |
| 启停与维护 | `start.ps1`、`stop.ps1`、`manage.py` | 本机数据库/队列；doctor 会调用真实模型 |
| 教学导入 | `demo/load_teaching_demo.py` | 已登录 API，入库会调用配置的 Embedding |
| 两组官方资料评测 | `evaluation/run_public_evaluation.py`、`prepare_public_evaluation.py` | 前者收费模型及独立栈；后者下载官方资料 |
| 已存候选 BGE 离线对照 | `evaluation/compare_saved_reranker.py` | 本地模型与独立推理环境；不发模型 API |
| 已存候选上下文选择对照 | `evaluation/compare_context_selection.py` | 本地参考 Tokenizer；不调用模型或数据库 |
| 中文词法召回离线对照 | `benchmarks/benchmark_lexical_recall.py` | 本机只读 SQLite、私有范围题集、已存排序/可选子问句向量；无模型请求，逐题输出仅限 `tmp/` |
| 宽召回、BGE 与后续原文块对照 | `benchmarks/benchmark_context_recall.py prepare/score` | 只读 SQLite、冻结查询向量、本地 BGE；推理经 `models/local_rerank.py` 单独运行，逐题数据仅限 `tmp/`，不改变网页默认 |
| 中文词法实际存储验证 | `benchmarks/validate_lexical_runtime.py` | 复制本机快照/向量到临时 PostgreSQL/Milvus；不调用模型，结束清理临时库和集合 |
| 分块、检索、质量和向量存储实验 | `benchmarks/` | 因入口而异，部分调用收费模型/真实数据库 |
| 模型/存储/角色/配额冒烟 | `smoke/` | 真实服务；模型冒烟会产生费用 |
| 隔离工程验收 | `acceptance/isolated_acceptance.py`、`browser_acceptance.py`、`acceptance_concurrency.py` | 独立临时存储/队列及模型替身；浏览器需额外运行依赖 |
| 本地模型下载与推理 | `models/download_reranker.py`、`local_rerank.py` | 权重下载/本地推理环境 |

常用命令：

```powershell
scripts\start.ps1
scripts\stop.ps1
.venv\Scripts\python.exe scripts\manage.py --help
.venv\Scripts\python.exe scripts\evaluation\run_public_evaluation.py --help
```

新生成的常规实验/冒烟报告默认放在本地 `tmp/reports/`；已冻结的历史报告放在 `docs/records/benchmarks/`。两组正式保存的原始执行记录仍在 `evaluation/results/`，不移动、不覆盖。原脚本路径与新位置的对照见[整理记录](../docs/records/2026-11/项目目录整理_2026-11-15.md)。

当前收费评测仍受 ¥3 总预算和未知费用停调约束，不能把更换目录当作新预算。此导航不表示这些实验本轮都执行过。
