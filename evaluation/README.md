# 评测导航

问答效果、工程回归、Agent 评分和真人评分分开报告。未执行项、失败、取消、未知用量均保留，不通过删题或补跑凑分。

长文本开发集的宽召回/本地 BGE/后续原文块实验使用 `scripts/benchmarks/benchmark_context_recall.py prepare/score`。题集及数据库哈希、模型文件哈希、完整推理输入哈希一并绑定；模型只接收问题和候选原文，不接收标注。实际 80% 门槛及剩余限制以[本轮记录](../docs/records/2026-11/召回覆盖率80达标对照_2026-11-17.md)为准；该开发诊断不能当作独立评测或网页默认质量。

后续子问句与上下文名额实验使用 `scripts/benchmarks/benchmark_facet_recall.py prepare/score`，复用冻结 BGE 分数并绑定实现哈希；同题开发集最终 12 块覆盖 41/43（95.35%）。目标、父结果及失败变体见[95% 开发对照](../docs/records/2026-11/召回覆盖率95开发对照_2026-11-17.md)；这不是独立评测或网页默认质量。

## 已保存的材料与结果

| 路径 | 内容与边界 |
| --- | --- |
| [public_docs_v1/](public_docs_v1/) | PostgreSQL 17.6 / Celery 5.5.3 六份官方摘录、原始文件、许可证、价格依据和 12 题；不是两套完整手册 |
| [两组问答结果](results/public-3yuan-v1/) | 基础 RAG / 完整 Agent 各 12 题，逐题 trace、失败、manifest、总账和 Agent 评分 |
| [BGE 离线对照](results/public-3yuan-v1-bge-offline-v2/report.json) | 冻结候选的排序比较，不是端到端答案质量 |
| [上下文选择对照](../docs/records/2026-11/上下文选择离线对照_2026-11-15.md) | 当前块级筛选 / 补邻段 / 返回整块；9 题标注原文覆盖相同，当前方式最紧凑，非答案正确率 |
| [本机长文本覆盖测试](../docs/records/2026-11/七堇年长文本覆盖率测试_2026-11-16.md) | 17 道复合题、43 条原文标签；隔离 SQLite RRF@5 为 7/43，逐题语料和数据库留在本机 `tmp/` |
| [中文召回改进对照](../docs/records/2026-11/中文长文本召回改进对照_2026-11-16.md) | 同题同片段数的 BM25/混合/小窗口/子问句实验；混合 Recall@12 从 10/43 到 18/43，有退步题，未切换产品默认 |
| [中文词法实际接入](../docs/records/2026-11/中文词法召回实际接入_2026-11-16.md) | 可选后端、任务配置冻结和容量限制；正式 PostgreSQL/Milvus 同题复验 10/43→18/43，新收费调用 0，保留 legacy 默认 |
| [评测规范](评测要求.md) | 正式至少 30 道独立人工审核题、指标定义与证据要求 |
| [发布与回退](版本发布与回退.md) | 版本组合及回退检查 |
| `quality_suite.json`、`hallucination_dev.jsonl` | 虚构教学控制材料，只作开发验证 |

两组已执行 24 份记录，含 7 份结构化输出失败；详见[原始收尾记录](../docs/records/2026-11/3元精简收尾_2026-11-14.md)。已有 Agent 二次复核，真人评分仍待完成。应用汇总的费用未知与持久化总账的估算口径不同，不能混报为供应商账单。

## 匿名人工试评

现有 12 题已备包，本机路径 `tmp/independent-review-20261115/`。评审者只收 `reviewer.zip`，操作者单独保管 `operator-key.json`；评分表初始全部待审。A/B 按题随机，包含原文、草案评分要求和 24 份答案/失败记录。

```powershell
# 导出到一个不存在的新目录；不调用模型。
.venv\Scripts\python.exe -m evaluation.blind_review export `
  evaluation/results/public-3yuan-v1 --corpus evaluation/public_docs_v1 `
  --output tmp/independent-review-新的唯一目录

# 实际真人填写后回收，不覆盖原结果或原 summary。
.venv\Scripts\python.exe -m evaluation.blind_review import `
  evaluation/results/public-3yuan-v1 `
  --key tmp/independent-review-20261115/operator-key.json `
  --scorecard tmp/returned-scorecard.json --output tmp/human-review-新的唯一目录
```

导入校验源资料、manifest、逐题结果、阅读册哈希、评分表身份和显式真人声明；程序不能认证评分者是真人或独立。空表/部分审核不生成总体质量率。12 题全部评完仍为小样本试评，详见[填写与独立性要求](../docs/records/2026-11/独立人工试评说明_2026-11-15.md)。

## 运行入口

| 用途 | 入口 |
| --- | --- |
| 两组官方资料实验、预算与定向续跑 | `scripts/evaluation/run_public_evaluation.py` |
| 已存请求的本地 BGE 排序对照 | `scripts/evaluation/compare_saved_reranker.py` |
| 已存候选的三种上下文组装对照 | `scripts/evaluation/compare_context_selection.py`，不调用模型 |
| API 配置选择/交错执行 | `evaluation/run.py` |
| 空人审或 Agent 标注模板 | `python -m evaluation.review` |
| Agent 逐题注释绑定 | `python -m evaluation.apply_agent_review` |
| 客观计数与分离的评分汇总 | `python -m evaluation.report` |
| 分块/检索/历史质量控制 | `scripts/benchmarks/`、`evaluation/chunking.py`、`evaluation/query_rewrite.py` |

收费运行只允许专用预算入口。总费用上限 ¥3、¥2.5 停止新增请求；未知价格/用量或未核实费用会停止。当前历史专用重排 smoke 的账外费用未核清，**收费续跑仍暂停**。通用 API 运行器没有同一本实验预算账，不可用于绕过这条限制。

正式模式 `evaluation/run.py` 要求至少 30 题 `human_reviewed=true` 且非虚构；声明不代替实际审阅证据。Agent 模式使用 `--review-source agent`，与人审指标隔离。开发题必须显式标记，只能用于开发运行。

实验应在独立验收栈内运行；正常产品的答案发布审批保持开启。隔离自动评测显式关闭 `ANSWER_REVIEW_REQUIRED` 后，该设置冻结到 run。不要临时修改运行任务，也不要自动审批被阻挡的答案。

## 汇总口径

- 以 manifest 的全部分配题为分母；失败/取消/缺失不删，技术失败不计为正确拒答。
- 质量评分绑定 manifest 与逐题 SHA；改结果后旧表失效。整组审核不完整时总体质量率为 null。
- `manual_metrics` 与 `agent_metrics` 分栏；Verifier 是被测流程的一部分，其通过位不作质量标签。
- 无引用时支持率不适用；状态匹配/Hit@5 只是限定口径的代理指标，不能改称业务准确率或 Recall@5。
- 所有 attempt、重试、Embedding 均进入用量；未知不能记零。P95 报观测数，采用最近秩；小样本不能外推生产 SLA。
- 累计原文、失败、价格/预算、运行身份和调用轨迹保持原样。人审、Agent 复核、工程测试和真实模型效果不相互替代。

完整历史命令和早期开发说明已移到[历史说明](../docs/records/2026-11/评测入口历史说明_2026-11-15.md)。当前剩余任务看[TODO](../TODO.md)，脚本位置看[脚本导航](../scripts/README.md)。
