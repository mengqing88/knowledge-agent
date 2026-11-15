# 历史实验与冒烟报告

本目录保存原来散放在 `docs/` 的 JSON，内容字节保持原样。失败、中断和早期尝试仍保留；不能只选最好一次作为结论。

| 主题 | 文件 |
| --- | --- |
| 分块控制集与结构对照 | [fixture](chunk-strategy-fixture-evaluation.json)、[结构](chunk-strategy-structural-evaluation.json) |
| 历史分块真实向量实验 | [最终记录](chunking-benchmark.json)、[首轮](chunking-benchmark-attempt1.json)、[中断](chunking-benchmark-interrupted.json)、[语义策略前](chunking-benchmark-before-semantic.json) |
| 检索与向量存储控制 | [检索](retrieval-benchmark.json)、[向量存储](vector-store-benchmark.json) |
| 查询改写 | [离线](query-rewrite-offline-evaluation.json)、[真实依赖](query-rewrite-live-evaluation.json) |
| 模型与角色冒烟 | [连接](connection-check.json)、[教学资料模型冒烟](live-model-smoke.json)、[角色冒烟](role-model-live-smoke.json) |
| 存储与配额冒烟 | [LlamaIndex](llamaindex-storage-check.json)、[Milvus](milvus-storage-check.json)、[Redis 配额](provider-quota-live-check.json) |

两组官方资料的问答结果在 [evaluation/results/public-3yuan-v1](../../../evaluation/results/public-3yuan-v1/)，BGE 冻结候选对照在 [v2 报告](../../../evaluation/results/public-3yuan-v1-bge-offline-v2/report.json)。本目录里的旧向量存储或旧分块实验不替代它们，也不代表生产准确率或性能承诺。

整理后的脚本默认把新生成报告放入 `tmp/reports/`，避免直接覆盖本目录的历史证据。原报告里记录的旧源码路径、哈希和运行身份保持原样，可通过[迁移表](../2026-11/项目目录整理_2026-11-15.md)定位新代码。
