# 文档导航

第一次看项目，从 [项目首页](../README.md) → [启动与日常使用](guides/启动与日常使用.md) → [架构与数据契约](architecture/architecture.md) 阅读。当前待办只看 [TODO](../TODO.md)，历史记录按日期理解。

## 使用与维护

| 内容 | 入口 |
| --- | --- |
| 安装、启动、停止、注册、上传和提问 | [启动与日常使用](guides/启动与日常使用.md) |
| 模型配置、手动启动、API、教学材料和检查命令 | [开发与运维](guides/开发与运维.md) |
| 本地数据、临时文件、评审包和模型的位置 | [本地工作区](guides/本地工作区.md) |
| MCP、外部操作审批和确认记忆 | [使用说明](guides/MCP_审批与记忆_使用说明.md) |
| 题集、两组结果、人工评审和预算 | [评测导航](../evaluation/README.md) |
| 可执行脚本按用途查找 | [脚本目录](../scripts/README.md) |
| 后端模块位置 | [后端目录](../backend/README.md) |
| 前端模块位置 | [前端目录](../frontend/README.md) |

## 架构与实现

| 主题 | 文档 |
| --- | --- |
| 产品边界、数据契约和权限 | [范围](architecture/scope.md)、[架构](architecture/architecture.md) |
| LlamaIndex / LangGraph、Milvus | [框架组合](architecture/LlamaIndex与LangGraph组合实现.md)、[向量存储](architecture/Milvus迁移与验证.md) |
| 解析与分块 | [多格式解析](architecture/多格式数据解析实现记录.md)、[分块对照](architecture/分块策略与对比评测.md) |
| 查询改写、MCP 编排、角色模型与配额 | [四项升级](architecture/查询改写_MCP编排_角色模型与共享配额.md) |
| 证据与幻觉控制 | [检索门控](architecture/检索质量门控与对照实验.md)、[防幻觉](architecture/防幻觉增强与开发评测.md) |
| 等待进度、取消、恢复 | [流式进度](architecture/流式进度优化.md) |
| 配置示例 | [MCP 服务](examples/mcp_servers.example.json)、[查询术语](examples/query_terms.example.json) |

## 验证记录与历史资料

| 类别 | 入口 |
| --- | --- |
| 本次目录整理与路径迁移表 | [整理记录](records/2026-11/项目目录整理_2026-11-15.md) |
| 3 元两组小样本、费用和失败 | [收尾记录](records/2026-11/3元精简收尾_2026-11-14.md) |
| 真实存储/队列的隔离联调 | [联调记录](records/2026-11/隔离联调验收_2026-11-14.md) |
| 匿名人工试评准备，仍待真人评分 | [试评说明](records/2026-11/独立人工试评说明_2026-11-15.md) |
| 发送模型请求前的时间与权限复核 | [修复记录](records/2026-11/模型请求发送前复核修复_2026-11-14.md) |
| 历史控制实验与冒烟 JSON | [结果索引](records/benchmarks/README.md) |
| 五部分需求、实施核对、旧界面提案 | [历史资料索引](archive/README.md) |

[项目共享说明](../项目共享说明.md) 是开发协作约定及累计记录；[项目档案](../项目档案_供简历Agent使用.md) 保留原文。它们不替代当前操作手册。档案里的历史代码路径可通过整理记录中的迁移表定位。
