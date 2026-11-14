# Milvus 迁移与验证

更新：2026-11-12。用户已确认完整迁移：Milvus 存向量，PostgreSQL 保留业务数据、原文、权限、版本、快照与全文检索；运行方式为 Docker Milvus Standalone。

## 当前分工

| 部分 | 当前实现 |
| --- | --- |
| 向量生成 | 原有 Embedding 服务；模型、维度、地址/revision 指纹仍校验 |
| 向量写入/召回 | pymilvus 2.4.15；Milvus 2.4.15；每个索引版本独立 collection |
| 索引/距离 | HNSW（M=16、efConstruction=200），COSINE（ef=64）；分数从高到低排序 |
| 原文、权限、版本与全文 | PostgreSQL 17；原生 simple FTS + 必要子串匹配，仍非 BM25 |
| 融合与重排 | 原有 RRF(k=60) + 聊天模型重排；没有独立 Cross-Encoder |
| 本地单机部署 | Docker Standalone，内嵌 etcd + 本地持久卷 `paperqa_milvus` |
| SQLite Demo | 保留 JSON/Python 余弦，仅供既有演示与确定性测试；不使用 Milvus |

生产 `Chunk.embedding` 不再写入向量；nullable JSON 列只为旧库回填过渡和 SQLite 测试保留。Python pgvector 依赖和 SQL `cosine_distance` 已移除。Compose 的 PostgreSQL 镜像仍为 `pgvector/pgvector:pg17`，目的是让含旧扩展/旧向量列的数据卷可原地升级；**业务检索已不使用该扩展**。没有对已有卷执行扩展级联删除。

## 场景、理由与实现

用户要求把向量检索交给 Milvus。分工后，上传版本的解析结果仍保存到 PostgreSQL，Embedding 数组以 chunk_id 为主键写到索引自己的 collection。`vector_store.py` 创建 HNSW/COSINE collection、校验维度，按批 upsert、flush，再读回主键核对。

`ingestion.py` 在写入核对通过后才提交索引记录及 active 版本切换。若新版本写入失败，业务事务回滚，旧版本继续启用。这个边界必须放在入库发布之前，否则“文件上传成功”会被误当作“可查询的新版本成功”。

`retrieval.py` 只查询快照列出的索引 collection；Milvus 返回的每个 ID 再到 PostgreSQL 核对其索引归属，然后按 COSINE 分数降序合并向量候选，和原生全文候选做 RRF。外部向量库返回了另一版本的 ID，也不能进入本快照证据。权限校验、原文摘录和引用定位继续由原有业务层负责。

## 启动和旧库迁移

已有生产库升级前应保留 PostgreSQL 备份。停止针对该库的旧 API/worker，避免旧 ORM 同时操作已更改的数据类型。当前单机只绑定 127.0.0.1：PostgreSQL 55432、Redis 56379、Milvus SDK 19530、Milvus 健康接口 19091。

```powershell
uv sync --frozen
docker compose up -d --wait db redis milvus
.venv\Scripts\python.exe -X utf8 scripts\manage.py migrate
.venv\Scripts\python.exe -X utf8 scripts\manage.py migrate-vectors
```

`scripts/start.ps1` 已按以上顺序接入，再初始化账号并启动应用；`-Demo` 跳过 Milvus 回填。实际 `.env` 密钥与原演示进程未更改。本次正式 PostgreSQL 是新建空库，已升级到 0009；`migrate-vectors` 返回 0 个旧索引，不能写成迁移了真实业务语料。原 `data/demo.db` 和 `data/team-demo.db` 没有执行数据搬迁。

0009 将已有 PostgreSQL `VECTOR NOT NULL` 转成 nullable JSON，并把没有 Milvus collection 的旧索引标成 `legacy_pgvector`，不会虚构一个已上传的 collection。回填按原 chunk_id 上传，不调用模型、不改快照或原文。每个索引的全部批次核对成功后，才在同一个 PostgreSQL 事务中切换为 Milvus 并清空旧向量。中途失败保留该索引的全部原向量；已完成的其他索引不重传。再次运行同一命令即可，upsert 保证稳定主键不重复。

可选限定范围：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\manage.py migrate-vectors --index-id 实际索引ID --batch-size 128
```

未回填索引不能创建新的生产快照，也不能走生产向量查询。数据库 schema downgrade 不会还原已经清空的向量，0009 明确拒绝降级；回退应恢复升级前 PostgreSQL 备份。已有 run 的冻结代码/依赖校验可能拒绝新代码，需创建新的运行，不能改写历史运行清单。

## 实际验证

| 验证 | 结果与边界 |
| --- | --- |
| Compose 配置 | `docker compose config --quiet` 通过，PostgreSQL/Redis/Milvus 三服务健康 |
| 正式空业务库 | Alembic 从空库升级至单 head 0009，通过；旧生产向量数量为 0 |
| 真实 Milvus 4096 维 | 写入、flush、主键读回、重复 ensure_collection、余弦检索通过；精确匹配 1.0，正交 0.0 |
| 真实 PostgreSQL 旧类型转换 | 隔离临时库重建旧 VECTOR NOT NULL，实际运行 0009 转 JSON/nullable；保留 4096 维数据 |
| 真实回填 | 1 个合成索引、2 个 Chunk，批大小为 1；写完才切换，重复命令跳过，PostgreSQL 向量清空 |
| 正式入库与范围 | Markdown 新版本经生产 PostgreSQL/Milvus 路径建索引，业务库只留文本和 FTS；故意放入另一索引的 ID，被快照过滤排除 |
| 写失败保护 | 实际 PostgreSQL 事务中注入 Milvus 写失败，旧版本继续 active，失败新索引未提交 |
| 重启持久化 | 重启本项目 Milvus 后实际查询成功；出现一次数据面查询超时后恢复，不能宣称无中断 |
| 确定性回归 | 新增 7 项迁移/排序/范围/失败核对；最终全量 **159 passed, 4 warnings in 143.75s**，4 条均为 Alembic 配置弃用提示 |
| 静态检查 | 全仓 Ruff 通过；该变更不涉及前端 |

实机报告：[milvus-storage-check.json](milvus-storage-check.json)。重现命令：

```powershell
.venv\Scripts\python.exe -X utf8 scripts\smoke_milvus.py --restart-milvus
```

脚本只创建并删除自己的随机临时 PostgreSQL 数据库与 Milvus collection，使用合成 4096 维向量，不调用付费模型。`--restart-milvus` 会重启当前 Compose 的 Milvus；省略时不做重启测试。报告不能证明聊天模型质量、真实语料 Recall@5、性能优势、登录/引用全链路或独立 Celery 整栈验收。

## 已遇到的问题及代价

最初三容器配置无法下载 MinIO，官方旧社区下载入口返回 410，镜像源返回拒绝访问。改用 Milvus 官方已提供的 embedded Standalone 方式，保留用户确认的 Docker 单机部署；来源：[官方 standalone_embed.sh](https://github.com/milvus-io/milvus/blob/v2.4.15/scripts/standalone_embed.sh)。当前使用已验证的 v2.4.15 镜像及独立持久卷。

pymilvus 2.4 仍导入 pkg_resources，因此将 setuptools 限定为 `<81`；启动会产生该上游依赖的弃用提示。该提示与 pytest 的 Alembic 配置警告分开记录。

第一轮重启检查在 HTTP 健康通过后即查询，遇到旧节点租约尚未清除的超时，且清理报错一度掩盖原错误。临时资源已清理，检查脚本改为在有限恢复窗口内实际查询；第二轮通过，报告保留 `restart_data_ready_retries=1`。产品运行时仍按单 RPC 默认 20 秒超时，重启窗口内请求可能失败，不提供透明零中断承诺。

Milvus 和 PostgreSQL 之间没有分布式事务。向量写成功而 SQL 提交失败可能留下不可达 collection；检索通过 PostgreSQL 归属过滤不会公开它，但占用空间，当前需人工按未引用 collection 核对清理，没有自动垃圾回收。每索引一个 collection 的简单隔离会增加内存和管理成本；入库/回填期间持有 SQL 行锁。它适合当前单机演示，没有集群高可用、容量与并发性能验收。重要数据应同时备份 PostgreSQL、原文件与 Milvus 持久卷。

## 面试表述

“我把向量检索迁到 Milvus，PostgreSQL 继续管理业务、原文和全文检索，再用 RRF 融合两路结果。为了避免新文档写到一半就替代旧版本，我在 Milvus 写入和读回核对成功后才提交版本切换；旧向量迁移保留 chunk_id，失败回滚，可重跑。做过真实 Docker Milvus 和 PostgreSQL 的合成向量验证，但还没有大规模性能或真实题集效果数据。”

可能追问：为什么写入成功还要核对，以及为什么不能只相信向量库返回的 ID？回答：两套存储的发布不是一个原子事务，主键核对确定待发布向量可见；返回 ID 还需在业务库确认它属于本次快照的索引，防止版本和引用范围混用。
