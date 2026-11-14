# 结构与数据契约

模块化单体：FastAPI 接受文档上传和问答请求；Celery worker 执行任务，Celery beat 每 3 秒触发数据库任务领取；Redis 提供 Celery broker，PostgreSQL 保存用户、知识库、文档、任务、租约、事件和全文索引；Milvus 保存片段向量并执行相似度查询；PostgresSaver 保存每个 attempt 的 LangGraph 检查点。浏览器通过 API/SSE 读取持久化状态。Celery 消息不作为业务状态的唯一来源。

## 处理链路

PDF → PyMuPDF 页/文本块；可检测的有边框表格按表头和数据行提取，并保留页码及行 bbox；DOCX → 段落与表格；Markdown/TXT → 标题或段落；CSV/TSV → 表头、列值和源行号；JSON/JSONL → 标量字段和值及字段路径/行号；HTML → 静态文本、标题和表格文本（跳过脚本、样式、模板及有明确隐藏标记的元素）；XLSX → 工作表、表头和行值。各格式都保存为带文档版本及原文字符范围的 SourceBlock，再按 SHA-256、解析配置和索引配置创建版本化记录。切块与原文范围映射 → 实际 Embedding → Milvus HNSW/COSINE + PostgreSQL simple 全文索引 → ready。

支持 PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML/HTM、XLSX；不会仅凭扩展名接纳任意二进制格式。文本类文件要求 UTF-8，CSV/TSV 第一条非空记录作为表头；JSON 只把标量叶子转成可引用块；HTML 不运行脚本；XLSX 使用只读工作簿解析、限制工作表/行/列/单元格和 ZIP 解压体积，公式按公式文本索引而不计算。旧版 `.xls`、图片 OCR、无边框 PDF 表格恢复、HTML 动态内容、XLSX 公式计算均未实现。有边框 PDF 表格结构检测失败时保留普通文本解析结果。非 PDF 引用使用章节、工作表/行号或 JSON 字段路径，不伪造页码。

分块默认 fixed/512/64，按参考 token 窗口和重叠保存原文范围，保留页/章节边界；legacy 字符规则及其他策略可配置。长度不等于供应商账单 tokenizer；改变配置需要重建索引/快照。保留原文、页序、bbox 和印刷页标签；检索清洗不会覆盖来源。重复边缘文字仅从检索表示排除。双栏、无边框表格、公式与扫描件限制保存在版本 warnings 中。

每个新快照明确包含 paper/version/parse/index/hash/model/dimension，以及 vector_backend/vector_collection。旧快照不重写，仍按原索引 ID 解析当前存储位置。索引配置还绑定 Embedding 服务地址的哈希、期望维度和手动 revision。相同模型名但服务或配置改变时不能直接混查；原文件可重新建索引，新问题显式创建新快照。

## 受控问答

Planner → 查询改写 → Retriever/重排 → 可选 MCP 只读循环 → EvidenceGate → Analyst → Verifier → route → 人审发布；route 可回 Retriever 或 Analyst，但轮数、调用数、上下文和时长都由创建任务时冻结的后端配置限制。按模式和配置可跳过部分模型节点，不能据此跳过原文引用与权限检查。

查询改写保留 500 字符内的原问题，增加当前知识库术语表达并过滤部分字面量漂移；更长问题明确跳过该模块。`workflow_tools.py` 在检索后只向模型提供已授权的只读工具，默认最多两次执行；内置 stdio 工具只读取本人在当前库的项目运行日志。成功回执通过 invocation/hash/字符范围形成独立 `mcp_observation` 引用，不入文档索引、不伪造页码，查看时继续检查证据权限。迁移 0011 扩展引用来源，保留旧文档引用。

`model_routing.py` 解析冻结的角色路由，支持专用 `/rerank`；未指定角色沿用原模型。Provider 每次 HTTP 尝试前可用 `provider_quota.py` 在 Redis 原子预留 60 秒 RPM/估算 TPM，按服务端点和凭据共池；失败不退预留，Redis 不可用时不发送请求。原有熔断仍在进程内。开启方式、实际证据与限制见 [四项升级](查询改写_MCP编排_角色模型与共享配额.md)。

专用 rerank API 使用一条请求；`query` 包含原问题和去重后的 Planner 子问题目标，候选仍映射回服务端已授权的 chunk 与 block ID。应用内 LLM 重排还会收到每个候选的 `retrieved_for_subquestion_ids`；该列表仅作召回来源提示。

Retriever 不调用聊天模型。原生 PostgreSQL 全文召回限制索引 ID 白名单；Milvus 为每个索引版本使用独立 collection（HNSW/COSINE，M=16、efConstruction=200、ef=64），返回的 Chunk ID 再逐 collection 对照 PostgreSQL 中的索引归属，以 Chunk ID 做 RRF(k=60)。COSINE 分数越大越相似，合并时按降序排名。原生 simple 分词不等于 BM25 或高质量中文分词；中文有必要实体子串匹配，跨语言语义依赖实际向量模型与 Planner 的英文查询。多论文按对象交错取证并限制每篇数量。

同一 Chunk 可能由多个子问题的查询命中；工作流合并候选时保留全部 `retrieved_for_subquestion_ids`，并传给 Reranker 与重排后的 EvidenceGate，避免只留下首次命中的子问题。该字段只表示检索来源关系，不代表证据已支持对应问题；两个提示都要求模型检查原文，不把该字段当成相关或支持结论。

证据 ID 由后端先分配。摘录必须等于 `SourceBlock.text[start:end]`，范围是 Python Unicode 字符下标，左闭右开。页序和版本从数据库派生，不信任模型字段。最终文本直接取逐条论点，发布后不再自由润色。

full_agent 只发布 Verifier 明确批准且不存在未解决问题的论点；基础 RAG 和关闭 Verifier 仍做相同的引用定位与快照检查，`semantic_verified=false`。Verifier 并非事实裁判，必须另做人审评测。

## 执行与恢复

API 不使用进程内后台任务。任务、状态、attempt 和事件写入数据库，worker 用 `FOR UPDATE SKIP LOCKED` 领取任务。10 秒心跳维护 90 秒租约；进程异常退出后租约到期可自动用新 attempt 从头执行，保留旧模型与检索记录。发布事务锁定任务，校验 attempt、状态和取消标记，并通过 `(run_id,revision)` 唯一约束防止重复答案。旧 attempt 的迟到结果无权发布。

检查点不提供恰好一次外部调用保证。本版重启恢复采用新 attempt 从头执行，可能再次产生模型费用；没有宣称自动从上一条外部请求续跑。业务 state 每个节点持久化，SSE 持久事件先提交再推送，支持 Last-Event-ID/after_seq 重连。断开浏览器不取消后台任务。

2026-11-14 用户明确确认保留该行为：保证答案不重复发布，允许恢复后重新请求模型及再次计费。恢复仍受累计预算与最大 attempt 限制，不新增完成调用复用或未知结果停止自动重试机制；这项确认不扩大真实模型测试或生产操作范围。

取消是协作式：立即保存 cancelled，进行中的网络请求可能持续到返回或超时，但之后不再运行节点或发布。运行时长也在节点与检索查询之间检查，单次模型请求有独立超时。

## 身份与访问范围

本版面向本机团队演示，API/工作台监听 127.0.0.1。管理员初始化账号，管理员在知识库内创建或邀请成员；登录使用 scrypt 密码哈希和服务端会话，Cookie 为 HttpOnly/SameSite=Strict。身份认证回答“是谁”，知识库成员角色回答“能做什么”；后端对文档、快照、运行、文件、证据、答案和反馈执行归属及角色校验。授权粒度为整个知识库，不提供单文档权限、公开注册或公网多租户服务；原文移出知识库后保留以支持历史引用。

## 目录

`backend/app/ingestion.py` 负责来源与索引；`retrieval.py` 负责混合搜索；`evidence.py` 负责定位；`workflow.py` 管理图；`worker.py` 管理持久任务；`services.py` 管理快照、幂等和序列；`main.py` 为 API；`evaluation/` 提供开发/正式评测格式和三配置运行器。

## 答案人工审核

`answer_review_required` 默认开启并写入冻结的执行清单。Verifier 和来源检查通过后，worker 保存不可变答案版本并将 run 置为 `review_pending`，此时任务已停止但答案尚未发布。阅读者只收到待审核状态；管理员/编辑者通过 `/api/corpora/{id}/answer-reviews` 取队列并读取待审论点与引用。

审核 API 将答案哈希、答案版本、审核人、决定和拒绝理由写入 `answer_approval_records`。接受时转为 `completed`，并创建 `accepted_answer_experiences`；任何答案变化都会让发布读取的哈希检查失败。拒绝时转 `rejected`，理由保留，发起人 replay 创建新 run、沿用相同快照并把理由附加到修订问题。历史答案和 trace 不覆盖。

此门禁增加人工等待与审核队列工作量；它解决模型自我核验不能替代独立人工决定的问题。自动评测进程需显式设置 `ANSWER_REVIEW_REQUIRED=false`，否则 run 会按产品规则停留在待审状态，不会被评测运行器当作已完成。

## 独立代码修复流程

`backend/app/code_repair.py` 与 `code_api.py` 提供本机、按用户隔离的代码修复任务 API。创建任务会拒绝脏源仓库，解析并保存固定 commit，在源仓库外创建 detached worktree。模型只能看到显式选中的 Python 文件；补丁应用前由程序校验范围并保护测试、构建和 CI 配置。

测试器不在可修改 worktree 里直接执行。它把固定提交与补丁复制到临时 staging，剔除密钥文件和路径越界项，再用 digest 固定的本机 Docker 镜像运行固定 pytest 命令；容器无网、只读、非 root，并受 CPU/内存/PID/时间限制。该环境信息、测试输出摘要和补丁哈希共同形成审核绑定。

测试通过后代码修复任务进入 `review_pending`。接受写入绑定具体补丁的经验；拒绝要求原因、持久化审核记录并把副本重置到基线。清理失败会留下可见状态并允许重试。它与上面的问答答案审核分表、分角色决定；代码修复仍没有自动生成专用复现测试。

## Milvus 写入、迁移与边界

新索引先向 Milvus 分批 upsert、flush 并按主键读回核对，成功后才提交 PostgreSQL 索引记录和 active 版本切换。跨存储没有分布式事务：失败会回滚业务数据，Milvus 可能留下不可达 collection，重试会产生新索引；当前没有自动孤儿回收。网络写入期间持有文档行锁，单 RPC 默认 20 秒超时，不承诺整次调用恰好一次。

0009 把旧 PostgreSQL VECTOR 转成 nullable JSON，标记 legacy_pgvector；migrate-vectors 按原 chunk_id 回填，不调用 Embedding、不重写快照。整索引核对成功后，事务切换 vector_backend/vector_collection 并清空 PostgreSQL 向量；部分失败保留全部原向量，可安全重跑。schema downgrade 不支持数据恢复，回退需 PostgreSQL 备份。

Docker 采用官方 embedded Standalone 方式：内嵌 etcd、本地持久卷，未使用 MinIO。旧 MinIO 社区镜像/下载入口不可用；官方参考来源、操作及验证见 [Milvus迁移与验证](Milvus迁移与验证.md)。单机及每索引一个 collection 方便隔离模型和快照，但增加 collection/内存成本，没有集群高可用、容量或百万级性能证据。HTTP 健康与重启后数据可查是两个条件。

## LlamaIndex 与 LangGraph 的接入边界

`llama_rag.py` 将既有 Chunk/原文范围映射为 TextNode；ProviderEmbedding 复用项目模型、超时和运行账本。入库经 VectorStoreIndex 写 Milvus，查询经 VectorIndexRetriever 从 Milvus 取候选并到 PostgreSQL 核对归属/恢复原文，之后仍做全文 RRF 和现有重排。LangGraph 继续管理规划、分析、核验和有限补检，答案仍走既有人审。默认切分和默认模型均不使用，已有 Milvus schema/数据格式兼容；实现、代价与实机证据见 [组合实现](LlamaIndex与LangGraph组合实现.md)。
