# MCP、外部操作审批与确认记忆

记录日期：2026-11-12。用户已明确确认本轮新增三项能力。本轮不启用真实第三方服务，不执行真实业务写入，不修改 `.env` 凭据。基础能力通过登录后的后端接口使用；本项目 OpenAPI 的 `/docs` 可查看请求结构，原 Vue 工作台暂未新增这些操作入口。

## 默认开关与配置

新增 `MCP_ENABLED=false`、`MCP_WRITES_ENABLED=false`、`MCP_SERVERS_FILE`。没有配置时，查询 `/api/mcp/servers` 返回关闭状态；不会在应用启动时启动 MCP 进程或探测远端。

管理员确认具体服务、可读/可写工具、用户和知识库后，配置服务 JSON 文件路径，再在运行环境显式开启。示例见 `docs/examples/mcp_servers.example.json`：域名、用户、知识库和工具名都是占位，不能直接用于真实服务。JSON 不保存密码/Token，`headers_env` 的值是环境变量名；例如对应的变量保存完整 Authorization 头值。服务配置是管理员控制的本地文件，不接受模型或普通用户改端点。

stdio 示例字段为 `transport=stdio`、`command=可执行文件绝对路径`、`args=固定参数数组`、`env={进程环境名:宿主环境变量名}`。不使用 shell 拼接命令；SDK 传递基本操作系统环境与显式 env 引用，不传递整个 `.env`，不继承任意宿主密钥。stdio 是被配置的宿主进程，不是 OS 沙箱；仅配置已明确授权的服务。HTTP 使用 Streamable HTTP，远端需 HTTPS，本机协议测试允许 HTTP；不跟随重定向、不使用环境代理，也不支持内嵌 URL 凭据。

每项服务必须声明精确的 `user_ids`、`corpus_ids`、`read_tools`。远端 `readOnlyHint` 不能授权调用；本地白名单仍依赖配置者了解工具实际语义。客户端支持工具分页发现、JSON Schema 参数校验、调用限时和进程退出清理；没有 OAuth 登录管理和自动重试。凭据缺失、非法参数、发现失败均明确失败。

## 只读工具

1. 已登录用户调用 `POST /api/mcp/servers/{server_id}/discover?corpus_id=...`，后端先检查知识库成员及服务授权，只返回白名单工具。
2. 调用 `POST /api/mcp/servers/{server_id}/call`，提交 `run_id`、原始 `tool_name`、`arguments`。run 必须属于本人，使用其知识库权限；工具调用前再次检查身份范围及参数。
3. 请求意图先保存在 `mcp_invocations`，完成后保存回执或失败编码。`GET /api/mcp/invocations` 和原 run 的 `/trace` 可以查询。外部返回明确标为 `external_unverified_not_indexed`，不会自动写入知识库或成为答案引用。

已识别的凭据拒绝放进参数，结果中的已知配置凭据及常见敏感字段被脱敏。未知敏感值识别并不完备。stdio 服务诊断写入空设备，避免未经处理的外部 stderr 泄露凭据；应用保存结构化失败编码，当前不保留该原始诊断流。

## 外部写操作与结果未知

写工具还需在服务配置声明 `write_tools`，例如：

```json
{"save_note": {"resource_argument": "resource", "idempotency_argument": "idempotency_key"}}
```

这只说明参数契约，不会创建远端工具，也不保证第三方实现幂等。资源与幂等参数必须是顶层字段。服务本身须接受幂等参数；不支持这种契约的写工具当前拒绝接入。

1. 本人的知识库编辑者或管理员调用 `POST /api/external-operations`，提交 run、attempt、server_id、tool_name、resource 及完整参数，并给 `Idempotency-Key`。参数内的资源和幂等键必须与该提案一致。
2. 服务端发现并校验 Schema，保存 `proposed`。摘要绑定本人身份、任务/attempt、任务配置、端点与配置、凭据指纹、工具、Schema、完整参数哈希、资源和幂等键。`GET /api/external-operations/{id}` 展示完整参数、端点身份、摘要与操作事件供本人审核；敏感认证值不保存到审批内容。
3. 本人调用 `POST /api/external-operations/{id}/decision`，提交当前 digest 和 `approve`/`reject`。模型不能代替这个显式用户请求；本实现是本人确认，不是双人复核或组织审批。
4. 本人调用 `/execute`，提交 digest。后端重新检查当前成员角色、用户及知识库白名单、run attempt、任务配置、工具 Schema、端点/凭据配置和完整参数；改变后要求重新提案审批。通过原子状态更新领取一次执行，并先提交审计，再发送请求；审计失败不调用。一个提案最多发送一次，成功、拒绝、失败或未知均不能直接再次执行。
5. 请求已可能发送后的超时、连接丢失、远端工具错误或回执异常记 `unknown`；不能声称“没有写入”。未发送前的校验/连接失败记 failed。成功回执保存为 succeeded。
6. 若进程在已发送后退出，记录可能停在 executing。超过 150 秒后读取详情会保守转 unknown；没有后台自动重放。仅 unknown 可通过 `/resolve` 登记本人核对后的 succeeded/failed，必须填写核对证据。界面/记录会保留 `manual_resolution`，不是伪造供应商回执。

当前未实现供应商专用状态查询或补偿操作；先在第三方核对资源和幂等键，再手动登记结论。重新创建其他提案仍可能产生真实副作用，不能把本地一次发送等同第三方 exactly-once。断电或回执数据库不可写时保留 executing，后续按未知处理；本地工作区回滚不能撤销远端写入。

## 私有经验与偏好记忆

`POST /api/memories` 创建候选，kind 为 preference/success/failure，包含 key、content、可选 source_run_id 和带时区的 expires_at。内容上限 2000 字符。个人偏好可以由本人直接提出，经验必须关联本人的已结束问答；成功经验关联 completed，失败经验关联 completed/failed。它们都是用户确认的提示，不证明答案事实正确。

- `GET /api/memories` 或 `/{id}` 查看本人内容；知识库经验还检查当前成员权限。
- `POST /{id}/confirm` 提交当前 version，只有候选可确认；同类同 key 的有效记忆冲突时拒绝静默覆盖。
- `PATCH /{id}` 提交 version 和新内容，版本递增并返回 candidate，必须再次确认。记录来源、类型、版本、确认和修改元数据。
- `DELETE /{id}?version=...` 擦除记忆正文、key 和来源问题文字，不再用于新调用；审计保留无正文的删除事件。本人被移出知识库后仍可按已知 ID 删除自己的记忆。
- 过期记忆立即不参与检索；本人调用 `POST /api/memories/cleanup` 擦除本人过期正文。没有全局定时清理，不删除原任务、原模型调用或历史冻结清单。

创建新 run 时，以服务端身份检索已确认、未过期内容并冻结进执行清单。个人偏好最多 4 条；经验按成功和失败**分别查询、分别各保留最多 2 条**，候选池每类最多 100 条。当前使用英文词与中文双字片段匹配，是确定性的轻量检索，不是记忆向量库。经验还严格匹配知识库与文档 manifest 哈希；文档、版本或索引变化后不宽松复用旧经验，代价是召回减少。

冻结内容最多每条 600 字符，提供给 Planner 和 Analyst 的 `memory_hints`；读取前检查原记忆仍确认、版本相同、未过期且权限有效。被改动或删除后在后续调用中排除。已经发出的模型请求无法撤回，旧 trace/config 仍保留当时输入。偏好只影响表达，经验只影响搜索参考或避坑，事实仍须使用当前 sources 引用并通过核验。模型不会自动提炼、确认或收录聊天偏好，用户反馈也不会自动转为经验。

## 存储与验证边界

新增 migration `0006_tools_memory.py`，接在 `0005_code_repair` 后。五张表分别保存写操作、操作事件、只读调用、记忆和记忆审计。正式部署需先备份并执行 Alembic upgrade；本轮只升级教学数据库副本验证，不修改 `data/demo.db`。

本轮验证包括真实本机协议进程和本机 HTTP 服务、临时 SQLite 状态机及故障注入；这能证明当前实现的调用和拒绝路径，不证明真实第三方权限、真实账号费用、远端幂等或 PostgreSQL 多进程并发。实际测试命令和最终结果见第二部分实施核对和项目档案的本轮追加记录。
