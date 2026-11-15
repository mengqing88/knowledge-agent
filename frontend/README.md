# 前端代码导航

Vue 3 + TypeScript 工作台；目录整理不改变界面和 API。

| 路径 | 职责 |
| --- | --- |
| `src/App.vue` | 工作台入口，登录/注册、材料、问答与审核交互 |
| `src/components/CodeRepairPanel.vue` | 代码修复任务面板 |
| `src/api/client.ts` | HTTP 调用、错误及审核请求 |
| `src/features/runs/` | 运行进度、SSE 恢复和答案视图逻辑 |
| `src/types/index.ts` | 前后端数据类型 |
| `src/styles/` | 工作台及团队样式 |
| `src/main.ts` | Vue 初始化 |
| `public/` | 当前静态资源 |
| `tests/` | 审核提交、运行视图与 SSE 测试 |

从 `frontend/` 执行：

```powershell
npm.cmd ci
npm.cmd run dev
# 或生成后端可直接提供的静态工作台
npm.cmd run build
```

开发地址 `http://127.0.0.1:5174/`，代理 `/api` 到 `http://127.0.0.1:8000`。检查命令为 `npm.cmd test` 和 `npm.cmd run build`。最早的静态布局提案已归入 [历史设计](../docs/archive/design/layout-preview.html)。
