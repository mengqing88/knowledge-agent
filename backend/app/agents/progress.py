"""Public progress is a projection of the audit log, never a draft answer."""

MESSAGES = {
    "query_rewritten": "检索表达已整理", "tool_started": "正在查询授权工具",
    "tool_finished": "工具查询结束", "tools_completed": "外部查询阶段完成",
    "queued": "问题已排队", "started": "开始处理", "planner": "问题规划已完成",
    "retriever": "证据检索与筛选完成", "analyst": "候选答案已生成，等待检查",
    "verifier": "本轮核验已完成", "review_pending": "答案等待人工审核",
    "completed": "答案已发布", "cancelled": "已取消后续处理", "failed": "处理失败",
    "rejected": "答案审核未通过", "retrieval_degraded": "正在使用可用的检索方式",
    "evidence_gate": "生成前证据检查完成",
    "local_rerank_started": "正在精排检索候选",
    "local_rerank_finished": "候选精排已完成", "context_selection": "原文上下文已选定",
}
NUMBERS = {"round", "revisions", "evidence_count", "new_evidence", "claim_count", "approved",
           "queries_done", "queries_total", "candidate_count", "seconds", "blocked_count"}
ENUMS = {
    "node": {"planner", "retriever", "tools", "evidence_gate", "analyst", "verifier", "route", "publish"},
    "role": {"Planner", "Embedding", "Reranker", "EvidenceGate", "Analyst", "Verifier", "ToolRouter"},
    "next": {"retriever", "analyst", "verifier", "publish"},
}


def progress_event(event):
    payload = {key: value for key, value in event.payload.items()
               if (key in NUMBERS and type(value) in (int, float)) or
               (key in ENUMS and isinstance(value, str) and value in ENUMS[key])}
    if event.kind in MESSAGES:
        payload["message"] = MESSAGES[event.kind]
    return {"seq": event.seq, "kind": event.kind, "attempt": event.attempt,
            "payload": payload, "created_at": event.created_at.isoformat()}
