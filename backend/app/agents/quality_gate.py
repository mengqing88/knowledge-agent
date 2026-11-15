"""Validate model readiness judgements against the final, scoped evidence supplied to it."""
from app.core.errors import ProviderError


def validate_readiness(plan, sources, report, executed_queries):
    targets = {q["id"]: q["target"] for q in plan["subquestions"]}
    allowed = {s["id"] for s in sources}
    reviews = report["reviews"]
    ids = [r["subquestion_id"] for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != set(targets):
        raise ProviderError("生成前证据检查必须覆盖且只覆盖全部子问题")
    pending, gaps, blocked = [], [], []
    seen = {query.strip().lower() for query in executed_queries}
    for review in reviews:
        qid, status = review["subquestion_id"], review["status"]
        evidence = review["evidence_ids"]
        if len(evidence) != len(set(evidence)) or not set(evidence) <= allowed:
            raise ProviderError("生成前证据检查引用了未知或重复证据")
        if status in {"ready", "partial"} and not evidence:
            raise ProviderError("生成前证据检查未提供可回答部分的原文依据")
        if status != "ready":
            gaps.append(f"生成前检查：子问题 {qid}（{targets[qid]}）尚不完整：{review['reason']}")
            if status in {"missing", "uncertain"}:
                blocked.append(qid)
            for query in review["followup_queries"]:
                normalized = query.strip().lower()
                if normalized not in seen and len(pending) < 3:
                    pending.append((qid, query.strip()))
                    seen.add(normalized)
    return {"reviews": reviews, "gaps": gaps, "blocked_ids": blocked, "pending_queries": pending}
