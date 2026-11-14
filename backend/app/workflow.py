import time
from datetime import timezone
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select

from .access import lock_snapshot_publication
from .errors import BudgetExhausted, DomainError, InterruptedRun, ProviderError
from .evidence import evidence_data, validate_evidence, validate_range
from .governance import (
    code_identity,
    digest,
    environment_identity,
    estimate_tokens,
    provider_identity,
    usage_summary,
)
from .grounding import coverage_issues, numeric_issues, uncovered_questions
from .ingestion import index_config
from .memory import active_frozen_context
from .model_routing import role_model, role_rates
from .models import (
    AnswerClaim,
    AnswerRevision,
    Chunk,
    ClaimEvidence,
    EvidenceSpan,
    IndexRevision,
    ModelCall,
    QuestionRun,
    RetrievalRecord,
    RunEvent,
    Snapshot,
    now,
    uid,
)
from .prompts import ANALYST, EVIDENCE_GATE, PLANNER, RERANKER, VERIFIER
from .quality_gate import validate_readiness
from .query_rewrite import rewrite_queries
from .schemas import Analysis, EvidenceGate, Plan, Reranking, Verification
from .security import redact
from .services import check_attempt, emit
from .tools import ScopedTools
from .vector_store import production_store


class AgentState(TypedDict, total=False):
    run_id: str
    snapshot_id: str
    question: str
    mode: str
    plan: dict
    sources: list
    seen_chunks: list
    executed_queries: list
    pending_queries: list
    draft: dict
    approved_claims: list
    issues: list
    approved_ids: list
    verification_reviews: list
    coverage: list
    round: int
    revisions: int
    calls: int
    embedding_calls: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
    no_improvement: bool
    stop_reason: str
    next_node: str
    budget_category: str
    tool_history: list
    evidence_readiness: dict
    gate_next: str


class Workflow:
    def __init__(self, session_factory, settings, provider, run_id, attempt):
        self.sessions, self.settings, self.provider = session_factory, settings, provider
        self.run_id, self.attempt = run_id, attempt
        self.vector_store = None
        self.started = time.monotonic()
        with self.sessions() as db:
            run = check_attempt(db, run_id, attempt)
            self.limits = run.config
            self.owner_id = run.owner_id
            self.parent_id = run.parent_run_id
            if self.limits.get("model") != settings.model:
                raise ProviderError("运行固定的聊天模型与当前配置不同，请重新创建问题")
            if "manifest_sha256" in self.limits:
                frozen = {k: v for k, v in self.limits.items() if k != "manifest_sha256"}
                if digest(frozen) != self.limits["manifest_sha256"]:
                    raise ProviderError("执行清单校验失败，未调用外部服务")
                if digest(run.question) != self.limits["question_sha256"]:
                    raise ProviderError("问题内容与执行清单不一致，未调用外部服务")
                if (self.limits["provider_identity"] != provider_identity(settings) or
                        self.limits["application_files"] != code_identity()):
                    raise ProviderError("模型服务或应用代码已改变，请重新创建问题")
                packages, locks = environment_identity()
                if (packages != self.limits["runtime_packages"] or locks != self.limits["lock_files"]):
                    raise ProviderError("运行依赖或锁文件已改变，请重新创建问题")
            first = db.scalar(select(RunEvent).where(RunEvent.run_id == run_id,
                              RunEvent.kind == "started").order_by(RunEvent.created_at).limit(1))
            if first:
                # Recovery does not reset the run's wall-clock allowance.
                self.started -= max(0., (now() - first.created_at.replace(tzinfo=timezone.utc)).total_seconds())
            if hasattr(provider, "settings"):
                provider.settings = settings.model_copy(update={
                    k: self.limits[k] for k in type(settings).model_fields if k in self.limits})
            if hasattr(provider, "deadline"):
                provider.deadline = min(provider.deadline or float("inf"),
                                        self.started + self.limits["max_run_seconds"])

    def guard(self):
        with self.sessions() as db:
            check_attempt(db, self.run_id, self.attempt)

    def save(self, state, kind, payload):
        state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        with self.sessions.begin() as db:
            run = check_attempt(db, self.run_id, self.attempt, lock=True)
            run.state = dict(state)
            emit(db, run.id, self.attempt, kind, payload)
        return state

    def budget_available(self, state, reserve=1):
        with self.sessions() as db:
            usage = usage_summary(db, self.run_id, self.limits)
        category = ("time" if time.monotonic() - self.started >= self.limits["max_run_seconds"] else
                    "model_calls" if usage["model_calls"] + reserve > self.limits["max_model_calls"] else
                    "tokens" if usage["reserved_or_actual_tokens"] >=
                    self.limits.get("max_run_tokens", self.settings.max_run_tokens) else None)
        if category:
            state["budget_category"] = category
        return category is None

    def begin_call(self, state, role, request, estimated):
        with self.sessions.begin() as db:
            run = check_attempt(db, self.run_id, self.attempt, lock=True)
            usage = usage_summary(db, self.run_id, self.limits)
            if time.monotonic() - self.started >= self.limits["max_run_seconds"]:
                raise BudgetExhausted("time")
            if role != "Embedding" and usage["model_calls"] >= self.limits["max_model_calls"]:
                raise BudgetExhausted("model_calls")
            cap = self.limits.get("max_embedding_batch_tokens" if role == "Embedding" else
                                  "max_request_input_tokens", 100000)
            if estimated > cap:
                raise BudgetExhausted("request_input_tokens")
            remaining = self.limits.get("max_run_tokens", self.settings.max_run_tokens) - usage["reserved_or_actual_tokens"]
            output = min(self.limits.get("max_output_tokens", 6000), remaining - estimated) if role != "Embedding" else 0
            if remaining < estimated or (role != "Embedding" and output < 1):
                raise BudgetExhausted("tokens")
            if self.limits.get("max_run_cost") is not None:
                rates = role_rates(self.limits, role)
                if None in rates or usage["reserved_or_actual_cost"] is None:
                    raise ProviderError("费用预算需要配置当前模型单价、币种和价格来源；未发送请求")
                remaining_cost = self.limits["max_run_cost"] - usage["reserved_or_actual_cost"]
                input_cost = estimated * rates[0] / 1000000
                if remaining_cost < input_cost:
                    raise BudgetExhausted("cost")
                if rates[1]:
                    output = min(output, int((remaining_cost-input_cost)*1000000/rates[1]))
                if role != "Embedding" and output < 1:
                    raise BudgetExhausted("cost")
            request = redact({**request, "output_limit": output, "estimator": "utf8-bytes-plus-256-v1"},
                             self.settings)
            call = ModelCall(id=uid(), run_id=self.run_id, attempt=self.attempt, role=role,
                             model=self.limits["embedding_model"] if role == "Embedding" else role_model(self.limits, role),
                             request=request, response={}, status="started", elapsed_seconds=0., usage_known=False,
                             estimated_input_tokens=estimated, reserved_tokens=estimated + output)
            db.add(call)
            state["embedding_calls" if role == "Embedding" else "calls"] = usage[
                "embedding_calls" if role == "Embedding" else "model_calls"] + 1
            run.state = dict(state)
            emit(db, run.id, self.attempt, "model_started", {"role": role, "call_id": call.id,
                 "estimated_input_tokens": estimated, "reserved_tokens": estimated + output})
        return call.id, output

    def reserve_retry(self, state, role, request, estimated, previous_call_id, elapsed):
        # Keep the exact reservation made before the previous HTTP attempt.
        # A smaller retry cap must never reduce that attempt's unknown usage.
        self.finish_call(previous_call_id, {"reason": "http_retry_previous_usage_unknown"},
                         {}, elapsed, "usage_unknown")
        return self.begin_call(state, role, {**request, "retry_of_call_id": previous_call_id}, estimated)

    def finish_call(self, call_id, response, usage, elapsed, status):
        with self.sessions.begin() as db:
            run = db.scalar(select(QuestionRun).where(QuestionRun.id == self.run_id).with_for_update())
            call = db.get(ModelCall, call_id)
            call.response, call.status, call.elapsed_seconds = redact(response, self.settings), status, round(elapsed, 3)
            call.usage_known = usage.get("usage_known", all(type(usage.get(k)) is int for k in
                                                         ("input_tokens", "output_tokens")))
            call.input_tokens, call.output_tokens = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            call.provider_request_id = usage.get("provider_request_id")
            if run.attempt == self.attempt and run.status == "running":
                emit(db, run.id, self.attempt, "model_finished", {"call_id": call.id, "role": call.role,
                     "status": status, "usage_known": call.usage_known})

    def embed(self, texts, expected_dim=0):
        self.guard()
        state = self.embedding_state
        call_id, _ = self.begin_call(state, "Embedding", {"input": texts, "expected_dim": expected_dim},
                                     estimate_tokens(texts))
        start = time.monotonic()
        previous_retry = getattr(self.provider, "before_retry", None)
        if previous_retry is not None:
            def before_retry():
                nonlocal call_id, start
                call_id, output = self.reserve_retry(state, "Embedding", {"input": texts},
                    estimate_tokens(texts), call_id, time.monotonic()-start)
                start = time.monotonic()
                return output
            self.provider.before_retry = before_retry
        try:
            result = self.provider.embed(texts, expected_dim)
        except Exception as exc:
            self.finish_call(call_id, {"error_type": type(exc).__name__,
                                       "code": getattr(exc, "code", type(exc).__name__),
                                       "message": redact(str(exc), self.settings)},
                             getattr(self.provider, "last_embedding_usage", {}), time.monotonic()-start, "failed")
            raise
        finally:
            if previous_retry is not None:
                self.provider.before_retry = previous_retry
        self.finish_call(call_id, {"count": len(result), "dimensions": len(result[0])},
                         getattr(self.provider, "last_embedding_usage", {}), time.monotonic()-start, "completed")
        self.guard()
        return result

    def call(self, state, role, prompt, data, schema):
        self.guard()
        if not self.budget_available(state):
            raise BudgetExhausted(state["budget_category"])
        prompt_record = self.limits.get("prompts", {}).get(role)
        if prompt_record:
            if prompt_record["schema"] != schema.model_json_schema():
                raise ProviderError("模型 Schema 与执行清单不一致")
            prompt = prompt_record["text"]
        request = {"data": data, "prompt": prompt, "schema": schema.model_json_schema()}
        # Includes the adapter instruction's framing, role and Schema.
        estimated = estimate_tokens(request) + 1024
        call_id, output_limit = self.begin_call(state, role, request, estimated)
        if hasattr(self.provider, "request_output_limit"):
            self.provider.request_output_limit = output_limit
        start = time.monotonic()
        previous_retry = getattr(self.provider, "before_retry", None)
        if previous_retry is not None:
            def before_retry():
                nonlocal call_id, start
                call_id, output = self.reserve_retry(state, role, request, estimated,
                                                    call_id, time.monotonic()-start)
                start = time.monotonic()
                return output
            self.provider.before_retry = before_retry
        try:
            result, usage = self.provider.structured(role, prompt, data, schema)
        except Exception as exc:
            self.finish_call(call_id, {"error_type": type(exc).__name__}, getattr(self.provider, "last_usage", {}),
                             time.monotonic()-start, "failed")
            raise
        finally:
            if previous_retry is not None:
                self.provider.before_retry = previous_retry
        self.finish_call(call_id, result.model_dump(), usage, time.monotonic()-start, "completed")
        self.guard()
        state["input_tokens"] += usage["input_tokens"]
        state["output_tokens"] += usage["output_tokens"]
        return result

    def planner(self, state):
        context = []
        with self.sessions() as db:
            snapshot = db.get(Snapshot, state["snapshot_id"])
            parent_id = None if "previous_questions" in self.limits else self.parent_id
            # Only previous questions are needed for reference resolution; no history becomes evidence.
            for _ in range(4):
                if not parent_id:
                    break
                parent = db.get(QuestionRun, parent_id)
                if not parent or parent.owner_id != self.owner_id or parent.snapshot_id != snapshot.id:
                    raise DomainError("追问上下文不属于当前快照", 403)
                context.append(parent.question)
                parent_id = parent.parent_run_id
            manifest = snapshot.manifest
            if "snapshot_sha256" in self.limits and digest(manifest) != self.limits["snapshot_sha256"]:
                raise ProviderError("材料快照与执行清单不一致")
            memory_context = active_frozen_context(db, self.owner_id, self.limits.get("memory_context", []))
        if state["mode"] == "baseline_rag":
            state["plan"] = Plan(normalized_question=state["question"], subquestions=[
                {"id": "q1", "target": state["question"], "queries": [state["question"][:500]]},
            ]).model_dump()
        else:
            plan = self.call(state, "Planner", PLANNER,
                             {"question": state["question"], "previous_questions": self.limits.get("previous_questions", context[::-1]),
                              "memory_hints": memory_context,
                              "papers": [{"title": m["title"], "paper_version_id": m["paper_version_id"]}
                                         for m in manifest]}, Plan)
            if not plan.needs_clarification and not plan.subquestions:
                raise ProviderError("Planner 未提供检索计划")
            if plan.needs_clarification and not plan.clarification:
                raise ProviderError("Planner 未提供澄清问题")
            if len({q.id for q in plan.subquestions}) != len(plan.subquestions):
                raise ProviderError("Planner 子问题 ID 重复")
            state["plan"] = plan.model_dump()
        if self.limits.get("query_rewrite_enabled") and state["plan"]["subquestions"]:
            subs, rewrites = rewrite_queries(state["question"], state["plan"]["subquestions"], snapshot.corpus_id,
                self.limits.get("query_term_rules", []), self.limits.get("query_max_per_subquestion", 5))
            state["plan"]["subquestions"] = subs
            self.save(state, "query_rewritten", {"records": rewrites, "original_question": state["question"]})
        return self.save(state, "planner", {"message": "问题规划已完成", "plan": state["plan"]})

    def retrieve(self, state):
        self.guard()
        if not self.budget_available(state, reserve=0) or state["round"] >= self.limits["max_retrieval_rounds"]:
            raise BudgetExhausted(state.get("budget_category", "retrieval_rounds"))
        state["round"] += 1
        before = {item["id"] for item in state["sources"]}
        pending = state.get("pending_queries") or [
            (sub["id"], query) for sub in state["plan"]["subquestions"] for query in sub["queries"]
        ]
        if not state.get("pending_queries") and self.limits.get("query_rewrite_enabled"):
            subs = state["plan"]["subquestions"]
            pending = [(sub["id"], sub["queries"][i])
                       for i in range(max((len(s["queries"]) for s in subs), default=0))
                       for sub in subs if i < len(sub["queries"])]
        queries, candidates = set(state["executed_queries"]), {}
        queries_total = len({query.strip().lower() for _, query in pending[:15]} - queries)
        queries_done = 0
        for sub_id, query in pending[:15]:
            self.guard()
            if time.monotonic() - self.started >= self.limits["max_run_seconds"]:
                raise BudgetExhausted("time")
            normalized = query.strip().lower()
            if normalized in queries:
                continue
            queries.add(normalized)
            with self.sessions.begin() as db:
                check_attempt(db, self.run_id, self.attempt)
                snapshot = db.get(Snapshot, state["snapshot_id"])
                self.embedding_state = state
                if self.vector_store is None:
                    self.vector_store = production_store(self.settings, db)
                tools = ScopedTools(db, snapshot, self.run_id, self, self.settings.embedding_model,
                                    index_config(self.settings), self.vector_store)
                ranked = tools.search_evidence(query)
                if ranked.degradation:
                    emit(db, self.run_id, self.attempt, "retrieval_degraded", {
                        "subquestion_id": sub_id, "round": state["round"], **ranked.degradation})
                for candidate in ranked:
                    chunk = db.get(Chunk, candidate["chunk_id"])
                    member = next((m for m in snapshot.manifest
                                   if m["index_revision_id"] == chunk.index_revision_id), None)
                    index = db.get(IndexRevision, chunk.index_revision_id)
                    if not member or not index or index.parse_revision_id != member["parse_revision_id"]:
                        raise DomainError("候选片段索引与快照的解析版本不一致", 409)
                    entry = candidates.setdefault(chunk.id, {"chunk_id": chunk.id, "score": 0.,
                        "index_id": chunk.index_revision_id, "retrieved_for_subquestion_ids": [], "blocks": []})
                    entry["score"] += candidate["score"]
                    if sub_id not in entry["retrieved_for_subquestion_ids"]:
                        entry["retrieved_for_subquestion_ids"].append(sub_id)
                    if not entry["blocks"]:
                        for source_range in chunk.source_ranges:
                            block = tools.read_source_blocks([source_range["block_id"]])[0]
                            if block.parse_revision_id != index.parse_revision_id:
                                raise DomainError("候选片段来源不属于其解析版本", 409)
                            validate_range(block.text, source_range["char_start"], source_range["char_end"])
                            entry["blocks"].append({**source_range,
                                "text": block.text[source_range["char_start"]:source_range["char_end"]]})
                    candidate.update(title=member["title"], paper_version_id=member["paper_version_id"])
                db.add(RetrievalRecord(run_id=self.run_id, attempt=self.attempt, round=state["round"],
                                       subquestion_id=sub_id, query=query, rankings=ranked))
                queries_done += 1
                check_attempt(db, self.run_id, self.attempt, lock=True)
                emit(db, self.run_id, self.attempt, "retrieval_progress", {
                    "round": state["round"], "queries_done": queries_done, "queries_total": queries_total,
                    "candidate_count": len(candidates)})
        # Balance candidate documents before semantic reranking; never discard a chunk just
        # because an earlier round saw it. New gap queries may make its result blocks useful.
        grouped = {}
        for item in sorted(candidates.values(), key=lambda x: -x["score"]):
            grouped.setdefault(item["index_id"], []).append(item)
        pool = [items[i] for i in range(max((len(v) for v in grouped.values()), default=0))
                for items in grouped.values() if i < len(items)][:24]
        if pool and state["mode"] != "baseline_rag":
            result = self.call(state, "Reranker", RERANKER, {
                "question": state["question"], "plan": state["plan"],
                "candidates": [{"chunk_id": c["chunk_id"],
                                "retrieved_for_subquestion_ids": c["retrieved_for_subquestion_ids"],
                                "blocks": c["blocks"]} for c in pool],
            }, Reranking)
            by_id = {c["chunk_id"]: c for c in pool}
            selected, seen_ids = [], set()
            for item in sorted(result.items, key=lambda x: -x.score):
                if item.chunk_id not in by_id or item.chunk_id in seen_ids:
                    raise ProviderError("Reranker 返回未知或重复片段")
                seen_ids.add(item.chunk_id)
                candidate = by_id[item.chunk_id]
                valid_ids = {b["block_id"] for b in candidate["blocks"]}
                if not set(item.relevant_block_ids) <= valid_ids:
                    raise ProviderError("Reranker 返回不属于候选片段的来源块")
                if item.score > 0 and item.relevant_block_ids:
                    selected.append({**candidate, "blocks": [b for b in candidate["blocks"]
                                     if b["block_id"] in item.relevant_block_ids]})
        else:
            selected = pool
        # Take relevant spans first. Titles/background no longer fill the per-document cap
        # before the actual result paragraphs in a lower-ranked chunk are considered.
        sources, counts, size = {}, {}, 0
        with self.sessions.begin() as db:
            snapshot = db.get(Snapshot, state["snapshot_id"])
            tools = ScopedTools(db, snapshot, self.run_id, self.provider, self.settings.embedding_model,
                                vector_store=self.vector_store)
            infos = []
            for candidate in selected:
                for block in candidate["blocks"]:
                    span = tools.get_evidence_span(block["block_id"], block["char_start"], block["char_end"])
                    infos.append({**evidence_data(db, span, self.owner_id),
                                  "retrieved_for_subquestion_ids": candidate["retrieved_for_subquestion_ids"]})
            for info in infos + state["sources"]:
                key = info["paper_version_id"]
                if info["id"] in sources:
                    existing = sources[info["id"]]
                    existing["retrieved_for_subquestion_ids"] = list(dict.fromkeys(
                        existing.get("retrieved_for_subquestion_ids", []) +
                        info.get("retrieved_for_subquestion_ids", [])))
                    continue
                if counts.get(key, 0) >= 16 or len(sources) >= 32:
                    continue
                if size + len(info["quote"]) > self.limits["max_context_chars"]:
                    continue
                sources[info["id"]] = info
                counts[key] = counts.get(key, 0) + 1
                size += len(info["quote"])
        state.update(sources=list(sources.values()), seen_chunks=sorted(candidates), executed_queries=sorted(queries),
                     pending_queries=[], no_improvement=not (set(sources) - before))
        return self.save(state, "retriever", {"message": "混合检索、精排与原文定位完成", "round": state["round"],
                                               "evidence_count": len(sources), "new_evidence": len(set(sources)-before)})

    def deterministic_issues(self, state):
        issues = coverage_issues(state.get("plan", {}), state["draft"])
        blocked = set(state.get("evidence_readiness", {}).get("blocked_ids", []))
        for claim in state["draft"]["claims"]:
            if blocked.intersection(claim.get("subquestion_ids", [])):
                issues.append({"claim_id": claim["id"], "type": "missing_evidence",
                               "message": "该子问题未通过生成前证据检查，不能发布推测论点"})
        allowed = {item["id"] for item in state["sources"]}
        validated_sources = {}
        ids = set()
        with self.sessions() as db:
            snapshot = db.get(Snapshot, state["snapshot_id"])
            for claim in state["draft"]["claims"]:
                if claim["id"] in ids:
                    raise ProviderError("Analyst 论点 ID 重复")
                ids.add(claim["id"])
                if not any(link["relation"] == "supports" for link in claim["evidence"]):
                    issues.append({"claim_id": claim["id"], "type": "unsupported_claim",
                                   "message": "事实论点缺少支持关系，背景或限定不能替代支持"})
                for link in claim["evidence"]:
                    span = db.get(EvidenceSpan, link["evidence_id"])
                    try:
                        if not span or span.id not in allowed or span.run_id != self.run_id:
                            raise DomainError("证据 ID 未提供或不属于本次运行")
                        validate_evidence(db, snapshot, span)
                        if span.invocation_id and not any(label in claim["text"] for label in ("工具记录", "日志记录")):
                            issues.append({"claim_id": claim["id"], "type": "unsupported_claim",
                                           "message": "外部观察论点必须明确标注工具记录或日志记录，不能冒充正式文档事实"})
                        validated_sources[span.id] = {"id": span.id, "quote": span.quote}
                    except DomainError as exc:
                        issues.append({"claim_id": claim["id"], "type": "invalid_reference", "message": str(exc)})
                issues.extend(numeric_issues(claim, validated_sources))
        bad = {i["claim_id"] for i in issues if i["claim_id"]}
        for gap in uncovered_questions(state.get("plan", {}), state["draft"],
                                       [c for c in state["draft"]["claims"] if c["id"] not in bad]):
            issues.append({"claim_id": None, "type": "missing_evidence",
                           "message": f"子问题 {gap['subquestion_id']}（{gap['target']}）：{gap['reason']}"})
        return issues

    def analyst(self, state):
        with self.sessions() as db:
            memory_context = active_frozen_context(db, self.owner_id, self.limits.get("memory_context", []))
        analysis = self.call(state, "Analyst", ANALYST, {
            "question": state["question"], "plan": state["plan"], "sources": state["sources"],
            "previous_draft": state.get("draft") if state["issues"] else None,
            "verification_feedback": state["issues"],
            "memory_hints": memory_context,
            "evidence_readiness": state.get("evidence_readiness", {}),
        }, Analysis)
        state["draft"] = analysis.model_dump()
        state["issues"] = self.deterministic_issues(state)
        if state["mode"] != "full_agent":
            bad = {i["claim_id"] for i in state["issues"]}
            state["approved_claims"] = [c for c in state["draft"]["claims"] if c["id"] not in bad]
        return self.save(state, "analyst", {"message": "已形成候选论点", "claim_count": len(analysis.claims),
                                             "gaps": analysis.gaps, "reference_issues": state["issues"]})

    def verifier(self, state):
        deterministic = self.deterministic_issues(state)
        verification = self.call(state, "Verifier", VERIFIER, {
            "question": state["question"], "plan": state["plan"], "draft": state["draft"],
            "sources": state["sources"], "executed_queries": state["executed_queries"],
            "reference_issues": deterministic,
        }, Verification)
        claim_ids = {c["id"] for c in state["draft"]["claims"]}
        if not set(verification.approved_claim_ids) <= claim_ids:
            raise ProviderError("Verifier 返回不存在的论点 ID")
        issues = deterministic + [i.model_dump() for i in verification.issues]
        reviews = {r.claim_id: r for r in verification.claim_reviews}
        if len(reviews) != len(verification.claim_reviews) or not set(reviews) <= claim_ids:
            raise ProviderError("Verifier 返回未知或重复的逐论点判定")
        inspectable = set()
        for claim in state["draft"]["claims"]:
            review = reviews.get(claim["id"])
            linked = {e["evidence_id"] for e in claim["evidence"]}
            supporting = {e["evidence_id"] for e in claim["evidence"] if e["relation"] == "supports"}
            if (review and review.verdict == "approved" and set(review.evidence_ids) <= linked and
                    set(review.evidence_ids) & supporting and review.rationale.strip()):
                inspectable.add(claim["id"])
            else:
                issues.append({"claim_id": claim["id"], "type": "unsupported_claim",
                               "message": "核验缺少明确通过、有效支持证据或判定理由；不确定论点暂不发布"})
        bad = {i["claim_id"] for i in issues if i["claim_id"]}
        approved = (set(verification.approved_claim_ids) & inspectable) - bad
        # Omitted claims are quarantined too; an empty issue list is not implicit approval.
        for claim_id in claim_ids - approved - bad:
            issues.append({"claim_id": claim_id, "type": "unsupported_claim", "message": "核验未批准该论点"})
        state["issues"] = issues
        state["approved_ids"] = sorted(approved)
        state["verification_reviews"] = [r.model_dump() for r in verification.claim_reviews]
        state["approved_claims"] = [c for c in state["draft"]["claims"] if c["id"] in approved]
        coverage = uncovered_questions(state["plan"], state["draft"], state["approved_claims"])
        for gap in coverage:
            issue = {"claim_id": None, "type": "missing_evidence",
                     "message": f"子问题 {gap['subquestion_id']}（{gap['target']}）：{gap['reason']}"}
            if issue not in issues:
                issues.append(issue)
        state["pending_queries"] = [("gap", q) for q in verification.followup_queries
                                    if q.strip().lower() not in state["executed_queries"]]
        return self.save(state, "verifier", {"message": "逐论点核验完成", "approved": len(approved),
                         "issues": issues, "claim_reviews": state["verification_reviews"], "uncovered": coverage})

    def gate_enabled(self, state):
        return state["mode"] == "full_agent" and self.limits.get("evidence_gate_enabled", True)

    def evidence_gate(self, state):
        self.guard()
        if not state["sources"]:
            report = {"reviews": [{"subquestion_id": q["id"], "status": "missing", "evidence_ids": [],
                      "reason": "当前检索未提供可用原文", "followup_queries": [q["target"][:500]]}
                      for q in state["plan"]["subquestions"]]}
        else:
            report = self.call(state, "EvidenceGate", EVIDENCE_GATE, {
                "question": state["question"], "plan": state["plan"], "sources": state["sources"],
                "executed_queries": state["executed_queries"],
            }, EvidenceGate).model_dump()
        readiness = validate_readiness(state["plan"], state["sources"], report, state["executed_queries"])
        state["evidence_readiness"] = readiness
        can_retry = (readiness["pending_queries"] and (state["round"] == 1 or not state["no_improvement"]) and
                     state["round"] < self.limits["max_retrieval_rounds"] and self.budget_available(state, reserve=4))
        if can_retry:
            state["pending_queries"] = readiness["pending_queries"]
            state["gate_next"] = "retriever"
        elif len(readiness["blocked_ids"]) == len(state["plan"]["subquestions"]):
            state["gate_next"], state["stop_reason"] = "publish", "evidence_gate_insufficient"
        else:
            state["gate_next"] = "analyst"
        return self.save(state, "evidence_gate", {"round": state["round"],
                         "blocked_count": len(readiness["blocked_ids"]), "next": state["gate_next"],
                         "reviews": readiness["reviews"]})

    def after_verifier(self, state):
        if not self.budget_available(state, reserve=0):
            state["stop_reason"] = "budget_exhausted"
            return "publish"
        gaps = state["draft"]["gaps"] or any(i["type"] == "missing_evidence" for i in state["issues"])
        if (gaps and state["pending_queries"] and not state["no_improvement"] and
                state["round"] < self.limits["max_retrieval_rounds"]):
            if not self.budget_available(state, reserve=4 if self.gate_enabled(state) else 3):
                state["stop_reason"] = "budget_exhausted"
                return "publish"
            return "retriever"
        if (any(i.get("claim_id") for i in state["issues"]) and
                state["revisions"] < self.limits["max_revision_rounds"]):
            if not self.budget_available(state, reserve=2):
                state["stop_reason"] = "budget_exhausted"
                return "publish"
            state["revisions"] += 1
            return "analyst"
        state["stop_reason"] = ("no_new_evidence" if state["no_improvement"] else
                                "bounded_completion" if state["issues"] or gaps else "verified")
        return "publish"

    def route(self, state):
        # Conditional edge functions only select an edge. Persist counters in a node
        # so revisions and stop reasons actually become LangGraph state updates.
        state["next_node"] = self.after_verifier(state)
        return self.save(state, "routing", {"next": state["next_node"],
                                              "stop_reason": state.get("stop_reason")})

    def publish(self, state):
        publication_started = time.monotonic()
        self.start_node("publish", state)
        gaps = list(state.get("draft", {}).get("gaps", []))
        gaps.extend(state.get("evidence_readiness", {}).get("gaps", []))
        gaps.extend(i["message"] for i in state["issues"]
                    if i["type"] == "missing_evidence" and not i.get("claim_id"))
        # Recheck references at the publication boundary, including baseline/ablation runs.
        issues = self.deterministic_issues(state) if state.get("draft") else []
        bad = {i["claim_id"] for i in issues}
        claims = [c for c in state.get("approved_claims", []) if c["id"] not in bad]
        coverage = uncovered_questions(state.get("plan", {}), state.get("draft", {}), claims)
        state["coverage"] = coverage
        for gap in coverage:
            gaps.append(f"子问题 {gap['subquestion_id']}（{gap['target']}）：{gap['reason']}")
        omitted = len(state.get("draft", {}).get("claims", [])) - len(claims)
        if omitted:
            gaps.append(f"{omitted} 条候选论点未通过引用或语义检查，已从答案移除。")
        if state.get("stop_reason") == "budget_exhausted":
            labels = {"time": "运行时间", "tokens": "总 Token 估算额度", "model_calls": "模型调用次数",
                      "cost": "估算费用额度", "request_input_tokens": "单次输入估算上限", "retrieval_rounds": "检索轮数"}
            gaps.append(f"达到{labels.get(state.get('budget_category'), '本次运行预算')}，保留已完成检查的部分；其余问题尚未确定。")
        needs_clarification = state.get("plan", {}).get("needs_clarification", False)
        if needs_clarification:
            status = "needs_clarification"
        elif not claims:
            status = "insufficient"
            if not gaps:
                gaps.append("当前材料及本次检索未找到足以支持回答的证据。")
        elif state.get("draft", {}).get("has_conflict"):
            status = "conflicted"
        else:
            status = "partial" if gaps else "supported"
        usage = {"model_calls": state["calls"], "embedding_calls": state["embedding_calls"],
                 "retrieval_rounds": state["round"], "revision_rounds": state["revisions"],
                 "input_tokens": state["input_tokens"], "output_tokens": state["output_tokens"],
                 "elapsed_seconds": round(time.monotonic()-self.started, 3), "attempt": self.attempt}
        with self.sessions.begin() as db:
            candidate = db.get(QuestionRun, self.run_id)
            snapshot = db.get(Snapshot, candidate.snapshot_id) if candidate else None
            if not snapshot:
                raise InterruptedRun()
            # Keep the same lock order as human approval and member removal.
            # check_attempt refreshes and validates the run after these locks.
            lock_snapshot_publication(db, snapshot, candidate.owner_id)
            run = check_attempt(db, self.run_id, self.attempt, lock=True)
            usage["accounting"] = usage_summary(db, run.id, self.limits)
            usage["budget_category"] = state.get("budget_category")
            existing = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run.id,
                                                             AnswerRevision.revision == 1))
            if not existing:
                answer = AnswerRevision(id=uid(), run_id=run.id, evidence_status=status,
                                        gaps=list(dict.fromkeys(gaps)), issues=state["issues"] + issues,
                                        stop_reason=state.get("stop_reason") or "completed",
                                        usage=usage, clarification=state.get("plan", {}).get("clarification"))
                db.add(answer)
                db.flush()
                for position, draft in enumerate(claims):
                    claim = AnswerClaim(id=uid(), answer_id=answer.id, position=position, text=draft["text"],
                                        conditions=draft["conditions"], semantic_verified=state["mode"] == "full_agent")
                    db.add(claim)
                    db.flush()
                    links = {(link["evidence_id"], link["relation"]) for link in draft["evidence"]}
                    for evidence_id, relation in links:
                        db.add(ClaimEvidence(claim_id=claim.id, evidence_id=evidence_id, relation=relation))
            require_human_review = bool(run.config.get("answer_review_required", False))
            run.status = "review_pending" if require_human_review else "completed"
            run.lease_until = None
            run.state = dict(state, elapsed_seconds=usage["elapsed_seconds"])
            emit(db, run.id, self.attempt, "node_finished", {"node": "publish",
                 "seconds": round(time.monotonic()-publication_started, 3), "error": None})
            emit(db, run.id, self.attempt,
                 "review_pending" if require_human_review else "completed",
                 {"message": "答案等待人工审核" if require_human_review else "答案已保存",
                  "evidence_status": status})
        return state

    def start_node(self, name, state):
        with self.sessions.begin() as db:
            check_attempt(db, self.run_id, self.attempt, lock=True)
            emit(db, self.run_id, self.attempt, "node_started", {
                "node": name, "round": state["round"] + (1 if name == "retriever" else 0),
                "revisions": state["revisions"]})

    def timed_node(self, name, function):
        def execute(state):
            started, error = time.monotonic(), None
            self.start_node(name, state)
            try:
                return function(state)
            except Exception as exc:
                error = type(exc).__name__
                raise
            finally:
                with self.sessions.begin() as db:
                    run = db.scalar(select(QuestionRun).where(QuestionRun.id == self.run_id)
                                    .execution_options(populate_existing=True).with_for_update())
                    if run and run.attempt == self.attempt and run.status == "running" and not run.cancel_requested:
                        emit(db, self.run_id, self.attempt, "node_finished", {
                            "node": name, "seconds": round(time.monotonic()-started, 3), "error": error})
        return execute

    def graph(self, checkpointer=None):
        from .workflow_tools import WorkflowTools
        builder = StateGraph(AgentState)
        builder.add_node("planner", self.timed_node("planner", self.planner))
        builder.add_node("retriever", self.timed_node("retriever", self.retrieve))
        builder.add_node("tools", self.timed_node("tools", WorkflowTools(self).run))
        builder.add_node("evidence_gate", self.timed_node("evidence_gate", self.evidence_gate))
        builder.add_node("analyst", self.timed_node("analyst", self.analyst))
        builder.add_node("verifier", self.timed_node("verifier", self.verifier))
        builder.add_node("route", self.timed_node("route", self.route))
        builder.add_node("publish", self.publish)
        builder.add_edge(START, "planner")
        builder.add_conditional_edges("planner", lambda s: "publish" if s["plan"]["needs_clarification"] else "retriever")
        builder.add_conditional_edges("retriever", lambda s: "tools" if self.limits.get("workflow_mcp_enabled") and
            s["mode"] != "baseline_rag" else "evidence_gate" if self.gate_enabled(s) else "analyst")
        builder.add_conditional_edges("tools", lambda s: "evidence_gate" if self.gate_enabled(s) else "analyst")
        builder.add_conditional_edges("evidence_gate", lambda s: s["gate_next"])
        builder.add_conditional_edges("analyst", lambda s: "verifier" if s["mode"] == "full_agent" else "publish")
        builder.add_edge("verifier", "route")
        builder.add_conditional_edges("route", lambda s: s["next_node"])
        builder.add_edge("publish", END)
        return builder.compile(checkpointer=checkpointer)

    def execute(self, initial, checkpointer=None):
        self.save(initial, "execution_manifest", {"manifest_sha256": self.limits.get("manifest_sha256"),
                  "message": "按固定配置开始执行；未完成调用的资源预留跨 attempt 保留"})
        try:
            self.graph(checkpointer).invoke(initial, {"configurable": {
                "thread_id": f"{self.run_id}:{self.attempt}",
            }, "recursion_limit": 40})
        except BudgetExhausted as exc:
            with self.sessions() as db:
                state = db.get(QuestionRun, self.run_id).state
            state["stop_reason"] = "budget_exhausted"
            state["budget_category"] = exc.category
            self.publish(state)
        finally:
            if self.vector_store is not None:
                self.vector_store.close()
                self.vector_store = None


def initial_state(run):
    return AgentState(run_id=run.id, snapshot_id=run.snapshot_id, question=run.question, mode=run.mode,
                      sources=[], seen_chunks=[], executed_queries=[], pending_queries=[], draft={},
                      approved_claims=[], issues=[], approved_ids=[], round=0, revisions=0, calls=0,
                      embedding_calls=0, input_tokens=0, output_tokens=0, no_improvement=False)
