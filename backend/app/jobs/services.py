import hashlib
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from app.agents.governance import digest, freeze_execution
from app.agents.memory import freeze_memory_context
from app.core.access import check_snapshot_access, corpus_role, require_role
from app.core.errors import DomainError, InterruptedRun
from app.core.models import (
    AnswerApprovalRecord,
    AnswerClaim,
    AnswerRevision,
    ClaimEvidence,
    Corpus,
    EvidenceSpan,
    IndexRevision,
    Paper,
    PaperVersion,
    ParseRevision,
    QuestionRun,
    RunEvent,
    Snapshot,
    now,
    uid,
)
from app.jobs.capacity import admit_job, lock_workload, running_jobs
from app.knowledge.evidence import evidence_data

TERMINAL = {"completed", "failed", "cancelled", "review_pending", "rejected"}


def owned(db, model, object_id: str, owner_id: str):
    obj = db.get(model, object_id)
    if not obj:
        raise DomainError("对象不存在或不可访问", 404)
    if model is Corpus:
        require_role(db, obj.id, owner_id)
    elif model is Snapshot:
        check_snapshot_access(db, obj, owner_id)
    elif model is QuestionRun:
        if obj.owner_id != owner_id:
            raise DomainError("对象不存在或不可访问", 404)
        check_snapshot_access(db, db.get(Snapshot, obj.snapshot_id), owner_id)
    else:
        if obj.owner_id != owner_id:
            raise DomainError("对象不存在或不可访问", 404)
    return obj


def owned_version(db, version_id: str, owner_id: str):
    version = db.get(PaperVersion, version_id)
    paper = db.get(Paper, version.paper_id) if version else None
    if not paper:
        raise DomainError("文档版本不存在或不可访问", 404)
    require_role(db, paper.corpus_id, owner_id)
    return version, paper


def version_data(version, paper):
    return {"id": version.id, "paper_id": paper.id, "title": paper.title, "filename": version.filename,
            "sha256": version.sha256, "status": version.status, "error": version.error,
            "page_count": version.page_count, "warnings": version.warnings,
            "created_at": version.created_at.isoformat(), "byte_size": version.byte_size,
            "folder": paper.folder, "tags": paper.tags, "is_active": version.is_active,
            "format": version.filename.rsplit(".", 1)[-1].lower(),
            "file_url": f"/api/paper-versions/{version.id}/file"}


def create_snapshot(db, corpus_id, version_ids, owner_id):
    owned(db, Corpus, corpus_id, owner_id)
    manifest = []
    for version_id in dict.fromkeys(version_ids):
        version, paper = owned_version(db, version_id, owner_id)
        if (paper.corpus_id != corpus_id or paper.deleted or
                version.status != "ready" or not version.is_active):
            raise DomainError("只能选择本知识库中已就绪且启用的文档版本", 409)
        row = db.execute(select(IndexRevision, ParseRevision).join(
            ParseRevision, IndexRevision.parse_revision_id == ParseRevision.id).where(
            ParseRevision.paper_version_id == version_id, IndexRevision.status == "ready",
            ParseRevision.status == "ready").order_by(IndexRevision.created_at.desc()).limit(1)).first()
        if not row:
            raise DomainError("论文索引尚未就绪", 409)
        index, parse = row
        if db.bind.dialect.name == "postgresql" and (
                index.vector_backend != "milvus" or not index.vector_collection):
            raise DomainError("文档索引尚未迁入 Milvus，请先运行 migrate-vectors", 409)
        manifest.append({"paper_id": paper.id, "title": paper.title, "paper_version_id": version.id,
                         "sha256": version.sha256, "parse_revision_id": parse.id,
                         "index_revision_id": index.id, "embedding_model": index.embedding_model,
                         "embedding_dimensions": index.embedding_dimensions,
                         "vector_backend": index.vector_backend,
                         "vector_collection": index.vector_collection})
    snapshot = Snapshot(id=uid(), corpus_id=corpus_id, owner_id=owner_id, manifest=manifest)
    db.add(snapshot)
    db.commit()
    return snapshot


def emit(db, run_id: str, attempt: int, kind: str, payload: dict):
    # Locking the run serializes the per-run sequence across API and worker processes.
    db.scalar(select(QuestionRun.id).where(QuestionRun.id == run_id).with_for_update())
    seq = (db.scalar(select(func.max(RunEvent.seq)).where(RunEvent.run_id == run_id)) or 0) + 1
    event = RunEvent(run_id=run_id, seq=seq, attempt=attempt, kind=kind, payload=payload)
    db.add(event)
    db.flush()
    return event


def create_run(db, request, key: str, owner_id: str, settings, replay_of=None):
    snapshot = owned(db, Snapshot, request.snapshot_id, owner_id)
    check_snapshot_access(db, snapshot, owner_id, current=True)
    lock_workload(db)
    existing = db.scalar(select(QuestionRun).where(
        QuestionRun.owner_id == owner_id, QuestionRun.idempotency_key == key))
    def validate_existing(existing):
        same = (existing.snapshot_id == request.snapshot_id and existing.question == request.question and
                existing.parent_run_id == request.parent_run_id and existing.mode == request.mode and
                existing.config.get("replay_of") == replay_of)
        if not same:
            raise DomainError("幂等键已用于其他请求", 409)
        return existing

    if existing:
        return validate_existing(existing)
    admit_job(db, settings, owner_id)
    if request.parent_run_id:
        parent = owned(db, QuestionRun, request.parent_run_id, owner_id)
        if parent.snapshot_id != snapshot.id:
            raise DomainError("追问默认沿用原快照；更新材料请开始新问题", 409)
        if parent.status not in TERMINAL or parent.status in {"review_pending", "rejected"}:
            raise DomainError("请等待上一问结束", 409)
    for member in snapshot.manifest:
        owned_version(db, member["paper_version_id"], owner_id)
    context, parent_id = [], request.parent_run_id
    for _ in range(4):
        if not parent_id:
            break
        parent = owned(db, QuestionRun, parent_id, owner_id)
        if parent.snapshot_id != snapshot.id:
            raise DomainError("追问上下文不属于当前快照", 403)
        context.append(parent.question)
        parent_id = parent.parent_run_id
    run = QuestionRun(id=uid(), owner_id=owner_id, snapshot_id=snapshot.id,
                      question=request.question, parent_run_id=request.parent_run_id, mode=request.mode,
                      idempotency_key=key, status="queued",
                      config=freeze_execution(settings, snapshot, request.question, context[::-1],
                          freeze_memory_context(db, owner_id, snapshot.corpus_id, request.question,
                                                digest(snapshot.manifest)), replay_of=replay_of))
    try:
        db.add(run)
        db.flush()
        emit(db, run.id, 0, "queued", {"message": "问题已进入持久任务队列"})
        db.commit()
    except IntegrityError:
        db.rollback()
        # Only a committed winner with this key can resolve an idempotency race.
        existing = db.scalar(select(QuestionRun).where(
            QuestionRun.owner_id == owner_id, QuestionRun.idempotency_key == key))
        if not existing:
            raise
        snapshot = owned(db, Snapshot, request.snapshot_id, owner_id)
        check_snapshot_access(db, snapshot, owner_id, current=True)
        return validate_existing(existing)
    return run


def check_attempt(db, run_id: str, attempt: int, lock=False):
    stmt = select(QuestionRun).where(QuestionRun.id == run_id).execution_options(populate_existing=True)
    if lock:
        stmt = stmt.with_for_update()
    run = db.scalar(stmt)
    if not run or run.attempt != attempt or run.status != "running" or run.cancel_requested:
        raise InterruptedRun()
    check_snapshot_access(db, db.get(Snapshot, run.snapshot_id), run.owner_id, current=True)
    return run


def claim_job(session_factory, settings):
    with session_factory.begin() as db:
        lock_workload(db)
        # Expired attempts are bounded too; heartbeat must not turn crashes into infinite billing.
        for model, statuses in ((QuestionRun, ["running"]), (PaperVersion, ["parsing", "indexing"])):
            for exhausted in db.scalars(select(model).where(
                    model.status.in_(statuses), model.lease_until < now(),
                    model.attempt >= settings.max_job_attempts).with_for_update()):
                exhausted.status, exhausted.error, exhausted.lease_until = (
                    "failed", "租约恢复次数已达上限，请检查后台进程后重新发起", None)
                if model is QuestionRun:
                    emit(db, exhausted.id, exhausted.attempt, "failed", {
                        "message": exhausted.error, "code": "recovery_limit"})
        db.flush()
        if running_jobs(db) >= settings.max_running_jobs:
            return None
        run = db.scalar(select(QuestionRun).where(
            QuestionRun.cancel_requested.is_(False),
            ((QuestionRun.status == "queued") | ((QuestionRun.status == "running") &
              (QuestionRun.lease_until < now()))),
        ).order_by(QuestionRun.created_at).with_for_update(skip_locked=True).limit(1))
        if run:
            run.status = "running"
            run.attempt += 1
            run.lease_until = now() + timedelta(seconds=settings.worker_lease_seconds)
            run.state = {}
            emit(db, run.id, run.attempt, "started", {
                "message": "开始处理" if run.attempt == 1 else "租约过期，新 attempt 从头执行；保留旧记录",
                "attempt": run.attempt,
            })
            return "run", run.id, run.attempt
        version = db.scalar(select(PaperVersion).where(
            (PaperVersion.status == "uploaded") | (PaperVersion.status.in_(["parsing", "indexing"]) &
                                                   (PaperVersion.lease_until < now())),
        ).join(Paper).where(Paper.deleted.is_(False)).order_by(
            PaperVersion.created_at).with_for_update(of=PaperVersion, skip_locked=True).limit(1))
        if version:
            version.status = "parsing"
            version.attempt += 1
            version.lease_until = now() + timedelta(seconds=settings.worker_lease_seconds)
            return "ingest", version.id, version.attempt
    return None


def heartbeat(session_factory, settings, job):
    kind, object_id, attempt = job
    model = QuestionRun if kind == "run" else PaperVersion
    active = ((QuestionRun.status == "running", QuestionRun.cancel_requested.is_(False)) if kind == "run" else
              (PaperVersion.status.in_(["parsing", "indexing"]),))
    with session_factory.begin() as db:
        db.execute(update(model).where(model.id == object_id, model.attempt == attempt).values(
            lease_until=now() + timedelta(seconds=settings.worker_lease_seconds)).where(*active))


def answer_data(db, answer: AnswerRevision, owner_id: str):
    run = db.get(QuestionRun, answer.run_id)
    snapshot = db.get(Snapshot, run.snapshot_id) if run else None
    if not run or not snapshot:
        raise DomainError("答案不存在或不可访问", 404)
    check_snapshot_access(db, snapshot, owner_id)
    claims = []
    evidence = {}
    for claim in db.scalars(select(AnswerClaim).where(AnswerClaim.answer_id == answer.id).order_by(AnswerClaim.position)):
        links = []
        for link in db.scalars(select(ClaimEvidence).where(ClaimEvidence.claim_id == claim.id)):
            span = db.get(EvidenceSpan, link.evidence_id)
            evidence[span.id] = evidence_data(db, span, owner_id)
            links.append({"evidence_id": span.id, "relation": link.relation})
        claims.append({"id": claim.id, "text": claim.text, "conditions": claim.conditions,
                       "semantic_verified": claim.semantic_verified, "evidence": links})
    return {"id": answer.id, "run_id": answer.run_id, "revision": answer.revision,
            "evidence_status": answer.evidence_status, "claims": claims, "evidence": list(evidence.values()),
            "gaps": answer.gaps, "issues": answer.issues, "stop_reason": answer.stop_reason,
            "clarification": answer.clarification, "usage": answer.usage}


def published_answer_data(db, answer: AnswerRevision, viewer_id: str):
    run = db.get(QuestionRun, answer.run_id)
    if not run or run.status != "completed":
        raise DomainError("答案尚未通过人工审核", 404, "answer_not_published")
    data = answer_data(db, answer, viewer_id)
    if run.config.get("answer_review_required"):
        approval = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run.id))
        if (not approval or approval.answer_id != answer.id or approval.decision != "accept" or
                (approval.answer_sha256 != answer_content_hash(data) and approval.answer_sha256 != digest(data))):
            raise DomainError("答案版本与审核记录不匹配，请重新审核", 409, "answer_approval_mismatch")
    return data


def answer_content_hash(data):
    """Hash displayed content with stable ordering for evidence sets.

    Claim order remains meaningful. Link and source order can vary with the SQL
    query plan, so canonicalize those sets without changing the displayed payload.
    """
    return digest({**data, "claims": [{**claim, "evidence": sorted(
        claim["evidence"], key=lambda link: (link["evidence_id"], link["relation"]))} for claim in data["claims"]],
        "evidence": sorted(data["evidence"], key=lambda source: source["id"])})


def run_data(db, run, viewer_id: str | None = None):
    answer = db.scalar(select(AnswerRevision).where(AnswerRevision.run_id == run.id).order_by(
        AnswerRevision.revision.desc()).limit(1))
    snapshot = db.get(Snapshot, run.snapshot_id)
    viewer_id = viewer_id or run.owner_id
    role = corpus_role(db, snapshot.corpus_id, viewer_id)
    can_review = role in {"admin", "editor"}
    visible_answer = answer if (run.status == "completed" or
                                (run.status == "review_pending" and can_review)) else None
    review = (db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run.id))
              if run.status == "rejected" else None)
    visible_state = run.status == "completed" or (run.status == "review_pending" and can_review)
    answer_payload = None
    if visible_answer:
        answer_payload = (published_answer_data(db, visible_answer, viewer_id)
                          if run.status == "completed" else answer_data(db, visible_answer, viewer_id))
    return {"id": run.id, "request_id": run.id, "question": run.question, "snapshot_id": run.snapshot_id,
            "is_owner": viewer_id == run.owner_id,
            "corpus_id": snapshot.corpus_id, "parent_run_id": run.parent_run_id,
            "status": run.status, "mode": run.mode, "attempt": run.attempt, "error": run.error,
            "config": run.config, "state": run.state if visible_state else {},
            "created_at": run.created_at.isoformat(),
            "answer": answer_payload,
            "answer_sha256": answer_content_hash(answer_payload) if answer_payload is not None else None,
            "answer_review": ({"decision": review.decision, "reason": review.reason}
                              if review and viewer_id == run.owner_id else None)}


def source_hash_key(content: bytes, suffix=".pdf") -> tuple[str, str]:
    sha = hashlib.sha256(content).hexdigest()
    return sha, f"{sha}{suffix}"
