import asyncio
import hashlib
import json
import mimetypes
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import Depends, FastAPI, File, Form, Header, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import or_, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session as DBSession

from app.agents.governance import digest, usage_summary
from app.agents.progress import progress_event
from app.core.access import check_snapshot_access, corpus_role, lock_snapshot_publication, require_role
from app.core.auth import CurrentUser, current_user
from app.core.config import ROOT, get_settings
from app.core.db import Session, get_db
from app.core.errors import DomainError
from app.core.models import (
    AcceptedAnswerExperience,
    AnswerApprovalRecord,
    AnswerRevision,
    Corpus,
    EvidenceSpan,
    ExternalOperation,
    MCPInvocation,
    Membership,
    ModelCall,
    Paper,
    PaperVersion,
    QuestionRun,
    RetrievalRecord,
    ReviewRecord,
    RunEvent,
    Snapshot,
    uid,
)
from app.core.schemas import (
    AnswerApprovalCreate,
    CorpusCreate,
    DocumentUpdate,
    ReviewCreate,
    RunCreate,
    SnapshotCreate,
)
from app.jobs.capacity import admit_job, lock_workload
from app.jobs.services import (
    TERMINAL,
    answer_content_hash,
    answer_data,
    create_run,
    create_snapshot,
    emit,
    owned,
    owned_version,
    published_answer_data,
    run_data,
    source_hash_key,
    version_data,
)
from app.knowledge.evidence import evidence_data
from app.knowledge.ingestion import SUPPORTED_SUFFIXES

settings = get_settings()


@asynccontextmanager
async def lifespan(_app):
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="团队知识库智能问答", version="0.2.0", lifespan=lifespan)
DB = Annotated[DBSession, Depends(get_db)]


@app.exception_handler(DomainError)
async def domain_error(_request, exc):
    return JSONResponse({"detail": str(exc), "code": exc.code}, status_code=exc.status,
                        headers={"Retry-After": "3"} if exc.status == 429 else None)


@app.exception_handler(SQLAlchemyError)
async def database_error(_request, _exc):
    # Do not expose connection strings or database stack traces to the browser.
    return JSONResponse({"detail": "数据库不可用，请启动 PostgreSQL 并执行迁移"}, status_code=503)


@app.get("/api/health")
def health(db: DB):
    try:
        db.execute(text("SELECT 1"))
        return {"status": "ok", "database": db.bind.dialect.name,
                "model_configured": bool(settings.chat_api_key and settings.model),
                "embedding_configured": bool(settings.embedding_api_key and settings.embedding_model),
                "execution": settings.task_backend, "deployment": "team-local",
                "diagnostic_mode": db.bind.dialect.name != "postgresql"}
    except Exception:
        return JSONResponse({"status": "unavailable", "detail": "数据库不可用，请启动依赖并执行迁移"}, status_code=503)


@app.get("/api/corpora")
def corpora(db: DB, user: CurrentUser):
    member_ids = select(Membership.corpus_id).where(Membership.user_id == user.id)
    return [{"id": c.id, "name": c.name, "role": corpus_role(db, c.id, user.id),
             "created_at": c.created_at.isoformat()} for c in db.scalars(select(Corpus).where(
                 or_(Corpus.owner_id == user.id, Corpus.id.in_(member_ids))).order_by(Corpus.created_at))]


@app.post("/api/corpora", status_code=201)
def add_corpus(body: CorpusCreate, user: CurrentUser, db: DB):
    if not body.name.strip():
        raise DomainError("论文集名称不能为空")
    corpus = Corpus(id=uid(), owner_id=user.id, name=body.name.strip())
    db.add(corpus)
    db.flush()
    db.add(Membership(corpus_id=corpus.id, user_id=user.id, role="admin"))
    db.commit()
    return {"id": corpus.id, "name": corpus.name, "role": "admin", "created_at": corpus.created_at.isoformat()}


@app.get("/api/corpora/{corpus_id}/papers")
def papers(corpus_id: str, user: CurrentUser, db: DB, include_history: bool = False):
    owned(db, Corpus, corpus_id, user.id)
    stmt = select(PaperVersion, Paper).join(Paper).where(Paper.corpus_id == corpus_id, Paper.deleted.is_(False))
    if not include_history:
        stmt = stmt.where(or_(PaperVersion.is_active.is_(True), PaperVersion.status != "ready"))
    return [version_data(v, p) for v, p in db.execute(stmt.order_by(PaperVersion.created_at.desc()))]


@app.post("/api/corpora/{corpus_id}/papers", status_code=202)
def upload(corpus_id: str, user: CurrentUser, db: DB, file: UploadFile = File(...), title: str = Form(""),
                 paper_id: str | None = Form(None), folder: str = Form(""), tags: str = Form("")):
    require_role(db, corpus_id, user.id, "editor")
    filename = Path((file.filename or "document.pdf").replace("\\", "/")).name[:255]
    metadata = DocumentUpdate(title=title.strip() or Path(filename).stem,
                              folder=folder, tags=[t.strip() for t in tags.split(",") if t.strip()])
    paper = db.get(Paper, paper_id) if paper_id else None
    if paper_id and (not paper or paper.corpus_id != corpus_id or paper.deleted):
        raise DomainError("指定文档不存在或不属于当前知识库", 404)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DomainError("支持 PDF、DOCX、Markdown、TXT、CSV/TSV、JSON/JSONL、HTML 和 XLSX")
    lock_workload(db)
    admit_job(db, settings)
    db.rollback()
    data = bytearray()
    try:
        while part := file.file.read(1024 * 1024):
            data.extend(part)
            if len(data) > settings.max_upload_mb * 1024 * 1024:
                raise DomainError(f"文件超过 {settings.max_upload_mb} MB 限制", 413)
    finally:
        file.file.close()
    if not data or (suffix == ".pdf" and b"%PDF-" not in data[:1024]):
        raise DomainError("文件为空或内容与 PDF 格式不符")
    if suffix in (".docx", ".xlsx") and not data.startswith(b"PK"):
        raise DomainError(f"文件内容不是有效的 {suffix}")
    if suffix not in (".pdf", ".docx", ".xlsx"):
        try:
            data.decode("utf-8-sig")
        except UnicodeError:
            raise DomainError("文本类文件需要使用 UTF-8 编码") from None
        if b"\x00" in data:
            raise DomainError("文本类文件不能包含二进制内容")
    corpus = db.get(Corpus, corpus_id)
    db.refresh(corpus, with_for_update=True)
    require_role(db, corpus_id, user.id, "editor")
    digest, storage_key = source_hash_key(bytes(data), suffix)
    duplicate_version = db.scalar(select(PaperVersion).join(Paper).where(
        Paper.corpus_id == corpus_id, Paper.deleted.is_(False), PaperVersion.sha256 == digest))
    if duplicate_version:
        if paper and duplicate_version.paper_id == paper.id:
            raise DomainError("该文档已有相同内容的版本", 409)
        raise DomainError("该知识库已有相同内容的文档；如需更新请明确指定原文档", 409)
    settings.storage_dir.mkdir(parents=True, exist_ok=True)
    path, temp = settings.storage_dir / storage_key, settings.storage_dir / f"{uid()}.upload"
    try:
        temp.write_bytes(data)
        lock_workload(db)
        admit_job(db, settings)
        require_role(db, corpus_id, user.id, "editor")
        db.refresh(corpus, with_for_update=True)
        if paper_id:
            # File reads release the initial transaction. Recheck the target
            # under its publication lock before moving bytes or queuing a version.
            paper = db.scalar(select(Paper).where(Paper.id == paper_id).with_for_update()
                              .execution_options(populate_existing=True))
            if not paper or paper.corpus_id != corpus_id or paper.deleted:
                raise DomainError("指定文档不存在或不属于当前知识库", 404)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    if not paper:
        paper = Paper(corpus_id=corpus_id, title=metadata.title, folder=metadata.folder, tags=metadata.tags)
        db.add(paper)
        db.flush()
    version = PaperVersion(paper_id=paper.id, sha256=digest, storage_key=storage_key, filename=filename,
                           byte_size=len(data), status="uploaded", is_active=False)
    db.add(version)
    db.commit()
    return version_data(version, paper)


@app.get("/api/paper-versions/{version_id}")
def version(version_id: str, user: CurrentUser, db: DB):
    return version_data(*owned_version(db, version_id, user.id))


@app.post("/api/paper-versions/{version_id}/retry", status_code=202)
def retry_version(version_id: str, user: CurrentUser, db: DB):
    version, paper = owned_version(db, version_id, user.id)
    require_role(db, paper.corpus_id, user.id, "editor")
    lock_workload(db)
    db.refresh(version, with_for_update=True)
    if paper.deleted or version.status != "failed":
        raise DomainError("仅失败的入库任务可以重试", 409)
    admit_job(db, settings)
    version.status, version.error, version.lease_until = "uploaded", None, None
    db.commit()
    return version_data(version, paper)


@app.post("/api/paper-versions/{version_id}/reindex", status_code=202)
def reindex_version(version_id: str, user: CurrentUser, db: DB):
    version, paper = owned_version(db, version_id, user.id)
    require_role(db, paper.corpus_id, user.id, "editor")
    lock_workload(db)
    db.refresh(version, with_for_update=True)
    if paper.deleted or version.status != "ready":
        raise DomainError("请等待当前入库结束", 409)
    admit_job(db, settings)
    version.status = "uploaded"
    db.commit()
    return version_data(version, paper)


@app.get("/api/paper-versions/{version_id}/file")
def pdf_file(version_id: str, request: Request, user: CurrentUser, db: DB):
    version, _ = owned_version(db, version_id, user.id)
    path = settings.storage_dir / version.storage_key
    snapshot = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
    try:
        checksum = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > version.byte_size:
                    raise DomainError("原文件与记录的版本不一致，请重新上传", 409)
                checksum.update(chunk)
                snapshot.write(chunk)
        if size != version.byte_size or checksum.hexdigest() != version.sha256:
            raise DomainError("原文件与记录的版本不一致，请重新上传", 409)
        snapshot.seek(0)
    except OSError:
        snapshot.close()
        raise DomainError("原文件不可访问或已移除", 404) from None
    except BaseException:
        snapshot.close()
        raise

    disposition = "inline" if version.filename.lower().endswith(".pdf") else "attachment"
    etag = f'"{version.sha256}"'
    headers = {"Content-Disposition": f"{disposition}; filename*=utf-8''{quote(version.filename, safe='')}",
               "Cache-Control": "no-store", "Accept-Ranges": "bytes", "ETag": etag}
    start, end, status = 0, size, 200
    range_header = request.headers.get("range", "")
    if_range = request.headers.get("if-range")
    # Multiple or malformed ranges are ignored, as HTTP permits. A single
    # satisfiable range reads only from the already verified snapshot.
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header, re.IGNORECASE) if len(range_header) <= 256 else None
    if match and (match[1] or match[2]) and (if_range is None or if_range == etag):
        if match[1]:
            start = int(match[1])
            end = min(int(match[2]) + 1, size) if match[2] else size
        else:
            start = max(0, size - int(match[2]))
        if start >= size or start >= end:
            snapshot.close()
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}", **headers})
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end - 1}/{size}"
    headers["Content-Length"] = str(end - start)
    snapshot.seek(start)

    def stream_snapshot():
        remaining = end - start
        while remaining:
            chunk = snapshot.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk

    class SnapshotResponse(StreamingResponse):
        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                snapshot.close()

    return SnapshotResponse(stream_snapshot(), status_code=status,
        media_type=mimetypes.guess_type(version.filename)[0] or "application/octet-stream", headers=headers)


@app.post("/api/corpora/{corpus_id}/snapshots", status_code=201)
def snapshot(corpus_id: str, user: CurrentUser, body: SnapshotCreate, db: DB):
    snap = create_snapshot(db, corpus_id, body.paper_version_ids, user.id)
    return {"id": snap.id, "corpus_id": snap.corpus_id, "manifest": snap.manifest}


@app.get("/api/snapshots/{snapshot_id}")
def read_snapshot(snapshot_id: str, user: CurrentUser, db: DB):
    snap = owned(db, Snapshot, snapshot_id, user.id)
    return {"id": snap.id, "corpus_id": snap.corpus_id, "manifest": snap.manifest,
            "created_at": snap.created_at.isoformat()}


@app.post("/api/question-runs", status_code=202)
def ask(body: RunCreate, user: CurrentUser, db: DB, idempotency_key: Annotated[str, Header(min_length=8, max_length=128)]):
    run = create_run(db, body, idempotency_key, user.id, settings)
    return run_data(db, run, user.id)


@app.get("/api/corpora/{corpus_id}/question-runs")
def history(corpus_id: str, user: CurrentUser, db: DB):
    owned(db, Corpus, corpus_id, user.id)
    rows = db.scalars(select(QuestionRun).join(Snapshot).where(
        Snapshot.corpus_id == corpus_id, QuestionRun.owner_id == user.id).order_by(
        QuestionRun.created_at.desc()).limit(100))
    return [{"id": r.id, "question": r.question, "status": r.status, "created_at": r.created_at.isoformat()}
            for r in rows]


@app.get("/api/question-runs/{run_id}")
def read_run(run_id: str, user: CurrentUser, db: DB):
    run = db.get(QuestionRun, run_id)
    if not run:
        raise DomainError("对象不存在或不可访问", 404)
    if run.owner_id != user.id:
        snapshot = db.get(Snapshot, run.snapshot_id)
        try:
            require_role(db, snapshot.corpus_id, user.id, "editor")
        except DomainError:
            raise DomainError("对象不存在或不可访问", 404) from None
        if run.status != "review_pending":
            raise DomainError("对象不存在或不可访问", 404)
    else:
        owned(db, QuestionRun, run_id, user.id)
    return run_data(db, run, user.id)


@app.post("/api/question-runs/{run_id}/cancel")
def cancel(run_id: str, user: CurrentUser, db: DB):
    run = owned(db, QuestionRun, run_id, user.id)
    db.refresh(run, with_for_update=True)
    if run.status not in TERMINAL:
        run.cancel_requested, run.status, run.lease_until = True, "cancelled", None
        emit(db, run.id, run.attempt, "cancelled", {"message": "已取消；进行中的模型请求结束后停止后续节点"})
        db.commit()
    return run_data(db, run, user.id)


@app.get("/api/question-runs/{run_id}/events")
def events(run_id: str, request: Request, user: CurrentUser, db: DB, after_seq: int = Query(0, ge=0), stream: bool = True,
           last_event_id: Annotated[str | None, Header()] = None):
    owned(db, QuestionRun, run_id, user.id)
    if last_event_id:
        try:
            if int(last_event_id) < 0:
                raise ValueError
            after_seq = max(after_seq, int(last_event_id))
        except ValueError:
            raise DomainError("Last-Event-ID 需要为整数") from None
    def read(session, cursor):
        return [progress_event(e) for e in session.scalars(select(RunEvent).where(
                     RunEvent.run_id == run_id, RunEvent.seq > cursor).order_by(RunEvent.seq).limit(200))]
    if not stream:
        return read(db, after_seq)

    user_id = user.id
    db.rollback()  # Do not hold the request's read transaction for the lifetime of SSE.

    def read_batch(cursor):
        with Session() as session:
            if current_user(request, session).id != user_id:
                raise DomainError("请先登录", 401)
            run = owned(session, QuestionRun, run_id, user_id)
            return read(session, cursor), run.status in TERMINAL

    async def iterator():
        cursor, heartbeat_at = after_seq, time.monotonic()
        yield ": connected\nretry: 2000\n\n"
        while True:
            if await request.is_disconnected():
                return
            try:
                rows, terminal = await run_in_threadpool(read_batch, cursor)
            except DomainError:
                return
            for event in rows:
                cursor = event["seq"]
                yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if terminal and len(rows) < 200:
                return
            if time.monotonic() - heartbeat_at >= 15:
                yield "event: heartbeat\ndata: {}\n\n"
                heartbeat_at = time.monotonic()
            if len(rows) < 200:
                await asyncio.sleep(0.35)
    return StreamingResponse(iterator(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/question-runs/{run_id}/trace")
def trace(run_id: str, user: CurrentUser, db: DB):
    run = owned(db, QuestionRun, run_id, user.id)
    role = corpus_role(db, db.get(Snapshot, run.snapshot_id).corpus_id, user.id)
    if run.owner_id != user.id and role not in {"admin", "editor"}:
        raise DomainError("运行轨迹仅对发起人和审核角色开放", 404)
    if run.status != "completed":
        raise DomainError("答案通过审核后才能读取完整运行轨迹", 409, "answer_not_published")
    retrievals = [{"attempt": r.attempt, "round": r.round, "subquestion_id": r.subquestion_id,
                   "query": r.query, "rankings": r.rankings}
                  for r in db.scalars(select(RetrievalRecord).where(RetrievalRecord.run_id == run_id).order_by(
                      RetrievalRecord.created_at))]
    calls = [{"id": c.id, "attempt": c.attempt, "role": c.role, "model": c.model,
              "input_tokens": c.input_tokens if c.usage_known else None,
              "output_tokens": c.output_tokens if c.usage_known else None,
              "elapsed_seconds": c.elapsed_seconds, "status": c.status, "usage_known": c.usage_known,
              "estimated_input_tokens": c.estimated_input_tokens, "reserved_tokens": c.reserved_tokens,
              "provider_request_id": c.provider_request_id,
              "request": c.request, "response": c.response} for c in db.scalars(select(ModelCall).where(ModelCall.run_id == run_id).order_by(
                  ModelCall.created_at))]
    run = db.get(QuestionRun, run_id)
    snapshot = db.get(Snapshot, run.snapshot_id)
    nodes = [{"node": e.payload.get("node"), "seconds": e.payload.get("seconds"),
              "attempt": e.attempt, "error": e.payload.get("error")}
             for e in db.scalars(select(RunEvent).where(RunEvent.run_id == run_id,
                                 RunEvent.kind == "node_finished").order_by(RunEvent.seq))]
    provider_events = [{"attempt": e.attempt, **e.payload} for e in db.scalars(select(RunEvent).where(
        RunEvent.run_id == run_id, RunEvent.kind == "provider_http").order_by(RunEvent.seq))]
    return {"request_id": run.id, "retrievals": retrievals, "model_calls": calls, "nodes": nodes,
            "external_tool_calls": [{"id": c.id, "server_id": c.server_id, "tool_name": c.tool_name,
                                      "arguments": c.arguments, "status": c.status, "result": c.result}
                                     for c in db.scalars(select(MCPInvocation).where(
                                         MCPInvocation.run_id == run.id, MCPInvocation.owner_id == user.id))],
            "external_operations": [{"id": op.id, "server_id": op.server_id, "tool_name": op.tool_name,
                                     "status": op.status, "digest": op.digest, "receipt": op.receipt}
                                    for op in db.scalars(select(ExternalOperation).where(
                                        ExternalOperation.run_id == run.id, ExternalOperation.owner_id == user.id))],
            "provider_events": provider_events,
            "degradations": [{"attempt": e.attempt, **e.payload} for e in db.scalars(select(RunEvent).where(
                RunEvent.run_id == run_id, RunEvent.kind == "retrieval_degraded").order_by(RunEvent.seq))],
            "execution_manifest": run.config, "accounting": usage_summary(db, run.id, run.config),
            "documents": snapshot.manifest, "status": run.status, "error": run.error,
            "followup_count": max(0, run.state.get("round", 0) - 1)}


@app.get("/api/corpora/{corpus_id}/answer-reviews")
def pending_answer_reviews(corpus_id: str, user: CurrentUser, db: DB):
    require_role(db, corpus_id, user.id, "editor")
    runs = db.scalars(select(QuestionRun).join(Snapshot).where(
        Snapshot.corpus_id == corpus_id, QuestionRun.status == "review_pending"
    ).order_by(QuestionRun.created_at).limit(100))
    return [{"run": run_data(db, run, user.id)} for run in runs]


@app.get("/api/answers/{answer_id}")
def read_answer(answer_id: str, user: CurrentUser, db: DB):
    answer = db.get(AnswerRevision, answer_id)
    if not answer:
        raise DomainError("答案不存在", 404)
    return published_answer_data(db, answer, user.id)


@app.get("/api/answers/{answer_id}/approval")
def answer_approval(answer_id: str, user: CurrentUser, db: DB):
    answer = db.get(AnswerRevision, answer_id)
    if not answer:
        raise DomainError("答案不存在", 404)
    run = db.get(QuestionRun, answer.run_id)
    snapshot = db.get(Snapshot, run.snapshot_id) if run else None
    if not snapshot:
        raise DomainError("答案不存在或不可访问", 404)
    require_role(db, snapshot.corpus_id, user.id, "editor")
    record = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run.id))
    if run.status == "completed" and record and record.answer_id != answer_id:
        raise DomainError("审核记录与答案版本不匹配", 409)
    if run.status not in {"review_pending", "completed", "rejected"}:
        raise DomainError("当前答案不在审核流程中", 409, "answer_not_pending")
    return {"status": run.status, "decision": record.decision if record else None,
            "reason": record.reason if record else None, "reviewer_id": record.reviewer_id if record else None,
            "answer_sha256": record.answer_sha256 if record else answer_content_hash(answer_data(db, answer, user.id))}


@app.post("/api/answers/{answer_id}/approval")
def decide_answer(answer_id: str, body: AnswerApprovalCreate, user: CurrentUser, db: DB):
    answer = db.get(AnswerRevision, answer_id)
    if not answer:
        raise DomainError("答案不存在", 404)
    run = db.get(QuestionRun, answer.run_id)
    snapshot = db.get(Snapshot, run.snapshot_id) if run else None
    if not snapshot:
        raise DomainError("答案不存在或不可访问", 404)
    lock_snapshot_publication(db, snapshot, run.owner_id, user.id)
    require_role(db, snapshot.corpus_id, user.id, "editor")
    run = db.scalar(select(QuestionRun).where(QuestionRun.id == run.id).with_for_update()
                    .execution_options(populate_existing=True))
    current_answer = db.scalar(select(AnswerRevision).where(
        AnswerRevision.run_id == run.id, AnswerRevision.revision == answer.revision))
    if not current_answer:
        raise DomainError("答案版本不存在", 404)
    existing = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == run.id))
    reason = body.reason.strip() if body.reason and body.reason.strip() else None
    if existing:
        if (existing.answer_id, existing.reviewer_id, existing.decision, existing.reason, existing.answer_sha256) == (
                answer.id, user.id, body.decision, reason, body.answer_sha256):
            return {"status": run.status, "decision": existing.decision, "reason": existing.reason,
                    "answer_sha256": existing.answer_sha256}
        raise DomainError("该运行已存在绑定版本的最终审核", 409, "approval_exists")
    if run.status != "review_pending" or current_answer.id != answer_id:
        raise DomainError("只有当前待审核答案可以作决定", 409, "answer_not_pending")
    if body.decision == "accept":
        # A draft may have waited for review after its owner lost access or its
        # source version was replaced. Recheck at the publication boundary.
        check_snapshot_access(db, snapshot, run.owner_id, current=True)
    current_hash = answer_content_hash(answer_data(db, current_answer, user.id))
    if body.answer_sha256 != current_hash:
        raise DomainError("答案内容已变化，请重新查看后再审核", 409, "answer_content_changed")
    record = AnswerApprovalRecord(id=uid(), run_id=run.id, answer_id=answer.id,
                                  reviewer_id=user.id, decision=body.decision,
                                  answer_sha256=current_hash, reason=reason)
    if body.decision == "accept":
        experience = AcceptedAnswerExperience(
            id=uid(), run_id=run.id, answer_id=answer.id, corpus_id=snapshot.corpus_id,
            reviewer_id=user.id, answer_sha256=current_hash, question=run.question)
        db.add(experience)
        run.status = "completed"
        event_kind, message = "completed", "答案已通过人工审核"
    else:
        run.status = "rejected"
        event_kind, message = "rejected", "答案已拒绝并记录理由；可在原快照上新建重试"
    db.add(record)
    emit(db, run.id, run.attempt, event_kind, {"message": message, "answer_sha256": current_hash})
    db.commit()
    return {"status": run.status, "decision": record.decision, "reason": record.reason,
            "answer_sha256": record.answer_sha256}


@app.get("/api/evidence/{evidence_id}")
def read_evidence(evidence_id: str, user: CurrentUser, db: DB):
    span = db.get(EvidenceSpan, evidence_id)
    if not span:
        raise DomainError("证据不存在", 404)
    return evidence_data(db, span, user.id)


@app.post("/api/answers/{answer_id}/reviews", status_code=201)
def review(answer_id: str, user: CurrentUser, body: ReviewCreate, db: DB,
           idempotency_key: Annotated[str | None, Header(min_length=8, max_length=128)] = None):
    answer = feedback_answer(answer_id, user, db)
    if idempotency_key:
        existing = db.scalar(select(ReviewRecord).where(ReviewRecord.owner_id == user.id,
                             ReviewRecord.idempotency_key == idempotency_key))
        if existing:
            if (existing.answer_id, existing.text, existing.category) != (answer_id, body.text, body.category):
                raise DomainError("反馈幂等键已用于其他内容", 409)
            return review_data(existing)
    record = ReviewRecord(id=uid(), owner_id=user.id, answer_id=answer_id, text=body.text,
                          category=body.category, answer_sha256=digest(answer), idempotency_key=idempotency_key)
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        if not idempotency_key:
            raise
        return review(answer_id, user, body, db, idempotency_key)
    return review_data(record)


def review_data(record):
    return {"id": record.id, "answer_id": record.answer_id, "text": record.text, "category": record.category,
            "classification_source": "user_selected" if record.category != "unclear" else "unclassified",
            "answer_sha256": record.answer_sha256, "created_at": record.created_at.isoformat(),
            "status": "needs_clarification" if record.category == "unclear" else "recorded",
            "clarification": "具体是偏好不合适、事实错误、遗漏内容、执行失败，还是需求发生了变化？"
            if record.category == "unclear" else None,
            "message": "反馈已保存；不会自动成为事实或正式回归用例"}


@app.get("/api/answers/{answer_id}/reviews")
def list_reviews(answer_id: str, user: CurrentUser, db: DB):
    feedback_answer(answer_id, user, db)
    return [review_data(r) for r in db.scalars(select(ReviewRecord).where(
        ReviewRecord.answer_id == answer_id, ReviewRecord.owner_id == user.id).order_by(ReviewRecord.created_at))]


def feedback_answer(answer_id: str, user: CurrentUser, db: DB) -> dict:
    data = read_answer(answer_id, user, db)
    answer = db.get(AnswerRevision, answer_id)
    run = db.get(QuestionRun, answer.run_id)
    if not run or run.owner_id != user.id:
        raise DomainError("仅运行发起人可提交或读取该答案反馈", 404)
    return data


@app.get("/api/question-runs/{run_id}/recording")
def recording(run_id: str, user: CurrentUser, db: DB):
    run = owned(db, QuestionRun, run_id, user.id)
    role = corpus_role(db, db.get(Snapshot, run.snapshot_id).corpus_id, user.id)
    if run.owner_id != user.id and role not in {"admin", "editor"}:
        raise DomainError("运行记录仅对发起人和审核角色开放", 404)
    if run.status != "completed":
        raise DomainError("答案通过审核后才能读取完整运行记录", 409, "answer_not_published")
    events = [{"seq": e.seq, "attempt": e.attempt, "kind": e.kind, "payload": e.payload,
               "created_at": e.created_at.isoformat()} for e in db.scalars(select(RunEvent).where(
                   RunEvent.run_id == run_id).order_by(RunEvent.seq))]
    history = trace(run_id, user, db)
    return {"kind": "stored_recording", "external_calls": False, "run": run_data(db, run, user.id),
            "events": events, "trace": history,
            "incomplete_call_ids": [c["id"] for c in history["model_calls"] if c["status"] == "started"],
            "limitations": ["只呈现已保存记录，缺失内容不重新生成", "旧记录可能没有调用开始或完整执行清单"]}


from app.api.team_api import router as team_router  # noqa: E402

app.include_router(team_router)

from app.api.extensions_api import router as extensions_router  # noqa: E402

app.include_router(extensions_router)

from app.repair.api import router as code_repair_router  # noqa: E402

app.include_router(code_repair_router)

dist = ROOT / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="workbench")
