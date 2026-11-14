import re
from datetime import timezone

from sqlalchemy import or_, select

from .access import require_role
from .errors import DomainError
from .governance import digest
from .models import Corpus, MemoryEntry, MemoryEvent, QuestionRun, Snapshot, User, now, uid

TOKEN = re.compile(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}")


def _words(value):
    words = set()
    for part in TOKEN.findall(value):
        words.add(part.lower())
        if re.fullmatch(r"[\u4e00-\u9fff]{2,}", part):
            words.update(part[index:index+2] for index in range(len(part)-1))
    return words


def current_corpus(db, run):
    snapshot = db.get(Snapshot, run.snapshot_id)
    return db.get(Corpus, snapshot.corpus_id) if snapshot else None


def validate_source(db, owner_id, kind, run_id, corpus_id):
    if not run_id:
        if kind != "preference":
            raise DomainError("经验记忆需要关联本人任务，避免没有来源的经验结论")
        return None
    run = db.get(QuestionRun, run_id)
    if not run or run.owner_id != owner_id:
        raise DomainError("任务不存在或不可访问", 404)
    corpus = current_corpus(db, run)
    if not corpus or (corpus_id and corpus_id != corpus.id):
        raise DomainError("任务与记忆范围不一致", 409)
    require_role(db, corpus.id, owner_id)
    if kind == "success" and run.status != "completed":
        raise DomainError("成功经验只能关联已完成任务")
    if kind == "failure" and run.status not in ("completed", "failed"):
        raise DomainError("失败经验只能关联已结束任务")
    return run, corpus


def create_candidate(db, owner_id, body):
    source_run_id, corpus_id, question, scope_hash = body.source_run_id, None, "", None
    if body.kind == "preference":
        corpus_id = None
        if source_run_id:
            run, corpus = validate_source(db, owner_id, body.kind, source_run_id, None)
            question = run.question
            # Preference provenance is retained, but source question text is not reused across corpora.
            question = ""
    else:
        run, corpus = validate_source(db, owner_id, body.kind, source_run_id, None)
        corpus_id, question = corpus.id, run.question
        scope_hash = digest(db.get(Snapshot, run.snapshot_id).manifest)
    if body.expires_at and body.expires_at <= now():
        raise DomainError("过期时间必须晚于现在")
    row = MemoryEntry(id=uid(), owner_id=owner_id, corpus_id=corpus_id, kind=body.kind,
                      key=body.key.strip(), content=body.content.strip(), source_run_id=source_run_id,
                      source_question=question, scope_sha256=scope_hash, expires_at=body.expires_at,
                      version=1, status="candidate")
    db.add(row)
    db.flush()
    db.add(MemoryEvent(memory_id=row.id, owner_id=owner_id, version=row.version,
                       kind="proposed", detail={"memory_kind": row.kind, "source_run_id": source_run_id}))
    db.commit()
    return row


def list_memories(db, owner_id, *, status=None, kind=None):
    stmt = select(MemoryEntry).where(MemoryEntry.owner_id == owner_id,
                                     MemoryEntry.status != "deleted")
    if status:
        stmt = stmt.where(MemoryEntry.status == status)
    if kind:
        stmt = stmt.where(MemoryEntry.kind == kind)
    visible = []
    for row in db.scalars(stmt.order_by(MemoryEntry.created_at.desc()).limit(200)):
        try:
            if row.corpus_id:
                require_role(db, row.corpus_id, owner_id)
            visible.append(row)
        except DomainError:
            continue
    return visible


def memory_data(row):
    return {"id": row.id, "kind": row.kind, "key": row.key, "content": row.content,
            "corpus_id": row.corpus_id, "source_run_id": row.source_run_id,
            "source_question": row.source_question, "scope_sha256": row.scope_sha256,
            "version": row.version, "status": row.status,
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "confirmed_at": row.confirmed_at.isoformat() if row.confirmed_at else None,
            "created_at": row.created_at.isoformat()}


def get_owned(db, memory_id, owner_id):
    row = db.get(MemoryEntry, memory_id)
    if not row or row.owner_id != owner_id or row.status == "deleted":
        raise DomainError("记忆不存在或不可访问", 404)
    if row.corpus_id:
        require_role(db, row.corpus_id, owner_id)
    return row


def lock_memory(db, row, owner_id, *, require_corpus=True):
    # Serialize preference conflict checks for the same account on PostgreSQL.
    db.scalar(select(User).where(User.id == owner_id).with_for_update())
    db.refresh(row, with_for_update=True)
    if row.owner_id != owner_id or row.status == "deleted":
        raise DomainError("记忆不存在或不可访问", 404)
    if require_corpus and row.corpus_id:
        require_role(db, row.corpus_id, owner_id)


def confirm_memory(db, row, owner_id, version):
    lock_memory(db, row, owner_id)
    if row.version != version:
        raise DomainError("记忆内容已更新，请刷新后重新确认", 409)
    if row.status != "candidate":
        raise DomainError("只有候选记忆可以确认", 409)
    if row.expires_at and row.expires_at.replace(tzinfo=timezone.utc) <= now():
        raise DomainError("记忆已过期，请修改有效期后重新确认", 409)
    validate_source(db, owner_id, row.kind, row.source_run_id, row.corpus_id)
    conflict = db.scalar(select(MemoryEntry.id).where(
        MemoryEntry.owner_id == owner_id, MemoryEntry.kind == row.kind, MemoryEntry.key == row.key,
        MemoryEntry.id != row.id, MemoryEntry.corpus_id.is_(None) if row.corpus_id is None else
        MemoryEntry.corpus_id == row.corpus_id, MemoryEntry.status == "confirmed",
        or_(MemoryEntry.expires_at.is_(None), MemoryEntry.expires_at > now())))
    if conflict:
        raise DomainError("已有同名有效记忆，请先明确修改或删除旧记忆，再确认新候选", 409)
    row.status, row.confirmed_at = "confirmed", now()
    db.add(MemoryEvent(memory_id=row.id, owner_id=owner_id, version=row.version,
                       kind="confirmed", detail={"memory_kind": row.kind}))
    db.commit()
    return row


def update_memory(db, row, owner_id, body):
    lock_memory(db, row, owner_id)
    if row.version != body.version:
        raise DomainError("记忆内容已更新，请刷新后再修改", 409)
    if body.expires_at and body.expires_at <= now():
        raise DomainError("过期时间必须晚于现在")
    row.content, row.expires_at = body.content.strip(), body.expires_at
    row.version += 1
    row.status, row.confirmed_at = "candidate", None
    db.add(MemoryEvent(memory_id=row.id, owner_id=owner_id, version=row.version,
                       kind="edited", detail={"memory_kind": row.kind, "confirmation_required": True}))
    db.commit()
    return row


def delete_memory(db, row, owner_id, version):
    lock_memory(db, row, owner_id, require_corpus=False)
    if row.version != version:
        raise DomainError("记忆内容已更新，请刷新后再删除", 409)
    old_version = row.version
    row.status, row.key, row.content, row.source_question = "deleted", "deleted", "", ""
    row.version += 1
    row.confirmed_at, row.source_run_id, row.scope_sha256 = None, None, None
    db.add(MemoryEvent(memory_id=row.id, owner_id=owner_id, version=old_version,
                       kind="deleted", detail={"memory_kind": row.kind}))
    db.commit()


def freeze_memory_context(db, owner_id, corpus_id, question, snapshot_hash):
    stmt = select(MemoryEntry).where(
        MemoryEntry.owner_id == owner_id, MemoryEntry.status == "confirmed",
        or_(MemoryEntry.expires_at.is_(None), MemoryEntry.expires_at > now()),
        or_(MemoryEntry.corpus_id.is_(None), MemoryEntry.corpus_id == corpus_id),
    ).order_by(MemoryEntry.confirmed_at.desc())
    # Each class is queried separately, so a large failure collection cannot crowd out successes.
    candidates = []
    for kind in ("preference", "success", "failure"):
        filtered = stmt.where(MemoryEntry.kind == kind)
        if kind != "preference":
            filtered = filtered.where(MemoryEntry.corpus_id == corpus_id, MemoryEntry.scope_sha256 == snapshot_hash)
        candidates.extend(db.scalars(filtered.limit(100)))
    query_words = _words(question)
    preferences = [row for row in candidates if row.kind == "preference"][:4]
    experiences = {"success": [], "failure": []}
    for row in candidates:
        if row.kind not in experiences or row.corpus_id != corpus_id or row.scope_sha256 != snapshot_hash:
            continue
        score = len(query_words & _words(f"{row.key} {row.content} {row.source_question}"))
        if score:
            experiences[row.kind].append((score, row))
    selected = [("preference", row) for row in preferences]
    for kind in ("success", "failure"):
        experiences[kind].sort(key=lambda value: (-value[0], value[1].id))
        selected.extend(("experience", row) for _, row in experiences[kind][:2])
    return [{"id": row.id, "kind": category, "memory_kind": row.kind, "key": row.key,
             "content": row.content[:600], "version": row.version,
             "scope": "user_private" if row.corpus_id is None else "corpus_private"}
            for category, row in selected]


def context_for_run(db, run: QuestionRun):
    corpus = current_corpus(db, run)
    return freeze_memory_context(db, run.owner_id, corpus.id, run.question,
                                 digest(db.get(Snapshot, run.snapshot_id).manifest)) if corpus else []


def active_frozen_context(db, owner_id, frozen):
    result = []
    for item in frozen:
        row = db.get(MemoryEntry, item["id"])
        if (row and row.owner_id == owner_id and row.version == item["version"] and row.status == "confirmed"
                and (not row.expires_at or row.expires_at.replace(tzinfo=timezone.utc) > now())):
            if row.corpus_id:
                require_role(db, row.corpus_id, owner_id)
            result.append(item)
    return result
