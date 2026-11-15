"""One short DB write transaction serializes admission and job claims, including SQLite demos."""
from sqlalchemy import func, select, update

from app.core.errors import DomainError
from app.core.models import Paper, PaperVersion, QuestionRun, WorkloadGate, now


def lock_workload(db):
    # An UPDATE also acquires SQLite's writer lock; SELECT FOR UPDATE alone does not.
    result = db.execute(update(WorkloadGate).where(WorkloadGate.id == 1).values(id=1))
    if result.rowcount != 1:
        raise DomainError("任务容量记录缺失，请执行数据库迁移", 503, "capacity_unavailable")


def pending_jobs(db):
    runs = db.scalar(select(func.count()).select_from(QuestionRun).where(
        QuestionRun.status.in_(["queued", "running"])))
    versions = db.scalar(select(func.count()).select_from(PaperVersion).join(Paper).where(
        Paper.deleted.is_(False), PaperVersion.status.in_(["uploaded", "parsing", "indexing"])))
    return runs + versions


def admit_job(db, settings, owner_id=None):
    # Caller holds lock_workload until commit/rollback, and checks idempotency first.
    if pending_jobs(db) >= settings.max_pending_jobs:
        raise DomainError("任务队列已满，请稍后重试", 429, "queue_full")
    if owner_id is not None:
        count = db.scalar(select(func.count()).select_from(QuestionRun).where(
            QuestionRun.owner_id == owner_id, QuestionRun.status.in_(["queued", "running"])))
        if count >= settings.max_pending_runs_per_user:
            raise DomainError("你的待处理问题已达上限，请等待或取消已有任务", 429, "user_queue_full")


def running_jobs(db):
    runs = db.scalar(select(func.count()).select_from(QuestionRun).where(
        QuestionRun.status == "running", QuestionRun.lease_until >= now()))
    versions = db.scalar(select(func.count()).select_from(PaperVersion).where(
        PaperVersion.status.in_(["parsing", "indexing"]), PaperVersion.lease_until >= now()))
    return runs + versions
