from datetime import timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .auth import CurrentUser
from .code_repair import (
    artifact_hash,
    create_task,
    propose,
    restore_checkpoint,
    run_tests,
    tracked_source_files,
    verification_hash,
)
from .config import get_settings
from .db import get_db
from .errors import DomainError, ProviderError
from .models import CodeRepairReview, CodeRepairTask, RepairExperience, uid
from .providers import ModelProvider
from .schemas import CodeTaskCreate, CodeTaskPropose, CodeTaskReview

router = APIRouter(prefix="/api/code-repair")
DB = Annotated[Session, Depends(get_db)]


def owned_task(db: Session, task_id: str, user_id: str) -> CodeRepairTask:
    task = db.get(CodeRepairTask, task_id)
    if not task or task.owner_id != user_id:
        raise DomainError("代码任务不存在", 404)
    return task


def task_data(task: CodeRepairTask) -> dict:
    return {
        "id": task.id,
        "repository": Path(task.repository_path).name,
        "objective": task.objective,
        "base_ref": task.base_ref,
        "base_commit": task.base_commit,
        "selected_files": task.selected_files or [],
        "status": task.status,
        "summary": task.proposed_summary,
        "patch_sha256": task.artifact_sha256,
        "patch": task.patch_text,
        "test_exit_code": task.test_exit_code,
        "test_output": task.test_output,
        "test_environment": task.test_environment,
        "verification_sha256": task.verification_sha256,
        "workspace_state": task.workspace_state,
        "error": task.error,
        "created_at": task.created_at.astimezone(timezone.utc).isoformat(),
    }


def review_data(review: CodeRepairReview) -> dict:
    return {
        "id": review.id,
        "task_id": review.task_id,
        "decision": review.decision,
        "artifact_sha256": review.artifact_sha256,
        "bundle_sha256": review.bundle_sha256,
        "reason": review.reason,
        "feedback_category": review.feedback_category,
        "root_cause_layer": review.root_cause_layer,
        "uncertainty": review.uncertainty,
        "reviewer_id": review.reviewer_id,
        "created_at": review.created_at.astimezone(timezone.utc).isoformat(),
    }


def restore_rejected_workspace(task: CodeRepairTask, settings, db: Session) -> CodeRepairTask:
    try:
        restore_checkpoint(task, settings)
    except Exception:
        db.rollback()
        task = owned_task(db, task.id, task.owner_id)
        task.workspace_state = "cleanup_failed"
        task.error = "拒绝反馈已保存，但任务副本清理失败；重试相同审核请求可继续清理"
        db.commit()
        raise DomainError(task.error, 503, "workspace_cleanup_failed") from None
    task.workspace_state = "restored"
    task.error = None
    db.commit()
    return task


@router.get("/tasks")
def list_tasks(user: CurrentUser, db: DB):
    return [task_data(task) for task in db.scalars(select(CodeRepairTask).where(
        CodeRepairTask.owner_id == user.id).order_by(CodeRepairTask.created_at.desc()).limit(100))]


@router.post("/tasks", status_code=201)
def add_task(body: CodeTaskCreate, user: CurrentUser, db: DB):
    settings = get_settings()
    task = create_task(settings, user.id, body.repository_path, body.base_ref, body.objective)
    db.add(task)
    try:
        db.commit()
    except Exception:
        db.rollback()
        try:
            restore_checkpoint(task, settings)
        except Exception:
            pass
        raise
    return task_data(task)


@router.get("/tasks/{task_id}")
def get_task(task_id: str, user: CurrentUser, db: DB):
    return task_data(owned_task(db, task_id, user.id))


@router.get("/tasks/{task_id}/review")
def get_task_review(task_id: str, user: CurrentUser, db: DB):
    owned_task(db, task_id, user.id)
    review = db.scalar(select(CodeRepairReview).where(CodeRepairReview.task_id == task_id))
    return review_data(review) if review else None


@router.get("/tasks/{task_id}/files")
def task_files(task_id: str, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    return {"files": tracked_source_files(task, get_settings())}


@router.post("/tasks/{task_id}/propose")
def generate_patch(task_id: str, body: CodeTaskPropose, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    claim = db.execute(update(CodeRepairTask).where(
        CodeRepairTask.id == task_id,
        CodeRepairTask.owner_id == user.id,
        CodeRepairTask.status == "ready",
    ).values(status="generating"))
    if claim.rowcount != 1:
        raise DomainError("只有新任务可以生成补丁", 409)
    db.commit()
    task = owned_task(db, task_id, user.id)
    settings = get_settings()
    try:
        summary, patch, fingerprint = propose(task, settings, body.files, ModelProvider(settings))
    except (DomainError, ProviderError) as exc:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", str(exc)[:4000]
        db.commit()
        raise
    except Exception:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", "补丁生成或范围校验失败；任务副本已恢复"
        db.commit()
        raise DomainError("生成或应用补丁失败，任务副本已恢复", 502, "patch_proposal_failed") from None
    task.selected_files = body.files
    task.proposed_summary = summary
    task.patch_text = patch
    task.artifact_sha256 = fingerprint
    task.verification_sha256 = None
    task.test_environment = None
    task.workspace_state = "active"
    task.status = "patched"
    try:
        db.commit()
    except Exception:
        db.rollback()
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        raise DomainError("补丁记录失败，任务副本已恢复", 503, "patch_persist_failed") from None
    return task_data(task)


@router.post("/tasks/{task_id}/test")
def execute_tests(task_id: str, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    settings = get_settings()
    claim = db.execute(update(CodeRepairTask).where(
        CodeRepairTask.id == task_id,
        CodeRepairTask.owner_id == user.id,
        CodeRepairTask.status == "patched",
    ).values(status="testing"))
    if claim.rowcount != 1:
        raise DomainError("只有已生成补丁的任务可以运行测试", 409)
    db.commit()
    task = owned_task(db, task_id, user.id)
    try:
        current = artifact_hash(task, settings)
    except DomainError:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", "测试前任务副本已变化；任务副本已恢复到固定基线"
        db.commit()
        raise DomainError(task.error, 409, "artifact_changed")
    if current != task.artifact_sha256:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", "测试前产物哈希与待测补丁不一致；任务副本已恢复到固定基线"
        db.commit()
        raise DomainError(task.error, 409, "artifact_changed")
    try:
        code, output, environment = run_tests(task, settings)
    except Exception as exc:
        restore_checkpoint(task, settings)
        code, output, environment = 1, "隔离测试执行异常，任务副本将恢复到固定基线。", None
        task.error = str(exc)[:4000]
    task.test_exit_code, task.test_output, task.test_environment = code, output, environment
    if code == 0:
        task.verification_sha256 = verification_hash(task)
        task.status, task.error = "review_pending", None
    else:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.verification_sha256 = None
        task.status, task.error = "failed", "自动化测试失败；任务副本已恢复到固定基线"
    db.commit()
    return task_data(task)


@router.post("/tasks/{task_id}/cleanup")
def cleanup_rejected_task(task_id: str, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    review = db.scalar(select(CodeRepairReview).where(CodeRepairReview.task_id == task_id))
    if task.status != "rejected" or not review or review.decision != "reject":
        raise DomainError("只有已拒绝的任务可以重试副本清理", 409)
    if task.workspace_state != "restored":
        task = restore_rejected_workspace(task, get_settings(), db)
    return task_data(task)


@router.post("/tasks/{task_id}/review")
def review_task(task_id: str, body: CodeTaskReview, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    reason = body.reason.strip() if body.reason and body.reason.strip() else None
    uncertainty = body.uncertainty.strip() if body.uncertainty and body.uncertainty.strip() else None
    existing = db.scalar(select(CodeRepairReview).where(CodeRepairReview.task_id == task_id))
    if existing:
        same_review = (
            existing.reviewer_id, existing.decision, existing.reason, existing.feedback_category,
            existing.root_cause_layer, existing.uncertainty,
        ) == (user.id, body.decision, reason, body.feedback_category,
              body.root_cause_layer, uncertainty)
        if same_review:
            if existing.decision == "reject" and task.workspace_state != "restored":
                restore_rejected_workspace(task, get_settings(), db)
            return review_data(existing)
        raise DomainError("该任务已有绑定产物版本的最终审核", 409, "review_exists")
    if task.status != "review_pending" or task.test_exit_code != 0 or not task.artifact_sha256:
        raise DomainError("只有测试通过且等待审核的任务可以验收", 409)
    settings = get_settings()
    try:
        current = artifact_hash(task, settings)
    except DomainError:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", "审核前任务副本已变化；任务副本已恢复到固定基线"
        db.commit()
        raise DomainError(task.error, 409, "artifact_changed")
    if current != task.artifact_sha256:
        restore_checkpoint(task, settings)
        task.workspace_state = "restored"
        task.status, task.error = "failed", "审核前产物哈希已变化；任务副本已恢复"
        db.commit()
        raise DomainError(task.error, 409, "artifact_changed")

    bundle = task.verification_sha256
    if not bundle or verification_hash(task) != bundle:
        raise DomainError("测试证明已变化，必须重新运行验收", 409, "verification_changed")
    record = CodeRepairReview(
        id=uid(), task_id=task.id, reviewer_id=user.id, decision=body.decision,
        artifact_sha256=current, bundle_sha256=bundle, reason=reason,
        feedback_category=body.feedback_category, root_cause_layer=body.root_cause_layer,
        uncertainty=uncertainty,
    )
    task.status = "accepted" if body.decision == "accept" else "rejected"
    task.error = None
    if body.decision == "reject":
        task.workspace_state = "cleanup_pending"
    db.add(record)
    if body.decision == "accept":
        db.add(RepairExperience(
            id=uid(), task_id=task.id, owner_id=user.id, artifact_sha256=current,
            objective=task.objective, summary=task.proposed_summary or "",
            verified_command="python -m pytest -q",
            verified_result=(task.test_output or "")[-20000:],
        ))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.scalar(select(CodeRepairReview).where(CodeRepairReview.task_id == task_id))
        if existing and (
                existing.reviewer_id, existing.decision, existing.reason, existing.feedback_category,
                existing.root_cause_layer, existing.uncertainty,
        ) == (user.id, body.decision, reason, body.feedback_category,
              body.root_cause_layer, uncertainty):
            if existing.decision == "reject":
                task = owned_task(db, task_id, user.id)
                restore_rejected_workspace(task, settings, db)
            return review_data(existing)
        raise DomainError("该任务已有最终审核", 409, "review_exists") from None
    if body.decision == "reject":
        restore_rejected_workspace(task, settings, db)
    return review_data(record)


@router.get("/tasks/{task_id}/patch")
def download_patch(task_id: str, user: CurrentUser, db: DB):
    task = owned_task(db, task_id, user.id)
    if not task.patch_text:
        raise DomainError("任务还没有补丁", 404)
    return PlainTextResponse(
        task.patch_text,
        headers={"Content-Disposition": f'attachment; filename="{task.id}.patch"',
                 "X-Artifact-SHA256": task.artifact_sha256 or ""},
    )


@router.get("/experiences")
def experiences(user: CurrentUser, db: DB):
    return [{
        "id": item.id,
        "task_id": item.task_id,
        "artifact_sha256": item.artifact_sha256,
        "objective": item.objective,
        "summary": item.summary,
        "verified_command": item.verified_command,
        "verified_result": item.verified_result,
        "created_at": item.created_at.astimezone(timezone.utc).isoformat(),
    } for item in db.scalars(select(RepairExperience).where(
        RepairExperience.owner_id == user.id).order_by(RepairExperience.created_at.desc()).limit(100))]
