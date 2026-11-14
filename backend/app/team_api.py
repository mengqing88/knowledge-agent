from typing import Annotated

from fastapi import APIRouter, Depends, Header, Request, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from .access import require_role
from .auth import CurrentUser, create_session, hash_password, token_hash, verify_password
from .config import get_settings
from .db import get_db
from .errors import DomainError
from .models import AnswerApprovalRecord, Corpus, LoginSession, Membership, Paper, QuestionRun, Snapshot, User
from .schemas import CorpusCreate, DocumentUpdate, Login, MemberCreate, RoleUpdate, RunCreate
from .services import TERMINAL, create_run, emit, owned, run_data

router = APIRouter()
DB = Annotated[Session, Depends(get_db)]


def user_data(user):
    return {"id": user.id, "username": user.username}


@router.post("/api/auth/login")
def login(body: Login, response: Response, db: DB):
    user = db.scalar(select(User).where(User.username == body.username))
    if not user or not verify_password(body.password, user.password_hash):
        raise DomainError("账号或密码错误", 401)
    hours = get_settings().session_hours
    token = create_session(db, user, hours)
    response.set_cookie("kb_session", token, httponly=True, samesite="strict", max_age=hours * 3600)
    return user_data(user)


@router.get("/api/auth/me")
def me(user: CurrentUser):
    return user_data(user)


@router.post("/api/auth/logout")
def logout(request: Request, response: Response, db: DB):
    token = request.cookies.get("kb_session", "")
    session = db.scalar(select(LoginSession).where(LoginSession.token_hash == token_hash(token)))
    if session:
        db.delete(session)
        db.commit()
    response.delete_cookie("kb_session")
    return {"message": "已退出登录"}


@router.patch("/api/corpora/{corpus_id}")
def rename_corpus(corpus_id: str, body: CorpusCreate, db: DB, user: CurrentUser):
    require_role(db, corpus_id, user.id, "admin")
    if not body.name.strip():
        raise DomainError("知识库名称不能为空")
    corpus = db.get(Corpus, corpus_id)
    corpus.name = body.name.strip()
    db.commit()
    return {"id": corpus.id, "name": corpus.name}


@router.get("/api/corpora/{corpus_id}/members")
def members(corpus_id: str, db: DB, user: CurrentUser):
    require_role(db, corpus_id, user.id, "admin")
    corpus = db.get(Corpus, corpus_id)
    return [{"id": u.id, "username": u.username, "role": m.role, "is_owner": u.id == corpus.owner_id}
            for m, u in db.execute(select(Membership, User).join(User).where(Membership.corpus_id == corpus_id))]


@router.post("/api/corpora/{corpus_id}/members", status_code=201)
def add_member(corpus_id: str, body: MemberCreate, db: DB, user: CurrentUser):
    require_role(db, corpus_id, user.id, "admin")
    member = db.scalar(select(User).where(User.username == body.username))
    if member and db.scalar(select(Membership).where(
            Membership.corpus_id == corpus_id, Membership.user_id == member.id)):
        raise DomainError("该账号已是本知识库成员", 409)
    if not member:
        if not body.password:
            raise DomainError("新账号需要设置至少 8 位密码")
        member = User(username=body.username, password_hash=hash_password(body.password))
        db.add(member)
        db.flush()
    elif body.password:
        raise DomainError("已有账号请留空密码；共享知识库不会重置其密码")
    db.add(Membership(corpus_id=corpus_id, user_id=member.id, role=body.role))
    db.commit()
    return {**user_data(member), "role": body.role}


def editable_member(db, corpus_id, user_id):
    if db.get(Corpus, corpus_id).owner_id == user_id:
        raise DomainError("不能移除或降级知识库创建者", 409)
    membership = db.scalar(select(Membership).where(
        Membership.corpus_id == corpus_id, Membership.user_id == user_id))
    if not membership:
        raise DomainError("成员不存在", 404)
    return membership


@router.patch("/api/corpora/{corpus_id}/members/{user_id}")
def change_role(corpus_id: str, user_id: str, body: RoleUpdate, db: DB, user: CurrentUser):
    require_role(db, corpus_id, user.id, "admin")
    editable_member(db, corpus_id, user_id).role = body.role
    db.commit()
    return {"message": "角色已更新"}


@router.delete("/api/corpora/{corpus_id}/members/{user_id}")
def remove_member(corpus_id: str, user_id: str, db: DB, user: CurrentUser):
    require_role(db, corpus_id, user.id, "admin")
    db.delete(editable_member(db, corpus_id, user_id))
    for run in db.scalars(select(QuestionRun).join(Snapshot).where(
            Snapshot.corpus_id == corpus_id, QuestionRun.owner_id == user_id,
            QuestionRun.status.in_(["queued", "running"])).with_for_update()):
        run.cancel_requested, run.status, run.lease_until = True, "cancelled", None
        emit(db, run.id, run.attempt, "cancelled", {"message": "知识库访问权限已被撤销"})
    db.commit()
    return {"message": "成员已移除"}


@router.patch("/api/documents/{paper_id}")
def edit_document(paper_id: str, body: DocumentUpdate, db: DB, user: CurrentUser):
    paper = db.get(Paper, paper_id)
    if not paper or paper.deleted:
        raise DomainError("文档不存在", 404)
    require_role(db, paper.corpus_id, user.id, "editor")
    if not body.title:
        raise DomainError("标题不能为空")
    paper.title, paper.folder, paper.tags = body.title, body.folder, body.tags
    db.commit()
    return {"id": paper.id, **body.model_dump()}


@router.delete("/api/documents/{paper_id}")
def delete_document(paper_id: str, db: DB, user: CurrentUser):
    paper = db.get(Paper, paper_id)
    if not paper or paper.deleted:
        raise DomainError("文档不存在", 404)
    require_role(db, paper.corpus_id, user.id, "editor")
    paper.deleted = True
    db.commit()
    return {"message": "文档已移出知识库，停止新检索；历史引用仍保留"}


@router.post("/api/question-runs/{run_id}/replay", status_code=202)
def replay(run_id: str, db: DB, user: CurrentUser,
           idempotency_key: Annotated[str, Header(min_length=8, max_length=128)]):
    previous = owned(db, QuestionRun, run_id, user.id)
    if previous.owner_id != user.id:
        raise DomainError("只有问题发起人可以重试", 403)
    if previous.status not in TERMINAL:
        raise DomainError("请等待当前任务结束", 409)
    question = previous.question
    if previous.status == "rejected":
        decision = db.scalar(select(AnswerApprovalRecord).where(AnswerApprovalRecord.run_id == previous.id))
        if decision and decision.reason:
            question += "\n\n请根据人工审核意见修订答案。审核意见：" + decision.reason
    run = create_run(db, RunCreate(snapshot_id=previous.snapshot_id, question=question,
                                   mode=previous.mode, parent_run_id=previous.parent_run_id),
                     idempotency_key, user.id, get_settings(), replay_of=run_id)
    return run_data(db, run, user.id)
