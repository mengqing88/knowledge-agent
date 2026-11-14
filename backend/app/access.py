"""Knowledge-base role checks shared by API, retrieval, worker and citation reads."""
from sqlalchemy import select

from .errors import DomainError
from .models import Corpus, Membership, Paper, PaperVersion

LEVELS = {"reader": 1, "editor": 2, "admin": 3}


def corpus_role(db, corpus_id, user_id):
    corpus = db.get(Corpus, corpus_id)
    if not corpus:
        raise DomainError("知识库不存在或不可访问", 404)
    if corpus.owner_id == user_id:
        return "admin"
    role = db.scalar(select(Membership.role).where(
        Membership.corpus_id == corpus_id, Membership.user_id == user_id))
    if role not in LEVELS:
        raise DomainError("知识库不存在或不可访问", 404)
    return role


def require_role(db, corpus_id, user_id, minimum="reader"):
    role = corpus_role(db, corpus_id, user_id)
    if LEVELS[role] < LEVELS[minimum]:
        raise DomainError("当前角色没有此操作权限", 403)
    return role


def check_snapshot_access(db, snapshot, user_id, *, current=False):
    require_role(db, snapshot.corpus_id, user_id)
    if current:
        for item in snapshot.manifest:
            version = db.get(PaperVersion, item["paper_version_id"])
            paper = db.get(Paper, version.paper_id) if version else None
            if (not paper or paper.corpus_id != snapshot.corpus_id or paper.deleted or
                    not version.is_active or version.status != "ready"):
                raise DomainError("快照包含已停用或删除的版本，请选择当前文档开始新问题", 409)


def lock_snapshot_publication(db, snapshot, owner_id, reviewer_id=None):
    """Lock existing membership, document and version rows before publication.

    PostgreSQL holds these row locks until commit. Stable ordering reduces
    deadlocks between approvals; a database deadlock still aborts the decision.
    """
    user_ids = sorted({owner_id, *([reviewer_id] if reviewer_id else [])})
    db.scalars(select(Membership).where(
        Membership.corpus_id == snapshot.corpus_id, Membership.user_id.in_(user_ids)
    ).order_by(Membership.user_id).with_for_update()).all()
    version_ids = sorted({item["paper_version_id"] for item in snapshot.manifest})
    # Use the version's actual parent, not the redundant manifest paper ID.
    # Verify the mapping again after locking in case it changed meanwhile.
    parents = dict(db.execute(select(PaperVersion.id, PaperVersion.paper_id).where(
        PaperVersion.id.in_(version_ids))).all())
    paper_ids = sorted(set(parents.values()))
    db.scalars(select(Paper).where(Paper.id.in_(paper_ids)).order_by(Paper.id)
               .with_for_update().execution_options(populate_existing=True)).all()
    versions = db.scalars(select(PaperVersion).where(PaperVersion.id.in_(version_ids)).order_by(PaperVersion.id)
                          .with_for_update().execution_options(populate_existing=True)).all()
    if any(parents.get(version.id) != version.paper_id for version in versions):
        raise DomainError("文档版本归属已变化，请重试审核", 409)
