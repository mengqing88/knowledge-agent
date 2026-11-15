"""Commit-pinned code repair tasks and artifact-bound human reviews."""
from alembic import op
from app.core import models

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    for name in ("code_repair_tasks", "code_repair_reviews", "repair_experiences"):
        models.Base.metadata.tables[name].create(bind, checkfirst=True)


def downgrade():
    bind = op.get_bind()
    for name in ("repair_experiences", "code_repair_reviews", "code_repair_tasks"):
        models.Base.metadata.tables[name].drop(bind, checkfirst=True)
