"""Team membership, authentication and document metadata (also upgrades existing demos)."""
from alembic import op
from app.core import models
from sqlalchemy import inspect, text

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    for table in (models.User.__table__, models.LoginSession.__table__, models.Membership.__table__):
        table.create(bind, checkfirst=True)
    # 0001 historically uses live metadata; guard both fresh and existing installations.
    for name, columns in {
        "papers": ["folder", "tags", "deleted"],
        "paper_versions": ["is_active"],
        "source_blocks": ["section"],
    }.items():
        existing = {col["name"] for col in inspect(bind).get_columns(name)}
        for col in columns:
            if col not in existing:
                op.add_column(name, models.Base.metadata.tables[name].c[col]._copy())
    bind.execute(text("UPDATE paper_versions SET is_active = false WHERE id NOT IN "
                      "(SELECT id FROM (SELECT id, ROW_NUMBER() OVER "
                      "(PARTITION BY paper_id ORDER BY created_at DESC, id DESC) AS rn "
                      "FROM paper_versions WHERE status = 'ready') ranked WHERE rn = 1)"))


def downgrade():
    raise RuntimeError("Team data cannot be downgraded without an explicit data migration")
