"""Persist isolated test evidence and durable code-repair review feedback."""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade():
    # 0001 and 0005 historically create tables from live application metadata.
    # Fresh databases can therefore already contain these fields before 0007.
    fields = {
        "code_repair_tasks": [
            sa.Column("test_environment", sa.JSON(), nullable=True),
            sa.Column("verification_sha256", sa.String(length=64), nullable=True),
            sa.Column("workspace_state", sa.String(length=24), server_default="active", nullable=False),
        ],
        "code_repair_reviews": [
            sa.Column("bundle_sha256", sa.String(length=64), nullable=True),
            sa.Column("feedback_category", sa.String(length=32), server_default="unclear", nullable=False),
            sa.Column("root_cause_layer", sa.String(length=32), server_default="unknown", nullable=False),
            sa.Column("uncertainty", sa.Text(), nullable=True),
        ],
    }
    for table, columns in fields.items():
        existing = {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}
        for column in columns:
            if column.name not in existing:
                op.add_column(table, column)
    op.execute("UPDATE code_repair_tasks SET workspace_state='restored' WHERE status IN ('failed', 'rejected')")


def downgrade():
    op.drop_column("code_repair_reviews", "uncertainty")
    op.drop_column("code_repair_reviews", "root_cause_layer")
    op.drop_column("code_repair_reviews", "feedback_category")
    op.drop_column("code_repair_reviews", "bundle_sha256")
    op.drop_column("code_repair_tasks", "workspace_state")
    op.drop_column("code_repair_tasks", "verification_sha256")
    op.drop_column("code_repair_tasks", "test_environment")
