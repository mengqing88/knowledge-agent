"""Persist the expiry time for external operation approvals."""
import sqlalchemy as sa
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("external_operations")}
    if "approval_expires_at" not in columns:
        op.add_column("external_operations", sa.Column(
            "approval_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade():
    columns = {column["name"] for column in sa.inspect(op.get_bind()).get_columns("external_operations")}
    if "approval_expires_at" in columns:
        op.drop_column("external_operations", "approval_expires_at")
