"""Keep external observations separate from document source blocks."""
import sqlalchemy as sa
from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade():
    columns = {c["name"]: c for c in sa.inspect(op.get_bind()).get_columns("evidence_spans")}
    with op.batch_alter_table("evidence_spans") as batch:
        if not columns["block_id"]["nullable"]:
            batch.alter_column("block_id", existing_type=sa.String(40), nullable=True)
        if "invocation_id" not in columns:
            batch.add_column(sa.Column("invocation_id", sa.String(40), nullable=True))
            batch.create_foreign_key("fk_evidence_invocation", "mcp_invocations", ["invocation_id"], ["id"])
        if "source_sha256" not in columns:
            batch.add_column(sa.Column("source_sha256", sa.String(64), nullable=True))


def downgrade():
    raise RuntimeError("外部证据不能静默降级为文档引用；请恢复升级前备份")
