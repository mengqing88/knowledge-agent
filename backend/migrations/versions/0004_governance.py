"""Feedback provenance and durable request reservations; preserve historical records."""
from alembic import op
from app.core import models
from sqlalchemy import inspect

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    fields = {
        "review_records": ["category", "answer_sha256", "idempotency_key"],
        "model_calls": ["status", "usage_known", "estimated_input_tokens", "reserved_tokens", "provider_request_id"],
    }
    for name, columns in fields.items():
        existing = {c["name"] for c in inspect(bind).get_columns(name)}
        for column in columns:
            if column not in existing:
                op.add_column(name, models.Base.metadata.tables[name].c[column]._copy())
    # 0001 creates from live metadata. Existing installations need this index;
    # fresh installations already have an equivalent unique constraint.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS ix_review_request "
               "ON review_records (owner_id, idempotency_key)")


def downgrade():
    raise RuntimeError("保留反馈与调用审计；回退应用前需显式规划兼容迁移")
