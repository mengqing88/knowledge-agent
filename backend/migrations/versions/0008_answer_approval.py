"""Require human approval before publishing knowledge-base answers."""
import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "answer_approval_records" not in existing:
        op.create_table(
            "answer_approval_records",
            sa.Column("id", sa.String(length=40), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("run_id", sa.String(length=40), sa.ForeignKey("question_runs.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("answer_id", sa.String(length=40), sa.ForeignKey("answer_revisions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("reviewer_id", sa.String(length=40), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("decision", sa.String(length=12), nullable=False),
            sa.Column("answer_sha256", sa.String(length=64), nullable=False),
            sa.Column("reason", sa.Text(), nullable=True),
            sa.UniqueConstraint("run_id", name="uq_answer_approval_records_run_id"),
            sa.UniqueConstraint("answer_id", name="uq_answer_approval_records_answer_id"),
        )
        op.create_index("ix_answer_approval_records_run_id", "answer_approval_records", ["run_id"])
        op.create_index("ix_answer_approval_records_answer_id", "answer_approval_records", ["answer_id"])
        op.create_index("ix_answer_approval_records_reviewer_id", "answer_approval_records", ["reviewer_id"])
    if "accepted_answer_experiences" not in existing:
        op.create_table(
            "accepted_answer_experiences",
            sa.Column("id", sa.String(length=40), primary_key=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("run_id", sa.String(length=40), sa.ForeignKey("question_runs.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("answer_id", sa.String(length=40), sa.ForeignKey("answer_revisions.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("corpus_id", sa.String(length=40), sa.ForeignKey("corpora.id", ondelete="CASCADE"),
                      nullable=False),
            sa.Column("reviewer_id", sa.String(length=40), sa.ForeignKey("users.id"), nullable=False),
            sa.Column("answer_sha256", sa.String(length=64), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.UniqueConstraint("run_id", name="uq_accepted_answer_experiences_run_id"),
        )
        op.create_index("ix_accepted_answer_experiences_run_id", "accepted_answer_experiences", ["run_id"])
        op.create_index("ix_accepted_answer_experiences_answer_id", "accepted_answer_experiences", ["answer_id"])
        op.create_index("ix_accepted_answer_experiences_corpus_id", "accepted_answer_experiences", ["corpus_id"])
        op.create_index("ix_accepted_answer_experiences_reviewer_id", "accepted_answer_experiences", ["reviewer_id"])


def downgrade():
    existing = set(sa.inspect(op.get_bind()).get_table_names())
    if "accepted_answer_experiences" in existing:
        op.drop_table("accepted_answer_experiences")
    if "answer_approval_records" in existing:
        op.drop_table("answer_approval_records")
