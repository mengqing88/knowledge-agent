"""Serialize task admission and capacity checks across API/worker processes."""
from alembic import op
from app.models import WorkloadGate

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade():
    WorkloadGate.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    op.drop_table("workload_gate")
