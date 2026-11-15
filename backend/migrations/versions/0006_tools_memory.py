"""Explicit external operations and user-confirmed private memory."""
from alembic import op
from app.core.models import ExternalOperation, ExternalOperationEvent, MCPInvocation, MemoryEntry, MemoryEvent

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade():
    for model in (ExternalOperation, ExternalOperationEvent, MCPInvocation, MemoryEntry, MemoryEvent):
        model.__table__.create(op.get_bind(), checkfirst=True)


def downgrade():
    raise RuntimeError("审批、回执与记忆审计保留；降级前需显式规划数据备份和兼容迁移")
