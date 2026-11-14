"""Record Milvus identity and make the old PostgreSQL vector payload portable.

Existing indexes are marked ``legacy_pgvector`` until the explicit
``migrate-vectors`` command uploads and verifies every chunk.
"""
import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def _columns(table):
    return {column["name"]: column for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade():
    bind = op.get_bind()
    index_columns = _columns("index_revisions")
    if "vector_backend" not in index_columns:
        op.add_column("index_revisions", sa.Column("vector_backend", sa.String(length=24),
                                                     nullable=False, server_default="milvus"))
    if "vector_collection" not in index_columns:
        op.add_column("index_revisions", sa.Column("vector_collection", sa.String(length=120),
                                                     nullable=True))

    # Old installations may still have pgvector's VECTOR column. Convert it to
    # JSON so the uploader can read it and successful migration can clear it.
    if bind.dialect.name == "postgresql":
        type_name = bind.execute(sa.text(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'chunks' "
            "AND column_name = 'embedding'"
        )).scalar_one()
        if type_name == "vector":
            op.execute(sa.text(
                "ALTER TABLE chunks ALTER COLUMN embedding TYPE JSON "
                "USING embedding::text::json"
            ))
        op.execute(sa.text("ALTER TABLE chunks ALTER COLUMN embedding DROP NOT NULL"))

    # Never invent a collection name for data that has not been uploaded.
    op.execute(sa.text(
        "UPDATE index_revisions SET vector_backend = 'legacy_pgvector' "
        "WHERE vector_collection IS NULL"
    ))


def downgrade():
    raise RuntimeError("Milvus 数据迁移不可通过 schema downgrade 撤销；请恢复迁移前的 PostgreSQL 备份")
