from app.db import Base
from app.models import Chunk
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable


def test_postgres_schema_and_vector_query_compile():
    dialect = postgresql.dialect()
    tables = {table.name: str(CreateTable(table).compile(dialect=dialect))
              for table in Base.metadata.sorted_tables}
    assert "embedding JSON" in tables["chunks"]
    assert "search_vector TSVECTOR" in tables["chunks"]
    assert "vector_backend VARCHAR(24)" in tables["index_revisions"]
    assert "vector_collection VARCHAR(120)" in tables["index_revisions"]
    indexes = [str(CreateIndex(index).compile(dialect=dialect)) for index in Chunk.__table__.indexes]
    assert any("USING gin (search_vector)" in index for index in indexes)
    stmt = select(Chunk.id).where(Chunk.index_revision_id.in_(["allowed-index"])).limit(12)
    compiled = str(stmt.compile(dialect=dialect))
    assert "index_revision_id IN" in compiled
