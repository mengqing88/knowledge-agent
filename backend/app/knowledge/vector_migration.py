"""Restartable transfer of immutable legacy chunk vectors to Milvus."""
import math

from sqlalchemy import null, or_, select, update

from app.core.errors import DomainError, ProviderError
from app.core.models import Chunk, IndexRevision
from app.knowledge.vector_store import collection_name


def migrate_vectors(session_factory, settings, vector_store, *, index_ids=None, batch_size=128):
    """Publish one index only after every batch is written and verified.

    Stable chunk IDs and collections make retries idempotent. A failed SQL
    transaction restores all original vectors and leaves the index unpublished.
    The CLI restricts this maintenance command to PostgreSQL; tests may use SQLite.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise DomainError("向量回填批次必须在 1–512 之间")
    with session_factory() as db:
        pending = select(IndexRevision.id).where(
            IndexRevision.status == "ready",
            or_(IndexRevision.vector_backend != "milvus", IndexRevision.vector_collection.is_(None)),
        ).order_by(IndexRevision.id)
        if index_ids:
            pending = pending.where(IndexRevision.id.in_(index_ids))
        pending_ids = list(db.scalars(pending))

    migrated = []
    for index_id in pending_ids:
        with session_factory.begin() as db:
            index = db.scalar(select(IndexRevision).where(IndexRevision.id == index_id).with_for_update())
            if index.vector_backend == "milvus" and index.vector_collection:
                continue
            name = collection_name(index.id, settings.milvus_collection_prefix)
            cursor, total = "", 0
            while True:
                chunks = list(db.scalars(select(Chunk).where(
                    Chunk.index_revision_id == index.id, Chunk.id > cursor,
                ).order_by(Chunk.id).limit(batch_size)))
                if not chunks:
                    break
                rows = []
                for chunk in chunks:
                    vector = chunk.embedding
                    valid = (isinstance(vector, list) and len(vector) == index.embedding_dimensions and
                             all(type(value) in (int, float) and math.isfinite(value) for value in vector) and
                             any(value != 0 for value in vector))
                    if not valid:
                        raise ProviderError("旧索引向量缺失、维度错误或含无效值；保留原数据并停止回填",
                                            "vector_migration_invalid")
                    rows.append({"chunk_id": chunk.id, "embedding": vector})
                if total == 0:
                    vector_store.ensure_collection(name, index.embedding_dimensions)
                vector_store.insert(name, rows)
                ids = [row["chunk_id"] for row in rows]
                vector_store.verify_ids(name, ids)
                db.execute(update(Chunk).where(Chunk.id.in_(ids)).values(embedding=null()))
                total += len(rows)
                cursor = chunks[-1].id
            if total == 0:
                raise ProviderError("旧索引没有可回填片段，未切换 Milvus", "vector_migration_invalid")
            index.vector_backend, index.vector_collection = "milvus", name
        migrated.append({"index_revision_id": index_id, "collection": name, "chunks": total})
    return {"migrated_indexes": len(migrated), "migrated_chunks": sum(i["chunks"] for i in migrated),
            "indexes": migrated}
