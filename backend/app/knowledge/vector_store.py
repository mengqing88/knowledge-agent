"""Vector storage adapters.

Milvus is the production vector backend.  The SQLite adapter is deliberately
kept for deterministic unit tests and the local demo; it never claims to be a
Milvus integration test.
"""

from __future__ import annotations

import math
from typing import Any

from app.core.errors import ProviderError


def collection_name(index_id: str, prefix: str = "paperqa") -> str:
    """Return a deterministic, Milvus-safe collection name for one index revision."""
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(index_id))[:48]
    return f"{prefix}_{safe}"


class MilvusVectorStore:
    """Small wrapper around ``MilvusClient`` used by ingestion and retrieval."""

    def __init__(self, settings):
        self.settings = settings
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as exc:  # pragma: no cover - depends on deployment extras
            raise ProviderError("Milvus 客户端依赖未安装", "vector_store_unavailable") from exc
        self._DataType = DataType
        self.timeout = settings.milvus_timeout_seconds
        kwargs = {"uri": settings.milvus_uri, "timeout": settings.milvus_timeout_seconds}
        if settings.milvus_token:
            kwargs["token"] = settings.milvus_token
        if settings.milvus_db_name:
            kwargs["db_name"] = settings.milvus_db_name
        try:
            self.client = MilvusClient(**kwargs)
        except Exception as exc:  # pragma: no cover - requires a live Milvus service
            raise ProviderError("Milvus 服务不可用", "vector_store_unavailable", retryable=True) from exc

    def collection_for(self, index_id: str) -> str:
        return collection_name(index_id, self.settings.milvus_collection_prefix)

    def ensure_collection(self, name: str, dimension: int) -> None:
        if dimension <= 0:
            raise ProviderError("Milvus 向量维度必须为正数", "vector_dimension")
        try:
            if self.client.has_collection(collection_name=name, timeout=self.timeout):
                description = self.client.describe_collection(collection_name=name, timeout=self.timeout)
                fields = description.get("fields", []) if isinstance(description, dict) else []
                vector_field = next((field for field in fields if field.get("name") == "embedding"), None)
                configured_dimension = (vector_field or {}).get("params", {}).get("dim")
                if configured_dimension is None or int(configured_dimension) != dimension:
                    raise ProviderError("Milvus collection 向量维度不匹配", "vector_dimension")
                self.client.load_collection(collection_name=name, timeout=self.timeout)
                return
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("chunk_id", self._DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", self._DataType.FLOAT_VECTOR, dim=dimension)
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=self.settings.milvus_index_type,
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.client.create_collection(collection_name=name, schema=schema, index_params=index_params,
                                         timeout=self.timeout)
            self.client.load_collection(collection_name=name, timeout=self.timeout)
        except ProviderError:
            raise
        except Exception as exc:  # pragma: no cover - requires a live Milvus service
            raise ProviderError("Milvus collection 创建失败", "vector_store_write", retryable=True) from exc

    def insert(self, name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        try:
            result = self.client.upsert(collection_name=name, data=rows, timeout=self.timeout)
            if isinstance(result, dict) and result.get("upsert_count") not in (None, len(rows)):
                raise ValueError("upsert_count mismatch")
            self.client.flush(collection_name=name, timeout=self.timeout)
        except Exception as exc:  # pragma: no cover - requires a live Milvus service
            raise ProviderError("Milvus 向量写入失败", "vector_store_write", retryable=True) from exc

    def search(self, name: str, vector: list[float], limit: int = 30) -> list[tuple[str, float]]:
        try:
            result = self.client.search(
                collection_name=name,
                data=[vector],
                anns_field="embedding",
                limit=limit,
                output_fields=["chunk_id"],
                search_params={"metric_type": "COSINE", "params": {"ef": 64}},
                consistency_level="Strong",
                timeout=self.timeout,
            )
            hits = result[0] if result else []
            values = []
            for hit in hits:
                entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
                chunk_id = entity.get("chunk_id") or hit.get("id")
                distance = hit.get("distance", 0.0)
                if isinstance(chunk_id, str) and math.isfinite(float(distance)):
                    values.append((chunk_id, float(distance)))
            return values
        except Exception as exc:  # pragma: no cover - requires a live Milvus service
            raise ProviderError("Milvus 向量检索失败", "vector_store_read", retryable=True) from exc

    def verify_ids(self, name: str, chunk_ids: list[str]) -> None:
        try:
            rows = self.client.get(collection_name=name, ids=chunk_ids, output_fields=["chunk_id"],
                                   consistency_level="Strong", timeout=self.timeout)
            if {row["chunk_id"] for row in rows} != set(chunk_ids):
                raise ValueError("Milvus chunk verification mismatch")
        except Exception as exc:
            raise ProviderError("Milvus 向量写入核对失败", "vector_store_write", retryable=True) from exc

    def close(self) -> None:
        self.client.close()

    def health(self) -> dict:
        try:
            collections = self.client.list_collections(timeout=self.timeout)
            return {"status": "ok", "collections": len(collections)}
        except Exception:  # pragma: no cover - requires a live Milvus service
            return {"status": "failed", "error": "Milvus 服务不可用"}


def production_store(settings, db) -> MilvusVectorStore | None:
    """Create the configured production store; SQLite remains test-only local retrieval."""
    if db.bind.dialect.name != "postgresql":
        return None
    if settings.vector_store != "milvus":
        raise ProviderError("生产向量后端必须配置为 Milvus", "vector_store_config")
    return MilvusVectorStore(settings)
