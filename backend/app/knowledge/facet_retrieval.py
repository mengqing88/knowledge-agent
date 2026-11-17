"""Bounded three-route recall and BGE/facet selection on the current snapshot."""
import hashlib
import math

from sqlalchemy import select
from sqlalchemy.orm import load_only

from app.core.errors import DomainError, ProviderError
from app.core.models import Chunk, IndexRevision, SourceBlock
from app.knowledge.context_selection import (
    WideCandidates,
    query_facets,
    source_successors,
    with_facet_context,
)
from app.knowledge.llama_rag import ProviderEmbedding
from app.knowledge.local_reranker import rerank_local, validate_local_model
from app.knowledge.retrieval import SearchResults, collapse_duplicate_hits, milvus_search


def scoped_context(db, snapshot, embedding_model, expected_index_config, config, cache, guard):
    if not snapshot.manifest:
        raise DomainError("快照为空", 409)
    members = {m["index_revision_id"]: m for m in snapshot.manifest}
    indexes = list(db.scalars(select(IndexRevision).where(IndexRevision.id.in_(members))))
    if len(indexes) != len(members) or any(i.status != "ready" for i in indexes):
        raise DomainError("快照包含不可用索引", 409)
    if any(i.parse_revision_id != members[i.id]["parse_revision_id"] for i in indexes):
        raise DomainError("快照解析版本不一致", 409)
    if {i.embedding_model for i in indexes} != {embedding_model} or len({i.embedding_dimensions for i in indexes}) != 1:
        raise ProviderError("快照向量模型或维度不兼容；请重新入库建立新快照")
    if expected_index_config and any(i.config_version != expected_index_config for i in indexes):
        raise ProviderError("Embedding 服务或索引配置已改变；请重新入库并明确创建新快照")
    if db.bind.dialect.name == "postgresql" and any(
            i.vector_backend != "milvus" or not i.vector_collection for i in indexes):
        raise ProviderError("索引尚未迁入 Milvus，请先运行 migrate-vectors", "vector_store_config")
    max_chunks, max_chars = config["lexical_max_chunks"], config["lexical_max_chars"]
    key = ("facet_context", db.get_bind(), tuple(sorted(members)), max_chunks, max_chars)
    guard()
    value = cache.get(key)
    if value is None:
        chunks, characters = {}, 0
        rows = db.scalars(select(Chunk).options(load_only(
            Chunk.id, Chunk.index_revision_id, Chunk.text, Chunk.source_ranges)).where(
            Chunk.index_revision_id.in_(members)).order_by(Chunk.id).limit(max_chunks + 1).execution_options(yield_per=64))
        for number, chunk in enumerate(rows, 1):
            if number % 64 == 1:
                guard()
            characters += len(chunk.text)
            if number > max_chunks or characters > max_chars:
                raise ProviderError("中文词法检索超过本次快照容量上限；请缩小范围或使用 legacy 配置",
                                    "lexical_capacity")
            chunks[chunk.id] = chunk
        block_ids = {r["block_id"] for c in chunks.values() for r in c.source_ranges}
        blocks = {}
        # Keep SQL parameter batches bounded for SQLite as well as PostgreSQL.
        keys = sorted(block_ids)
        for start in range(0, len(keys), 500):
            guard()
            blocks.update((b.id, b) for b in db.scalars(select(SourceBlock).where(
                SourceBlock.id.in_(keys[start:start+500]))))
        try:
            successors = source_successors(chunks, blocks, {i.id: i.parse_revision_id for i in indexes})
        except (ValueError, KeyError, TypeError) as exc:
            raise DomainError("片段来源不属于当前快照或来源顺序无效", 409) from exc
        ranker = WideCandidates({key: chunk.text for key, chunk in chunks.items()}, guard=guard)
        value = chunks, successors, ranker
        cache[key] = value
    guard()
    return indexes, value


def context_search(db, snapshot, query, provider, embedding_model, expected_index_config, vector_store,
                   config, cache, *, guard, deadline, observe=None):
    if not isinstance(query, str) or not query.strip() or len(query) > 4000:
        raise DomainError("检索表达需要为 1–4000 字符")
    # Fail before any paid embedding when the local model/snapshot cannot run.
    guard()
    validate_local_model(config)
    indexes, (chunks, successors, ranker) = scoped_context(
        db, snapshot, embedding_model, expected_index_config, config, cache, guard)
    if not chunks:
        return SearchResults()
    degradation, dense = None, []
    try:
        vector = ProviderEmbedding(provider, embedding_model, indexes[0].embedding_dimensions,
                                   guard).get_query_embedding(query)
    except ProviderError as exc:
        if not exc.retryable:
            raise
        degradation = {"code": exc.code, "route": "keyword_only"}
    else:
        if db.bind.dialect.name == "postgresql":
            dense = milvus_search(db, indexes, vector, vector_store, limit=100)
        else:
            # Deterministic test/demo path. Production vectors are never read from SQL.
            rows = db.execute(select(Chunk.id, Chunk.embedding).where(Chunk.id.in_(chunks)))
            norm = math.sqrt(sum(v*v for v in vector))
            similarities = []
            for key, embedding in rows:
                value = sum(a*b for a, b in zip(embedding, vector)) / (
                    math.sqrt(sum(v*v for v in embedding)) * norm)
                similarities.append((key, value))
            dense = [key for key, _ in sorted(similarities, key=lambda row: (-row[1], row[0]))[:100]]
    guard()
    if not set(dense) <= set(chunks):
        raise DomainError("向量候选超出当前快照", 403)
    routes, combined = ranker.rankings(query, dense, limit=100)
    if not combined:
        return SearchResults(degradation=degradation)
    candidates = [{"id": key, "text": chunks[key].text} for key in combined]
    questions = [query, *query_facets(query)]
    requests = [{"id": str(i), "question": text, "candidates": candidates} for i, text in enumerate(questions)]
    orders = rerank_local(requests, config, guard=guard, deadline=deadline, observe=observe)
    selected = with_facet_context(orders[0], orders[1:], successors, chunks)
    folded = collapse_duplicate_hits(db, [{"chunk_id": key, "score": 1/(60+rank),
        "ranks": {"context": rank}} for rank, key in enumerate(selected, 1)])
    if observe:
        observe("context_selection", {"strategy": "facet_context", "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "candidate_count": len(combined), "routes": routes, "selected_chunk_ids": [h["chunk_id"] for h in folded],
                "facet_count": len(orders)-1, "selected_count": len(folded), "final_limit": 12})
    return SearchResults(folded, degradation)
