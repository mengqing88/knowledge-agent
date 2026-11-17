import hashlib
import math
import re
import unicodedata

from sqlalchemy import func, literal_column, or_, select

from app.core.errors import DomainError, ProviderError
from app.core.models import Chunk, IndexRevision, Snapshot
from app.knowledge.lexical import scoped_bm25
from app.knowledge.llama_rag import ProviderEmbedding, retrieve_revision


class SearchResults(list):
    def __init__(self, values=(), degradation=None):
        super().__init__(values)
        self.degradation = degradation


def query_terms(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+-]*|[\u4e00-\u9fff]{2,}", query)
    return list(dict.fromkeys(w.lower() for w in words))[:24]


def rrf(rankings: list[list[str]], k: int = 60) -> list[dict]:
    fused = {}
    for route, values in enumerate(rankings):
        seen = set()
        for rank, chunk_id in enumerate(values, 1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            item = fused.setdefault(chunk_id, {"chunk_id": chunk_id, "score": 0., "ranks": {}})
            item["score"] += 1 / (k + rank)
            item["ranks"][str(route)] = rank
    return sorted(fused.values(), key=lambda v: (-v["score"], v["chunk_id"]))


def _duplicate_text(text: str) -> str:
    """Normalize canonical Unicode forms and whitespace for duplicate matching."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def collapse_duplicate_hits(db, hits: list[dict]) -> list[dict]:
    """Fold duplicate search hits and retain their IDs in the trace.

    Duplicates within the same index are identified after canonical Unicode
    and whitespace normalization. Different indexes keep their own provenance.
    Compatibility characters, punctuation, operators, and case are preserved because they
    can distinguish values or units. Similar wording is kept separate because
    it may express different conditions. The database rows and source files remain
    untouched; ``duplicate_chunk_ids`` lets callers audit what was folded.
    """
    if len(hits) < 2:
        return hits
    ids = [item["chunk_id"] for item in hits]
    chunks = {chunk.id: chunk for chunk in db.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
    representatives: list[dict] = []
    by_exact: dict[tuple[str, str], dict] = {}
    for hit in hits:
        chunk = chunks.get(hit["chunk_id"])
        if chunk is None:
            representatives.append(hit)
            continue
        normalized = _duplicate_text(chunk.text)
        exact_key = (chunk.index_revision_id, hashlib.sha256(normalized.encode("utf-8")).hexdigest()) if normalized else None
        representative = by_exact.get(exact_key) if exact_key else None
        if representative is not None:
            representative.setdefault("duplicate_chunk_ids", []).append(chunk.id)
            continue
        item = {**hit, "duplicate_chunk_ids": []}
        representatives.append(item)
        if exact_key:
            by_exact[exact_key] = item
    return representatives


def milvus_search(db, indexes, vector, vector_store, limit=30) -> list[str]:
    """Rank Milvus hits and resolve each ID against its own PostgreSQL index."""
    if vector_store is None:
        raise ProviderError("Milvus 向量后端未配置", "vector_store_config")
    semantic_hits = []
    for index in indexes:
        # QueryBundle carries the one already-accounted query embedding, so a
        # multi-document snapshot does not issue another model call per index.
        hits = retrieve_revision(db, index, vector_store, None, "", embedding=vector, limit=limit)
        semantic_hits.extend((hit.node.node_id, hit.score) for hit in hits)
    # Milvus COSINE returns similarity: larger values are more relevant.
    return [chunk_id for chunk_id, _score in sorted(
        semantic_hits, key=lambda item: (-item[1], item[0]))][:limit]


def hybrid_search(db, snapshot: Snapshot, query: str, provider, embedding_model: str, limit=12,
                  expected_index_config: str | None = None, vector_store=None, **lexical_options) -> list[dict]:
    if type(limit) is not int or not 1 <= limit <= 30:
        raise DomainError("检索参数超出范围")
    keyword, semantic, degradation = retrieve_rankings(db, snapshot, query, provider, embedding_model,
                                                      expected_index_config, vector_store, **lexical_options)
    rankings = [keyword] if degradation else [keyword, semantic]
    fused = collapse_duplicate_hits(db, rrf(rankings))
    return SearchResults(fused[:limit], degradation)


def retrieve_rankings(db, snapshot: Snapshot, query: str, provider, embedding_model: str,
                      expected_index_config: str | None = None, vector_store=None, *, lexical_backend="legacy",
                      lexical_max_chunks=5000, lexical_max_chars=2_000_000, lexical_cache=None, guard=None):
    """Shared scoped recall for offline route comparisons; production still uses equal RRF."""
    if (not isinstance(query, str) or not query.strip() or len(query) > 500 or
            not snapshot.manifest):
        raise DomainError("检索参数超出范围")
    if (lexical_backend not in ("legacy", "ngram_bm25") or
            type(lexical_max_chunks) is not int or not 1 <= lexical_max_chunks <= 20000 or
            type(lexical_max_chars) is not int or not 1 <= lexical_max_chars <= 8_000_000):
        raise DomainError("词法检索配置超出范围")
    index_ids = [member["index_revision_id"] for member in snapshot.manifest]
    indexes = list(db.scalars(select(IndexRevision).where(IndexRevision.id.in_(index_ids))))
    if len(indexes) != len(set(index_ids)) or any(i.status != "ready" for i in indexes):
        raise DomainError("快照包含不可用索引", 409)
    if {i.embedding_model for i in indexes} != {embedding_model} or len({i.embedding_dimensions for i in indexes}) != 1:
        raise ProviderError("快照向量模型或维度不兼容；请使用对应配置或重新入库建立新快照")
    if expected_index_config and any(i.config_version != expected_index_config for i in indexes):
        raise ProviderError("Embedding 服务或索引配置已改变；请重新入库并明确创建新快照")
    scope = Chunk.index_revision_id.in_(index_ids)
    terms = query_terms(query)
    if db.bind.dialect.name == "postgresql":
        if any(i.vector_backend != "milvus" or not i.vector_collection for i in indexes):
            raise ProviderError("索引尚未迁入 Milvus，请先运行 migrate-vectors", "vector_store_config")
    lexical = None
    if lexical_backend == "ngram_bm25":
        lexical = scoped_bm25(db, index_ids, query, max_chunks=lexical_max_chunks, max_chars=lexical_max_chars,
                              cache=lexical_cache, guard=guard)
    if db.bind.dialect.name == "postgresql":
        # OR terms preserves recall for method IDs and English expansions; this is native FTS, not BM25.
        tsquery = func.plainto_tsquery(literal_column("'simple'"), terms[0] if terms else query)
        for term in terms[1:]:
            tsquery = tsquery.op("||")(func.plainto_tsquery(literal_column("'simple'"), term))
        match = Chunk.search_vector.op("@@")(tsquery)
        exact = [Chunk.text.ilike(f"%{term.replace('%', '').replace('_', '')}%") for term in terms]
        keyword = lexical if lexical is not None else list(db.scalars(select(Chunk.id).where(
            scope, or_(match, *exact)).order_by(func.ts_rank_cd(Chunk.search_vector, tsquery).desc(), Chunk.id).limit(30)))
        try:
            vector = ProviderEmbedding(provider, embedding_model, indexes[0].embedding_dimensions).get_query_embedding(query)
        except ProviderError as exc:
            if not exc.retryable:
                raise
            return keyword, [], {"code": exc.code, "route": "keyword_only"}
        semantic = milvus_search(db, indexes, vector, vector_store)
    else:
        # Test-only SQLite backend. Normal application startup rejects SQLite.
        chunks = list(db.scalars(select(Chunk).where(scope)))
        if lexical is not None:
            keyword = lexical
        else:
            scores = [(c.id, sum(c.text.lower().count(t) for t in terms)) for c in chunks]
            keyword = [cid for cid, s in sorted(scores, key=lambda x: (-x[1], x[0])) if s > 0][:30]
        try:
            vector = ProviderEmbedding(provider, embedding_model, indexes[0].embedding_dimensions).get_query_embedding(query)
        except ProviderError as exc:
            if not exc.retryable:
                raise
            return keyword, [], {"code": exc.code, "route": "keyword_only"}

        def distance(c):
            a, b = c.embedding, vector
            return 1 - sum(x*y for x, y in zip(a, b)) / (math.sqrt(sum(x*x for x in a)) *
                                                       math.sqrt(sum(y*y for y in b)))
        semantic = [c.id for c in sorted(chunks, key=lambda c: (distance(c), c.id))][:30]
    return keyword, semantic, None
