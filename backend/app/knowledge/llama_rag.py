"""LlamaIndex owns vector indexing/retrieval; LangGraph owns the answer workflow.

The vector-store adapter composes the existing Milvus schema with PostgreSQL
source records. It never lets an external vector ID supply unverified text.
"""
from collections.abc import Sequence
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import BaseNode, QueryBundle, TextNode, TransformComponent
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from pydantic import PrivateAttr
from sqlalchemy import select

from app.core.errors import ProviderError
from app.core.models import Chunk


class ProviderEmbedding(BaseEmbedding):
    """Use the project's provider (or Workflow ledger), with explicit per-call state."""

    _provider: Any = PrivateAttr()
    _dimension: int = PrivateAttr(default=0)
    _guard: Any = PrivateAttr()

    def __init__(self, provider, model_name, dimension=0, guard=None):
        # The configured gateway resets response indexes after 8 inputs in a
        # larger batch. Stay within that limit; never relax index validation.
        super().__init__(model_name=model_name, embed_batch_size=8, callback_manager=CallbackManager([]))
        self._provider, self._dimension = provider, dimension
        self._guard = guard or (lambda: None)

    def _get_text_embeddings(self, texts):
        self._guard()
        if self._provider is None:
            raise ProviderError("LlamaIndex 必须使用项目配置的 Embedding 服务", "vector_store_config")
        vectors = self._provider.embed(texts, self._dimension)
        if not self._dimension and vectors:
            self._dimension = len(vectors[0])
        return vectors

    def _get_text_embedding(self, text):
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query):
        return self._get_text_embeddings([query])[0]

    async def _aget_query_embedding(self, query):
        return self._get_query_embedding(query)


class KeepSourceChunks(TransformComponent):
    """Chunks already carry exact source ranges; do not split them a second time."""

    def __call__(self, nodes, **kwargs):
        return nodes


def chunk_node(chunk, embedding=None):
    metadata = {"index_revision_id": chunk.index_revision_id, "source_ranges": chunk.source_ranges}
    return TextNode(id_=chunk.id, text=chunk.text, metadata=metadata, embedding=embedding,
                    excluded_embed_metadata_keys=list(metadata), excluded_llm_metadata_keys=list(metadata))


class RevisionVectorStore(BasePydanticVectorStore):
    """Milvus vectors + SQL text as one LlamaIndex store, scoped to one revision.

    ``stores_text=False`` is intentional: Milvus contains only IDs and vectors.
    Query results hydrate TextNodes from SQL and never require a persisted
    LlamaIndex docstore or a second authoritative copy of source text.
    """

    stores_text: bool = False
    _db: Any = PrivateAttr()
    _index: Any = PrivateAttr()
    _store: Any = PrivateAttr()
    _guard: Any = PrivateAttr()

    def __init__(self, db, index, vector_store, guard=None):
        super().__init__()
        self._db, self._index, self._store = db, index, vector_store
        self._guard = guard or (lambda: None)

    @property
    def client(self):
        return self._store

    def add(self, nodes: Sequence[BaseNode], **kwargs):
        self._guard()
        allowed = set(self._db.scalars(select(Chunk.id).where(
            Chunk.index_revision_id == self._index.id,
            Chunk.id.in_([node.node_id for node in nodes]))))
        if any(node.node_id not in allowed or node.metadata.get("index_revision_id") != self._index.id
               for node in nodes):
            raise ProviderError("LlamaIndex 写入片段不属于当前索引", "vector_store_config")
        rows = [{"chunk_id": node.node_id, "embedding": node.get_embedding()} for node in nodes]
        self._store.insert(self._index.vector_collection, rows)
        ids = [row["chunk_id"] for row in rows]
        self._store.verify_ids(self._index.vector_collection, ids)
        return ids

    def delete(self, ref_doc_id, **kwargs):
        raise ProviderError("不可通过 LlamaIndex 删除不可变文档索引", "vector_store_config")

    def query(self, query: VectorStoreQuery, **kwargs):
        if (query.mode != VectorStoreQueryMode.DEFAULT or query.filters is not None or query.doc_ids):
            raise ProviderError("向量查询范围必须由服务端索引版本确定", "vector_store_config")
        self._guard()
        # A retriever created from a freshly built VectorStoreIndex supplies its
        # node IDs automatically. Permit that restriction after checking SQL
        # ownership; it cannot broaden the revision's collection scope.
        if query.node_ids:
            allowed = set(self._db.scalars(select(Chunk.id).where(
                Chunk.index_revision_id == self._index.id, Chunk.id.in_(query.node_ids))))
            if allowed != set(query.node_ids):
                raise ProviderError("向量查询范围必须由服务端索引版本确定", "vector_store_config")
        hits = self._store.search(self._index.vector_collection, query.query_embedding, query.similarity_top_k)
        chunks = {chunk.id: chunk for chunk in self._db.scalars(select(Chunk).where(
            Chunk.index_revision_id == self._index.id, Chunk.id.in_([key for key, _ in hits])))}
        if query.node_ids:
            chunks = {key: chunk for key, chunk in chunks.items() if key in allowed}
        # COSINE similarities descend. Deduplicate a malformed external result.
        scores = {}
        for key, score in hits:
            if key in chunks:
                scores[key] = max(score, scores.get(key, float("-inf")))
        ids = sorted(scores, key=lambda key: (-scores[key], key))[:query.similarity_top_k]
        return VectorStoreQueryResult(nodes=[chunk_node(chunks[key]) for key in ids], ids=ids,
                                      similarities=[scores[key] for key in ids])


def embed_chunks(chunks, provider, model_name, guard=None):
    return ProviderEmbedding(provider, model_name, guard=guard).get_text_embedding_batch(
        [chunk["text"] for chunk in chunks])


def vector_index(db, index, store, provider, *, nodes=None, guard=None):
    adapter = RevisionVectorStore(db, index, store, guard)
    return VectorStoreIndex(
        nodes=nodes or [], storage_context=StorageContext.from_defaults(vector_store=adapter),
        embed_model=ProviderEmbedding(provider, index.embedding_model, index.embedding_dimensions, guard),
        insert_batch_size=128, callback_manager=CallbackManager([]), transformations=[KeepSourceChunks()],
    )


def index_chunks(db, index, chunks, vectors, store, provider, guard=None):
    if len(chunks) != len(vectors):
        raise ProviderError("索引片段与向量条数不匹配", "vector_dimension")
    store.ensure_collection(index.vector_collection, index.embedding_dimensions)
    nodes = [chunk_node(chunk, vector) for chunk, vector in zip(chunks, vectors)]
    return vector_index(db, index, store, provider, nodes=nodes, guard=guard)


def retrieve_revision(db, index, store, provider, query, *, embedding=None, limit=30):
    if index.vector_backend != "milvus" or not index.vector_collection:
        raise ProviderError("索引尚未迁入 Milvus，请先运行 migrate-vectors", "vector_store_config")
    retriever = vector_index(db, index, store, provider).as_retriever(similarity_top_k=limit)
    return retriever.retrieve(QueryBundle(query_str=query, embedding=embedding))
