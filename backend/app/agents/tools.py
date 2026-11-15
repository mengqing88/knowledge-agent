from sqlalchemy import select

from app.core.errors import DomainError
from app.core.models import SourceBlock
from app.knowledge.evidence import allowed_parse_ids, get_or_create_span
from app.knowledge.retrieval import hybrid_search


class ScopedTools:
    """No caller-supplied owner, snapshot or database scope. Context is injected by the server."""

    def __init__(self, db, snapshot, run_id, provider, embedding_model, expected_index_config=None,
                 vector_store=None):
        self.db, self.snapshot, self.run_id = db, snapshot, run_id
        self.provider, self.embedding_model = provider, embedding_model
        self.expected_index_config = expected_index_config
        self.vector_store = vector_store

    def search_evidence(self, query: str, limit=12):
        return hybrid_search(self.db, self.snapshot, query, self.provider, self.embedding_model, limit,
                             self.expected_index_config, self.vector_store)

    def read_source_blocks(self, block_ids: list[str]):
        if (not isinstance(block_ids, list) or not 1 <= len(block_ids) <= 24 or
                any(not isinstance(bid, str) or not bid.strip() for bid in block_ids)):
            raise DomainError("来源块参数需要为 1–24 个非空字符串 ID")
        blocks = list(self.db.scalars(select(SourceBlock).where(
            SourceBlock.id.in_(block_ids), SourceBlock.parse_revision_id.in_(allowed_parse_ids(self.snapshot)))))
        if {b.id for b in blocks} != set(block_ids):
            raise DomainError("来源不属于当前快照", 403)
        if sum(len(b.text) for b in blocks) > 40000:
            raise DomainError("来源读取量超出上限")
        return blocks

    def get_evidence_span(self, block_id: str, char_start: int, char_end: int):
        return get_or_create_span(self.db, self.run_id, self.snapshot, block_id, char_start, char_end)
