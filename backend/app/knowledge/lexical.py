"""Dictionary-free Chinese lexical recall over a caller-scoped snapshot."""
import math
import re
from collections import Counter, defaultdict

from sqlalchemy import select

from app.core.errors import ProviderError
from app.core.models import Chunk

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_.+-]*")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def lexical_tokens(text):
    """Han bigrams/trigrams, preserving punctuation boundaries and Latin IDs."""
    output = []
    for piece in TOKEN_PATTERN.findall(text.casefold()):
        if HAN_PATTERN.fullmatch(piece):
            if len(piece) == 1:
                output.append(piece)
            for n in (2, 3):
                output.extend(piece[start:start+n] for start in range(len(piece)-n+1))
        else:
            output.append(piece)
    return output


class LexicalBM25:
    """BM25 over exactly the supplied documents; never reads another index."""

    def __init__(self, documents, guard=None):
        check = guard or (lambda: None)
        self.postings = defaultdict(dict)
        self.lengths = {}
        for number, (key, text) in enumerate(sorted(documents.items())):
            if number % 64 == 0:
                check()
            frequencies = Counter(lexical_tokens(text))
            self.lengths[key] = sum(frequencies.values())
            for term, frequency in frequencies.items():
                self.postings[term][key] = frequency
        count = len(self.lengths)
        self.average_length = sum(self.lengths.values()) / count if count else 0.
        self.idf = {term: math.log(1 + (count - len(hits) + .5) / (len(hits) + .5))
                    for term, hits in self.postings.items()}
        check()

    def search(self, query, limit=30):
        if type(limit) is not int or limit < 1:
            raise ValueError("A positive integer result limit is required")
        if not self.average_length:
            return []
        scores = defaultdict(float)
        for term in sorted(set(lexical_tokens(query))):
            for key, frequency in self.postings.get(term, {}).items():
                length_factor = .25 + .75 * self.lengths[key] / self.average_length
                scores[key] += self.idf[term] * frequency * 2.2 / (frequency + 1.2 * length_factor)
        return sorted(scores, key=lambda key: (-scores[key], key))[:limit]


def scoped_bm25(db, index_ids, query, *, max_chunks=5000, max_chars=2_000_000, cache=None, guard=None):
    """Bounded per-run index. Cache keys include the DB engine and exact scope.

    The caller must revalidate snapshot readiness before every use. This cache
    belongs to one Workflow, never a global service cache; it is released when
    that run ends. Source rows and embeddings are not modified.
    """
    check = guard or (lambda: None)
    key = (db.get_bind(), tuple(sorted(set(index_ids))), max_chunks, max_chars)
    check()
    index = cache.get(key) if cache is not None else None
    if index is None:
        documents, characters = {}, 0
        statement = select(Chunk.id, Chunk.text).where(Chunk.index_revision_id.in_(index_ids)).order_by(
            Chunk.id).limit(max_chunks + 1).execution_options(yield_per=64)
        for number, (chunk_id, text) in enumerate(db.execute(statement), 1):
            if number % 64 == 1:
                check()
            characters += len(text)
            if number > max_chunks or characters > max_chars:
                raise ProviderError("中文词法检索超过本次快照容量上限；请缩小范围或使用 legacy 配置",
                                    "lexical_capacity")
            documents[chunk_id] = text
        index = LexicalBM25(documents, guard=check)
        if cache is not None:
            cache[key] = index
    result = index.search(query)
    check()
    return result
