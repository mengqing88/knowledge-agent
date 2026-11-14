"""Source-preserving chunk policies with a named reference tokenizer.

All offsets refer to the original SourceBlock, including hard length splits.
Semantic boundaries use the project's provider, never framework defaults.
"""
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache

import tiktoken

from .errors import ProviderError

VERSION = "source-token-v4"
TOKENIZER = "cl100k_base"


def clean(text):
    text = re.sub(r"([A-Za-z])-\r?\n([a-z])", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def tokenizer():
    return tiktoken.get_encoding(TOKENIZER)


def token_count(text):
    return len(tokenizer().encode(text, disallowed_special=()))


@dataclass(frozen=True)
class ChunkPolicy:
    strategy: str = "fixed"
    size: int = 512
    overlap: int = 64
    semantic_threshold: float = .55
    semantic_max_units: int = 128

    def __post_init__(self):
        if self.strategy not in {"legacy", "fixed", "paragraph", "structure", "semantic"}:
            raise ValueError("未知分块策略")
        if not 32 <= self.size <= 4096 or not 0 <= self.overlap < self.size:
            raise ValueError("分块长度需为 32–4096，重叠需小于分块长度")
        if not 0 <= self.semantic_threshold <= 1:
            raise ValueError("语义边界阈值需为 0–1")
        if not 1 <= self.semantic_max_units <= 1024:
            raise ValueError("语义切分句段上限需为 1–1024")

    @classmethod
    def from_settings(cls, settings):
        return cls(settings.chunk_strategy, settings.chunk_size_tokens, settings.chunk_overlap_tokens,
                   settings.chunk_semantic_threshold, settings.chunk_semantic_max_units)

    def manifest(self):
        return {"version": VERSION, "tokenizer": TOKENIZER, **asdict(self)}


def searchable_blocks(blocks, page_heights, guard):
    def margin(block):
        height = page_heights.get(block.page_index, 842)
        return len(block.bbox) >= 4 and (block.bbox[3] < height * .09 or block.bbox[1] > height * .91)

    pages = defaultdict(set)
    for block in blocks:
        guard()
        if margin(block):
            pages[clean(block.text)].add(block.page_index)
    for block in blocks:
        guard()
        if not (margin(block) and len(pages[clean(block.text)]) >= 3):
            yield block


def _text(parts):
    return "\n\n".join(clean(block.text[start:end]) for block, start, end in parts)


def _label(block):
    section = block.section or ""
    if not section or section == "正文":
        return ""
    # Keep JSON paths/sheet-row locations searchable without changing source
    # offsets. Very long headings cannot consume the entire chunk budget.
    while token_count(section) > 16:
        section = section[:max(1, len(section) // 2)]
    return f"[{section}]\n\n"


def _chunk(parts):
    # Merge adjacent sentence ranges so citations do not fragment into one span
    # per sentence. Overlaps between different chunks remain explicit.
    merged = []
    for block, start, end in parts:
        if merged and merged[-1][0].id == block.id and merged[-1][2] == start:
            merged[-1] = (block, merged[-1][1], end)
        else:
            merged.append((block, start, end))
    return {"text": _label(merged[0][0]) + _text(merged), "source_ranges": [
        {"block_id": block.id, "char_start": start, "char_end": end} for block, start, end in merged]}


def _prefix_end(text, start, end, budget, guard):
    """Find a Unicode-safe prefix. Count the rendered text, not decoded tokens."""
    low, high = start + 1, end
    best = start
    while low <= high:
        guard()
        middle = (low + high) // 2
        if token_count(clean(text[start:middle])) <= budget:
            best, low = middle, middle + 1
        else:
            high = middle - 1
    if best == start:
        raise ValueError("单字符超过分块 token 上限")
    return best


def _bounded(block, start, end, size, guard):
    while start < end:
        guard()
        stop = end if token_count(clean(block.text[start:end])) <= size else _prefix_end(
            block.text, start, end, size, guard)
        if clean(block.text[start:stop]):
            yield block, start, stop
        start = stop


def _units(block, policy, guard):
    # Table rows and JSON scalars are already separate SourceBlocks/sections.
    # Keep their raw location; only an oversized row/field needs a hard split.
    section = block.section or ""
    structured = (" · 第 " in section or " · 表头" in section or " · 单元格 " in section)
    size = policy.size - token_count(_label(block)) - 2
    if policy.strategy in {"fixed", "paragraph"} or structured:
        yield from _bounded(block, 0, len(block.text), size, guard)
        return
    start = 0
    for match in re.finditer(r"[。！？!?]+[’”\"']*\s*|\.(?=\s|$)\s*|\n\s*\n", block.text):
        guard()
        end = match.end()
        yield from _bounded(block, start, end, size, guard)
        start = end
    yield from _bounded(block, start, len(block.text), size, guard)


def _fixed(units, policy, guard):
    """Fill token windows, splitting original ranges at Unicode characters."""
    chunks, current, fresh = [], [], False
    for block, start, end in units:
        while start < end:
            guard()
            if not clean(block.text[start:end]):
                break
            if token_count(_chunk([*current, (block, start, end)])["text"]) <= policy.size:
                current.append((block, start, end))
                fresh = True
                break
            low, high, best = start + 1, end, start
            while low <= high:
                guard()
                middle = (low + high) // 2
                if token_count(_chunk([*current, (block, start, middle)])["text"]) <= policy.size:
                    best, low = middle, middle + 1
                else:
                    high = middle - 1
            if best > start and clean(block.text[start:best]):
                current.append((block, start, best))
                start, fresh = best, True
            if fresh:
                chunks.append(_chunk(current))
                current, fresh = _tail(current, policy.overlap, guard), False
            elif current:
                current = []
            else:
                raise ValueError("分块 token 上限不足以容纳单字符及位置标签")
    if fresh:
        chunks.append(_chunk(current))
    return chunks


def _tail(parts, budget, guard):
    if not budget:
        return []
    result = []
    for block, start, end in reversed(parts):
        guard()
        if token_count(_text([(block, start, end), *result])) <= budget:
            result.insert(0, (block, start, end))
            continue
        # Retain a bounded source suffix when a sentence alone exceeds overlap.
        low, high, best = start, end - 1, end
        while low <= high:
            middle = (low + high) // 2
            if token_count(_text([(block, middle, end), *result])) <= budget:
                best, high = middle, middle - 1
            else:
                low = middle + 1
        if best < end and clean(block.text[best:end]):
            result.insert(0, (block, best, end))
        break
    return result


def _similarity(a, b):
    denominator = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not denominator:
        raise ProviderError("语义切分向量无效", "vector_dimension")
    return sum(x*y for x, y in zip(a, b)) / denominator


def split_blocks(blocks, page_heights, policy, *, provider=None, model_name="", guard=None):
    guard = guard or (lambda: None)
    groups = []
    for block in searchable_blocks(blocks, page_heights, guard):
        location = (block.page_index, block.section)
        if not groups or groups[-1][0] != location:
            groups.append((location, []))
        groups[-1][1].extend(_units(block, policy, guard))
    if policy.strategy == "semantic" and sum(len(units) for _, units in groups) > policy.semantic_max_units:
        raise ProviderError("语义分块句段数超过配置上限；本次未发送边界 Embedding", "chunk_semantic_limit")
    chunks = []
    for _location, units in groups:
        guard()
        if policy.strategy == "fixed":
            chunks.extend(_fixed(units, policy, guard))
            continue
        if policy.strategy == "paragraph":
            # Independent source paragraphs/rows do not overlap. An oversized
            # paragraph still gets proper sliding windows within that source.
            paragraphs = []
            for unit in units:
                if not paragraphs or paragraphs[-1][-1][0].id != unit[0].id:
                    paragraphs.append([])
                paragraphs[-1].append(unit)
            for paragraph in paragraphs:
                chunks.extend(_fixed(paragraph, policy, guard))
            continue
        vectors = None
        if policy.strategy == "semantic" and units:
            from .llama_rag import embed_chunks
            vectors = embed_chunks([{"text": _text([unit])} for unit in units], provider, model_name, guard)
        current = []
        for number, unit in enumerate(units):
            guard()
            topic_break = (vectors is not None and number > 0 and
                           _similarity(vectors[number - 1], vectors[number]) < policy.semantic_threshold)
            overflow = current and token_count(_chunk([*current, unit])["text"]) > policy.size
            if current and (overflow or topic_break):
                chunks.append(_chunk(current))
                current = [] if topic_break else _tail(current, policy.overlap, guard)
                # Overlap is a ceiling: drop its oldest ranges if keeping them
                # would force the new source unit beyond the hard token limit.
                while current and token_count(_chunk([*current, unit])["text"]) > policy.size:
                    current.pop(0)
            current.append(unit)
        if current:
            chunks.append(_chunk(current))
    return chunks
