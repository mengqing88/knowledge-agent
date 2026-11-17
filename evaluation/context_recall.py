"""Label-free, bounded candidate generation and forward source context selection.

This experimental path accepts caller-scoped chunks, blocks and cached vectors.
It does not read evidence labels, credentials or a model provider.
"""
import math
from collections import defaultdict

from app.knowledge.evidence import validate_range
from app.knowledge.lexical import LexicalBM25, lexical_windows


class WideCandidates:
    def __init__(self, texts):
        self.full = LexicalBM25(texts)
        windows, self.parents = {}, {}
        for chunk_id, text in sorted(texts.items()):
            for number, window in enumerate(lexical_windows(text)):
                key = f"{chunk_id}:{number}"
                windows[key], self.parents[key] = window, chunk_id
        self.window = LexicalBM25(windows)

    def rankings(self, query, dense, limit=100):
        if type(limit) is not int or not 1 <= limit <= 150:
            raise ValueError("Route limit must be between 1 and 150")
        full = self.full.search(query, limit)
        short = []
        if self.parents:
            short = list(dict.fromkeys(self.parents[key] for key in self.window.search(query, len(self.parents))))
        routes = {"full": full, "window": short[:limit], "dense": dense[:limit]}
        combined = list(dict.fromkeys(
            routes[name][rank] for rank in range(limit) for name in routes if rank < len(routes[name])))
        return routes, combined


def source_successors(chunks, blocks, index_parse_ids):
    """Follow original source coordinates, never UUID order or another index."""
    grouped = defaultdict(list)
    for chunk_id, chunk in chunks.items():
        parse_id = index_parse_ids.get(chunk.index_revision_id)
        if parse_id is None or not chunk.source_ranges:
            raise ValueError("Chunk has no valid scoped index or source ranges")
        positions = []
        for span in chunk.source_ranges:
            block = blocks.get(span["block_id"])
            if block is None or block.parse_revision_id != parse_id:
                raise ValueError("Chunk source is outside its scoped parse revision")
            start, end = span["char_start"], span["char_end"]
            if type(start) is not int or type(end) is not int:
                raise ValueError("Source offsets must be integers")
            validate_range(block.text, start, end)
            positions.append((block.page_index, block.block_index, start))
        grouped[chunk.index_revision_id].append((min(positions), chunk_id))
    successors = {}
    for positions in grouped.values():
        ordered = [chunk_id for _position, chunk_id in sorted(positions)]
        successors.update(zip(ordered, ordered[1:]))
    return successors


def with_forward_context(ranking, successors, allowed_ids, seed_count=6):
    """Interleave each leading hit with one following chunk, then backfill.

    The caller folds exact duplicates and applies the unchanged final top-12
    limit. A neighbor consumes a result slot, just like a direct hit.
    """
    if type(seed_count) is not int or not 1 <= seed_count <= 15:
        raise ValueError("Seed count must be between 1 and 15")
    allowed = set(allowed_ids)
    if (not set(ranking) <= allowed or not set(successors) <= allowed or
            not set(successors.values()) <= allowed or len(ranking) != len(set(ranking))):
        raise ValueError("Ranking or successors are duplicated or outside scope")
    leading = []
    for chunk_id in ranking[:seed_count]:
        leading.append(chunk_id)
        if chunk_id in successors:
            leading.append(successors[chunk_id])
    return list(dict.fromkeys(leading + list(ranking)))


def validated_order(result, request, digest):
    """Reject incomplete, foreign, nonfinite or unbound inference results."""
    expected = {candidate["id"] for candidate in request["candidates"]}
    if (result.get("id") != request["id"] or result.get("request_sha256") != digest or
            len(expected) != len(request["candidates"])):
        raise ValueError("Reranker input identity differs")
    scores = result.get("scores")
    if not isinstance(scores, list) or len(scores) != len(expected):
        raise ValueError("Reranker omitted candidates")
    seen = set()
    for item in scores:
        key, value = item.get("id"), item.get("score")
        if (key not in expected or key in seen or type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError("Reranker returned an invalid candidate or score")
        seen.add(key)
    return [row["id"] for row in sorted(scores, key=lambda row: (-row["score"], row["id"]))]
