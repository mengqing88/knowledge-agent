"""Experimental Chinese lexical recall; never changes application search defaults.

The ranker accepts only text and IDs. Source evidence labels belong to the
benchmark scorer and cannot affect tokenization, IDF, or ranking.
"""
import re

from app.knowledge.lexical import LexicalBM25 as LexicalBM25
from app.knowledge.lexical import lexical_tokens as lexical_tokens


def query_clauses(question):
    """Additional diagnostic queries; the benchmark always retains the original.

    This syntactic split does not resolve pronouns or infer missing entities.
    It is intentionally separate from the application's guarded query rewriter.
    """
    return list(dict.fromkeys(part.strip(" ，,：: ") for part in re.split(r"[？?；;。]+", question)
                              if len(part.strip()) >= 5))


def child_windows(text):
    """Short lexical windows mapped back to the unchanged source chunk."""
    if not text.strip():
        return []
    buffer, windows = "", []
    for unit in (piece for piece in re.split(r"(?<=[。！？!?\n])", text) if piece):
        if len(buffer) + len(unit) > 160 and buffer:
            windows.append(buffer)
            buffer = ""
        if len(unit) > 200:
            windows.extend(unit[start:start+160] for start in range(0, len(unit), 130))
        else:
            buffer += unit
    if buffer:
        windows.append(buffer)
    return windows


def round_robin(rankings):
    seen, output = set(), []
    for rank in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking) and ranking[rank] not in seen:
                seen.add(ranking[rank])
                output.append(ranking[rank])
    return output
