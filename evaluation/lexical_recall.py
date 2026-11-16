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


def round_robin(rankings):
    seen, output = set(), []
    for rank in range(max(map(len, rankings), default=0)):
        for ranking in rankings:
            if rank < len(ranking) and ranking[rank] not in seen:
                seen.add(ranking[rank])
                output.append(ranking[rank])
    return output
