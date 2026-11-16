"""Experimental Chinese lexical recall; never changes application search defaults.

The ranker accepts only text and IDs. Source evidence labels belong to the
benchmark scorer and cannot affect tokenization, IDF, or ranking.
"""
import math
import re
from collections import Counter, defaultdict

TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_.+-]*")
HAN_PATTERN = re.compile(r"[\u4e00-\u9fff]+")


def lexical_tokens(text):
    """Overlapping Han bigrams/trigrams; keep Latin identifiers intact.

    Punctuation separates runs, so unrelated words across a comma or line break
    cannot create a synthetic phrase. No dictionary or model download is needed.
    """
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
    """An in-memory benchmark index; caller supplies a snapshot-scoped corpus."""

    def __init__(self, documents):
        self.postings = defaultdict(dict)
        self.lengths = {}
        for key, text in sorted(documents.items()):
            frequencies = Counter(lexical_tokens(text))
            self.lengths[key] = sum(frequencies.values())
            for term, frequency in frequencies.items():
                self.postings[term][key] = frequency
        count = len(self.lengths)
        self.average_length = sum(self.lengths.values()) / count if count else 0.
        self.idf = {term: math.log(1 + (count - len(hits) + .5) / (len(hits) + .5))
                    for term, hits in self.postings.items()}

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
