"""Experimental routing only. No profile automatically changes production configuration."""
import math
import re

PROFILES = ("keyword", "vector", "rrf", "keyword_weighted", "adaptive")


def query_route(query):
    if re.search(r"`[^`]+`|[A-Za-z][\w-]*(?:[._][\w-]+)+|\b[A-Z][A-Z0-9_]{2,}\b", query):
        return "exact_identifier", (.8, .2)
    if re.search(r"比较|分别|区别|差异|以及|compare|versus|\bvs\b", query, re.I):
        return "comparison", (.5, .5)
    return "semantic_question", (.3, .7)


def weighted_rrf(keyword, vector, weights, k=60):
    if len(weights) != 2 or any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("Invalid fusion weights")
    scores = {}
    for values, weight in zip((keyword, vector), weights):
        seen = set()
        for rank, key in enumerate(values, 1):
            if key not in seen:
                scores[key] = scores.get(key, 0.) + weight / (k + rank)
                seen.add(key)
    return sorted(scores, key=lambda key: (-scores[key], key))


def rank_profile(query, keyword, vector, profile):
    if profile == "keyword":
        return list(dict.fromkeys(keyword))
    if profile == "vector":
        return list(dict.fromkeys(vector))
    weights = {"rrf": (.5, .5), "keyword_weighted": (.7, .3), "adaptive": query_route(query)[1]}
    if profile not in weights:
        raise ValueError("Unknown retrieval experiment profile")
    return weighted_rrf(keyword, vector, weights[profile])
