"""Auditable query expansion: the original question is never replaced."""
import json
import re

from pydantic import BaseModel, ConfigDict, Field

from .errors import DomainError


class TermRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    corpus_ids: list[str] = Field(min_length=1, max_length=100)
    term: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(min_length=1, max_length=8)


def load_rules(settings):
    if not settings.query_terms_file:
        return []
    try:
        if settings.query_terms_file.stat().st_size > 128000:
            raise ValueError()
        raw = json.loads(settings.query_terms_file.read_text(encoding="utf-8-sig"))
        if set(raw) != {"rules"} or len(raw["rules"]) > 200:
            raise ValueError()
        rules = [TermRule.model_validate(item).model_dump() for item in raw["rules"]]
        if any(not alias.strip() or len(alias) > 100 for r in rules for alias in r["aliases"]):
            raise ValueError()
        return rules
    except (OSError, ValueError, TypeError):
        raise DomainError("查询术语配置不合法，未执行检索", 503, "query_terms_invalid") from None


def _has_term(text, term):
    return bool(re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text, re.I))


def protected_tokens(text):
    # Literal IDs, numbers, comparison operators and negation must survive.
    return set(re.findall(r"[A-Za-z][A-Za-z0-9_.:/-]*|\d+(?:\.\d+)?|[<>≤≥]=?|不超过|不少于|不得|不能|不含|未|无", text))


def rewrite_queries(question, subquestions, corpus_id, rules, max_per_subquestion=5):
    # Search expressions are capped at 500 characters. Do not silently replace a
    # long question with a shorter target and claim its constraints survived.
    if len(question) > 500:
        return subquestions, [{"kind": "skipped", "reason": "original_question_over_search_limit",
                               "original_question": question}]
    output, records = [], []
    for sub in subquestions:
        original = question
        candidates = [(original, "original")]
        # Additive expansion retains all of the original entities and conditions.
        for rule in rules:
            if corpus_id not in rule["corpus_ids"]:
                continue
            names = [rule["term"], *rule["aliases"]]
            if any(_has_term(original, name) for name in names):
                candidates.append((original + " " + " ".join(names), "terminology"))
        required = protected_tokens(original)
        for query in sub["queries"]:
            numbers = set(re.findall(r"\d+(?:\.\d+)?", query))
            if not required <= protected_tokens(query) or not numbers <= set(re.findall(r"\d+(?:\.\d+)?", original)):
                records.append({"subquestion_id": sub["id"], "kind": "rejected_planner",
                                "query": query, "reason": "protected_literal_missing"})
                continue
            anchored = query if original in query else original + " " + query
            candidates.append((anchored, "planner"))
        seen, queries = set(), []
        for query, kind in candidates:
            key = query.strip().casefold()
            if not key or key in seen or len(query) > 500 or len(queries) >= max_per_subquestion:
                continue
            seen.add(key)
            queries.append(query.strip())
            records.append({"subquestion_id": sub["id"], "kind": kind, "query": query.strip()})
        output.append({**sub, "queries": queries})
    return output, records
