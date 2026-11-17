"""Compatibility exports for the shared runtime/evaluation selector."""
from app.knowledge.context_selection import (
    WideCandidates,
    query_facets,
    source_successors,
    validated_order,
    with_facet_context,
    with_forward_context,
)

__all__ = ['WideCandidates', 'query_facets', 'source_successors', 'validated_order', 'with_facet_context', 'with_forward_context']
