from uuid import UUID

import pytest

from graph_rag.fact_traversal import GraphFactTraverser
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")
PROVENANCE = TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID)


def make_graph() -> KnowledgeGraph:
    """Create a small directed fact chain."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=PROVENANCE,
    )
    graph.add_triple(
        Triple(subject="acme", relation="located in", object="paris"),
        provenance=PROVENANCE,
    )
    graph.add_triple(
        Triple(subject="acme", relation="is available", object="yes"),
        provenance=PROVENANCE,
    )
    graph.add_triple(
        Triple(subject="yes", relation="unrelated to", object="noise"),
        provenance=PROVENANCE,
    )
    return graph


def test_traverses_facts_to_configured_depth() -> None:
    """Traversal follows canonical facts through subject-capable nodes."""

    facts = GraphFactTraverser(max_depth=2).traverse(make_graph(), ["alice"])

    assert [
        (fact.subject, fact.relation, fact.object, fact.depth) for fact in facts
    ] == [
        ("alice", "works at", "acme", 1),
        ("acme", "is available", "yes", 2),
        ("acme", "located in", "paris", 2),
    ]


def test_tracks_every_seed_that_reaches_a_fact() -> None:
    """One canonical fact preserves coverage across multiple roots."""

    facts = GraphFactTraverser(max_depth=1).traverse(
        make_graph(),
        ["alice", "acme"],
    )

    works_at = next(fact for fact in facts if fact.relation == "works at")
    assert works_at.seeds == frozenset({"alice", "acme"})


def test_returns_empty_for_unknown_nodes() -> None:
    """Unknown traversal roots produce no facts."""

    assert GraphFactTraverser().traverse(make_graph(), ["unknown"]) == []


def test_rejects_invalid_limits() -> None:
    """Traversal bounds must be positive."""

    with pytest.raises(ValueError, match="max_depth"):
        GraphFactTraverser(max_depth=0)
    with pytest.raises(ValueError, match="max_candidates"):
        GraphFactTraverser(max_candidates=0)
