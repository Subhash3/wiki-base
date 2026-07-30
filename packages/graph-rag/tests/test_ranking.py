from uuid import UUID

import networkx as nx
import pytest

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RankedChunk, Triple, TripleProvenance
from graph_rag.ranking import (
    aggregate_chunk_scores,
    build_ranking_graph,
    personalized_page_rank,
)

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def add_fact(graph: KnowledgeGraph, triple: Triple) -> None:
    """Add a fact with fixed test provenance."""

    graph.add_triple(
        triple,
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=CHUNK_ID,
        ),
    )


def test_builds_undirected_entity_only_graph() -> None:
    """Factual direction does not restrict associative traversal."""

    graph = KnowledgeGraph()
    add_fact(graph, Triple(subject="alice", relation="works at", object="acme"))
    add_fact(graph, Triple(subject="acme", relation="located in", object="paris"))

    ranking_graph = build_ranking_graph(graph)

    assert isinstance(ranking_graph, nx.Graph)
    assert not ranking_graph.is_directed()
    assert set(ranking_graph.nodes) == {"alice", "acme", "paris"}
    assert ranking_graph.has_edge("alice", "acme")
    assert ranking_graph.has_edge("acme", "alice")
    assert all(not str(node).startswith("document:") for node in ranking_graph.nodes)


def test_collapses_multiple_relations_between_the_same_entities() -> None:
    """Repeated entity pairs form one unweighted ranking connection."""

    graph = KnowledgeGraph()
    add_fact(graph, Triple(subject="alice", relation="works at", object="acme"))
    add_fact(graph, Triple(subject="alice", relation="founded", object="acme"))

    ranking_graph = build_ranking_graph(graph)

    assert ranking_graph.number_of_edges() == 1
    assert ranking_graph.edges["alice", "acme"]["weight"] == 1.0


def test_empty_knowledge_graph_builds_empty_ranking_graph() -> None:
    """An empty index produces an empty ranking graph."""

    ranking_graph = build_ranking_graph(KnowledgeGraph())

    assert ranking_graph.number_of_nodes() == 0
    assert ranking_graph.number_of_edges() == 0


def test_personalized_page_rank_spreads_from_query_seed() -> None:
    """PageRank assigns probability across the seed's connected component."""

    graph = nx.Graph()
    graph.add_edges_from([("alice", "acme"), ("acme", "paris")])
    graph.add_edge("unrelated", "isolated")

    scores = personalized_page_rank(graph, ["alice"])

    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["alice"] > 0
    assert scores["acme"] > 0
    assert scores["paris"] > 0
    assert scores["unrelated"] == pytest.approx(0)
    assert scores["isolated"] == pytest.approx(0)


def test_personalized_page_rank_uses_distinct_valid_seeds() -> None:
    """Unknown and duplicate seeds do not affect personalization."""

    graph = nx.path_graph(["alice", "acme", "paris"])

    scores = personalized_page_rank(
        graph,
        ["alice", "unknown", "alice", "paris"],
    )

    assert scores["alice"] == pytest.approx(scores["paris"])


def test_personalized_page_rank_returns_empty_without_valid_seeds() -> None:
    """PageRank does not fall back to an unpersonalized global ranking."""

    graph = nx.path_graph(["alice", "acme"])

    assert personalized_page_rank(graph, []) == {}
    assert personalized_page_rank(graph, ["unknown"]) == {}


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_personalized_page_rank_rejects_invalid_alpha(alpha: float) -> None:
    """The damping factor must remain a probability."""

    with pytest.raises(ValueError, match="alpha"):
        personalized_page_rank(nx.Graph(), ["alice"], alpha=alpha)


def test_aggregates_node_scores_across_source_chunks() -> None:
    """Each chunk receives the sum of its associated node scores."""

    graph = KnowledgeGraph()
    provenance_one = TripleProvenance(
        document_id=DOCUMENT_ID,
        chunk_id=CHUNK_ID,
    )
    provenance_two = TripleProvenance(
        document_id=UUID("10000000-0000-0000-0000-000000000002"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000002"),
    )
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=provenance_one,
    )
    graph.add_triple(
        Triple(subject="acme", relation="located in", object="paris"),
        provenance=provenance_two,
    )

    ranked = aggregate_chunk_scores(
        graph,
        {"alice": 0.4, "acme": 0.3, "paris": 0.2},
    )

    assert ranked == [
        RankedChunk(
            document_id=provenance_one.document_id,
            chunk_id=provenance_one.chunk_id,
            score=pytest.approx(0.7),
        ),
        RankedChunk(
            document_id=provenance_two.document_id,
            chunk_id=provenance_two.chunk_id,
            score=pytest.approx(0.5),
        ),
    ]


def test_aggregation_ignores_unknown_and_nonpositive_nodes() -> None:
    """Only positive scores attached to known nodes produce results."""

    graph = KnowledgeGraph()
    add_fact(graph, Triple(subject="alice", relation="works at", object="acme"))

    ranked = aggregate_chunk_scores(
        graph,
        {"alice": 0.4, "acme": 0.0, "unknown": 1.0},
    )

    assert ranked == [
        RankedChunk(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID, score=0.4)
    ]


def test_aggregation_returns_empty_without_scores() -> None:
    """An empty score mapping produces no ranked chunks."""

    assert aggregate_chunk_scores(KnowledgeGraph(), {}) == []
