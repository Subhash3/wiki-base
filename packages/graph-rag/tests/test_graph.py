from uuid import UUID

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance

DOCUMENT_ONE = UUID("10000000-0000-0000-0000-000000000001")
DOCUMENT_TWO = UUID("10000000-0000-0000-0000-000000000002")
CHUNK_ONE = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_TWO = UUID("00000000-0000-0000-0000-000000000002")


def test_graph_merges_facts_and_retains_chunk_provenance() -> None:
    graph = KnowledgeGraph()
    fact = Triple(subject="alice", relation="works at", object="acme")
    provenance_one = TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE)
    provenance_two = TripleProvenance(document_id=DOCUMENT_TWO, chunk_id=CHUNK_TWO)

    graph.add_triple(fact, provenance=provenance_one)
    graph.add_triple(fact, provenance=provenance_two)

    assert graph.nodes == frozenset({"alice", "acme"})
    assert graph.provenance_for_node("acme") == frozenset(
        {provenance_one, provenance_two}
    )
    assert graph.triples_for_provenance(provenance_one) == frozenset({fact})
    assert list(graph.edges())[0].provenance == frozenset(
        {provenance_one, provenance_two}
    )


def test_merge_returns_new_graph_with_deduplicated_provenance() -> None:
    first = KnowledgeGraph()
    second = KnowledgeGraph()
    shared_fact = Triple(subject="alice", relation="works at", object="acme")
    second_fact = Triple(subject="acme", relation="located in", object="paris")
    provenance_one = TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE)
    provenance_two = TripleProvenance(document_id=DOCUMENT_TWO, chunk_id=CHUNK_TWO)
    first.add_triple(shared_fact, provenance=provenance_one)
    second.add_triple(shared_fact, provenance=provenance_one)
    second.add_triple(second_fact, provenance=provenance_two)

    merged = KnowledgeGraph.merge(first, second)

    assert merged is not first
    assert merged is not second
    assert merged.nodes == frozenset({"alice", "acme", "paris"})
    assert len(list(merged.edges())) == 2
    assert merged.provenance_for_node("alice") == frozenset({provenance_one})


def test_canonical_json_round_trip_preserves_graph() -> None:
    graph = KnowledgeGraph()
    fact = Triple(subject="alice", relation="works at", object="acme")
    provenance = TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE)
    graph.add_triple(fact, provenance=provenance)

    loaded = KnowledgeGraph.from_json(graph.to_json())

    assert loaded.nodes == graph.nodes
    assert list(loaded.edges()) == list(graph.edges())
    assert "document:" not in graph.to_json()


def test_canonical_dict_round_trip_preserves_graph() -> None:
    """Database-ready mappings retain graph facts and provenance."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE),
    )

    loaded = KnowledgeGraph.from_dict(graph.to_dict())

    assert loaded.nodes == graph.nodes
    assert list(loaded.edges()) == list(graph.edges())


def test_round_trip_preserves_isolated_entity_mentions_and_synonyms() -> None:
    """Canonical version 2 retains passage associations and semantic edges."""

    graph = KnowledgeGraph()
    provenance = TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE)
    graph.add_entity("glamour", provenance=provenance)
    graph.add_entity("honda glamour", provenance=provenance)
    assert graph.add_synonym("glamour", "honda glamour", similarity=0.91)

    payload = graph.to_dict()
    loaded = KnowledgeGraph.from_dict(payload)

    assert payload["version"] == 2
    assert loaded.nodes == frozenset({"glamour", "honda glamour"})
    assert loaded.entity_provenance_for_node("glamour") == frozenset({provenance})
    assert list(loaded.synonyms())[0].similarity == 0.91


def test_loads_legacy_version_one_graphs() -> None:
    """Existing edge-only document graphs remain readable during re-indexing."""

    graph = KnowledgeGraph.from_dict(
        {
            "version": 1,
            "edges": [
                {
                    "subject": "alice",
                    "relation": "works at",
                    "object": "acme",
                    "provenance": [
                        {
                            "document_id": str(DOCUMENT_ONE),
                            "chunk_id": str(CHUNK_ONE),
                        }
                    ],
                }
            ],
        }
    )

    assert graph.nodes == frozenset({"alice", "acme"})
    assert graph.entity_provenance_for_node("alice") == frozenset()
