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
