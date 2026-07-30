from uuid import UUID

from graph_rag.entity_linking import ExactEntityLinker
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def make_graph() -> KnowledgeGraph:
    """Create a small normalized graph."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=CHUNK_ID,
        ),
    )
    return graph


def test_links_normalized_exact_entities() -> None:
    """Formatting differences are normalized before matching."""

    nodes = ExactEntityLinker().link([" Alice. ", "ACME"], make_graph())

    assert nodes == ["alice", "acme"]


def test_ignores_unmatched_and_duplicate_entities() -> None:
    """Unknown entities are skipped and graph nodes remain distinct."""

    nodes = ExactEntityLinker().link(
        ["Unknown", "Alice", "alice", "Another"],
        make_graph(),
    )

    assert nodes == ["alice"]


def test_empty_entities_return_no_nodes() -> None:
    """An empty entity list produces no graph seeds."""

    assert ExactEntityLinker().link([], make_graph()) == []
