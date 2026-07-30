from uuid import UUID

import networkx as nx

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance
from graph_rag.visualization import GraphVisualizer

DOCUMENT_ONE = UUID("10000000-0000-0000-0000-000000000001")
DOCUMENT_TWO = UUID("10000000-0000-0000-0000-000000000002")
CHUNK_ONE = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_TWO = UUID("00000000-0000-0000-0000-000000000002")


def make_graph() -> KnowledgeGraph:
    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ONE,
            chunk_id=CHUNK_ONE,
        ),
    )
    graph.add_triple(
        Triple(subject="acme", relation="located in", object="paris"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_TWO,
            chunk_id=CHUNK_TWO,
        ),
    )
    return graph


def test_builds_network_for_one_document() -> None:
    network = GraphVisualizer(make_graph()).build(document_id=DOCUMENT_ONE)
    document_node = f"document:{DOCUMENT_ONE}"

    assert isinstance(network, nx.MultiDiGraph)
    assert set(network.nodes) == {"alice", "acme", document_node}
    assert network.nodes[document_node]["group"] == "document"
    assert network.number_of_edges() == 3
    factual_edges = [
        data
        for _source, _target, data in network.edges(data=True)
        if not data.get("visualization_only")
    ]
    assert factual_edges[0]["label"] == "works at"
    assert factual_edges[0]["provenance"][0]["document_id"] == str(DOCUMENT_ONE)
    assert {
        target
        for source, target, data in network.edges(data=True)
        if source == document_node and data.get("visualization_only")
    } == {"alice", "acme"}


def test_renders_interactive_pyvis_html() -> None:
    network = GraphVisualizer(make_graph()).build()

    html = GraphVisualizer.to_html(network, title="Test graph")

    assert "vis-network" in html
    assert "Test graph" in html
