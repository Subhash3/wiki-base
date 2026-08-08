from uuid import UUID

import networkx as nx
from pyvis.network import Network

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
    assert network.nodes[document_node]["size"] == 24
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
    provenance_edges = [
        data
        for _source, _target, data in network.edges(data=True)
        if data.get("visualization_only")
    ]
    assert all(edge["physics"] is False for edge in provenance_edges)


def test_sizes_entities_by_unique_connected_entities() -> None:
    """Highly connected entities are larger than leaf entities."""

    network = GraphVisualizer(make_graph()).build()

    assert network.nodes["acme"]["connections"] == 2
    assert network.nodes["alice"]["connections"] == 1
    assert network.nodes["acme"]["size"] > network.nodes["alice"]["size"]
    assert network.nodes["acme"]["connectivity"] == "connected"
    assert network.nodes["alice"]["connectivity"] == "leaf"
    assert network.nodes["acme"]["color"]["background"] == "#2563eb"
    assert network.nodes["alice"]["color"]["background"] == "#0d9488"
    assert "Connected entities: 2" in network.nodes["acme"]["title"]


def test_stabilizes_graphs_before_freezing_them() -> None:
    """Graphs receive their force-directed layout before becoming stationary."""

    options = GraphVisualizer._options()

    assert options["physics"]["enabled"] is True
    assert options["physics"]["solver"] == "forceAtlas2Based"
    assert options["physics"]["stabilization"]["iterations"] == 200
    assert options["edges"]["smooth"] == {"type": "dynamic"}


def test_preserves_all_nodes_edges_and_labels_during_color_conversion() -> None:
    """Color conversion does not remove or hide graph content."""

    network = GraphVisualizer(make_graph()).build()
    visualization = Network(directed=True)
    visualization.from_nx(network)
    expected_node_ids = {node["id"] for node in visualization.nodes}
    expected_edges = len(visualization.edges)

    GraphVisualizer._apply_entity_colors(visualization)

    entity_colors = {
        node["id"]: node["color"]
        for node in visualization.nodes
        if node.get("connectivity") is not None
    }
    assert entity_colors["acme"]["background"] == "#2563eb"
    assert entity_colors["alice"]["background"] == "#0d9488"
    assert all(
        "group" not in node for node in visualization.nodes if node.get("connectivity") is not None
    )
    assert {node["id"] for node in visualization.nodes} == expected_node_ids
    assert len(visualization.edges) == expected_edges
    assert any(edge.get("visualization_only") for edge in visualization.edges)
    assert all(edge["label"] for edge in visualization.edges)


def test_renders_interactive_pyvis_html() -> None:
    network = GraphVisualizer(make_graph()).build()

    html = GraphVisualizer.to_html(network, title="Test graph")

    assert "vis-network" in html
    assert "Test graph" in html
    assert "Connectivity" in html
    assert "enableHtmlTooltips(nodes);" in html
    assert "enableHtmlTooltips(edges);" in html
    assert "tooltip.innerHTML = item.title;" in html
    assert 'network.once("stabilizationIterationsDone"' in html
    assert "network.storePositions();" in html
    assert "network.setOptions({physics: {enabled: false}});" in html


def test_renders_self_contained_3d_plotly_html() -> None:
    network = GraphVisualizer(make_graph()).build()

    html = GraphVisualizer.to_3d_html(network, title="Test graph · 3D")

    assert "plotly.js" in html
    assert "scatter3d" in html
    assert "Test graph" in html
    assert "works at" in html
    assert "alice" in html
    assert "plotly_relayout" in html
    assert "initialEyeDistance / eyeDistance" in html
    assert "animateNodeSizes(baseSizes.map(size => size * zoomScale), zoomScale)" in html
    assert "clearTimeout(zoomTimer)" in html
    assert "setTimeout(function()" in html
    assert "requestAnimationFrame(renderFrame)" in html
    assert "easedProgress" in html
    assert "'textfont.size'" in html


def test_scales_3d_node_sizes() -> None:
    assert GraphVisualizer._3d_node_size({"size": 12}) == 10.8
    assert GraphVisualizer._3d_node_size({"size": 48}) == 32.4


def test_only_hubs_receive_persistent_3d_labels() -> None:
    assert GraphVisualizer._3d_node_label("alice", {"connectivity": "leaf"}) == ""
    assert GraphVisualizer._3d_node_label(
        "acme", {"connectivity": "hub", "label": "Acme Corp"}
    ) == "Acme Corp"


def test_groups_tooltip_chunks_by_document() -> None:
    """Provenance tooltips display each document identifier once."""

    second_chunk = UUID("00000000-0000-0000-0000-000000000003")
    title = GraphVisualizer._provenance_title(
        (
            TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE),
            TripleProvenance(document_id=DOCUMENT_ONE, chunk_id=second_chunk),
            TripleProvenance(document_id=DOCUMENT_TWO, chunk_id=CHUNK_TWO),
        )
    )

    assert title.count(f"Document: {DOCUMENT_ONE}") == 1
    assert title.count(f"Document: {DOCUMENT_TWO}") == 1
    assert title.count("Chunks:") == 2
    assert str(CHUNK_ONE) in title
    assert str(second_chunk) in title
    assert str(CHUNK_TWO) in title


def test_visualizes_synonym_edges_separately() -> None:
    """Semantic edges are distinguishable from factual relationships."""

    graph = make_graph()
    graph.add_synonym("alice", "paris", similarity=0.92)

    network = GraphVisualizer(graph).build()
    synonym_edges = [
        data for _source, _target, data in network.edges(data=True) if data.get("synonym")
    ]

    assert synonym_edges[0]["label"] == "synonym"
    assert synonym_edges[0]["similarity"] == 0.92
    assert synonym_edges[0]["dashes"] is True
