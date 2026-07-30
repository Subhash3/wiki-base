from pathlib import Path

import networkx as nx

from graph_rag.visualization import GraphVisualizer
from graph_rag.visualization_cli import visualize_file


def test_visualize_file_writes_html_next_to_json(tmp_path: Path) -> None:
    json_file = tmp_path / "document-id.json"
    network = nx.MultiDiGraph()
    network.add_edge("alice", "acme", label="works at", arrows="to")
    json_file.write_text(GraphVisualizer.to_json(network), encoding="utf-8")

    html_file = visualize_file(json_file)

    assert html_file == tmp_path / "document-id.html"
    assert "vis-network" in html_file.read_text(encoding="utf-8")
