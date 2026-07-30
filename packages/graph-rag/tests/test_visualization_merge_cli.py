from pathlib import Path

import networkx as nx

from graph_rag.visualization import GraphVisualizer
from graph_rag.visualization_merge_cli import find_json_files, merge_files


def write_graph(path: Path, *, source: str, target: str, document_id: str) -> None:
    network = nx.MultiDiGraph()
    provenance = [{"document_id": document_id, "chunk_id": f"chunk-{document_id}"}]
    network.add_node(source, label=source, provenance=provenance)
    network.add_node(target, label=target, provenance=provenance)
    network.add_edge(source, target, label="related to", provenance=provenance)
    path.write_text(GraphVisualizer.to_json(network), encoding="utf-8")


def test_merge_files_writes_combined_json_and_html(tmp_path: Path) -> None:
    write_graph(
        tmp_path / "one.json",
        source="alice",
        target="acme",
        document_id="document-one",
    )
    write_graph(
        tmp_path / "two.json",
        source="acme",
        target="paris",
        document_id="document-two",
    )

    output_json, output_html = merge_files([tmp_path])
    merged = GraphVisualizer.from_json(output_json.read_text(encoding="utf-8"))

    assert output_json == tmp_path / "merged.json"
    assert output_html == tmp_path / "merged.html"
    assert {"alice", "acme", "paris"}.issubset(merged.nodes)
    assert "vis-network" in output_html.read_text(encoding="utf-8")


def test_directory_scan_ignores_existing_merged_json(tmp_path: Path) -> None:
    write_graph(
        tmp_path / "one.json",
        source="alice",
        target="acme",
        document_id="document-one",
    )
    write_graph(
        tmp_path / "merged.json",
        source="old",
        target="graph",
        document_id="old-document",
    )

    files = find_json_files([tmp_path])

    assert files == [tmp_path / "one.json"]
