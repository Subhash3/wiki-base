from pathlib import Path
from uuid import UUID

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance
from graph_rag.visualization_cli import visualize_file


def test_visualize_file_writes_html_next_to_json(tmp_path: Path) -> None:
    json_file = tmp_path / "document-id.json"
    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=UUID("10000000-0000-0000-0000-000000000001"),
            chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        ),
    )
    json_file.write_text(graph.to_json(), encoding="utf-8")

    html_file = visualize_file(json_file)

    assert html_file == tmp_path / "document-id.html"
    assert "vis-network" in html_file.read_text(encoding="utf-8")


def test_visualize_file_optionally_writes_3d_html(tmp_path: Path) -> None:
    json_file = tmp_path / "document-id.json"
    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=UUID("10000000-0000-0000-0000-000000000001"),
            chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
        ),
    )
    json_file.write_text(graph.to_json(), encoding="utf-8")

    visualize_file(json_file, export_3d=True)

    html_3d = tmp_path / "document-id-3d.html"
    assert html_3d.is_file()
    assert "scatter3d" in html_3d.read_text(encoding="utf-8")
