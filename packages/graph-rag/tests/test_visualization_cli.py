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
