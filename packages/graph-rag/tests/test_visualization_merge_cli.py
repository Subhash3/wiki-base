from pathlib import Path
from uuid import UUID

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance
from graph_rag.visualization_merge_cli import find_json_files, merge_files


def write_graph(
    path: Path,
    *,
    source: str,
    target: str,
    document_id: UUID,
    chunk_id: UUID,
) -> None:
    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject=source, relation="related to", object=target),
        provenance=TripleProvenance(
            document_id=document_id,
            chunk_id=chunk_id,
        ),
    )
    path.write_text(graph.to_json(), encoding="utf-8")


def test_merge_files_writes_combined_json_and_html(tmp_path: Path) -> None:
    write_graph(
        tmp_path / "one.json",
        source="alice",
        target="acme",
        document_id=UUID("10000000-0000-0000-0000-000000000001"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    write_graph(
        tmp_path / "two.json",
        source="acme",
        target="paris",
        document_id=UUID("10000000-0000-0000-0000-000000000002"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000002"),
    )

    output_json, output_html = merge_files([tmp_path])
    merged = KnowledgeGraph.from_json(output_json.read_text(encoding="utf-8"))

    assert output_json == tmp_path / "merged.json"
    assert output_html == tmp_path / "merged.html"
    assert merged.nodes == frozenset({"alice", "acme", "paris"})
    assert "vis-network" in output_html.read_text(encoding="utf-8")


def test_directory_scan_ignores_existing_merged_json(tmp_path: Path) -> None:
    write_graph(
        tmp_path / "one.json",
        source="alice",
        target="acme",
        document_id=UUID("10000000-0000-0000-0000-000000000001"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
    )
    write_graph(
        tmp_path / "merged.json",
        source="old",
        target="graph",
        document_id=UUID("10000000-0000-0000-0000-000000000003"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000003"),
    )

    files = find_json_files([tmp_path])

    assert files == [tmp_path / "one.json"]


def test_merge_files_optionally_writes_3d_html(tmp_path: Path) -> None:
    source = tmp_path / "one.json"
    write_graph(
        source,
        source="alice",
        target="acme",
        document_id=UUID("10000000-0000-0000-0000-000000000001"),
        chunk_id=UUID("00000000-0000-0000-0000-000000000001"),
    )

    merge_files([source], export_3d=True)

    assert (tmp_path / "merged-3d.html").is_file()
