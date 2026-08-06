import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from graph_rag import GraphVisualizer, KnowledgeGraph

from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.database.queries.document_graphs import (
    get_document_graph,
    list_ready_wiki_base_graphs,
)
from wiki_base.database.queries.graph_synonyms import list_wiki_base_graph_synonyms


async def visualize_document(
    database: Database,
    *,
    document_id: UUID,
    output: Path,
    export_3d: bool = False,
) -> tuple[Path, Path]:
    """Export one stored document graph as canonical JSON and interactive HTML."""

    async with database.connection() as connection:
        payload = await get_document_graph(connection, document_id)
    if payload is None:
        raise ValueError(f"No stored graph found for document {document_id}")

    graph = KnowledgeGraph.from_dict(payload)
    visualizer = GraphVisualizer(graph)
    network = visualizer.build(document_id=document_id)
    html_output = _require_suffix(output, ".html")
    json_output = html_output.with_suffix(".json")
    html_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(graph.to_json(), encoding="utf-8")
    html_output.write_text(
        visualizer.to_html(network, title=f"Graph RAG · {document_id}"),
        encoding="utf-8",
    )
    if export_3d:
        html_3d_output = html_output.with_name(f"{html_output.stem}-3d.html")
        html_3d_output.write_text(
            visualizer.to_3d_html(network, title=f"Graph RAG · {document_id} · 3D"),
            encoding="utf-8",
        )
    return json_output, html_output


async def merge_wiki_base(
    database: Database,
    *,
    wiki_base_id: UUID,
    output: Path,
    embedding_model: str | None = None,
    synonym_similarity_threshold: float = 0.95,
    export_3d: bool = False,
) -> tuple[Path, Path]:
    """Merge ready graphs for a wiki base and write JSON and HTML artifacts."""

    async with database.connection() as connection:
        payloads = await list_ready_wiki_base_graphs(connection, wiki_base_id)
        synonyms = (
            await list_wiki_base_graph_synonyms(
                connection,
                wiki_base_id=wiki_base_id,
                embedding_model=embedding_model,
                similarity_threshold=synonym_similarity_threshold,
            )
            if embedding_model is not None
            else []
        )
    if not payloads:
        raise ValueError(f"No ready graphs found for wiki base {wiki_base_id}")

    graph = KnowledgeGraph()
    for payload in payloads:
        graph = KnowledgeGraph.merge(graph, KnowledgeGraph.from_dict(payload))
    for synonym in synonyms:
        graph.add_synonym(
            synonym.first,
            synonym.second,
            similarity=synonym.similarity,
        )

    output = _require_suffix(output, ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.to_json(), encoding="utf-8")

    visualizer = GraphVisualizer(graph)
    html_output = output.with_suffix(".html")
    html_output.write_text(
        visualizer.to_html(
            visualizer.build(),
            title=f"Graph RAG · Wiki Base {wiki_base_id}",
        ),
        encoding="utf-8",
    )
    if export_3d:
        html_3d_output = html_output.with_name(f"{html_output.stem}-3d.html")
        html_3d_output.write_text(
            visualizer.to_3d_html(
                visualizer.build(), title=f"Graph RAG · Wiki Base {wiki_base_id} · 3D"
            ),
            encoding="utf-8",
        )
    return output, html_output


def run_visualizer() -> None:
    """Load a document graph from PostgreSQL and render it."""

    parser = argparse.ArgumentParser(
        description="Render a document graph stored in PostgreSQL as interactive HTML."
    )
    parser.add_argument("document_id", type=UUID, help="Document UUID")
    parser.add_argument("--output", type=Path, help="Output HTML path")
    parser.add_argument("--3d", dest="export_3d", action="store_true", help="Also export 3D HTML")
    arguments = parser.parse_args()
    output = arguments.output or Path(f"{arguments.document_id}.html")
    output_json, output_html = asyncio.run(
        _visualize_from_settings(
            document_id=arguments.document_id,
            output=output,
            export_3d=arguments.export_3d,
        )
    )
    print(output_json)
    print(output_html)
    if arguments.export_3d:
        print(output_html.with_name(f"{output_html.stem}-3d.html"))


def run_merger() -> None:
    """Load and merge a wiki base's ready graphs from PostgreSQL."""

    parser = argparse.ArgumentParser(
        description="Merge ready document graphs for a wiki base from PostgreSQL."
    )
    parser.add_argument("wiki_base_id", type=UUID, help="Wiki base UUID")
    parser.add_argument("--output", type=Path, help="Output canonical JSON path")
    parser.add_argument("--3d", dest="export_3d", action="store_true", help="Also export 3D HTML")
    arguments = parser.parse_args()
    output = arguments.output or Path(f"{arguments.wiki_base_id}.json")
    output_json, output_html = asyncio.run(
        _merge_from_settings(
            wiki_base_id=arguments.wiki_base_id,
            output=output,
            export_3d=arguments.export_3d,
        )
    )
    print(output_json)
    print(output_html)
    if arguments.export_3d:
        print(output_html.with_name(f"{output_html.stem}-3d.html"))


async def _visualize_from_settings(
    *, document_id: UUID, output: Path, export_3d: bool = False
) -> tuple[Path, Path]:
    """Export one graph using the configured database."""

    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect(min_size=1, max_size=1)
    try:
        return await visualize_document(
            database,
            document_id=document_id,
            output=output,
            export_3d=export_3d,
        )
    finally:
        await database.disconnect()


async def _merge_from_settings(
    *,
    wiki_base_id: UUID,
    output: Path,
    export_3d: bool = False,
) -> tuple[Path, Path]:
    """Merge ready graphs using the configured database."""

    settings = get_settings()
    database = Database(settings.database_url)
    await database.connect(min_size=1, max_size=1)
    try:
        return await merge_wiki_base(
            database,
            wiki_base_id=wiki_base_id,
            output=output,
            embedding_model=settings.embedding_model,
            synonym_similarity_threshold=(settings.graph_synonym_similarity_threshold),
            export_3d=export_3d,
        )
    finally:
        await database.disconnect()


def _require_suffix(path: Path, suffix: str) -> Path:
    """Validate an artifact's expected file extension."""

    if path.suffix.lower() != suffix:
        raise ValueError(f"Output must be a {suffix} file: {path}")
    return path
