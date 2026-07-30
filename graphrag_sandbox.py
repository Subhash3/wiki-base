"""Small document-to-HippoRAG indexing sandbox.

Run from the repository root:

    uv run --package graph-rag python graphrag_sandbox.py path/to/documents

Optionally set OLLAMA_URL and OLLAMA_MODEL to override the defaults.
"""

import argparse
import asyncio
import os
from pathlib import Path
from uuid import UUID, uuid4

from document_processing.chunking.docling import DoclingDocumentChunker
from document_processing.models import DocumentChunk, DocumentSource
from document_processing.parsing import (
    DocxDocumentParser,
    PdfDocumentParser,
    PptxDocumentParser,
)
from document_processing.parsing.docling_converter import DoclingConverter
from document_processing.parsing.registry import ParserRegistry
from graph_rag import (
    GraphVisualizer,
    HippoRAGIndexer,
    IndexedChunk,
    KnowledgeGraph,
    LLMTripleExtractor,
)
from llm_providers.generation.ollama import OllamaGenerationProvider

MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing PDF, DOCX, or PPTX documents",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=700,
        help="Approximate maximum tokens per chunk (default: 700)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("graphrag-output"),
        help="Directory for JSON and HTML graphs (default: graphrag-output)",
    )
    return parser.parse_args()


def build_document_processors(
    *, max_tokens: int
) -> tuple[ParserRegistry, DoclingDocumentChunker]:
    converter = DoclingConverter()
    registry = ParserRegistry(
        [
            PdfDocumentParser(converter),
            DocxDocumentParser(converter),
            PptxDocumentParser(converter),
        ]
    )
    return registry, DoclingDocumentChunker(max_tokens=max_tokens)


def chunk_document(
    path: Path,
    *,
    registry: ParserRegistry,
    chunker: DoclingDocumentChunker,
) -> list[DocumentChunk]:
    media_type = MEDIA_TYPES[path.suffix.lower()]
    source = DocumentSource(path=path, name=path.name, media_type=media_type)
    parsed = registry.resolve(source.name, source.media_type).parse(source)
    return chunker.chunk(parsed, media_type=media_type)


def find_documents(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise ValueError(f"Directory does not exist: {directory}")

    documents = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_TYPES
    )
    if not documents:
        supported = ", ".join(sorted(MEDIA_TYPES))
        raise ValueError(f"No supported documents found; expected {supported}")
    return documents


def save_graph(
    graph: KnowledgeGraph,
    *,
    name: str,
    title: str,
    output_dir: Path,
    document_id: UUID | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = GraphVisualizer(graph)
    network = visualizer.build(document_id=document_id)
    (output_dir / f"{name}.json").write_text(
        graph.to_json(),
        encoding="utf-8",
    )
    (output_dir / f"{name}.html").write_text(
        visualizer.to_html(network, title=title),
        encoding="utf-8",
    )


async def main(directory: Path, *, max_tokens: int, output_dir: Path) -> None:
    documents = find_documents(directory)
    registry, chunker = build_document_processors(max_tokens=max_tokens)
    document_chunks: list[tuple[UUID, list[IndexedChunk]]] = []
    for document in documents:
        document_id = uuid4()
        chunks = chunk_document(document, registry=registry, chunker=chunker)
        indexed_chunks = [
            IndexedChunk(document_id=document_id, chunk=chunk) for chunk in chunks
        ]
        document_chunks.append((document_id, indexed_chunks))
        print(f"Document {document.name}: {document_id} ({len(chunks)} chunks)")

    if not any(indexed_chunks for _document_id, indexed_chunks in document_chunks):
        raise ValueError("The documents did not contain any usable text")

    generation = OllamaGenerationProvider(
        base_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        model=os.getenv("OLLAMA_MODEL", "gemma3:270m"),
        timeout_seconds=120,
    )

    try:
        indexer = HippoRAGIndexer(extractor=LLMTripleExtractor(generation=generation))
        document_graphs = [
            (document_id, await indexer.index(indexed_chunks))
            for document_id, indexed_chunks in document_chunks
        ]
    finally:
        await generation.close()

    for document_id, document_graph in document_graphs:
        save_graph(
            document_graph,
            name=str(document_id),
            title=f"Graph RAG · {document_id}",
            output_dir=output_dir,
            document_id=document_id,
        )

    graph = document_graphs[0][1]
    for _document_id, document_graph in document_graphs[1:]:
        graph = KnowledgeGraph.merge(graph, document_graph)
    save_graph(
        graph,
        name="merged",
        title="Graph RAG · Merged",
        output_dir=output_dir,
    )
    print(f"Saved graph JSON and HTML files to {output_dir}")

    print("PASSAGES")
    for _document_id, indexed_chunks in document_chunks:
        for indexed_chunk in indexed_chunks:
            chunk = indexed_chunk.chunk
            excerpt = " ".join(chunk.content.split())
            if len(excerpt) > 160:
                excerpt = f"{excerpt[:157]}..."
            print(f"- {indexed_chunk.document_id}/{chunk.id}: {excerpt}")

    print("\nNODES")
    for node in sorted(graph.nodes):
        sources = ", ".join(
            f"{source.document_id}/{source.chunk_id}"
            for source in graph.provenance_for_node(node)
        )
        print(f"- {node} [{sources}]")

    print("\nEDGES")
    for edge in graph.edges():
        sources = ", ".join(
            f"{source.document_id}/{source.chunk_id}" for source in edge.provenance
        )
        print(f"- ({edge.subject}) -[{edge.relation}]-> ({edge.object}) [{sources}]")


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(
        main(
            arguments.directory,
            max_tokens=arguments.max_tokens,
            output_dir=arguments.output_dir,
        )
    )
