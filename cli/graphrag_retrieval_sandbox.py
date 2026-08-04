"""Run HippoRAG retrieval against a canonical graph JSON file.

Run from the repository root:

    uv run python cli/graphrag_retrieval_sandbox.py \
      "Where is Alice's employer headquartered?" \
      .wiki-base-graphs/document-id.json
"""

import argparse
import asyncio
from pathlib import Path
from uuid import UUID

from graph_rag import (
    EmbeddingEntityLinker,
    KnowledgeGraph,
    LLMQueryEntityExtractor,
    PageRankRetriever,
    RankedChunk,
)
from llm_providers.embeddings.ollama import OllamaEmbeddingProvider

from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.generation import create_generation_provider, create_groq_rate_limiter


def parse_args() -> argparse.Namespace:
    """Parse the question and canonical graph path."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to retrieve evidence for")
    parser.add_argument("graph_json", type=Path, help="Canonical KnowledgeGraph JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum chunks to print (default: 5)",
    )
    return parser.parse_args()


def load_graph(path: Path) -> KnowledgeGraph:
    """Load a canonical graph produced by the indexing worker."""

    if not path.is_file():
        raise ValueError(f"Graph JSON does not exist: {path}")
    return KnowledgeGraph.from_json(path.read_text(encoding="utf-8"))


async def load_chunk_text(
    database: Database,
    ranked_chunks: list[RankedChunk],
) -> dict[UUID, str]:
    """Load ranked chunk text from PostgreSQL."""

    chunk_ids = [chunk.chunk_id for chunk in ranked_chunks]
    if not chunk_ids:
        return {}
    async with database.connection() as connection:
        rows = await connection.fetch(
            """
            SELECT id, content
            FROM chunks
            WHERE id = ANY($1::uuid[])
            """,
            chunk_ids,
        )
    return {row["id"]: row["content"] for row in rows}


async def main(question: str, graph_json: Path, *, limit: int) -> None:
    """Retrieve and print the highest-ranked chunks."""

    settings = get_settings()
    graph = load_graph(graph_json)
    extraction = create_generation_provider(
        settings,
        provider=settings.extraction_provider,
        model=settings.extraction_model,
        groq_rate_limiter=create_groq_rate_limiter(settings),
    )
    embeddings = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    database = Database(settings.database_url)
    await database.connect(
        min_size=settings.database_min_pool_size,
        max_size=settings.database_max_pool_size,
    )
    try:
        retriever = PageRankRetriever(
            entity_extractor=LLMQueryEntityExtractor(generation=extraction),
            entity_linker=EmbeddingEntityLinker(
                embeddings=embeddings,
                similarity_threshold=settings.graph_entity_similarity_threshold,
                relationship_similarity_threshold=(
                    settings.graph_relationship_similarity_threshold
                ),
                max_links_per_entity=settings.graph_entity_max_links,
                embedding_batch_size=settings.graph_entity_embedding_batch_size,
            ),
        )
        ranked_chunks = await retriever.retrieve(
            question,
            graph,
            limit=limit,
        )
        chunk_text = await load_chunk_text(database, ranked_chunks)
    finally:
        await embeddings.close()
        await extraction.close()
        await database.disconnect()

    if not ranked_chunks:
        print("No matching chunks found.")
        return

    for index, chunk in enumerate(ranked_chunks, start=1):
        content = chunk_text.get(chunk.chunk_id, "[chunk text not found]")
        print(
            f"\n[{index}] score={chunk.score:.6f} "
            f"document={chunk.document_id} chunk={chunk.chunk_id}"
        )
        print(content)


if __name__ == "__main__":
    arguments = parse_args()
    asyncio.run(
        main(
            arguments.question,
            arguments.graph_json,
            limit=arguments.limit,
        )
    )
