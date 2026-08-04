import asyncio

from graph_rag import HippoRAGIndexer, LLMTripleExtractor
from llm_providers.embeddings.ollama import OllamaEmbeddingProvider
from llm_providers.generation.ollama import OllamaGenerationProvider

from wiki_base.config.logging import configure_logging
from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.workers.graph_indexing import GraphIndexingWorker


async def run_worker() -> None:
    """Configure and run the graph indexing worker."""

    settings = get_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database_url)
    extraction = OllamaGenerationProvider(
        base_url=settings.ollama_url,
        model=settings.extraction_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    embeddings = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    worker = GraphIndexingWorker(
        database=database,
        indexer=HippoRAGIndexer(
            extractor=LLMTripleExtractor(generation=extraction),
        ),
        embeddings=embeddings,
        extraction_model=settings.extraction_model,
        index_version=settings.graph_index_version,
        embedding_batch_size=settings.graph_entity_embedding_batch_size,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
    )

    await database.connect(
        min_size=settings.database_min_pool_size,
        max_size=settings.database_max_pool_size,
    )
    try:
        await worker.run()
    finally:
        await embeddings.close()
        await extraction.close()
        await database.disconnect()


def run() -> None:
    """Run the graph worker until interrupted."""

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
