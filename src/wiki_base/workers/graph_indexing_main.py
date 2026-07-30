import asyncio

from graph_rag import HippoRAGIndexer, LLMTripleExtractor
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
    generation = OllamaGenerationProvider(
        base_url=settings.ollama_url,
        model=settings.generation_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    worker = GraphIndexingWorker(
        database=database,
        indexer=HippoRAGIndexer(
            extractor=LLMTripleExtractor(generation=generation),
        ),
        output_directory=settings.graph_directory,
        extraction_model=settings.generation_model,
        index_version=settings.graph_index_version,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
    )

    await database.connect(
        min_size=settings.database_min_pool_size,
        max_size=settings.database_max_pool_size,
    )
    try:
        await worker.run()
    finally:
        await generation.close()
        await database.disconnect()


def run() -> None:
    """Run the graph worker until interrupted."""

    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
