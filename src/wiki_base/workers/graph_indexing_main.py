import asyncio
import logging

from graph_rag import HippoRAGIndexer, LLMPassageEntityExtractor, LLMTripleExtractor

from wiki_base.config.logging import configure_logging
from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.embeddings import create_embedding_provider
from wiki_base.generation import create_generation_provider, create_groq_rate_limiter
from wiki_base.workers.graph_indexing import GraphIndexingWorker

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    """Configure and run the graph indexing worker."""

    settings = get_settings()
    configure_logging(
        settings.log_level,
        log_directory=settings.log_directory,
        process_name="graph-indexing-worker",
    )
    logger.info(
        "graph indexing worker configuration extraction_provider=%s "
        "extraction_model=%s embedding_provider=%s embedding_model=%s index_version=%s "
        "embedding_batch_size=%d synonym_similarity_threshold=%.3f "
        "synonym_max_links=%d poll_interval_seconds=%.3f",
        settings.extraction_provider,
        settings.extraction_model,
        settings.embedding_provider,
        settings.embedding_model,
        settings.graph_index_version,
        settings.graph_entity_embedding_batch_size,
        settings.graph_synonym_similarity_threshold,
        settings.graph_synonym_max_links,
        settings.worker_poll_interval_seconds,
    )
    database = Database(settings.database_url)
    extraction = create_generation_provider(
        settings,
        provider=settings.extraction_provider,
        model=settings.extraction_model,
        groq_rate_limiter=create_groq_rate_limiter(settings),
    )
    embeddings = create_embedding_provider(settings)
    worker = GraphIndexingWorker(
        database=database,
        indexer=HippoRAGIndexer(
            extractor=LLMTripleExtractor(generation=extraction),
            entity_extractor=LLMPassageEntityExtractor(generation=extraction),
        ),
        embeddings=embeddings,
        extraction_model=settings.extraction_model,
        index_version=settings.graph_index_version,
        embedding_batch_size=settings.graph_entity_embedding_batch_size,
        synonym_similarity_threshold=settings.graph_synonym_similarity_threshold,
        synonym_max_links=settings.graph_synonym_max_links,
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
