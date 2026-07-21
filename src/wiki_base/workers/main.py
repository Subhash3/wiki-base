import asyncio

from wiki_base.config.logging import configure_logging
from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.ingestion.chunking.docling import DoclingDocumentChunker
from wiki_base.ingestion.parsers import DocxDocumentParser, PdfDocumentParser, PptxDocumentParser
from wiki_base.ingestion.parsers.docling_converter import DoclingConverter
from wiki_base.ingestion.parsers.registry import ParserRegistry
from wiki_base.ingestion.pipeline import IngestionPipeline
from wiki_base.providers.embeddings.ollama import OllamaEmbeddingProvider
from wiki_base.workers.ingestion import IngestionWorker


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    database = Database(settings.database_url)
    provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    converter = DoclingConverter()
    parser_registry = ParserRegistry(
        [
            PdfDocumentParser(converter),
            DocxDocumentParser(converter),
            PptxDocumentParser(converter),
        ]
    )
    pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        chunker=DoclingDocumentChunker(max_tokens=settings.chunk_max_tokens),
        embedding_provider=provider,
        embedding_batch_size=settings.embedding_batch_size,
    )
    worker = IngestionWorker(
        database=database,
        pipeline=pipeline,
        poll_interval_seconds=settings.worker_poll_interval_seconds,
    )

    await database.connect(
        min_size=settings.database_min_pool_size,
        max_size=settings.database_max_pool_size,
    )
    try:
        await worker.run()
    finally:
        await provider.close()
        await database.disconnect()


def run() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        pass
