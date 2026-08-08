import asyncio
import logging

from document_processing.chunking.docling import DoclingDocumentChunker
from document_processing.parsing import DocxDocumentParser, PdfDocumentParser, PptxDocumentParser
from document_processing.parsing.docling_converter import DoclingConverter
from document_processing.parsing.registry import ParserRegistry

from wiki_base.config.logging import configure_logging
from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.embeddings import create_embedding_provider
from wiki_base.ingestion.pipeline import IngestionPipeline
from wiki_base.workers.ingestion import IngestionWorker

logger = logging.getLogger(__name__)


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(
        settings.log_level,
        log_directory=settings.log_directory,
        process_name="ingestion-worker",
    )
    logger.info(
        "ingestion worker configuration embedding_provider=%s embedding_model=%s "
        "embedding_batch_size=%d "
        "chunk_max_tokens=%d chunk_tokenizer_model=%s ocr_languages=%s "
        "ocr_force_full_page=%s poll_interval_seconds=%.3f",
        settings.embedding_provider,
        settings.embedding_model,
        settings.embedding_batch_size,
        settings.chunk_max_tokens,
        settings.chunk_tokenizer_model,
        settings.parsed_ocr_languages,
        settings.ocr_force_full_page,
        settings.worker_poll_interval_seconds,
    )

    database = Database(settings.database_url)
    provider = create_embedding_provider(settings)
    converter = DoclingConverter(
        ocr_languages=settings.parsed_ocr_languages,
        force_full_page_ocr=settings.ocr_force_full_page,
    )
    parser_registry = ParserRegistry(
        [
            PdfDocumentParser(converter),
            DocxDocumentParser(converter),
            PptxDocumentParser(converter),
        ]
    )
    pipeline = IngestionPipeline(
        parser_registry=parser_registry,
        chunker=DoclingDocumentChunker(
            max_tokens=settings.chunk_max_tokens,
            tokenizer_model=settings.chunk_tokenizer_model,
        ),
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
