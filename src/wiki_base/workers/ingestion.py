import asyncio
import logging

from wiki_base.database.connection import Database
from wiki_base.database.queries.ingestion_jobs import (
    claim_next_ingestion_job,
    complete_ingestion_job,
    fail_ingestion_job,
)
from wiki_base.ingestion.models import DocumentSource
from wiki_base.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class IngestionWorker:
    def __init__(
        self,
        *,
        database: Database,
        pipeline: IngestionPipeline,
        poll_interval_seconds: float,
    ) -> None:
        self._database = database
        self._pipeline = pipeline
        self._poll_interval_seconds = poll_interval_seconds

    async def run(self) -> None:
        logger.info("ingestion worker started")
        while True:
            processed_job = await self.run_once()
            if not processed_job:
                await asyncio.sleep(self._poll_interval_seconds)

    async def run_once(self) -> bool:
        async with self._database.connection() as connection:
            job = await claim_next_ingestion_job(connection)
        if job is None:
            return False

        logger.info("processing ingestion job %s for %s", job.id, job.document_name)
        cleanup_staged_file = False
        try:
            chunks = await self._pipeline.ingest(
                DocumentSource(
                    path=job.staging_path,
                    name=job.document_name,
                    media_type=job.media_type,
                )
            )
            async with self._database.connection() as connection:
                await complete_ingestion_job(connection, job, chunks)
            cleanup_staged_file = True
            logger.info("completed ingestion job %s with %d chunks", job.id, len(chunks))
        except Exception as error:
            logger.exception("ingestion job %s failed", job.id)
            async with self._database.connection() as connection:
                await fail_ingestion_job(
                    connection,
                    job,
                    error_code=type(error).__name__,
                    error_message=str(error)[:500] or "Document ingestion failed",
                )
            cleanup_staged_file = True
        finally:
            if cleanup_staged_file:
                job.staging_path.unlink(missing_ok=True)
        return True
