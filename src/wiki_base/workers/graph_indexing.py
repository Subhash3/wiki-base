import asyncio
import logging
from pathlib import Path
from uuid import UUID

from graph_rag import HippoRAGIndexer, KnowledgeGraph

from wiki_base.database.connection import Database
from wiki_base.database.queries.graph_indexing_jobs import (
    claim_next_graph_indexing_job,
    complete_graph_indexing_job,
    fail_graph_indexing_job,
    load_graph_indexing_chunks,
)

logger = logging.getLogger(__name__)


class GraphIndexingWorker:
    """Build and store one knowledge graph per indexed document."""

    def __init__(
        self,
        *,
        database: Database,
        indexer: HippoRAGIndexer,
        output_directory: Path,
        extraction_model: str,
        index_version: str,
        poll_interval_seconds: float,
    ) -> None:
        """Configure the graph indexing worker."""

        self._database = database
        self._indexer = indexer
        self._output_directory = output_directory
        self._extraction_model = extraction_model
        self._index_version = index_version
        self._poll_interval_seconds = poll_interval_seconds

    async def run(self) -> None:
        """Poll continuously for queued graph indexing jobs."""

        logger.info("graph indexing worker started")
        while True:
            processed_job = await self.run_once()
            if not processed_job:
                await asyncio.sleep(self._poll_interval_seconds)

    async def run_once(self) -> bool:
        """Process one queued graph indexing job."""

        async with self._database.connection() as connection:
            job = await claim_next_graph_indexing_job(connection)
        if job is None:
            return False

        logger.info("indexing graph for document %s", job.document_id)
        try:
            async with self._database.connection() as connection:
                chunks = await load_graph_indexing_chunks(connection, job.document_id)
            if not chunks:
                raise ValueError("Document has no chunks to index")

            graph = await self._indexer.index(chunks)
            output_path = self._write_graph(job.document_id, graph)
            async with self._database.connection() as connection:
                await complete_graph_indexing_job(
                    connection,
                    job,
                    output_path=output_path,
                    extraction_model=self._extraction_model,
                    index_version=self._index_version,
                )
            logger.info("indexed graph for document %s", job.document_id)
        except Exception as error:
            logger.exception("graph indexing failed for document %s", job.document_id)
            async with self._database.connection() as connection:
                await fail_graph_indexing_job(
                    connection,
                    job,
                    error_message=str(error)[:500] or "Graph indexing failed",
                )
        return True

    def _write_graph(self, document_id: UUID, graph: KnowledgeGraph) -> Path:
        """Write one canonical graph JSON file."""

        self._output_directory.mkdir(parents=True, exist_ok=True)
        output_path = self._output_directory / f"{document_id}.json"
        output_path.write_text(graph.to_json(), encoding="utf-8")
        return output_path
