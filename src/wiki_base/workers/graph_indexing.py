import asyncio
import logging

from graph_rag import GraphConcept, HippoRAGIndexer, graph_concepts
from llm_providers.embeddings.base import EmbeddingProvider

from wiki_base.database.connection import Database
from wiki_base.database.queries.document_graphs import upsert_document_graph
from wiki_base.database.queries.graph_concepts import replace_document_graph_concepts
from wiki_base.database.queries.graph_indexing_jobs import (
    claim_next_graph_indexing_job,
    complete_graph_indexing_job,
    fail_graph_indexing_job,
    load_graph_indexing_chunks,
)
from wiki_base.database.queries.graph_synonyms import replace_wiki_base_graph_synonyms

logger = logging.getLogger(__name__)


class GraphIndexingWorker:
    """Build and store one knowledge graph per indexed document."""

    def __init__(
        self,
        *,
        database: Database,
        indexer: HippoRAGIndexer,
        embeddings: EmbeddingProvider,
        extraction_model: str,
        index_version: str,
        embedding_batch_size: int,
        synonym_similarity_threshold: float,
        synonym_max_links: int,
        poll_interval_seconds: float,
    ) -> None:
        """Configure the graph indexing worker."""

        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")
        if not -1 <= synonym_similarity_threshold <= 1:
            raise ValueError("synonym_similarity_threshold must be between -1 and 1")
        if synonym_max_links < 1:
            raise ValueError("synonym_max_links must be positive")

        self._database = database
        self._indexer = indexer
        self._embeddings = embeddings
        self._extraction_model = extraction_model
        self._index_version = index_version
        self._embedding_batch_size = embedding_batch_size
        self._synonym_similarity_threshold = synonym_similarity_threshold
        self._synonym_max_links = synonym_max_links
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
            concepts = graph_concepts(graph)
            concept_embeddings = await self._embed_concepts(concepts)
            async with self._database.connection() as connection:
                async with connection.transaction():
                    await upsert_document_graph(
                        connection,
                        document_id=job.document_id,
                        graph=graph.to_dict(),
                        extraction_model=self._extraction_model,
                        index_version=self._index_version,
                    )
                    wiki_base_id = await replace_document_graph_concepts(
                        connection,
                        document_id=job.document_id,
                        concepts=concepts,
                        embeddings=concept_embeddings,
                        embedding_model=self._embeddings.model_info.model,
                    )
                    await complete_graph_indexing_job(connection, job)
                    await replace_wiki_base_graph_synonyms(
                        connection,
                        wiki_base_id=wiki_base_id,
                        embedding_model=self._embeddings.model_info.model,
                        similarity_threshold=self._synonym_similarity_threshold,
                        max_links_per_entity=self._synonym_max_links,
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

    async def _embed_concepts(
        self,
        concepts: list[GraphConcept],
    ) -> list[list[float]]:
        """Embed graph concepts in bounded batches."""

        embeddings: list[list[float]] = []
        for start in range(0, len(concepts), self._embedding_batch_size):
            batch = concepts[start : start + self._embedding_batch_size]
            embeddings.extend(
                await self._embeddings.embed_documents(
                    [concept.text for concept in batch]
                )
            )
        if len(embeddings) != len(concepts):
            raise ValueError("Embedding provider returned an unexpected vector count")
        return embeddings
