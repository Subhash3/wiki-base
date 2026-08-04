import logging
from dataclasses import dataclass, field
from uuid import UUID

import asyncpg
import httpx
import networkx as nx
from graph_rag import (
    FactRetriever,
    KnowledgeGraph,
    PageRankRetriever,
    RankedChunk,
    RankedFact,
)
from llm_providers.embeddings.base import EmbeddingProvider

from wiki_base.api.errors import ServiceError
from wiki_base.database.connection import Database
from wiki_base.database.queries.chunks import load_chunks_by_ids, search_chunks
from wiki_base.database.queries.document_graphs import (
    list_ready_wiki_base_graphs,
)
from wiki_base.database.queries.graph_synonyms import list_wiki_base_graph_synonyms
from wiki_base.database.queries.wiki_bases import (
    get_wiki_base,
    list_wiki_base_retrieval_statuses,
)
from wiki_base.database.records import IngestionStatus
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.retrieval.graph_concepts import PostgresSemanticConceptSearch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A ranked chunk with citation metadata."""

    id: UUID
    document_id: UUID
    document_name: str
    content: str
    score: float
    page: int | None
    slide: int | None
    section: str | None
    heading: str | None


@dataclass(frozen=True, slots=True)
class QueryChunksResult:
    """The chunks retrieved for one question."""

    wiki_base_id: UUID
    question: str
    chunks: list[RetrievedChunk]
    facts: list[RankedFact] = field(default_factory=list)
    mode: RetrievalMode = RetrievalMode.LITE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR


class QueryChunksService:
    """Retrieve chunks with vector or graph ranking."""

    def __init__(
        self,
        *,
        database: Database,
        embeddings: EmbeddingProvider,
        page_rank_retriever: PageRankRetriever,
        fact_retriever: FactRetriever,
        synonym_similarity_threshold: float = 0.95,
    ) -> None:
        """Configure vector, PageRank, and fact retrieval dependencies."""

        self._database = database
        self._embeddings = embeddings
        self._page_rank_retriever = page_rank_retriever
        self._fact_retriever = fact_retriever
        self._synonym_similarity_threshold = synonym_similarity_threshold

    async def query(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        limit: int,
        mode: RetrievalMode = RetrievalMode.LITE,
    ) -> QueryChunksResult:
        """Retrieve ranked chunks using the selected mode."""

        normalized_question = question.strip()
        if not normalized_question:
            raise ServiceError("invalid_question", "Question cannot be blank.", 422)

        logger.info(
            "retrieval started wiki_base_id=%s mode=%s limit=%d question=%r",
            wiki_base_id,
            mode.value,
            limit,
            normalized_question,
        )

        try:
            async with self._database.connection() as connection:
                wiki_base = await get_wiki_base(connection, wiki_base_id)
                if wiki_base is None:
                    raise ServiceError(
                        "wiki_base_not_found", "The requested wiki base was not found.", 404
                    )
                retrieval_statuses = await list_wiki_base_retrieval_statuses(
                    connection,
                    wiki_base_id,
                )
                status = retrieval_statuses.get(wiki_base_id, {}).get(
                    mode,
                    IngestionStatus.QUEUED,
                )
                logger.debug(
                    "retrieval readiness wiki_base_id=%s requested_mode=%s "
                    "requested_status=%s all_statuses=%s",
                    wiki_base_id,
                    mode.value,
                    status.value,
                    {
                        retrieval_mode.value: retrieval_status.value
                        for retrieval_mode, retrieval_status in retrieval_statuses.get(
                            wiki_base_id,
                            {},
                        ).items()
                    },
                )
                if status not in {
                    IngestionStatus.READY,
                    IngestionStatus.PARTIALLY_FAILED,
                }:
                    raise ServiceError(
                        "wiki_base_not_ready",
                        f"{mode.value.title()} retrieval is unavailable while its "
                        f"status is {status.value}.",
                        409,
                    )

            facts: list[RankedFact] = []
            if mode == RetrievalMode.PRO:
                chunks = await self._query_page_rank(
                    wiki_base_id=wiki_base_id,
                    question=normalized_question,
                    limit=limit,
                )
                retrieval_strategy = RetrievalStrategy.GRAPH
                if not chunks:
                    logger.warning(
                        "PageRank returned no chunks; using vector fallback "
                        "wiki_base_id=%s question=%r",
                        wiki_base_id,
                        normalized_question,
                    )
                    chunks = await self._query_vector(
                        wiki_base_id=wiki_base_id,
                        question=normalized_question,
                        limit=limit,
                    )
                    retrieval_strategy = RetrievalStrategy.VECTOR_FALLBACK
            elif mode == RetrievalMode.FACTS:
                chunks, facts = await self._query_facts(
                    wiki_base_id=wiki_base_id,
                    question=normalized_question,
                    limit=limit,
                )
                retrieval_strategy = RetrievalStrategy.FACT_GRAPH
                if not chunks:
                    logger.warning(
                        "fact traversal returned no chunks; using vector fallback "
                        "wiki_base_id=%s question=%r",
                        wiki_base_id,
                        normalized_question,
                    )
                    chunks = await self._query_vector(
                        wiki_base_id=wiki_base_id,
                        question=normalized_question,
                        limit=limit,
                    )
                    facts = []
                    retrieval_strategy = RetrievalStrategy.VECTOR_FALLBACK
            else:
                chunks = await self._query_vector(
                    wiki_base_id=wiki_base_id,
                    question=normalized_question,
                    limit=limit,
                )
                retrieval_strategy = RetrievalStrategy.VECTOR
        except ServiceError:
            raise
        except (
            asyncpg.PostgresError,
            httpx.HTTPError,
            nx.NetworkXException,
            OSError,
            ValueError,
        ) as error:
            raise ServiceError(
                "retrieval_unavailable", "Chunks could not be retrieved right now.", 503
            ) from error

        result = QueryChunksResult(
            wiki_base_id=wiki_base_id,
            question=normalized_question,
            chunks=chunks,
            facts=facts,
            mode=mode,
            retrieval_strategy=retrieval_strategy,
        )
        logger.info(
            "retrieval completed wiki_base_id=%s requested_mode=%s strategy=%s "
            "chunks=%d facts=%d chunk_ids=%s",
            wiki_base_id,
            mode.value,
            retrieval_strategy.value,
            len(chunks),
            len(facts),
            [str(chunk.id) for chunk in chunks],
        )
        return result

    async def _query_vector(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using cosine similarity."""

        embedding = await self._embeddings.embed_query(question)
        async with self._database.connection() as connection:
            matches = await search_chunks(
                connection,
                wiki_base_id=wiki_base_id,
                embedding=embedding,
                limit=limit,
            )
        logger.debug(
            "vector retrieval wiki_base_id=%s question=%r matches=%s",
            wiki_base_id,
            question,
            [
                {
                    "chunk_id": str(match.id),
                    "document_id": str(match.document_id),
                    "document_name": match.document_name,
                    "score": round(match.score, 6),
                    "content": match.content,
                }
                for match in matches
            ],
        )
        return [
            RetrievedChunk(
                id=match.id,
                document_id=match.document_id,
                document_name=match.document_name,
                content=match.content,
                score=match.score,
                page=match.page_number,
                slide=match.slide_number,
                section=match.section,
                heading=match.heading,
            )
            for match in matches
        ]

    async def _query_page_rank(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks using Personalized PageRank."""

        graph = await self._load_graph(wiki_base_id)
        ranked = await self._page_rank_retriever.retrieve(
            question,
            graph,
            limit=limit,
            semantic_search=self._semantic_search(wiki_base_id),
        )
        return await self._hydrate_ranked_chunks(
            wiki_base_id=wiki_base_id,
            ranked=ranked,
            log_label="PageRank",
        )

    async def _query_facts(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        limit: int,
    ) -> tuple[list[RetrievedChunk], list[RankedFact]]:
        """Retrieve ranked facts and their supporting chunks."""

        graph = await self._load_graph(wiki_base_id)
        result = await self._fact_retriever.retrieve(
            question,
            graph,
            limit=limit,
            semantic_search=self._semantic_search(wiki_base_id),
        )
        chunks = await self._hydrate_ranked_chunks(
            wiki_base_id=wiki_base_id,
            ranked=result.chunks,
            log_label="Fact",
        )
        hydrated_ids = {chunk.id for chunk in chunks}
        facts = [
            fact
            for fact in result.facts
            if any(provenance.chunk_id in hydrated_ids for provenance in fact.fact.provenance)
        ]
        logger.debug(
            "fact provenance hydration wiki_base_id=%s hydrated_chunk_ids=%s retained_facts=%s",
            wiki_base_id,
            [str(chunk.id) for chunk in chunks],
            [
                {
                    "subject": item.fact.subject,
                    "relation": item.fact.relation,
                    "object": item.fact.object,
                    "score": round(item.score, 6),
                    "depth": item.fact.depth,
                    "seeds": sorted(item.fact.seeds),
                    "provenance": [
                        {
                            "document_id": str(source.document_id),
                            "chunk_id": str(source.chunk_id),
                        }
                        for source in sorted(
                            item.fact.provenance,
                            key=lambda source: (
                                source.document_id.int,
                                source.chunk_id.int,
                            ),
                        )
                    ],
                }
                for item in facts
            ],
        )
        return chunks, facts

    async def _load_graph(self, wiki_base_id: UUID) -> KnowledgeGraph:
        """Load and merge ready document graphs and synonym edges."""

        async with self._database.connection() as connection:
            stored_graphs = await list_ready_wiki_base_graphs(connection, wiki_base_id)
            synonyms = await list_wiki_base_graph_synonyms(
                connection,
                wiki_base_id=wiki_base_id,
                embedding_model=self._embeddings.model_info.model,
                similarity_threshold=self._synonym_similarity_threshold,
            )
        graph = KnowledgeGraph()
        for stored_graph in stored_graphs:
            graph = KnowledgeGraph.merge(
                graph,
                KnowledgeGraph.from_dict(stored_graph),
            )
        added_synonyms = sum(
            graph.add_synonym(
                synonym.first,
                synonym.second,
                similarity=synonym.similarity,
            )
            for synonym in synonyms
        )
        logger.debug(
            "knowledge graph loaded wiki_base_id=%s document_graphs=%d nodes=%d "
            "factual_edges=%d synonym_edges_loaded=%d synonym_edges_added=%d",
            wiki_base_id,
            len(stored_graphs),
            len(graph.nodes),
            sum(1 for _edge in graph.edges()),
            len(synonyms),
            added_synonyms,
        )
        return graph

    def _semantic_search(self, wiki_base_id: UUID) -> PostgresSemanticConceptSearch:
        """Create a semantic graph-concept search scoped to one wiki base."""

        return PostgresSemanticConceptSearch(
            database=self._database,
            wiki_base_id=wiki_base_id,
            embedding_model=self._embeddings.model_info.model,
        )

    async def _hydrate_ranked_chunks(
        self,
        *,
        wiki_base_id: UUID,
        ranked: list[RankedChunk],
        log_label: str,
    ) -> list[RetrievedChunk]:
        """Load ranked chunk contents while preserving graph order."""

        async with self._database.connection() as connection:
            stored = await load_chunks_by_ids(
                connection,
                wiki_base_id=wiki_base_id,
                chunk_ids=[item.chunk_id for item in ranked],
            )
        logger.debug(
            "%s ranked chunks=%s hydrated_chunk_ids=%s",
            log_label,
            [(str(item.chunk_id), round(item.score, 4)) for item in ranked],
            [str(chunk.id) for chunk in stored],
        )
        chunks_by_id = {chunk.id: chunk for chunk in stored}
        return [
            RetrievedChunk(
                id=chunk.id,
                document_id=chunk.document_id,
                document_name=chunk.document_name,
                content=chunk.content,
                score=item.score,
                page=chunk.page_number,
                slide=chunk.slide_number,
                section=chunk.section,
                heading=chunk.heading,
            )
            for item in ranked
            if (chunk := chunks_by_id.get(item.chunk_id)) is not None
        ]
