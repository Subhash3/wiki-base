from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import asyncpg
import httpx
import networkx as nx
from graph_rag import HippoRAGRetriever, KnowledgeGraph
from llm_providers.embeddings.base import EmbeddingProvider

from wiki_base.api.errors import ServiceError
from wiki_base.database.connection import Database
from wiki_base.database.queries.chunks import load_chunks_by_ids, search_chunks
from wiki_base.database.queries.graph_indexing_jobs import (
    list_ready_wiki_base_graph_paths,
)
from wiki_base.database.queries.wiki_bases import (
    get_wiki_base,
    list_wiki_base_retrieval_statuses,
)
from wiki_base.database.records import IngestionStatus
from wiki_base.retrieval import RetrievalMode


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
    mode: RetrievalMode = RetrievalMode.LITE


class QueryChunksService:
    """Retrieve chunks with vector or graph ranking."""

    def __init__(
        self,
        *,
        database: Database,
        embeddings: EmbeddingProvider,
        graph_retriever: HippoRAGRetriever,
    ) -> None:
        """Configure Lite and Pro retrieval dependencies."""

        self._database = database
        self._embeddings = embeddings
        self._graph_retriever = graph_retriever

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

            if mode == RetrievalMode.PRO:
                chunks = await self._query_graph(
                    wiki_base_id=wiki_base_id,
                    question=normalized_question,
                    limit=limit,
                )
            else:
                chunks = await self._query_vector(
                    wiki_base_id=wiki_base_id,
                    question=normalized_question,
                    limit=limit,
                )
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

        return QueryChunksResult(
            wiki_base_id=wiki_base_id,
            question=normalized_question,
            chunks=chunks,
            mode=mode,
        )

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

    async def _query_graph(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        limit: int,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks from merged ready document graphs."""

        async with self._database.connection() as connection:
            paths = await list_ready_wiki_base_graph_paths(connection, wiki_base_id)
        graph = KnowledgeGraph()
        for path in paths:
            graph = KnowledgeGraph.merge(graph, self._load_graph(path))

        ranked = await self._graph_retriever.retrieve(question, graph, limit=limit)
        async with self._database.connection() as connection:
            stored = await load_chunks_by_ids(
                connection,
                wiki_base_id=wiki_base_id,
                chunk_ids=[item.chunk_id for item in ranked],
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

    @staticmethod
    def _load_graph(path: Path) -> KnowledgeGraph:
        """Load one canonical document graph."""

        content = path.read_text(encoding="utf-8")
        return KnowledgeGraph.from_json(content)
