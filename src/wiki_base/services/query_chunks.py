from dataclasses import dataclass
from uuid import UUID

import asyncpg
import httpx
from llm_providers.embeddings.base import EmbeddingProvider

from wiki_base.api.errors import ServiceError
from wiki_base.database.connection import Database
from wiki_base.database.queries.chunks import search_chunks
from wiki_base.database.queries.wiki_bases import get_wiki_base
from wiki_base.database.records import IngestionStatus


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
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
    wiki_base_id: UUID
    question: str
    chunks: list[RetrievedChunk]


class QueryChunksService:
    def __init__(self, *, database: Database, embeddings: EmbeddingProvider) -> None:
        self._database = database
        self._embeddings = embeddings

    async def query(
        self, *, wiki_base_id: UUID, question: str, limit: int
    ) -> QueryChunksResult:
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
                if wiki_base.status not in {
                    IngestionStatus.READY,
                    IngestionStatus.PARTIALLY_FAILED,
                }:
                    raise ServiceError(
                        "wiki_base_not_ready",
                        f"The wiki base cannot be queried while its status is {wiki_base.status}.",
                        409,
                    )

            embedding = await self._embeddings.embed_query(normalized_question)
            async with self._database.connection() as connection:
                matches = await search_chunks(
                    connection,
                    wiki_base_id=wiki_base_id,
                    embedding=embedding,
                    limit=limit,
                )
        except ServiceError:
            raise
        except (asyncpg.PostgresError, httpx.HTTPError, ValueError) as error:
            raise ServiceError(
                "retrieval_unavailable", "Chunks could not be retrieved right now.", 503
            ) from error

        return QueryChunksResult(
            wiki_base_id=wiki_base_id,
            question=normalized_question,
            chunks=[
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
            ],
        )
