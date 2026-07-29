from dataclasses import dataclass
from uuid import UUID

import httpx
from llm_providers.generation.base import ChatMessage, GenerationProvider

from wiki_base.api.errors import ServiceError
from wiki_base.services.query_chunks import QueryChunksService, RetrievedChunk


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    chunk_id: UUID
    document_id: UUID
    document_name: str
    excerpt: str
    score: float
    page: int | None
    slide: int | None
    section: str | None
    heading: str | None


@dataclass(frozen=True, slots=True)
class QueryAnswer:
    wiki_base_id: UUID
    question: str
    answer: str
    citations: list[AnswerCitation]


class QueryService:
    def __init__(
        self,
        *,
        chunks: QueryChunksService,
        generation: GenerationProvider,
    ) -> None:
        self._chunks = chunks
        self._generation = generation

    async def query(
        self,
        *,
        wiki_base_id: UUID,
        question: str,
        history: list[ChatMessage],
        limit: int,
    ) -> QueryAnswer:
        retrieval = await self._chunks.query(
            wiki_base_id=wiki_base_id,
            question=question,
            limit=limit,
        )
        if not retrieval.chunks:
            return QueryAnswer(
                wiki_base_id=wiki_base_id,
                question=retrieval.question,
                answer="The available documents do not provide enough information.",
                citations=[],
            )

        source_map = {
            f"S{index}": chunk for index, chunk in enumerate(retrieval.chunks, start=1)
        }
        context = "\n\n".join(
            self._format_source(source_id, chunk)
            for source_id, chunk in source_map.items()
        )
        messages = [*history, ChatMessage(role="user", content=retrieval.question)]
        try:
            generated = await self._generation.generate(messages, context)
        except (httpx.HTTPError, ValueError) as error:
            raise ServiceError(
                "generation_unavailable", "An answer could not be generated right now.", 503
            ) from error

        cited_chunks = [
            source_map[source_id]
            for source_id in dict.fromkeys(generated.source_ids)
            if source_id in source_map
        ]
        return QueryAnswer(
            wiki_base_id=wiki_base_id,
            question=retrieval.question,
            answer=generated.text,
            citations=[self._citation(chunk) for chunk in cited_chunks],
        )

    @staticmethod
    def _format_source(source_id: str, chunk: RetrievedChunk) -> str:
        location_parts = []
        if chunk.page is not None:
            location_parts.append(f"page {chunk.page}")
        if chunk.slide is not None:
            location_parts.append(f"slide {chunk.slide}")
        if chunk.section:
            location_parts.append(f"section {chunk.section}")
        location = ", ".join(location_parts) or "location unavailable"
        return f"[{source_id}] {chunk.document_name} ({location})\n{chunk.content}"

    @staticmethod
    def _citation(chunk: RetrievedChunk) -> AnswerCitation:
        return AnswerCitation(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            excerpt=chunk.content,
            score=chunk.score,
            page=chunk.page,
            slide=chunk.slide,
            section=chunk.section,
            heading=chunk.heading,
        )
