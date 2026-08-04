import logging
from dataclasses import dataclass
from uuid import UUID

import httpx
from graph_rag import RankedFact
from llm_providers.generation.base import ChatMessage, GenerationProvider

from wiki_base.api.errors import ServiceError
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.services.query_chunks import QueryChunksService, RetrievedChunk

logger = logging.getLogger(__name__)


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
    mode: RetrievalMode = RetrievalMode.LITE
    retrieval_strategy: RetrievalStrategy = RetrievalStrategy.VECTOR


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
        mode: RetrievalMode = RetrievalMode.LITE,
    ) -> QueryAnswer:
        """Retrieve evidence and generate an answer."""

        retrieval = await self._chunks.query(
            wiki_base_id=wiki_base_id,
            question=question,
            limit=limit,
            mode=mode,
        )
        if not retrieval.chunks:
            return QueryAnswer(
                wiki_base_id=wiki_base_id,
                question=retrieval.question,
                answer="The available documents do not provide enough information.",
                citations=[],
                mode=retrieval.mode,
                retrieval_strategy=retrieval.retrieval_strategy,
            )

        source_map = {f"S{index}": chunk for index, chunk in enumerate(retrieval.chunks, start=1)}
        source_context = "\n\n".join(
            self._format_source(source_id, chunk) for source_id, chunk in source_map.items()
        )
        fact_context = self._format_facts(retrieval.facts, source_map)
        context = (
            f"GRAPH FACTS\n{fact_context}\n\nSOURCE PASSAGES\n{source_context}"
            if fact_context
            else source_context
        )
        logger.info(
            "answer generation started wiki_base_id=%s mode=%s strategy=%s "
            "question=%r facts=%d sources=%d",
            wiki_base_id,
            retrieval.mode.value,
            retrieval.retrieval_strategy.value,
            retrieval.question,
            len(retrieval.facts),
            len(source_map),
        )
        logger.debug(
            "Answer generation facts=%s sources=%s history_messages=%d",
            [
                (
                    fact.fact.subject,
                    fact.fact.relation,
                    fact.fact.object,
                    round(fact.score, 4),
                )
                for fact in retrieval.facts
            ],
            [
                (
                    source_id,
                    str(chunk.id),
                    round(chunk.score, 4),
                    " ".join(chunk.content[:180].split()),
                )
                for source_id, chunk in source_map.items()
            ],
            len(history),
        )
        logger.debug(
            "complete answer-generation context wiki_base_id=%s\n%s",
            wiki_base_id,
            context,
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
        unknown_source_ids = [
            source_id
            for source_id in dict.fromkeys(generated.source_ids)
            if source_id not in source_map
        ]
        logger.debug(
            "answer generation result wiki_base_id=%s answer=%r "
            "model_source_ids=%s accepted_source_ids=%s unknown_source_ids=%s "
            "resolved_chunk_ids=%s",
            wiki_base_id,
            generated.text,
            generated.source_ids,
            [
                source_id
                for source_id in dict.fromkeys(generated.source_ids)
                if source_id in source_map
            ],
            unknown_source_ids,
            [str(chunk.id) for chunk in cited_chunks],
        )
        if not cited_chunks:
            logger.warning(
                "answer has no valid citations wiki_base_id=%s mode=%s "
                "strategy=%s model_source_ids=%s answer=%r",
                wiki_base_id,
                retrieval.mode.value,
                retrieval.retrieval_strategy.value,
                generated.source_ids,
                generated.text,
            )
        return QueryAnswer(
            wiki_base_id=wiki_base_id,
            question=retrieval.question,
            answer=generated.text,
            citations=[self._citation(chunk) for chunk in cited_chunks],
            mode=retrieval.mode,
            retrieval_strategy=retrieval.retrieval_strategy,
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
    def _format_facts(
        facts: list[RankedFact],
        source_map: dict[str, RetrievedChunk],
    ) -> str:
        """Format ranked graph facts with supporting source identifiers."""

        source_ids_by_chunk = {chunk.id: source_id for source_id, chunk in source_map.items()}
        lines = []
        for item in facts:
            source_ids = sorted(
                {
                    source_ids_by_chunk[provenance.chunk_id]
                    for provenance in item.fact.provenance
                    if provenance.chunk_id in source_ids_by_chunk
                }
            )
            if not source_ids:
                continue
            citations = ", ".join(f"[{source_id}]" for source_id in source_ids)
            lines.append(
                f"- {item.fact.subject} {item.fact.relation} {item.fact.object}. {citations}"
            )
        return "\n".join(lines)

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
