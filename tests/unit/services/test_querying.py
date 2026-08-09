from uuid import UUID

from graph_rag import GraphFact, RankedFact, TripleProvenance
from llm_providers.generation.base import ChatMessage, GeneratedAnswer

from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.services.query_chunks import QueryChunksResult, RetrievedChunk
from wiki_base.services.querying import QueryService

WIKI_BASE_ID = UUID("10000000-0000-0000-0000-000000000001")
DOCUMENT_ID = UUID("20000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("30000000-0000-0000-0000-000000000001")


class StubChunks:
    """Return one fact-backed source chunk."""

    async def query(self, **_arguments) -> QueryChunksResult:
        """Return a Facts retrieval result."""

        provenance = TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID)
        return QueryChunksResult(
            wiki_base_id=WIKI_BASE_ID,
            question="What engine does Tiago have?",
            chunks=[
                RetrievedChunk(
                    id=CHUNK_ID,
                    document_id=DOCUMENT_ID,
                    document_name="tiago.pdf",
                    content="Tata Tiago has a 1199 cc petrol engine.",
                    score=0.91,
                    page=1,
                    slide=None,
                    section=None,
                    heading=None,
                )
            ],
            facts=[
                RankedFact(
                    fact=GraphFact(
                        subject="tata tiago",
                        relation="has petrol engine",
                        object="1199 cc",
                        provenance=frozenset({provenance}),
                        depth=1,
                        seeds=frozenset({"tata tiago"}),
                    ),
                    score=0.94,
                )
            ],
            mode=RetrievalMode.FACTS,
            retrieval_strategy=RetrievalStrategy.FACT_GRAPH,
        )


class StubGeneration:
    """Capture answer context and return one citation."""

    def __init__(self) -> None:
        self.context = ""

    async def generate(self, _messages, context: str) -> GeneratedAnswer:
        """Capture context and cite the first source."""

        self.context = context
        return GeneratedAnswer(text="It has a 1199 cc petrol engine.", source_ids=("S1",))


class CapturingChunks(StubChunks):
    def __init__(self) -> None:
        self.question = ""

    async def query(self, **arguments) -> QueryChunksResult:
        self.question = arguments["question"]
        return await super().query(**arguments)


class StubContextualization:
    async def generate_structured(self, messages, schema, *, max_tokens=4096):
        assert messages[-1].content == "What about its engine?"
        assert schema["required"] == ["standalone_question"]
        assert max_tokens == 256
        return {"standalone_question": "What engine does the Tata Tiago have?"}


async def test_fact_retrieval_adds_grounded_facts_to_answer_context() -> None:
    """Facts are supplied alongside their provenance source passages."""

    generation = StubGeneration()
    service = QueryService(chunks=StubChunks(), generation=generation)

    answer = await service.query(
        wiki_base_id=WIKI_BASE_ID,
        question="What engine does Tiago have?",
        history=[],
        limit=5,
        mode=RetrievalMode.FACTS,
    )

    assert "GRAPH FACTS" in generation.context
    assert "tata tiago has petrol engine 1199 cc. [S1]" in generation.context
    assert "SOURCE PASSAGES" in generation.context
    assert answer.retrieval_strategy == RetrievalStrategy.FACT_GRAPH
    assert answer.citations[0].chunk_id == CHUNK_ID


async def test_history_contextualizes_retrieval_but_preserves_original_question() -> None:
    chunks = CapturingChunks()
    generation = StubGeneration()
    service = QueryService(
        chunks=chunks,
        generation=generation,
        contextualization=StubContextualization(),
    )

    answer = await service.query(
        wiki_base_id=WIKI_BASE_ID,
        question="What about its engine?",
        history=[ChatMessage(role="user", content="Tell me about the Tata Tiago.")],
        limit=5,
    )

    assert chunks.question == "What engine does the Tata Tiago have?"
    assert answer.question == "What about its engine?"
