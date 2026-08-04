from uuid import UUID

from graph_rag import GraphFact, RankedFact, TripleProvenance

from wiki_base.api.routes.query_chunks import query_chunks
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.schemas.query_chunks import QueryChunksResponse
from wiki_base.services.query_chunks import QueryChunksResult, RetrievedChunk


class StubQueryChunksService:
    async def query(
        self, *, wiki_base_id: UUID, question: str, limit: int, mode: RetrievalMode
    ) -> QueryChunksResult:
        assert limit == 3
        assert mode == RetrievalMode.PRO
        return QueryChunksResult(
            wiki_base_id=wiki_base_id,
            question=question,
            chunks=[
                RetrievedChunk(
                    id=UUID("0190f3a1-a0ee-77ac-a76b-fb191cb0f8a0"),
                    document_id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
                    document_name="policy.pdf",
                    content="Contractors are eligible under the same conditions.",
                    score=0.8472,
                    page=7,
                    slide=None,
                    section="Eligibility",
                    heading="Who is eligible",
                )
            ],
            mode=mode,
            retrieval_strategy=RetrievalStrategy.GRAPH,
        )


async def test_query_chunks_returns_ranked_content_and_citation_metadata() -> None:
    wiki_base_id = UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705")

    response = await query_chunks(
        service=StubQueryChunksService(),
        wiki_base_id=wiki_base_id,
        question="Does this apply to contractors?",
        limit=3,
        mode=RetrievalMode.PRO,
    )

    assert response.wiki_base_id == wiki_base_id
    assert response.chunks[0].score == 0.8472
    assert response.chunks[0].document_name == "policy.pdf"
    assert response.chunks[0].page == 7
    assert response.mode == RetrievalMode.PRO
    assert response.retrieval_strategy == RetrievalStrategy.GRAPH


def test_query_chunks_schema_serializes_ranked_facts() -> None:
    """Fact traversal details remain available for debugging and benchmarks."""

    document_id = UUID("0190f3a0-b096-7af5-8392-cc61de46f6de")
    chunk_id = UUID("0190f3a1-a0ee-77ac-a76b-fb191cb0f8a0")
    result = QueryChunksResult(
        wiki_base_id=UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705"),
        question="Where does Alice work?",
        chunks=[],
        facts=[
            RankedFact(
                fact=GraphFact(
                    subject="alice",
                    relation="works at",
                    object="acme",
                    provenance=frozenset(
                        {
                            TripleProvenance(
                                document_id=document_id,
                                chunk_id=chunk_id,
                            )
                        }
                    ),
                    depth=1,
                    seeds=frozenset({"alice"}),
                ),
                score=0.9,
            )
        ],
        mode=RetrievalMode.FACTS,
        retrieval_strategy=RetrievalStrategy.FACT_GRAPH,
    )

    response = QueryChunksResponse.model_validate(result)

    assert response.facts[0].fact.relation == "works at"
    assert response.facts[0].fact.provenance[0].chunk_id == chunk_id
