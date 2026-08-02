from uuid import UUID

from wiki_base.api.routes.queries import query
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.schemas.queries import ConversationMessage, QueryRequest
from wiki_base.services.querying import AnswerCitation, QueryAnswer


class StubQueryService:
    async def query(
        self, *, wiki_base_id, question, history, limit, mode
    ) -> QueryAnswer:
        assert history[0].content == "Who is eligible?"
        assert limit == 5
        assert mode == RetrievalMode.PRO
        return QueryAnswer(
            wiki_base_id=wiki_base_id,
            question=question,
            answer="Eligible contractors are covered.",
            citations=[
                AnswerCitation(
                    chunk_id=UUID("0190f3a1-a0ee-77ac-a76b-fb191cb0f8a0"),
                    document_id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
                    document_name="policy.pdf",
                    excerpt="Eligible contractors are covered.",
                    score=0.9,
                    page=7,
                    slide=None,
                    section="Eligibility",
                    heading=None,
                )
            ],
            mode=mode,
            retrieval_strategy=RetrievalStrategy.GRAPH,
        )


async def test_query_returns_generated_answer_with_verified_citations() -> None:
    request = QueryRequest(
        wiki_base_id=UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705"),
        question="Does that include contractors?",
        history=[ConversationMessage(role="user", content="Who is eligible?")],
        mode=RetrievalMode.PRO,
    )

    response = await query(request=request, service=StubQueryService())

    assert response.answer == "Eligible contractors are covered."
    assert response.citations[0].document_name == "policy.pdf"
    assert response.citations[0].page == 7
    assert response.mode == RetrievalMode.PRO
    assert response.retrieval_strategy == RetrievalStrategy.GRAPH
