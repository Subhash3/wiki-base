from fastapi import APIRouter
from llm_providers.generation.base import ChatMessage

from wiki_base.api.dependencies import QueryServiceDependency
from wiki_base.schemas.queries import QueryRequest, QueryResponse

router = APIRouter(tags=["retrieval"])


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, service: QueryServiceDependency) -> QueryResponse:
    result = await service.query(
        wiki_base_id=request.wiki_base_id,
        question=request.question,
        history=[
            ChatMessage(role=message.role, content=message.content)
            for message in request.history
        ],
        limit=request.limit,
        mode=request.mode,
    )
    return QueryResponse.model_validate(result)
