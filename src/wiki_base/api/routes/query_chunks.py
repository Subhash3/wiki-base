from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query

from wiki_base.api.dependencies import QueryChunksServiceDependency
from wiki_base.retrieval import RetrievalMode
from wiki_base.schemas.query_chunks import QueryChunksResponse

router = APIRouter(tags=["retrieval"])


@router.get("/querychunks", response_model=QueryChunksResponse)
async def query_chunks(
    service: QueryChunksServiceDependency,
    wiki_base_id: Annotated[UUID, Query()],
    question: Annotated[str, Query(min_length=1, max_length=4000)],
    limit: Annotated[int, Query(ge=1, le=20)] = 5,
    mode: Annotated[RetrievalMode, Query()] = RetrievalMode.LITE,
) -> QueryChunksResponse:
    result = await service.query(
        wiki_base_id=wiki_base_id,
        question=question,
        limit=limit,
        mode=mode,
    )
    return QueryChunksResponse.model_validate(result)
