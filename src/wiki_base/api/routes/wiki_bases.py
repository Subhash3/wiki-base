from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status

from wiki_base.api.dependencies import WikiBaseServiceDependency
from wiki_base.schemas.wiki_bases import (
    WikiBaseQueuedResponse,
    WikiBaseStatusResponse,
    WikiBaseSummaryResponse,
)

router = APIRouter(prefix="/wiki-bases", tags=["wiki bases"])


@router.post("", response_model=WikiBaseQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_wiki_base(
    service: WikiBaseServiceDependency,
    name: Annotated[str, Form(min_length=1, max_length=200)],
    documents: Annotated[list[UploadFile], File()],
) -> WikiBaseQueuedResponse:
    result = await service.create(name=name, uploads=documents)
    return WikiBaseQueuedResponse.model_validate(result)


@router.get("", response_model=list[WikiBaseSummaryResponse])
async def list_wiki_bases(
    service: WikiBaseServiceDependency,
) -> list[WikiBaseSummaryResponse]:
    results = await service.list()
    return [WikiBaseSummaryResponse.model_validate(result) for result in results]


@router.get("/{wiki_base_id}/status", response_model=WikiBaseStatusResponse)
async def get_wiki_base_status(
    wiki_base_id: UUID,
    service: WikiBaseServiceDependency,
) -> WikiBaseStatusResponse:
    result = await service.get_status(wiki_base_id)
    return WikiBaseStatusResponse.model_validate(result)
