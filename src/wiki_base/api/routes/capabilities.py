from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["operations"])


class DocumentFormat(BaseModel):
    extension: Literal[".pdf", ".docx", ".pptx"]
    media_type: str


class CapabilitiesResponse(BaseModel):
    document_formats: list[DocumentFormat]
    reranking: bool = False
    streaming: bool = False


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(
        document_formats=[
            DocumentFormat(extension=".pdf", media_type="application/pdf"),
            DocumentFormat(
                extension=".docx",
                media_type=(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ),
            ),
            DocumentFormat(
                extension=".pptx",
                media_type=(
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                ),
            ),
        ]
    )
