from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

WikiBaseStatus = Literal["queued", "processing", "ready", "partially_failed", "failed"]
DocumentStatus = Literal["queued", "processing", "ready", "failed"]


class QueuedDocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    media_type: str
    status: Literal["queued"] = "queued"


class WikiBaseQueuedResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    status: Literal["queued"] = "queued"
    created_at: datetime
    documents: list[QueuedDocumentResponse]


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    media_type: str
    status: DocumentStatus
    error_code: str | None = None
    error_message: str | None = None


class WikiBaseStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    status: WikiBaseStatus
    document_count: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    documents: list[DocumentStatusResponse]


class WikiBaseSummaryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    status: WikiBaseStatus
    document_count: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
