from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from uuid import UUID


class IngestionStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    READY = "ready"
    PARTIALLY_FAILED = "partially_failed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class WikiBaseRecord:
    id: UUID
    name: str
    status: IngestionStatus
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: UUID
    wiki_base_id: UUID
    name: str
    media_type: str
    status: IngestionStatus
    created_at: datetime
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionJobRecord:
    id: UUID
    wiki_base_id: UUID
    document_id: UUID
    document_name: str
    media_type: str
    staging_path: Path


@dataclass(frozen=True, slots=True)
class GraphIndexingJobRecord:
    """A document claimed for graph indexing."""

    document_id: UUID
