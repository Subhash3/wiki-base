from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import asyncpg
from fastapi import UploadFile

from wiki_base.api.errors import ServiceError
from wiki_base.database.connection import Database
from wiki_base.database.queries.documents import list_wiki_base_documents
from wiki_base.database.queries.wiki_bases import (
    create_wiki_base_manifest,
    get_wiki_base,
    list_wiki_bases,
)
from wiki_base.database.records import IngestionStatus
from wiki_base.ingestion.staging import DocumentStaging, StagedDocument


@dataclass(frozen=True, slots=True)
class QueuedDocument:
    id: UUID
    name: str
    media_type: str
    status: str = "queued"


@dataclass(frozen=True, slots=True)
class QueuedWikiBase:
    id: UUID
    name: str
    created_at: datetime
    documents: list[QueuedDocument]
    status: str = "queued"


@dataclass(frozen=True, slots=True)
class DocumentStatus:
    id: UUID
    name: str
    media_type: str
    status: IngestionStatus
    error_code: str | None
    error_message: str | None


@dataclass(frozen=True, slots=True)
class WikiBaseStatus:
    id: UUID
    name: str
    status: IngestionStatus
    document_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    documents: list[DocumentStatus]


@dataclass(frozen=True, slots=True)
class WikiBaseSummary:
    id: UUID
    name: str
    status: IngestionStatus
    document_count: int
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class WikiBaseService:
    def __init__(
        self,
        *,
        database: Database,
        staging: DocumentStaging,
        max_documents: int,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self._database = database
        self._staging = staging
        self._max_documents = max_documents
        self._embedding_model = embedding_model
        self._embedding_dimensions = embedding_dimensions

    async def create(self, *, name: str, uploads: list[UploadFile]) -> QueuedWikiBase:
        normalized_name = name.strip()
        if not normalized_name:
            raise ServiceError("invalid_name", "Wiki base name cannot be blank.", 422)
        if not uploads:
            raise ServiceError("documents_required", "At least one document is required.", 422)
        if len(uploads) > self._max_documents:
            raise ServiceError(
                "too_many_documents",
                f"At most {self._max_documents} documents may be uploaded at once.",
                413,
            )

        staged_documents: list[StagedDocument] = []
        document_ids: list[UUID] = []
        request_bytes = 0
        try:
            for upload in uploads:
                doc_id = uuid4()
                staged = await self._staging.stage(
                    upload, document_id=doc_id, request_bytes=request_bytes
                )
                staged_documents.append(staged)
                document_ids.append(doc_id)
                request_bytes += staged.size_bytes

            checksums = [document.checksum for document in staged_documents]
            if len(checksums) != len(set(checksums)):
                raise ServiceError(
                    "duplicate_document",
                    "The upload contains duplicate document content.",
                    409,
                )

            wiki_base_id = uuid4()
            async with self._database.connection() as connection:
                created_at = await create_wiki_base_manifest(
                    connection,
                    wiki_base_id=wiki_base_id,
                    name=normalized_name,
                    documents=list(zip(document_ids, staged_documents, strict=True)),
                    embedding_model=self._embedding_model,
                    embedding_dimensions=self._embedding_dimensions,
                )

            return QueuedWikiBase(
                id=wiki_base_id,
                name=normalized_name,
                created_at=created_at,
                documents=[
                    QueuedDocument(
                        id=document_id,
                        name=document.name,
                        media_type=document.media_type,
                    )
                    for document_id, document in zip(
                        document_ids,
                        staged_documents,
                        strict=True,
                    )
                ],
            )
        except ServiceError:
            await self._staging.cleanup(staged_documents)
            raise
        except asyncpg.PostgresError as error:
            await self._staging.cleanup(staged_documents)
            raise ServiceError(
                "database_unavailable",
                "The wiki base could not be queued.",
                503,
            ) from error
        except BaseException:
            await self._staging.cleanup(staged_documents)
            raise

    async def get_status(self, wiki_base_id: UUID) -> WikiBaseStatus:
        async with self._database.connection() as connection:
            wiki_base = await get_wiki_base(connection, wiki_base_id)
            if wiki_base is None:
                raise ServiceError(
                    "wiki_base_not_found",
                    "The requested wiki base was not found.",
                    404,
                )
            documents = await list_wiki_base_documents(connection, wiki_base_id)

        document_statuses = [
            DocumentStatus(
                id=document.id,
                name=document.name,
                media_type=document.media_type,
                status=document.status,
                error_code=document.error_code,
                error_message=document.error_message,
            )
            for document in documents
        ]
        return WikiBaseStatus(
            id=wiki_base.id,
            name=wiki_base.name,
            status=wiki_base.status,
            document_count=len(document_statuses),
            created_at=wiki_base.created_at,
            started_at=wiki_base.started_at,
            completed_at=wiki_base.completed_at,
            documents=document_statuses,
        )

    async def list(self) -> list[WikiBaseSummary]:
        try:
            async with self._database.connection() as connection:
                records = await list_wiki_bases(connection)
        except asyncpg.PostgresError as error:
            raise ServiceError(
                "database_unavailable",
                "Wiki bases could not be listed right now.",
                503,
            ) from error

        return [
            WikiBaseSummary(
                id=wiki_base.id,
                name=wiki_base.name,
                status=wiki_base.status,
                document_count=document_count,
                created_at=wiki_base.created_at,
                started_at=wiki_base.started_at,
                completed_at=wiki_base.completed_at,
            )
            for wiki_base, document_count in records
        ]
