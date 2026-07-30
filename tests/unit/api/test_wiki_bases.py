from datetime import UTC, datetime
from io import BytesIO
from uuid import UUID

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from wiki_base.api.routes.wiki_bases import (
    create_wiki_base,
    get_wiki_base_status,
    list_wiki_bases,
)
from wiki_base.database.queries.wiki_bases import _aggregate_status
from wiki_base.database.records import IngestionStatus
from wiki_base.retrieval import RetrievalMode
from wiki_base.services.wiki_bases import (
    DocumentStatus,
    QueuedDocument,
    QueuedWikiBase,
    WikiBaseStatus,
    WikiBaseSummary,
)


class StubWikiBaseService:
    async def create(self, *, name: str, uploads: list[UploadFile]) -> QueuedWikiBase:
        assert len(uploads) == 2
        return QueuedWikiBase(
            id=UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705"),
            name=name,
            created_at=datetime(2026, 7, 21, tzinfo=UTC),
            retrieval_statuses={
                RetrievalMode.LITE: IngestionStatus.QUEUED,
                RetrievalMode.PRO: IngestionStatus.QUEUED,
            },
            documents=[
                QueuedDocument(
                    id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
                    name="policy.pdf",
                    media_type="application/pdf",
                ),
                QueuedDocument(
                    id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6df"),
                    name="handbook.docx",
                    media_type=(
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ),
                ),
            ],
        )

    async def get_status(self, wiki_base_id: UUID) -> WikiBaseStatus:
        return WikiBaseStatus(
            id=wiki_base_id,
            name="Engineering Handbook",
            retrieval_statuses={
                RetrievalMode.LITE: IngestionStatus.PROCESSING,
                RetrievalMode.PRO: IngestionStatus.QUEUED,
            },
            document_count=1,
            created_at=datetime(2026, 7, 21, tzinfo=UTC),
            started_at=datetime(2026, 7, 21, 0, 1, tzinfo=UTC),
            completed_at=None,
            documents=[
                DocumentStatus(
                    id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
                    name="policy.pdf",
                    media_type="application/pdf",
                    status=IngestionStatus.PROCESSING,
                    error_code=None,
                    error_message=None,
                )
            ],
        )

    async def list(self) -> list[WikiBaseSummary]:
        return [
            WikiBaseSummary(
                id=UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705"),
                name="Engineering Handbook",
                retrieval_statuses={
                    RetrievalMode.LITE: IngestionStatus.READY,
                    RetrievalMode.PRO: IngestionStatus.PROCESSING,
                },
                document_count=2,
                created_at=datetime(2026, 7, 21, tzinfo=UTC),
                started_at=datetime(2026, 7, 21, 0, 1, tzinfo=UTC),
                completed_at=datetime(2026, 7, 21, 0, 2, tzinfo=UTC),
            )
        ]


async def test_create_wiki_base_returns_queued_manifest() -> None:
    uploads = [
        UploadFile(
            file=BytesIO(b"%PDF-test"),
            filename="policy.pdf",
            headers=Headers({"content-type": "application/pdf"}),
        ),
        UploadFile(
            file=BytesIO(b"test"),
            filename="handbook.docx",
            headers=Headers(
                {
                    "content-type": (
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                }
            ),
        ),
    ]

    response = await create_wiki_base(
        service=StubWikiBaseService(),
        name="Engineering Handbook",
        documents=uploads,
    )

    assert response.name == "Engineering Handbook"
    assert response.retrieval_statuses == {
        RetrievalMode.LITE: "queued",
        RetrievalMode.PRO: "queued",
    }
    assert [document.name for document in response.documents] == [
        "policy.pdf",
        "handbook.docx",
    ]


async def test_get_wiki_base_status_returns_document_progress() -> None:
    wiki_base_id = UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705")

    response = await get_wiki_base_status(
        wiki_base_id=wiki_base_id,
        service=StubWikiBaseService(),
    )

    assert response.id == wiki_base_id
    assert response.retrieval_statuses[RetrievalMode.LITE] == "processing"
    assert response.retrieval_statuses[RetrievalMode.PRO] == "queued"
    assert response.document_count == 1
    assert response.documents[0].name == "policy.pdf"
    assert response.documents[0].status == "processing"


async def test_list_wiki_bases_returns_summaries() -> None:
    response = await list_wiki_bases(service=StubWikiBaseService())

    assert len(response) == 1
    assert response[0].name == "Engineering Handbook"
    assert response[0].retrieval_statuses[RetrievalMode.LITE] == "ready"
    assert response[0].retrieval_statuses[RetrievalMode.PRO] == "processing"
    assert response[0].document_count == 2


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([IngestionStatus.QUEUED], IngestionStatus.QUEUED),
        (
            [IngestionStatus.READY, IngestionStatus.QUEUED],
            IngestionStatus.PROCESSING,
        ),
        (
            [IngestionStatus.READY, IngestionStatus.PROCESSING],
            IngestionStatus.PROCESSING,
        ),
        ([IngestionStatus.READY, IngestionStatus.READY], IngestionStatus.READY),
        ([IngestionStatus.FAILED, IngestionStatus.FAILED], IngestionStatus.FAILED),
        (
            [IngestionStatus.READY, IngestionStatus.FAILED],
            IngestionStatus.PARTIALLY_FAILED,
        ),
    ],
)
def test_aggregates_retrieval_statuses(
    statuses: list[IngestionStatus],
    expected: IngestionStatus,
) -> None:
    assert _aggregate_status(statuses) == expected
