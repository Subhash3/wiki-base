from io import BytesIO
from pathlib import Path
from uuid import UUID

import pytest
from fastapi import UploadFile
from starlette.datastructures import Headers

from wiki_base.api.errors import ServiceError
from wiki_base.ingestion.staging import DocumentStaging


def make_upload(filename: str, content: bytes, media_type: str) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": media_type}),
    )


@pytest.fixture
def staging(tmp_path: Path) -> DocumentStaging:
    return DocumentStaging(
        directory=tmp_path / "staging",
        max_document_size_bytes=1024,
        max_request_size_bytes=2048,
    )


async def test_stages_and_cleans_up_pdf(staging: DocumentStaging) -> None:
    upload = make_upload("policy.pdf", b"%PDF-1.7\ncontent", "application/pdf")

    document = await staging.stage(
        upload,
        document_id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
        request_bytes=0,
    )

    assert document.name == "policy.pdf"
    assert document.size_bytes == 16
    assert document.path.exists()

    await staging.cleanup([document])

    assert not document.path.exists()


async def test_rejects_content_that_does_not_match_extension(
    staging: DocumentStaging,
) -> None:
    upload = make_upload("policy.pdf", b"not a PDF", "application/pdf")

    with pytest.raises(ServiceError) as error:
        await staging.stage(
            upload,
            document_id=UUID("0190f3a0-b096-7af5-8392-cc61de46f6de"),
            request_bytes=0,
        )

    assert error.value.code == "invalid_document_content"
