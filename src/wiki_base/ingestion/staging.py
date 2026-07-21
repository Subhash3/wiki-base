import hashlib
import zipfile
from dataclasses import dataclass
from io import BufferedIOBase
from pathlib import Path, PurePath
from typing import BinaryIO
from uuid import UUID

from fastapi import UploadFile

from wiki_base.api.errors import ServiceError

_CHUNK_SIZE = 1024 * 1024
_MEDIA_TYPES = {
    ".pdf": frozenset({"application/pdf"}),
    ".docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    ".pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
}
_OFFICE_MARKERS = {
    ".docx": "word/document.xml",
    ".pptx": "ppt/presentation.xml",
}


@dataclass(frozen=True, slots=True)
class StagedDocument:
    path: Path
    name: str
    media_type: str
    checksum: str
    size_bytes: int


def _validate_file_signature(path: Path, extension: str) -> bool:
    if extension == ".pdf":
        with path.open("rb") as document:
            return document.read(5) == b"%PDF-"

    marker = _OFFICE_MARKERS[extension]
    try:
        with zipfile.ZipFile(path) as archive:
            return marker in archive.namelist()
    except zipfile.BadZipFile:
        return False


def _copy_upload(
    source: BinaryIO | BufferedIOBase,
    destination: Path,
    *,
    request_bytes: int,
    max_document_size_bytes: int,
    max_request_size_bytes: int,
    name: str,
) -> tuple[str, int]:
    checksum = hashlib.sha256()
    size_bytes = 0
    source.seek(0)
    with destination.open("xb") as staged_file:
        while chunk := source.read(_CHUNK_SIZE):
            size_bytes += len(chunk)
            if size_bytes > max_document_size_bytes:
                raise ServiceError(
                    "document_too_large",
                    f"Document {name!r} exceeds the configured size limit.",
                    413,
                )
            if request_bytes + size_bytes > max_request_size_bytes:
                raise ServiceError(
                    "request_too_large",
                    "The combined document upload exceeds the configured size limit.",
                    413,
                )
            checksum.update(chunk)
            staged_file.write(chunk)
    return checksum.hexdigest(), size_bytes


class DocumentStaging:
    def __init__(
        self,
        *,
        directory: Path,
        max_document_size_bytes: int,
        max_request_size_bytes: int,
    ) -> None:
        self._directory = directory
        self._max_document_size_bytes = max_document_size_bytes
        self._max_request_size_bytes = max_request_size_bytes

    async def stage(
        self,
        upload: UploadFile,
        *,
        document_id: UUID,
        request_bytes: int,
    ) -> StagedDocument:
        name = upload.filename or ""
        if not name or PurePath(name).name != name:
            raise ServiceError("invalid_filename", "A safe document filename is required.", 422)

        extension = Path(name).suffix.lower()
        expected_media_types = _MEDIA_TYPES.get(extension)
        if expected_media_types is None or upload.content_type not in expected_media_types:
            raise ServiceError(
                "unsupported_document_type",
                "Only PDF, DOCX, and PPTX documents are supported.",
                415,
            )

        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._directory / f"{document_id}{extension}"

        try:
            checksum, size_bytes = _copy_upload(
                upload.file,
                path,
                request_bytes=request_bytes,
                max_document_size_bytes=self._max_document_size_bytes,
                max_request_size_bytes=self._max_request_size_bytes,
                name=name,
            )

            if size_bytes == 0:
                raise ServiceError("empty_document", f"Document {name!r} is empty.", 422)

            signature_is_valid = _validate_file_signature(path, extension)
            if not signature_is_valid:
                raise ServiceError(
                    "invalid_document_content",
                    f"Document {name!r} does not match its declared file type.",
                    422,
                )

            return StagedDocument(
                path=path,
                name=name,
                media_type=upload.content_type,
                checksum=checksum,
                size_bytes=size_bytes,
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        finally:
            upload.file.close()

    async def cleanup(self, documents: list[StagedDocument]) -> None:
        for document in documents:
            document.path.unlink(missing_ok=True)
