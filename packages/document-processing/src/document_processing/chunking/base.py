from typing import Protocol

from document_processing.models import DocumentChunk, ParsedDocument


class DocumentChunker(Protocol):
    def chunk(self, document: ParsedDocument, *, media_type: str) -> list[DocumentChunk]: ...
