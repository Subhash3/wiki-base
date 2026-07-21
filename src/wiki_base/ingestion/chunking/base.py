from typing import Protocol

from wiki_base.ingestion.models import IngestionChunk, ParsedDocument


class DocumentChunker(Protocol):
    def chunk(self, document: ParsedDocument, *, media_type: str) -> list[IngestionChunk]: ...
