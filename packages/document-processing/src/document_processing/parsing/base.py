from typing import Protocol

from document_processing.models import DocumentSource, ParsedDocument


class DocumentParser(Protocol):
    supported_extensions: frozenset[str]
    supported_media_types: frozenset[str]

    def parse(self, source: DocumentSource) -> ParsedDocument: ...
