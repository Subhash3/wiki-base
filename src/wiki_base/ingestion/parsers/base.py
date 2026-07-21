from typing import Protocol

from wiki_base.ingestion.models import DocumentSource, ParsedDocument


class DocumentParser(Protocol):
    supported_extensions: frozenset[str]
    supported_media_types: frozenset[str]

    def parse(self, source: DocumentSource) -> ParsedDocument: ...
