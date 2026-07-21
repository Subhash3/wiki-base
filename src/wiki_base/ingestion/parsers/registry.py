from pathlib import Path

from wiki_base.ingestion.parsers.base import DocumentParser


class UnsupportedDocumentTypeError(ValueError):
    pass


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._by_extension: dict[str, DocumentParser] = {}
        for parser in parsers:
            for extension in parser.supported_extensions:
                if extension in self._by_extension:
                    raise ValueError(f"Parser already registered for {extension}")
                self._by_extension[extension] = parser

    def resolve(self, filename: str, media_type: str) -> DocumentParser:
        extension = Path(filename).suffix.lower()
        parser = self._by_extension.get(extension)
        if parser is None or media_type not in parser.supported_media_types:
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type: extension={extension!r}, media_type={media_type!r}"
            )
        return parser
