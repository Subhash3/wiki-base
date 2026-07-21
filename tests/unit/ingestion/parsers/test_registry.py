import pytest

from wiki_base.ingestion.parsers.registry import (
    ParserRegistry,
    UnsupportedDocumentTypeError,
)


class StubPdfParser:
    supported_extensions = frozenset({".pdf"})
    supported_media_types = frozenset({"application/pdf"})

    def parse(self, source: object) -> object:
        return source


def test_resolves_parser_by_extension_and_media_type() -> None:
    parser = StubPdfParser()
    registry = ParserRegistry([parser])

    assert registry.resolve("POLICY.PDF", "application/pdf") is parser


def test_rejects_media_type_mismatch() -> None:
    registry = ParserRegistry([StubPdfParser()])

    with pytest.raises(UnsupportedDocumentTypeError):
        registry.resolve("policy.pdf", "text/plain")
