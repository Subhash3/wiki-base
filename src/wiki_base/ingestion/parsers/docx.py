from docling.datamodel.base_models import InputFormat

from wiki_base.ingestion.models import DocumentSource, ParsedDocument
from wiki_base.ingestion.parsers.docling_converter import DoclingConverter


class DocxDocumentParser:
    supported_extensions = frozenset({".docx"})
    supported_media_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    )

    def __init__(self, converter: DoclingConverter) -> None:
        self._converter = converter

    def parse(self, source: DocumentSource) -> ParsedDocument:
        document = self._converter.convert(str(source.path), InputFormat.DOCX)
        return ParsedDocument(name=source.name, native_document=document)
