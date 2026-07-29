from docling.datamodel.base_models import InputFormat

from document_processing.models import DocumentSource, ParsedDocument
from document_processing.parsing.docling_converter import DoclingConverter


class PdfDocumentParser:
    supported_extensions = frozenset({".pdf"})
    supported_media_types = frozenset({"application/pdf"})

    def __init__(self, converter: DoclingConverter) -> None:
        self._converter = converter

    def parse(self, source: DocumentSource) -> ParsedDocument:
        document = self._converter.convert(str(source.path), InputFormat.PDF)
        return ParsedDocument(name=source.name, native_document=document)
