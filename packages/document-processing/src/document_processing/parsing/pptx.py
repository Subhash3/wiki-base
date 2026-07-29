from docling.datamodel.base_models import InputFormat

from document_processing.models import DocumentSource, ParsedDocument
from document_processing.parsing.docling_converter import DoclingConverter


class PptxDocumentParser:
    supported_extensions = frozenset({".pptx"})
    supported_media_types = frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    )

    def __init__(self, converter: DoclingConverter) -> None:
        self._converter = converter

    def parse(self, source: DocumentSource) -> ParsedDocument:
        document = self._converter.convert(str(source.path), InputFormat.PPTX)
        return ParsedDocument(name=source.name, native_document=document)
