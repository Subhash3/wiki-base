from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument


class DoclingConverter:
    def __init__(self) -> None:
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX]
        )

    def convert(self, path: str, input_format: InputFormat) -> DoclingDocument:
        result = self._converter.convert(path, raises_on_error=True)
        if result.input.format != input_format:
            raise ValueError(
                f"Expected {input_format.value!r}, detected {result.input.format.value!r}"
            )
        return result.document
