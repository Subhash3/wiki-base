from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    RapidOcrOptions,
    ThreadedPdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument


class DoclingConverter:
    """Convert supported documents with an explicitly configured PDF pipeline."""

    def __init__(
        self,
        *,
        ocr_languages: list[str] | None = None,
        force_full_page_ocr: bool = False,
    ) -> None:
        """Configure RapidOCR for PDF conversion."""

        pdf_options = ThreadedPdfPipelineOptions(
            do_ocr=True,
            ocr_options=RapidOcrOptions(
                lang=ocr_languages if ocr_languages is not None else ["english"],
                force_full_page_ocr=force_full_page_ocr,
            ),
        )
        self._converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options),
            },
        )

    def convert(self, path: str, input_format: InputFormat) -> DoclingDocument:
        """Convert one document and validate its detected format."""

        result = self._converter.convert(path, raises_on_error=True)
        if result.input.format != input_format:
            raise ValueError(
                f"Expected {input_format.value!r}, detected {result.input.format.value!r}"
            )
        return result.document
