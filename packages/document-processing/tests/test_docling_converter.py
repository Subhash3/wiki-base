from typing import Any

from docling.datamodel.base_models import InputFormat

from document_processing.parsing import docling_converter
from document_processing.parsing.docling_converter import DoclingConverter


def test_configures_rapidocr_for_pdf_conversion(monkeypatch) -> None:
    """The converter passes explicit OCR settings to the PDF pipeline."""

    captured: dict[str, Any] = {}

    class StubDocumentConverter:
        """Capture converter initialization."""

        def __init__(self, **options: Any) -> None:
            captured.update(options)

    monkeypatch.setattr(
        docling_converter,
        "DocumentConverter",
        StubDocumentConverter,
    )

    DoclingConverter(
        ocr_languages=["english"],
        force_full_page_ocr=True,
    )

    pdf_format = captured["format_options"][InputFormat.PDF]
    pdf_options = pdf_format.pipeline_options
    assert pdf_options.do_ocr is True
    assert pdf_options.ocr_options.lang == ["english"]
    assert pdf_options.ocr_options.force_full_page_ocr is True
