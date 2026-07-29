from docling_core.types.doc import DocItemLabel, DoclingDocument

from document_processing.chunking.docling import DoclingDocumentChunker
from document_processing.models import ParsedDocument


def test_docling_chunker_creates_contextualized_chunks_offline() -> None:
    native_document = DoclingDocument(name="policy")
    native_document.add_text(DocItemLabel.SECTION_HEADER, "Remote Work")
    native_document.add_text(
        DocItemLabel.TEXT,
        "Employees may work remotely for three days per week.",
    )
    chunker = DoclingDocumentChunker(max_tokens=100)

    chunks = chunker.chunk(
        ParsedDocument(name="policy.docx", native_document=native_document),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Employees may work remotely for three days per week."
    assert "Remote Work" in chunks[0].embedding_content
    assert chunks[0].heading == "Remote Work"
    assert chunks[0].ordinal == 0
