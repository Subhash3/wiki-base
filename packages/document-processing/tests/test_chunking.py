from typing import Any

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.labels import DocItemLabel

from document_processing.chunking.docling import DoclingDocumentChunker
from document_processing.models import ParsedDocument


class WordTokenizer(BaseTokenizer):
    """Count whitespace-delimited words for offline tests."""

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        """Count words in the supplied text."""

        return len(text.split())

    def get_max_tokens(self) -> int:
        """Return the configured chunk limit."""

        return self.max_tokens

    def get_tokenizer(self) -> Any:
        """Return the underlying test tokenizer."""

        return self


def test_docling_chunker_creates_contextualized_chunks_offline() -> None:
    native_document = DoclingDocument(name="policy")
    native_document.add_text(DocItemLabel.SECTION_HEADER, "Remote Work")
    native_document.add_text(
        DocItemLabel.TEXT,
        "Employees may work remotely for three days per week.",
    )
    tokenizer = WordTokenizer(max_tokens=100)
    chunker = DoclingDocumentChunker(max_tokens=100, tokenizer=tokenizer)

    chunks = chunker.chunk(
        ParsedDocument(name="policy.docx", native_document=native_document),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert len(chunks) == 1
    assert chunks[0].content == "Employees may work remotely for three days per week."
    assert "Remote Work" in chunks[0].embedding_content
    assert chunks[0].heading == "Remote Work"
    assert chunks[0].ordinal == 0
    assert chunks[0].token_count == tokenizer.count_tokens(chunks[0].embedding_content)


def test_docling_chunker_loads_the_configured_hugging_face_tokenizer(
    monkeypatch,
) -> None:
    loaded: dict[str, object] = {}
    tokenizer = WordTokenizer(max_tokens=512)

    def from_pretrained(
        _cls,
        model_name: str,
        *,
        max_tokens: int,
    ) -> BaseTokenizer:
        loaded["model_name"] = model_name
        loaded["max_tokens"] = max_tokens
        return tokenizer

    monkeypatch.setattr(
        HuggingFaceTokenizer,
        "from_pretrained",
        classmethod(from_pretrained),
    )

    DoclingDocumentChunker(
        max_tokens=512,
        tokenizer_model="BAAI/bge-m3",
    )

    assert loaded == {
        "model_name": "BAAI/bge-m3",
        "max_tokens": 512,
    }
