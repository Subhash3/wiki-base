from uuid import uuid4

from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)

from document_processing.models import DocumentChunk, ParsedDocument

_PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"


class DoclingDocumentChunker:
    """Chunk Docling documents with an embedding-compatible tokenizer."""

    def __init__(
        self,
        *,
        max_tokens: int,
        tokenizer_model: str = _DEFAULT_TOKENIZER_MODEL,
        tokenizer: BaseTokenizer | None = None,
    ) -> None:
        """Configure chunk size and tokenizer."""

        self._tokenizer = tokenizer or HuggingFaceTokenizer.from_pretrained(
            tokenizer_model,
            max_tokens=max_tokens,
        )
        self._chunker = HybridChunker(tokenizer=self._tokenizer)

    def chunk(self, document: ParsedDocument, *, media_type: str) -> list[DocumentChunk]:
        """Create contextualized chunks from a parsed Docling document."""

        chunks: list[DocumentChunk] = []
        for chunk in self._chunker.chunk(document.native_document):
            content = chunk.text.strip()
            if not content:
                continue

            embedding_content = self._chunker.contextualize(chunk).strip()
            headings = list(getattr(chunk.meta, "headings", None) or [])
            page_numbers = sorted(
                {
                    provenance.page_no
                    for item in getattr(chunk.meta, "doc_items", [])
                    for provenance in getattr(item, "prov", [])
                }
            )
            location = page_numbers[0] if page_numbers else None
            is_presentation = media_type == _PPTX_MEDIA_TYPE

            chunks.append(
                DocumentChunk(
                    id=uuid4(),
                    ordinal=len(chunks),
                    content=content,
                    embedding_content=embedding_content,
                    token_count=self._tokenizer.count_tokens(embedding_content),
                    page_number=None if is_presentation else location,
                    slide_number=location if is_presentation else None,
                    section=" > ".join(headings) or None,
                    heading=headings[-1] if headings else None,
                    caption=None,
                    metadata={"source_pages": page_numbers},
                )
            )
        return chunks
