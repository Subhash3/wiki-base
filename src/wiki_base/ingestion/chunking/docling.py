import math
from typing import Any

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from wiki_base.ingestion.models import IngestionChunk, ParsedDocument

_PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class ApproximateTokenizer(BaseTokenizer):
    """Small offline token estimator used only to keep chunks reasonably bounded."""

    max_tokens: int = 700

    def count_tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text) / 4))

    def get_max_tokens(self) -> int:
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        return self.count_tokens


class DoclingDocumentChunker:
    def __init__(self, *, max_tokens: int) -> None:
        self._tokenizer = ApproximateTokenizer(max_tokens=max_tokens)
        self._chunker = HybridChunker(tokenizer=self._tokenizer)

    def chunk(self, document: ParsedDocument, *, media_type: str) -> list[IngestionChunk]:
        chunks: list[IngestionChunk] = []
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
                IngestionChunk(
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
