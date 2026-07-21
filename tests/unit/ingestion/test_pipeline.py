from pathlib import Path

from wiki_base.ingestion.models import DocumentSource, IngestionChunk, ParsedDocument
from wiki_base.ingestion.pipeline import IngestionPipeline


class StubParser:
    supported_extensions = frozenset({".pdf"})
    supported_media_types = frozenset({"application/pdf"})

    def parse(self, source: DocumentSource) -> ParsedDocument:
        return ParsedDocument(name=source.name, native_document=object())


class StubRegistry:
    def resolve(self, filename: str, media_type: str) -> StubParser:
        assert filename == "policy.pdf"
        assert media_type == "application/pdf"
        return StubParser()


class StubChunker:
    def chunk(self, document: ParsedDocument, *, media_type: str) -> list[IngestionChunk]:
        return [
            IngestionChunk(
                ordinal=index,
                content=f"chunk {index}",
                embedding_content=f"context chunk {index}",
                token_count=3,
                page_number=1,
                slide_number=None,
                section=None,
                heading=None,
                caption=None,
            )
            for index in range(3)
        ]


class StubEmbeddingProvider:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batch_sizes.append(len(texts))
        return [[float(index)] for index, _text in enumerate(texts)]

    async def embed_query(self, text: str) -> list[float]:
        return [0.0]


async def test_pipeline_embeds_chunks_in_configured_batches() -> None:
    provider = StubEmbeddingProvider()
    pipeline = IngestionPipeline(
        parser_registry=StubRegistry(),  # type: ignore[arg-type]
        chunker=StubChunker(),
        embedding_provider=provider,  # type: ignore[arg-type]
        embedding_batch_size=2,
    )

    chunks = await pipeline.ingest(
        DocumentSource(
            path=Path("policy.pdf"),
            name="policy.pdf",
            media_type="application/pdf",
        )
    )

    assert len(chunks) == 3
    assert provider.batch_sizes == [2, 1]
