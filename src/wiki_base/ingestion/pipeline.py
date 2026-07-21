from wiki_base.ingestion.chunking.base import DocumentChunker
from wiki_base.ingestion.models import DocumentSource, EmbeddedChunk
from wiki_base.ingestion.parsers.registry import ParserRegistry
from wiki_base.providers.embeddings.base import EmbeddingProvider


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser_registry: ParserRegistry,
        chunker: DocumentChunker,
        embedding_provider: EmbeddingProvider,
        embedding_batch_size: int,
    ) -> None:
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._embedding_batch_size = embedding_batch_size

    async def ingest(self, source: DocumentSource) -> list[EmbeddedChunk]:
        parser = self._parser_registry.resolve(source.name, source.media_type)
        parsed_document = parser.parse(source)
        chunks = self._chunker.chunk(parsed_document, media_type=source.media_type)
        if not chunks:
            raise ValueError("The document did not contain any usable text")

        embedded_chunks: list[EmbeddedChunk] = []
        for start in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[start : start + self._embedding_batch_size]
            embeddings = await self._embedding_provider.embed_documents(
                [chunk.embedding_content for chunk in batch]
            )
            embedded_chunks.extend(
                EmbeddedChunk(chunk=chunk, embedding=embedding)
                for chunk, embedding in zip(batch, embeddings, strict=True)
            )
        return embedded_chunks
