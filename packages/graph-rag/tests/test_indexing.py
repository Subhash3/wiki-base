from uuid import UUID

from document_processing.models import DocumentChunk
from llm_providers.generation.base import RecoverableGenerationError

from graph_rag.indexing import HippoRAGIndexer
from graph_rag.models import IndexedChunk, Triple, TripleProvenance

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        id=CHUNK_ID,
        ordinal=0,
        content="Alice works at Acme.",
        embedding_content="Alice works at Acme.",
        token_count=5,
        page_number=1,
        slide_number=None,
        section=None,
        heading=None,
        caption=None,
    )


class StubExtractor:
    async def extract(
        self,
        chunk: DocumentChunk,
        *,
        entities: list[str] | None = None,
    ) -> list[Triple]:
        assert chunk.id == CHUNK_ID
        assert entities == ["Alice", "Acme", "Standalone concept"]
        return [
            Triple(subject=" Alice ", relation=" WORKS AT ", object="Acme."),
            Triple(subject="", relation="invalid", object="fact"),
        ]


class StubEntityExtractor:
    async def extract(self, chunk: DocumentChunk) -> list[str]:
        assert chunk.id == CHUNK_ID
        return ["Alice", "Acme", "Standalone concept"]


async def test_indexer_extracts_normalizes_and_links_facts() -> None:
    indexed_chunk = IndexedChunk(document_id=DOCUMENT_ID, chunk=make_chunk())
    graph = await HippoRAGIndexer(
        extractor=StubExtractor(),
        entity_extractor=StubEntityExtractor(),
    ).index([indexed_chunk])
    provenance = TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID)

    assert graph.nodes == frozenset({"alice", "acme", "standalone concept"})
    assert graph.entity_provenance_for_node("standalone concept") == frozenset(
        {provenance}
    )
    assert graph.triples_for_provenance(provenance) == frozenset(
        {Triple(subject="alice", relation="works at", object="acme")}
    )


async def test_indexer_continues_when_entity_generation_is_recoverable() -> None:
    """A rejected entity response does not discard the document graph."""

    class FailingEntityExtractor:
        async def extract(self, chunk: DocumentChunk) -> list[str]:
            raise RecoverableGenerationError("structured output rejected")

    class FallbackTripleExtractor:
        async def extract(
            self,
            chunk: DocumentChunk,
            *,
            entities: list[str] | None = None,
        ) -> list[Triple]:
            assert entities == []
            return [Triple(subject="alice", relation="works at", object="acme")]

    graph = await HippoRAGIndexer(
        extractor=FallbackTripleExtractor(),
        entity_extractor=FailingEntityExtractor(),
    ).index([IndexedChunk(document_id=DOCUMENT_ID, chunk=make_chunk())])

    assert graph.nodes == frozenset({"alice", "acme"})


async def test_indexer_keeps_entities_when_triple_generation_is_recoverable() -> None:
    """A rejected triple response skips its facts but retains passage entities."""

    class FailingTripleExtractor:
        async def extract(
            self,
            chunk: DocumentChunk,
            *,
            entities: list[str] | None = None,
        ) -> list[Triple]:
            raise RecoverableGenerationError("structured output rejected")

    graph = await HippoRAGIndexer(
        extractor=FailingTripleExtractor(),
        entity_extractor=StubEntityExtractor(),
    ).index([IndexedChunk(document_id=DOCUMENT_ID, chunk=make_chunk())])

    assert graph.nodes == frozenset({"alice", "acme", "standalone concept"})
    assert list(graph.edges()) == []
