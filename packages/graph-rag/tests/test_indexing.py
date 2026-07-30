from uuid import UUID

from document_processing.models import DocumentChunk

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
    async def extract(self, chunk: DocumentChunk) -> list[Triple]:
        assert chunk.id == CHUNK_ID
        return [
            Triple(subject=" Alice ", relation=" WORKS AT ", object="Acme."),
            Triple(subject="", relation="invalid", object="fact"),
        ]


async def test_indexer_extracts_normalizes_and_links_facts() -> None:
    indexed_chunk = IndexedChunk(document_id=DOCUMENT_ID, chunk=make_chunk())
    graph = await HippoRAGIndexer(extractor=StubExtractor()).index([indexed_chunk])
    provenance = TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID)

    assert graph.nodes == frozenset({"alice", "acme"})
    assert graph.triples_for_provenance(provenance) == frozenset(
        {Triple(subject="alice", relation="works at", object="acme")}
    )
