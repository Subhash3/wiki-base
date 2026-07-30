from collections.abc import Iterable

from graph_rag.extraction import TripleExtractor
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import IndexedChunk, TripleProvenance
from graph_rag.normalization import normalize_triple


class HippoRAGIndexer:
    """Build a provenance-aware knowledge graph from document chunks."""

    def __init__(self, *, extractor: TripleExtractor) -> None:
        self._extractor = extractor

    async def index(self, chunks: Iterable[IndexedChunk]) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        for indexed_chunk in chunks:
            triples = await self._extractor.extract(indexed_chunk.chunk)
            provenance = TripleProvenance(
                document_id=indexed_chunk.document_id,
                chunk_id=indexed_chunk.chunk.id,
            )
            for triple in triples:
                normalized = normalize_triple(triple)
                if normalized is not None:
                    graph.add_triple(normalized, provenance=provenance)
        return graph
