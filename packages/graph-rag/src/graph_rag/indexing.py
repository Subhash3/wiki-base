import logging
from collections.abc import Iterable

from llm_providers.generation.base import RecoverableGenerationError

from graph_rag.extraction import PassageEntityExtractor, TripleExtractor
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import IndexedChunk, TripleProvenance
from graph_rag.normalization import normalize_text, normalize_triple

logger = logging.getLogger(__name__)


class HippoRAGIndexer:
    """Build a provenance-aware knowledge graph from document chunks."""

    def __init__(
        self,
        *,
        extractor: TripleExtractor,
        entity_extractor: PassageEntityExtractor | None = None,
    ) -> None:
        """Configure passage entity and triple extraction."""

        self._extractor = extractor
        self._entity_extractor = entity_extractor

    async def index(self, chunks: Iterable[IndexedChunk]) -> KnowledgeGraph:
        """Extract passage concepts and build a provenance-aware graph."""

        graph = KnowledgeGraph()
        for indexed_chunk in chunks:
            provenance = TripleProvenance(
                document_id=indexed_chunk.document_id,
                chunk_id=indexed_chunk.chunk.id,
            )
            try:
                entities = (
                    await self._entity_extractor.extract(indexed_chunk.chunk)
                    if self._entity_extractor is not None
                    else []
                )
            except RecoverableGenerationError as error:
                logger.warning(
                    "Skipping passage entities for document=%s chunk=%s: %s",
                    indexed_chunk.document_id,
                    indexed_chunk.chunk.id,
                    error,
                )
                entities = []
            for entity in entities:
                normalized_entity = normalize_text(entity)
                if normalized_entity:
                    graph.add_entity(normalized_entity, provenance=provenance)

            try:
                triples = await self._extractor.extract(
                    indexed_chunk.chunk,
                    entities=entities,
                )
            except RecoverableGenerationError as error:
                logger.warning(
                    "Skipping passage triples for document=%s chunk=%s: %s",
                    indexed_chunk.document_id,
                    indexed_chunk.chunk.id,
                    error,
                )
                continue
            for triple in triples:
                normalized = normalize_triple(triple)
                if normalized is not None:
                    graph.add_triple(normalized, provenance=provenance)
        return graph
