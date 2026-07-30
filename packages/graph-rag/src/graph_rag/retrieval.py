from graph_rag.entity_linking import EntityLinker
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RankedChunk
from graph_rag.query_extraction import QueryEntityExtractor
from graph_rag.ranking import (
    aggregate_chunk_scores,
    build_ranking_graph,
    personalized_page_rank,
)


class HippoRAGRetriever:
    """Retrieve chunks through entity linking and Personalized PageRank."""

    def __init__(
        self,
        *,
        entity_extractor: QueryEntityExtractor,
        entity_linker: EntityLinker,
        alpha: float = 0.85,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> None:
        """Configure the retrieval components and PageRank settings."""

        self._entity_extractor = entity_extractor
        self._entity_linker = entity_linker
        self._alpha = alpha
        self._max_iterations = max_iterations
        self._tolerance = tolerance

    async def retrieve(
        self,
        question: str,
        graph: KnowledgeGraph,
        *,
        limit: int,
    ) -> list[RankedChunk]:
        """Return the highest-ranked chunks for a question."""

        if limit < 1:
            raise ValueError("limit must be positive")

        entities = await self._entity_extractor.extract(question)
        seeds = self._entity_linker.link(entities, graph)
        if not seeds:
            return []

        ranking_graph = build_ranking_graph(graph)
        node_scores = personalized_page_rank(
            ranking_graph,
            seeds,
            alpha=self._alpha,
            max_iterations=self._max_iterations,
            tolerance=self._tolerance,
        )
        return aggregate_chunk_scores(graph, node_scores)[:limit]
