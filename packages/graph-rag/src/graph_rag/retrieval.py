import logging
import re

from graph_rag.entity_linking import EntityLinker, SemanticConceptSearch
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RankedChunk
from graph_rag.normalization import normalize_text
from graph_rag.query_extraction import QueryConcepts, QueryEntityExtractor
from graph_rag.ranking import (
    aggregate_chunk_scores,
    build_ranking_graph,
    personalized_page_rank,
)

logger = logging.getLogger(__name__)

_PAGE_RANK_LOG_LIMIT = 20


class PageRankRetriever:
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
        semantic_search: SemanticConceptSearch | None = None,
    ) -> list[RankedChunk]:
        """Return the highest-ranked chunks for a question."""

        if limit < 1:
            raise ValueError("limit must be positive")

        concepts = await self._entity_extractor.extract(question)
        logger.debug(
            "Graph query concepts entities=%s relationships=%s",
            concepts.entities,
            concepts.relationships,
        )
        mentioned_nodes = find_mentioned_nodes(question, graph)
        concepts = QueryConcepts(
            entities=[*mentioned_nodes, *concepts.entities],
            relationships=concepts.relationships,
        )
        seeds = await self._entity_linker.link(
            concepts,
            graph,
            semantic_search=semantic_search,
        )
        logger.debug(
            "Graph retrieval entities=%s relationships=%s mentioned_nodes=%s seeds=%s",
            concepts.entities,
            concepts.relationships,
            mentioned_nodes,
            seeds,
        )
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
        logger.debug(
            "PageRank nodes total=%d top=%s",
            len(node_scores),
            [
                (node, round(score, 6))
                for node, score in list(node_scores.items())[:_PAGE_RANK_LOG_LIMIT]
            ],
        )
        return aggregate_chunk_scores(graph, node_scores)[:limit]


def find_mentioned_nodes(question: str, graph: KnowledgeGraph) -> list[str]:
    """Return graph nodes explicitly mentioned in the question."""

    normalized_question = normalize_text(question)
    if not normalized_question:
        return []
    return [
        node
        for node in sorted(graph.nodes)
        if re.search(rf"(?<!\w){re.escape(node)}(?!\w)", normalized_question)
    ]
