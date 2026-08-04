import logging
import math

from llm_providers.embeddings.base import EmbeddingProvider

from graph_rag.concepts import edge_text
from graph_rag.entity_linking import EntityLinker, SemanticConceptSearch
from graph_rag.fact_traversal import GraphFactTraverser
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import (
    FactRetrievalResult,
    GraphFact,
    RankedChunk,
    RankedFact,
    TripleProvenance,
)
from graph_rag.query_extraction import QueryConcepts, QueryEntityExtractor
from graph_rag.retrieval import find_mentioned_nodes

logger = logging.getLogger(__name__)


class FactRetriever:
    """Retrieve query-relevant facts and their supporting chunks."""

    def __init__(
        self,
        *,
        entity_extractor: QueryEntityExtractor,
        entity_linker: EntityLinker,
        embeddings: EmbeddingProvider,
        traverser: GraphFactTraverser | None = None,
        max_facts: int = 12,
        depth_decay: float = 0.75,
        coverage_bonus: float = 0.1,
    ) -> None:
        """Configure fact traversal and relevance scoring."""

        if max_facts < 1:
            raise ValueError("max_facts must be positive")
        if not 0 < depth_decay <= 1:
            raise ValueError("depth_decay must be between zero and one")
        if coverage_bonus < 0:
            raise ValueError("coverage_bonus cannot be negative")
        self._entity_extractor = entity_extractor
        self._entity_linker = entity_linker
        self._embeddings = embeddings
        self._traverser = traverser or GraphFactTraverser()
        self._max_facts = max_facts
        self._depth_decay = depth_decay
        self._coverage_bonus = coverage_bonus

    async def retrieve(
        self,
        question: str,
        graph: KnowledgeGraph,
        *,
        limit: int,
        semantic_search: SemanticConceptSearch | None = None,
    ) -> FactRetrievalResult:
        """Return ranked facts and chunks for one question."""

        if limit < 1:
            raise ValueError("limit must be positive")

        concepts = await self._entity_extractor.extract(question)
        mentioned_nodes = find_mentioned_nodes(question, graph)
        query_entities = concepts.entities or mentioned_nodes
        entity_concepts = QueryConcepts(
            entities=query_entities,
            relationships=[],
        )
        seeds = await self._entity_linker.link(
            entity_concepts,
            graph,
            semantic_search=semantic_search,
        )
        logger.debug(
            "Fact retrieval entities=%s relationships=%s mentioned_nodes=%s seeds=%s",
            concepts.entities,
            concepts.relationships,
            mentioned_nodes,
            seeds,
        )
        if not seeds:
            return FactRetrievalResult(facts=[], chunks=[])

        candidates = self._traverser.traverse(graph, seeds)
        if not candidates:
            return FactRetrievalResult(facts=[], chunks=[])
        logger.debug(
            "Fact traversal produced %d candidates from seeds=%s max_depth=%d max_candidates=%d",
            len(candidates),
            seeds,
            self._traverser.max_depth,
            self._traverser.max_candidates,
        )

        query_phrases = list(dict.fromkeys([*concepts.relationships, question.strip()]))
        similarities = await self._score_facts(
            query_phrases,
            candidates,
            semantic_search=semantic_search,
        )
        ranked_facts = [
            RankedFact(
                fact=fact,
                score=similarities.get(_fact_key(fact), 0.0)
                * self._depth_decay ** (fact.depth - 1),
            )
            for fact in candidates
        ]
        ranked_facts.sort(
            key=lambda item: (
                -item.score,
                item.fact.depth,
                item.fact.subject,
                item.fact.relation,
                item.fact.object,
            )
        )
        selected = _select_facts(ranked_facts, seeds, limit=self._max_facts)
        logger.debug(
            "Fact retrieval ranked facts=%s",
            [
                {
                    "subject": item.fact.subject,
                    "relation": item.fact.relation,
                    "object": item.fact.object,
                    "score": round(item.score, 6),
                    "depth": item.fact.depth,
                    "seeds": sorted(item.fact.seeds),
                    "provenance": [
                        {
                            "document_id": str(source.document_id),
                            "chunk_id": str(source.chunk_id),
                        }
                        for source in sorted(
                            item.fact.provenance,
                            key=lambda source: (
                                source.document_id.int,
                                source.chunk_id.int,
                            ),
                        )
                    ],
                }
                for item in selected
            ],
        )
        return FactRetrievalResult(
            facts=selected,
            chunks=_rank_chunks(
                selected,
                coverage_bonus=self._coverage_bonus,
            )[:limit],
        )

    async def _score_facts(
        self,
        queries: list[str],
        facts: list[GraphFact],
        *,
        semantic_search: SemanticConceptSearch | None,
    ) -> dict[tuple[str, str, str], float]:
        """Return the best semantic similarity for each candidate fact."""

        if semantic_search is None:
            return await self._score_facts_in_memory(queries, facts)

        candidate_keys = frozenset(_fact_key(fact) for fact in facts)
        scores: dict[tuple[str, str, str], float] = {}
        for query in queries:
            embedding = await self._embeddings.embed_query(query)
            matches = await semantic_search.search_relationships(
                embedding,
                threshold=-1.0,
                limit=len(candidate_keys),
                candidate_keys=candidate_keys,
            )
            for match in matches:
                key = (match.subject, match.relationship, match.object)
                scores[key] = max(scores.get(key, -1.0), match.similarity)
        return scores

    async def _score_facts_in_memory(
        self,
        queries: list[str],
        facts: list[GraphFact],
    ) -> dict[tuple[str, str, str], float]:
        """Score standalone graph facts without persisted embeddings."""

        texts = [edge_text(fact.subject, fact.relation, fact.object) for fact in facts]
        fact_embeddings = await self._embeddings.embed_documents(texts)
        if len(fact_embeddings) != len(facts):
            raise ValueError("Embedding provider returned an unexpected vector count")

        scores = {_fact_key(fact): -1.0 for fact in facts}
        for query in queries:
            query_embedding = await self._embeddings.embed_query(query)
            for fact, fact_embedding in zip(facts, fact_embeddings, strict=True):
                key = _fact_key(fact)
                scores[key] = max(
                    scores[key],
                    _cosine_similarity(query_embedding, fact_embedding),
                )
        return scores


def _select_facts(
    ranked: list[RankedFact],
    seeds: list[str],
    *,
    limit: int,
) -> list[RankedFact]:
    """Select facts while preserving coverage across query seeds."""

    selected: list[RankedFact] = []
    selected_keys: set[tuple[str, str, str]] = set()
    for seed in seeds:
        match = next(
            (
                item
                for item in ranked
                if seed in item.fact.seeds and _fact_key(item.fact) not in selected_keys
            ),
            None,
        )
        if match is not None:
            selected.append(match)
            selected_keys.add(_fact_key(match.fact))
            if len(selected) == limit:
                break

    if len(selected) < limit:
        for item in ranked:
            key = _fact_key(item.fact)
            if key in selected_keys:
                continue
            selected.append(item)
            selected_keys.add(key)
            if len(selected) == limit:
                break
    return sorted(
        selected,
        key=lambda item: (
            -item.score,
            item.fact.depth,
            item.fact.subject,
            item.fact.relation,
            item.fact.object,
        ),
    )


def _rank_chunks(
    facts: list[RankedFact],
    *,
    coverage_bonus: float,
) -> list[RankedChunk]:
    """Project fact scores to provenance chunks without dense-sum bias."""

    scores: dict[TripleProvenance, list[float]] = {}
    for item in facts:
        for provenance in item.fact.provenance:
            scores.setdefault(provenance, []).append(item.score)

    ranked = []
    for provenance, fact_scores in scores.items():
        ordered = sorted(fact_scores, reverse=True)
        score = ordered[0] + coverage_bonus * sum(ordered[1:])
        ranked.append(
            RankedChunk(
                document_id=provenance.document_id,
                chunk_id=provenance.chunk_id,
                score=score,
            )
        )
    result = sorted(
        ranked,
        key=lambda item: (-item.score, item.document_id.int, item.chunk_id.int),
    )
    logger.debug(
        "Fact-to-chunk score projection coverage_bonus=%.3f chunks=%s",
        coverage_bonus,
        [
            {
                "document_id": str(item.document_id),
                "chunk_id": str(item.chunk_id),
                "fact_scores": sorted(
                    scores[
                        TripleProvenance(
                            document_id=item.document_id,
                            chunk_id=item.chunk_id,
                        )
                    ],
                    reverse=True,
                ),
                "final_score": round(item.score, 6),
            }
            for item in result
        ],
    )
    return result


def _fact_key(fact: GraphFact) -> tuple[str, str, str]:
    """Return a fact's canonical graph key."""

    return fact.subject, fact.relation, fact.object


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity between equal-length vectors."""

    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )
