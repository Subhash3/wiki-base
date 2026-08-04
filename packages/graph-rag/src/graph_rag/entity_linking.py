import logging
import math
import re
from typing import Protocol

from llm_providers.embeddings.base import EmbeddingProvider

from graph_rag.concepts import edge_text
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import (
    EntityConceptMatch,
    GraphEdge,
    RelationshipConceptMatch,
)
from graph_rag.normalization import normalize_text
from graph_rag.query_extraction import QueryConcepts

logger = logging.getLogger(__name__)

_RELATIONSHIP_STOP_WORDS = frozenset(
    {"a", "an", "and", "at", "by", "for", "from", "in", "is", "of", "on", "or", "the", "to", "with"}
)
_LEXICAL_RELATIONSHIP_BONUS = 0.15
_SEMANTIC_SEARCH_LIMIT = 20


class SemanticConceptSearch(Protocol):
    """Search persisted graph-concept embeddings."""

    async def search_entities(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
    ) -> list[EntityConceptMatch]:
        """Return semantically similar entities."""

        ...

    async def search_relationships(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
    ) -> list[RelationshipConceptMatch]:
        """Return semantically similar relationship facts."""

        ...


class EntityLinker(Protocol):
    """Link query concepts to knowledge graph nodes."""

    async def link(
        self,
        concepts: QueryConcepts,
        graph: KnowledgeGraph,
        *,
        semantic_search: SemanticConceptSearch | None = None,
    ) -> list[str]:
        """Return graph nodes matching query entities and relationships."""

        ...


class ExactEntityLinker:
    """Link entities and relationships through exact matching."""

    async def link(
        self,
        concepts: QueryConcepts,
        graph: KnowledgeGraph,
        *,
        semantic_search: SemanticConceptSearch | None = None,
    ) -> list[str]:
        """Return distinct nodes matching query concepts."""

        del semantic_search

        linked: list[str] = []
        seen: set[str] = set()
        exact_entities: list[tuple[str, str]] = []
        for entity in concepts.entities:
            node = normalize_text(entity)
            if node in graph.nodes and node not in seen:
                linked.append(node)
                seen.add(node)
                exact_entities.append((entity, node))
        relation_lookup = _relation_lookup(graph)
        exact_relationships: list[tuple[str, str]] = []
        for relationship in concepts.relationships:
            for relation in relation_lookup.get(normalize_text(relationship), []):
                exact_relationships.append((relationship, relation))
                _append_relation_endpoints(linked, seen, graph, relation)
        logger.debug(
            "Exact graph matches entities=%s relationships=%s",
            exact_entities,
            exact_relationships,
        )
        return linked


class EmbeddingEntityLinker:
    """Link exact concepts first and embed unmatched concepts."""

    def __init__(
        self,
        *,
        embeddings: EmbeddingProvider,
        similarity_threshold: float = 0.75,
        relationship_similarity_threshold: float = 0.6,
        max_links_per_entity: int = 1,
        embedding_batch_size: int = 128,
    ) -> None:
        """Configure embedding similarity and batching."""

        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if not -1 <= relationship_similarity_threshold <= 1:
            raise ValueError(
                "relationship_similarity_threshold must be between -1 and 1"
            )
        if max_links_per_entity < 1:
            raise ValueError("max_links_per_entity must be positive")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be positive")

        self._embeddings = embeddings
        self._similarity_threshold = similarity_threshold
        self._relationship_similarity_threshold = relationship_similarity_threshold
        self._max_links_per_entity = max_links_per_entity
        self._embedding_batch_size = embedding_batch_size
        self._node_embeddings: dict[str, list[float]] = {}
        self._edge_embeddings: dict[str, list[float]] = {}

    async def link(
        self,
        concepts: QueryConcepts,
        graph: KnowledgeGraph,
        *,
        semantic_search: SemanticConceptSearch | None = None,
    ) -> list[str]:
        """Return nodes matching entities and relationship labels."""

        if not graph.nodes or not (concepts.entities or concepts.relationships):
            return []

        linked: list[str] = []
        seen: set[str] = set()
        exact_entities: list[tuple[str, str]] = []
        unmatched_entities: list[str] = []
        unmatched_entity_keys: set[str] = set()
        for entity in concepts.entities:
            normalized = normalize_text(entity)
            if not normalized:
                continue
            if normalized in graph.nodes:
                if normalized not in seen:
                    linked.append(normalized)
                    seen.add(normalized)
                    exact_entities.append((entity, normalized))
            elif normalized not in unmatched_entity_keys:
                unmatched_entities.append(entity.strip())
                unmatched_entity_keys.add(normalized)

        relation_lookup = _relation_lookup(graph)
        exact_relationships: list[tuple[str, str]] = []
        unmatched_relationships: list[str] = []
        unmatched_relationship_keys: set[str] = set()
        for relationship in concepts.relationships:
            normalized = normalize_text(relationship)
            if not normalized:
                continue
            exact_relations = relation_lookup.get(normalized)
            if exact_relations:
                for relation in exact_relations:
                    exact_relationships.append((relationship, relation))
                    _append_relation_endpoints(linked, seen, graph, relation)
            elif normalized not in unmatched_relationship_keys:
                unmatched_relationships.append(relationship.strip())
                unmatched_relationship_keys.add(normalized)

        logger.debug(
            "Exact graph matches entities=%s relationships=%s",
            exact_entities,
            exact_relationships,
        )

        if semantic_search is not None:
            await self._link_stored_concepts(
                unmatched_entities,
                unmatched_relationships,
                graph,
                semantic_search,
                linked,
                seen,
            )
        else:
            await self._link_in_memory(
                unmatched_entities,
                unmatched_relationships,
                graph,
                linked,
                seen,
            )
        return linked

    async def _link_stored_concepts(
        self,
        entities: list[str],
        relationships: list[str],
        graph: KnowledgeGraph,
        semantic_search: SemanticConceptSearch,
        linked: list[str],
        seen: set[str],
    ) -> None:
        """Link unmatched concepts through persisted embeddings."""

        for entity in entities:
            query_embedding = await self._embeddings.embed_query(entity)
            matches = await semantic_search.search_entities(
                query_embedding,
                threshold=self._similarity_threshold,
                limit=max(_SEMANTIC_SEARCH_LIMIT, self._max_links_per_entity),
            )
            accepted = 0
            for match in matches:
                if match.similarity < self._similarity_threshold:
                    logger.debug(
                        "Rejected PostgreSQL entity match %r -> %r "
                        "(similarity=%.3f threshold=%.3f)",
                        entity,
                        match.entity,
                        match.similarity,
                        self._similarity_threshold,
                    )
                    continue
                logger.debug(
                    "PostgreSQL entity match %r -> %r (similarity=%.3f)",
                    entity,
                    match.entity,
                    match.similarity,
                )
                if match.entity in graph.nodes and match.entity not in seen:
                    linked.append(match.entity)
                    seen.add(match.entity)
                    accepted += 1
                    if accepted == self._max_links_per_entity:
                        break

        candidate_threshold = max(
            -1.0,
            self._relationship_similarity_threshold - _LEXICAL_RELATIONSHIP_BONUS,
        )
        for relationship in relationships:
            query_embedding = await self._embeddings.embed_query(relationship)
            candidates = await semantic_search.search_relationships(
                query_embedding,
                threshold=candidate_threshold,
                limit=max(_SEMANTIC_SEARCH_LIMIT, self._max_links_per_entity),
            )
            ranked = sorted(
                (
                    (
                        candidate.similarity
                        + _relationship_bonus(relationship, candidate.text),
                        candidate,
                    )
                    for candidate in candidates
                ),
                key=lambda item: (-item[0], item[1].text),
            )
            matches = 0
            for score, candidate in ranked:
                if score < self._relationship_similarity_threshold:
                    continue
                logger.debug(
                    "PostgreSQL relationship match %r -> %r "
                    "(similarity=%.3f score=%.3f)",
                    relationship,
                    candidate.text,
                    candidate.similarity,
                    score,
                )
                _append_nodes(
                    linked,
                    seen,
                    graph,
                    candidate.subject,
                    candidate.object,
                )
                matches += 1
                if matches == self._max_links_per_entity:
                    break

    async def _link_in_memory(
        self,
        entities: list[str],
        relationships: list[str],
        graph: KnowledgeGraph,
        linked: list[str],
        seen: set[str],
    ) -> None:
        """Link unmatched concepts using the standalone in-memory index."""

        nodes = sorted(graph.nodes)
        for _, node in await self._semantic_matches(
            entities,
            nodes,
            self._node_embeddings,
            threshold=self._similarity_threshold,
            kind="entity",
        ):
            if node not in seen:
                linked.append(node)
                seen.add(node)

        edge_lookup = {
            edge_text(edge.subject, edge.relation, edge.object): edge
            for edge in graph.edges()
        }
        edge_texts = sorted(edge_lookup)
        for relationship in relationships:
            candidates, bonus = _focus_relationship_candidates(
                relationship,
                edge_texts,
            )
            matches = await self._semantic_matches(
                [relationship],
                candidates,
                self._edge_embeddings,
                threshold=self._relationship_similarity_threshold,
                kind="relationship",
                score_bonus=bonus,
            )
            for _, matched_edge_text in matches:
                _append_edge_endpoints(linked, seen, edge_lookup[matched_edge_text])

    async def _semantic_matches(
        self,
        queries: list[str],
        candidates: list[str],
        cache: dict[str, list[float]],
        *,
        threshold: float,
        kind: str,
        score_bonus: float = 0.0,
    ) -> list[tuple[str, str]]:
        """Match query phrases to cached candidate embeddings."""

        if not queries or not candidates:
            return []
        await self._embed_missing(candidates, cache)
        linked: list[tuple[str, str]] = []
        for query in queries:
            query_embedding = await self._embeddings.embed_query(query)
            ranked = sorted(
                (
                    (_cosine_similarity(query_embedding, cache[candidate]), candidate)
                    for candidate in candidates
                ),
                key=lambda item: (-item[0], item[1]),
            )
            matches = 0
            best_similarity, best_candidate = ranked[0]
            if best_similarity + score_bonus < threshold:
                logger.debug(
                    "Rejected embedding %s match %r -> %r "
                    "(similarity=%.3f score=%.3f threshold=%.3f)",
                    kind,
                    query,
                    best_candidate,
                    best_similarity,
                    best_similarity + score_bonus,
                    threshold,
                )
                continue
            for similarity, candidate in ranked:
                score = similarity + score_bonus
                if score < threshold:
                    break
                logger.debug(
                    "Embedding %s match %r -> %r (similarity=%.3f score=%.3f)",
                    kind,
                    query,
                    candidate,
                    similarity,
                    score,
                )
                linked.append((query, candidate))
                matches += 1
                if matches == self._max_links_per_entity:
                    break
        return linked

    async def _embed_missing(
        self,
        values: list[str],
        cache: dict[str, list[float]],
    ) -> None:
        """Embed and cache candidate values not seen by this linker."""

        missing = [value for value in values if value not in cache]
        for start in range(0, len(missing), self._embedding_batch_size):
            batch = missing[start : start + self._embedding_batch_size]
            vectors = await self._embeddings.embed_documents(batch)
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned an unexpected vector count")
            cache.update(zip(batch, vectors, strict=True))


def _relation_lookup(graph: KnowledgeGraph) -> dict[str, list[str]]:
    """Group graph relationship labels by normalized text."""

    lookup: dict[str, set[str]] = {}
    for edge in graph.edges():
        lookup.setdefault(normalize_text(edge.relation), set()).add(edge.relation)
    return {key: sorted(values) for key, values in lookup.items()}


def _append_relation_endpoints(
    linked: list[str],
    seen: set[str],
    graph: KnowledgeGraph,
    relation: str,
) -> None:
    """Add endpoints of edges carrying a matched relationship label."""

    matching_edges = sorted(
        (edge for edge in graph.edges() if edge.relation == relation),
        key=lambda edge: (edge.subject, edge.object),
    )
    for edge in matching_edges:
        _append_edge_endpoints(linked, seen, edge)


def _append_edge_endpoints(
    linked: list[str],
    seen: set[str],
    edge: GraphEdge,
) -> None:
    """Add both endpoints of one matched graph edge."""

    for node in (edge.subject, edge.object):
        if node not in seen:
            linked.append(node)
            seen.add(node)


def _focus_relationship_candidates(
    relationship: str,
    candidates: list[str],
) -> tuple[list[str], float]:
    """Prefer facts containing the rarest meaningful query term."""

    query_tokens = _content_tokens(relationship)
    if not query_tokens:
        return candidates, 0.0
    candidate_tokens = {
        candidate: _content_tokens(candidate) for candidate in candidates
    }
    frequencies = {
        token: sum(token in tokens for tokens in candidate_tokens.values())
        for token in query_tokens
    }
    shared_frequencies = {
        token: frequency for token, frequency in frequencies.items() if frequency > 0
    }
    if not shared_frequencies:
        return candidates, 0.0

    rarest_frequency = min(shared_frequencies.values())
    rarest_tokens = {
        token
        for token, frequency in shared_frequencies.items()
        if frequency == rarest_frequency
    }
    focused = [
        candidate
        for candidate in candidates
        if candidate_tokens[candidate] & rarest_tokens
    ]
    if len(focused) == len(candidates):
        return candidates, 0.0
    logger.debug(
        "Lexical relationship focus %r tokens=%s candidates=%d->%d",
        relationship,
        sorted(rarest_tokens),
        len(candidates),
        len(focused),
    )
    return focused, _LEXICAL_RELATIONSHIP_BONUS


def _relationship_bonus(query: str, candidate: str) -> float:
    """Boost stored relationship facts sharing meaningful query terms."""

    return (
        _LEXICAL_RELATIONSHIP_BONUS
        if _content_tokens(query) & _content_tokens(candidate)
        else 0.0
    )


def _append_nodes(
    linked: list[str],
    seen: set[str],
    graph: KnowledgeGraph,
    *nodes: str,
) -> None:
    """Append matched nodes that exist in the active graph."""

    for node in nodes:
        if node in graph.nodes and node not in seen:
            linked.append(node)
            seen.add(node)


def _content_tokens(value: str) -> set[str]:
    """Return meaningful normalized tokens for relationship matching."""

    return {
        token
        for token in re.findall(r"\w+", normalize_text(value))
        if token not in _RELATIONSHIP_STOP_WORDS
    }


def _cosine_similarity(first: list[float], second: list[float]) -> float:
    """Return cosine similarity for two vectors."""

    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )
