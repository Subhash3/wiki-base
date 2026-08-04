from uuid import UUID

import pytest

from graph_rag.entity_linking import ExactEntityLinker
from graph_rag.fact_retrieval import FactRetriever
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RelationshipConceptMatch, Triple, TripleProvenance
from graph_rag.query_extraction import QueryConcepts

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
ENGINE_CHUNK = UUID("00000000-0000-0000-0000-000000000001")
HEIGHT_CHUNK = UUID("00000000-0000-0000-0000-000000000002")


class StubExtractor:
    """Return fixed query concepts."""

    async def extract(self, question: str) -> QueryConcepts:
        """Return a Tiago engine query."""

        del question
        return QueryConcepts(entities=["Tata Tiago"], relationships=["engine options"])


class StubEmbeddings:
    """Return deterministic two-dimensional embeddings."""

    vectors = {
        "engine options": [1.0, 0.0],
        "What engine options does Tiago have?": [1.0, 0.0],
        "What engines are available for the Tata Tiago?": [1.0, 0.0],
        "tata tiago has height 1535 mm": [0.0, 1.0],
        "tata tiago has petrol engine 1199 cc": [1.0, 0.0],
    }

    async def embed_query(self, text: str) -> list[float]:
        """Return a configured query vector."""

        return self.vectors[text]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return configured fact vectors."""

        return [self.vectors[text] for text in texts]


class StubSemanticSearch:
    """Return stored relationship scores for candidate facts."""

    def __init__(self) -> None:
        """Capture candidate-restricted searches."""

        self.candidate_keys: list[frozenset[tuple[str, str, str]] | None] = []

    async def search_entities(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
    ) -> list[object]:
        """Return no entity matches."""

        del embedding, threshold, limit
        return []

    async def search_relationships(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
        candidate_keys: frozenset[tuple[str, str, str]] | None = None,
    ) -> list[RelationshipConceptMatch]:
        """Return both candidate facts with different relevance."""

        del embedding, threshold, limit
        self.candidate_keys.append(candidate_keys)
        return [
            RelationshipConceptMatch(
                text="tata tiago has petrol engine 1199 cc",
                subject="tata tiago",
                relationship="has petrol engine",
                object="1199 cc",
                similarity=0.92,
            ),
            RelationshipConceptMatch(
                text="tata tiago has height 1535 mm",
                subject="tata tiago",
                relationship="has height",
                object="1535 mm",
                similarity=0.31,
            ),
        ]


def make_graph() -> KnowledgeGraph:
    """Create engine and height facts in separate chunks."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(
            subject="tata tiago",
            relation="has petrol engine",
            object="1199 cc",
        ),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=ENGINE_CHUNK,
        ),
    )
    graph.add_triple(
        Triple(subject="tata tiago", relation="has height", object="1535 mm"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=HEIGHT_CHUNK,
        ),
    )
    return graph


async def test_ranks_facts_and_chunks_without_pagerank() -> None:
    """Fact similarity directly controls fact and chunk ordering."""

    retriever = FactRetriever(
        entity_extractor=StubExtractor(),
        entity_linker=ExactEntityLinker(),
        embeddings=StubEmbeddings(),
    )

    result = await retriever.retrieve(
        "What engine options does Tiago have?",
        make_graph(),
        limit=2,
    )

    assert result.facts[0].fact.relation == "has petrol engine"
    assert result.facts[0].score == pytest.approx(1.0)
    assert result.chunks[0].chunk_id == ENGINE_CHUNK
    assert result.chunks[0].score == pytest.approx(1.0)


async def test_uses_candidate_restricted_stored_fact_search() -> None:
    """PostgreSQL scoring receives only facts reached through traversal."""

    search = StubSemanticSearch()
    retriever = FactRetriever(
        entity_extractor=StubExtractor(),
        entity_linker=ExactEntityLinker(),
        embeddings=StubEmbeddings(),
    )

    result = await retriever.retrieve(
        "What engine options does Tiago have?",
        make_graph(),
        limit=2,
        semantic_search=search,
    )

    assert result.chunks[0].chunk_id == ENGINE_CHUNK
    assert search.candidate_keys
    assert search.candidate_keys[0] == frozenset(
        {
            ("tata tiago", "has petrol engine", "1199 cc"),
            ("tata tiago", "has height", "1535 mm"),
        }
    )


async def test_returns_empty_when_query_entities_do_not_link() -> None:
    """Fact retrieval leaves vector fallback to its orchestrating service."""

    class UnknownExtractor:
        async def extract(self, question: str) -> QueryConcepts:
            del question
            return QueryConcepts(entities=["Unknown"], relationships=[])

    retriever = FactRetriever(
        entity_extractor=UnknownExtractor(),
        entity_linker=ExactEntityLinker(),
        embeddings=StubEmbeddings(),
    )

    result = await retriever.retrieve("Unknown question", make_graph(), limit=2)

    assert result.facts == []
    assert result.chunks == []


async def test_extracted_entities_prevent_generic_mentioned_node_seeds() -> None:
    """Words such as available do not become roots when entities were extracted."""

    graph = make_graph()
    graph.add_triple(
        Triple(subject="vehicle", relation="has usb ports", object="available"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=HEIGHT_CHUNK,
        ),
    )

    class CapturingLinker:
        def __init__(self) -> None:
            self.concepts: QueryConcepts | None = None

        async def link(self, concepts, _graph, *, semantic_search=None):
            del semantic_search
            self.concepts = concepts
            return ["tata tiago"]

    linker = CapturingLinker()
    retriever = FactRetriever(
        entity_extractor=StubExtractor(),
        entity_linker=linker,
        embeddings=StubEmbeddings(),
    )

    await retriever.retrieve(
        "What engines are available for the Tata Tiago?",
        graph,
        limit=2,
        semantic_search=StubSemanticSearch(),
    )

    assert linker.concepts is not None
    assert linker.concepts.entities == ["Tata Tiago"]
