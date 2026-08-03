import logging
from uuid import UUID

import pytest

from graph_rag.entity_linking import ExactEntityLinker
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance
from graph_rag.query_extraction import QueryConcepts
from graph_rag.retrieval import HippoRAGRetriever

DOCUMENT_ONE = UUID("10000000-0000-0000-0000-000000000001")
DOCUMENT_TWO = UUID("10000000-0000-0000-0000-000000000002")
CHUNK_ONE = UUID("00000000-0000-0000-0000-000000000001")
CHUNK_TWO = UUID("00000000-0000-0000-0000-000000000002")


class StubEntityExtractor:
    """Return configured query concepts."""

    def __init__(
        self,
        entities: list[str],
        relationships: list[str] | None = None,
    ) -> None:
        """Store the concepts returned during retrieval."""

        self.entities = entities
        self.relationships = relationships or []
        self.questions: list[str] = []

    async def extract(self, question: str) -> QueryConcepts:
        """Capture the question and return configured concepts."""

        self.questions.append(question)
        return QueryConcepts(
            entities=self.entities,
            relationships=self.relationships,
        )


def make_multihop_graph() -> KnowledgeGraph:
    """Create a two-chunk graph connected through Acme."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ONE,
            chunk_id=CHUNK_ONE,
        ),
    )
    graph.add_triple(
        Triple(subject="acme", relation="located in", object="paris"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_TWO,
            chunk_id=CHUNK_TWO,
        ),
    )
    return graph


async def test_retrieves_chunks_across_multiple_graph_hops(caplog) -> None:
    """A query seed retrieves evidence connected through shared entities."""

    extractor = StubEntityExtractor(["Alice"])
    retriever = HippoRAGRetriever(
        entity_extractor=extractor,
        entity_linker=ExactEntityLinker(),
    )
    caplog.set_level(logging.DEBUG, logger="graph_rag")

    ranked = await retriever.retrieve(
        "Where is Alice's employer headquartered?",
        make_multihop_graph(),
        limit=10,
    )

    assert extractor.questions == ["Where is Alice's employer headquartered?"]
    assert {chunk.chunk_id for chunk in ranked} == {CHUNK_ONE, CHUNK_TWO}
    assert ranked[0].score >= ranked[1].score
    assert "Graph query concepts entities=['Alice'] relationships=[]" in caplog.text
    assert "Exact graph matches entities=" in caplog.text
    assert "PageRank nodes total=3 top=" in caplog.text


async def test_returns_empty_when_entities_do_not_link() -> None:
    """Unmatched entities do not trigger a global PageRank fallback."""

    retriever = HippoRAGRetriever(
        entity_extractor=StubEntityExtractor(["Unknown"]),
        entity_linker=ExactEntityLinker(),
    )

    ranked = await retriever.retrieve(
        "What is unknown?",
        make_multihop_graph(),
        limit=5,
    )

    assert ranked == []


async def test_uses_graph_nodes_mentioned_in_question_when_extraction_is_wrong() -> None:
    """Explicit node mentions survive an incorrect LLM entity list."""

    retriever = HippoRAGRetriever(
        entity_extractor=StubEntityExtractor(["Tesla Model 3", "Austin Powers"]),
        entity_linker=ExactEntityLinker(),
    )

    ranked = await retriever.retrieve(
        "Where does Alice work?",
        make_multihop_graph(),
        limit=5,
    )

    assert {chunk.chunk_id for chunk in ranked} == {CHUNK_ONE, CHUNK_TWO}


async def test_uses_relationships_to_reach_disconnected_evidence() -> None:
    """A matched relationship seeds the endpoints of its graph edge."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="honda unicorn", relation="is a", object="motorcycle"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ONE,
            chunk_id=CHUNK_ONE,
        ),
    )
    graph.add_triple(
        Triple(
            subject="ex-showroom delhi",
            relation="loan amount for",
            object="total interest",
        ),
        provenance=TripleProvenance(
            document_id=DOCUMENT_TWO,
            chunk_id=CHUNK_TWO,
        ),
    )
    retriever = HippoRAGRetriever(
        entity_extractor=StubEntityExtractor(
            ["Honda Unicorn"],
            ["loan amount for"],
        ),
        entity_linker=ExactEntityLinker(),
    )

    ranked = await retriever.retrieve(
        "Any loan offers on Honda Unicorn?",
        graph,
        limit=5,
    )

    assert {chunk.chunk_id for chunk in ranked} == {CHUNK_ONE, CHUNK_TWO}


async def test_applies_result_limit() -> None:
    """Only the requested number of ranked chunks is returned."""

    retriever = HippoRAGRetriever(
        entity_extractor=StubEntityExtractor(["Alice"]),
        entity_linker=ExactEntityLinker(),
    )

    ranked = await retriever.retrieve(
        "Where is Alice's employer headquartered?",
        make_multihop_graph(),
        limit=1,
    )

    assert len(ranked) == 1


async def test_rejects_nonpositive_limit() -> None:
    """Retrieval requires space for at least one result."""

    retriever = HippoRAGRetriever(
        entity_extractor=StubEntityExtractor(["Alice"]),
        entity_linker=ExactEntityLinker(),
    )

    with pytest.raises(ValueError, match="limit"):
        await retriever.retrieve("Question", make_multihop_graph(), limit=0)
