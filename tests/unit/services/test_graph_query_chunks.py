from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from graph_rag import KnowledgeGraph, RankedChunk, Triple, TripleProvenance

from wiki_base.database.queries.chunks import StoredChunk
from wiki_base.database.records import IngestionStatus, WikiBaseRecord
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.services import query_chunks
from wiki_base.services.query_chunks import QueryChunksService, RetrievedChunk

WIKI_BASE_ID = UUID("0190f3a0-7d83-7a41-a27c-b7314f5ae705")
DOCUMENT_ONE = UUID("0190f3a0-b096-7af5-8392-cc61de46f6de")
DOCUMENT_TWO = UUID("0190f3a0-b096-7af5-8392-cc61de46f6df")
CHUNK_ONE = UUID("0190f3a1-a0ee-77ac-a76b-fb191cb0f8a0")
CHUNK_TWO = UUID("0190f3a1-a0ee-77ac-a76b-fb191cb0f8a1")


class StubDatabase:
    """Provide a connection context for service tests."""

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[object]:
        """Yield a placeholder database connection."""

        yield object()


class UnusedEmbeddings:
    """Fail if Pro retrieval tries to create an embedding."""

    async def embed_query(self, _text: str) -> list[float]:
        """Reject unexpected vector retrieval."""

        raise AssertionError("Pro retrieval must not embed the question")


class StubGraphRetriever:
    """Return a fixed ranking and capture the merged graph."""

    graph: KnowledgeGraph | None = None

    async def retrieve(
        self,
        _question: str,
        graph: KnowledgeGraph,
        *,
        limit: int,
    ) -> list[RankedChunk]:
        """Return two chunks in graph rank order."""

        assert limit == 2
        self.graph = graph
        return [
            RankedChunk(document_id=DOCUMENT_TWO, chunk_id=CHUNK_TWO, score=0.8),
            RankedChunk(document_id=DOCUMENT_ONE, chunk_id=CHUNK_ONE, score=0.6),
        ]


def write_graph(
    path: Path,
    *,
    triple: Triple,
    document_id: UUID,
    chunk_id: UUID,
) -> None:
    """Write one canonical document graph."""

    graph = KnowledgeGraph()
    graph.add_triple(
        triple,
        provenance=TripleProvenance(document_id=document_id, chunk_id=chunk_id),
    )
    path.write_text(graph.to_json(), encoding="utf-8")


async def test_pro_retrieval_merges_wiki_base_graphs_and_preserves_rank(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_graph(
        first_path,
        triple=Triple(subject="Alice", relation="works at", object="Acme"),
        document_id=DOCUMENT_ONE,
        chunk_id=CHUNK_ONE,
    )
    write_graph(
        second_path,
        triple=Triple(subject="Acme", relation="located in", object="Paris"),
        document_id=DOCUMENT_TWO,
        chunk_id=CHUNK_TWO,
    )

    async def get_wiki_base(_connection, _wiki_base_id):
        return WikiBaseRecord(
            id=WIKI_BASE_ID,
            name="Test",
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
        )

    async def list_statuses(_connection, _wiki_base_id):
        return {
            WIKI_BASE_ID: {
                RetrievalMode.LITE: IngestionStatus.READY,
                RetrievalMode.PRO: IngestionStatus.READY,
            }
        }

    async def list_paths(_connection, _wiki_base_id):
        return [first_path, second_path]

    async def load_chunks(_connection, *, wiki_base_id, chunk_ids):
        assert wiki_base_id == WIKI_BASE_ID
        assert chunk_ids == [CHUNK_TWO, CHUNK_ONE]
        return [
            StoredChunk(
                id=CHUNK_ONE,
                document_id=DOCUMENT_ONE,
                document_name="first.pdf",
                content="Alice works at Acme.",
                page_number=1,
                slide_number=None,
                section=None,
                heading=None,
            ),
            StoredChunk(
                id=CHUNK_TWO,
                document_id=DOCUMENT_TWO,
                document_name="second.pdf",
                content="Acme is located in Paris.",
                page_number=2,
                slide_number=None,
                section=None,
                heading=None,
            ),
        ]

    monkeypatch.setattr(query_chunks, "get_wiki_base", get_wiki_base)
    monkeypatch.setattr(
        query_chunks,
        "list_wiki_base_retrieval_statuses",
        list_statuses,
    )
    monkeypatch.setattr(
        query_chunks,
        "list_ready_wiki_base_graph_paths",
        list_paths,
    )
    monkeypatch.setattr(query_chunks, "load_chunks_by_ids", load_chunks)

    retriever = StubGraphRetriever()
    service = QueryChunksService(
        database=StubDatabase(),
        embeddings=UnusedEmbeddings(),
        graph_retriever=retriever,
    )
    result = await service.query(
        wiki_base_id=WIKI_BASE_ID,
        question="Where does Alice work?",
        limit=2,
        mode=RetrievalMode.PRO,
    )

    assert retriever.graph is not None
    assert retriever.graph.nodes == frozenset({"Alice", "Acme", "Paris"})
    assert [chunk.id for chunk in result.chunks] == [CHUNK_TWO, CHUNK_ONE]
    assert [chunk.score for chunk in result.chunks] == [0.8, 0.6]
    assert result.mode == RetrievalMode.PRO
    assert result.retrieval_strategy == RetrievalStrategy.GRAPH


async def test_pro_retrieval_falls_back_to_vector_chunks(monkeypatch) -> None:
    """Pro uses vector retrieval when the graph produces no chunks."""

    async def get_wiki_base(_connection, _wiki_base_id):
        return WikiBaseRecord(
            id=WIKI_BASE_ID,
            name="Test",
            created_at=datetime.now(UTC),
            started_at=None,
            completed_at=None,
        )

    async def list_statuses(_connection, _wiki_base_id):
        return {
            WIKI_BASE_ID: {
                RetrievalMode.LITE: IngestionStatus.READY,
                RetrievalMode.PRO: IngestionStatus.READY,
            }
        }

    monkeypatch.setattr(query_chunks, "get_wiki_base", get_wiki_base)
    monkeypatch.setattr(
        query_chunks,
        "list_wiki_base_retrieval_statuses",
        list_statuses,
    )
    service = QueryChunksService(
        database=StubDatabase(),
        embeddings=UnusedEmbeddings(),
        graph_retriever=StubGraphRetriever(),
    )
    fallback = RetrievedChunk(
        id=CHUNK_ONE,
        document_id=DOCUMENT_ONE,
        document_name="first.pdf",
        content="Alice works at Acme.",
        score=0.9,
        page=1,
        slide=None,
        section=None,
        heading=None,
    )

    async def query_graph(**_arguments):
        return []

    async def query_vector(**_arguments):
        return [fallback]

    monkeypatch.setattr(service, "_query_graph", query_graph)
    monkeypatch.setattr(service, "_query_vector", query_vector)

    result = await service.query(
        wiki_base_id=WIKI_BASE_ID,
        question="Who owns the car?",
        limit=5,
        mode=RetrievalMode.PRO,
    )

    assert result.chunks == [fallback]
    assert result.mode == RetrievalMode.PRO
    assert result.retrieval_strategy == RetrievalStrategy.VECTOR_FALLBACK
