from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from document_processing.models import DocumentChunk
from graph_rag import IndexedChunk, KnowledgeGraph, Triple, TripleProvenance
from llm_providers.embeddings.base import EmbeddingModelInfo

from wiki_base.database.records import GraphIndexingJobRecord
from wiki_base.workers import graph_indexing
from wiki_base.workers.graph_indexing import GraphIndexingWorker

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")
WIKI_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")


class StubDatabase:
    """Provide a connection context for worker tests."""

    class Connection:
        """Provide a transaction context for graph persistence."""

        @asynccontextmanager
        async def transaction(self) -> AsyncIterator[None]:
            """Yield one placeholder transaction."""

            yield

    def __init__(self) -> None:
        """Create the shared placeholder connection."""

        self._connection = self.Connection()

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[object]:
        """Yield a placeholder connection."""

        yield self._connection


class StubIndexer:
    """Return a fixed graph for worker tests."""

    async def index(self, chunks: list[IndexedChunk]) -> KnowledgeGraph:
        """Build a small graph from the supplied chunk."""

        assert chunks[0].document_id == DOCUMENT_ID
        graph = KnowledgeGraph()
        graph.add_triple(
            Triple(subject="alice", relation="works at", object="acme"),
            provenance=TripleProvenance(
                document_id=DOCUMENT_ID,
                chunk_id=CHUNK_ID,
            ),
        )
        return graph


class StubEmbeddings:
    """Embed graph concepts with deterministic test vectors."""

    def __init__(self) -> None:
        """Capture embedded concept batches."""

        self.batches: list[list[str]] = []

    @property
    def model_info(self) -> EmbeddingModelInfo:
        """Return the configured test model."""

        return EmbeddingModelInfo(model="test-embedding", dimensions=2, max_tokens=10)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector for every supplied concept."""

        self.batches.append(texts)
        return [[1.0, 0.0] for _ in texts]

    async def embed_query(self, _text: str) -> list[float]:
        """Reject query embedding in the indexing worker."""

        raise AssertionError("Graph indexing must not embed queries")


def make_chunk() -> IndexedChunk:
    """Create one stored chunk for a worker test."""

    return IndexedChunk(
        document_id=DOCUMENT_ID,
        chunk=DocumentChunk(
            id=CHUNK_ID,
            ordinal=0,
            content="Alice works at Acme.",
            embedding_content="Alice works at Acme.",
            token_count=5,
            page_number=None,
            slide_number=None,
            section=None,
            heading=None,
            caption=None,
        ),
    )


async def test_worker_indexes_and_completes_one_job(
    monkeypatch,
) -> None:
    """A queued document graph is stored and marked ready."""

    job = GraphIndexingJobRecord(document_id=DOCUMENT_ID)
    stored: dict[str, object] = {}
    stored_concepts: dict[str, object] = {}
    stored_synonyms: dict[str, object] = {}
    completed = False

    async def claim(_connection):
        return job

    async def load(_connection, document_id):
        assert document_id == DOCUMENT_ID
        return [make_chunk()]

    async def store(_connection, **values):
        stored.update(values)

    async def complete(_connection, _job):
        nonlocal completed
        completed = True

    async def store_concepts(_connection, **values):
        stored_concepts.update(values)
        return WIKI_BASE_ID

    async def store_synonyms(_connection, **values):
        stored_synonyms.update(values)

    monkeypatch.setattr(graph_indexing, "claim_next_graph_indexing_job", claim)
    monkeypatch.setattr(graph_indexing, "load_graph_indexing_chunks", load)
    monkeypatch.setattr(graph_indexing, "upsert_document_graph", store)
    monkeypatch.setattr(
        graph_indexing,
        "replace_document_graph_concepts",
        store_concepts,
    )
    monkeypatch.setattr(graph_indexing, "complete_graph_indexing_job", complete)
    monkeypatch.setattr(
        graph_indexing,
        "replace_wiki_base_graph_synonyms",
        store_synonyms,
    )
    embeddings = StubEmbeddings()
    worker = GraphIndexingWorker(
        database=StubDatabase(),  # type: ignore[arg-type]
        indexer=StubIndexer(),  # type: ignore[arg-type]
        embeddings=embeddings,
        extraction_model="test-model",
        index_version="1",
        embedding_batch_size=2,
        synonym_similarity_threshold=0.85,
        synonym_max_links=3,
        poll_interval_seconds=1,
    )

    processed = await worker.run_once()

    assert processed is True
    assert completed is True
    assert stored["document_id"] == DOCUMENT_ID
    assert stored["extraction_model"] == "test-model"
    assert stored["index_version"] == "1"
    assert KnowledgeGraph.from_dict(stored["graph"]).nodes == frozenset(
        {"alice", "acme"}
    )
    assert embeddings.batches == [
        ["acme", "alice"],
        ["alice works at acme"],
    ]
    assert stored_concepts["embedding_model"] == "test-embedding"
    assert len(stored_concepts["concepts"]) == 3
    assert len(stored_concepts["embeddings"]) == 3
    assert stored_synonyms == {
        "wiki_base_id": WIKI_BASE_ID,
        "embedding_model": "test-embedding",
        "similarity_threshold": 0.85,
        "max_links_per_entity": 3,
    }


async def test_worker_marks_failed_job(monkeypatch) -> None:
    """An empty document is marked failed."""

    job = GraphIndexingJobRecord(document_id=DOCUMENT_ID)
    failure: dict[str, str] = {}

    async def claim(_connection):
        return job

    async def load(_connection, _document_id):
        return []

    async def fail(_connection, _job, *, error_message):
        failure["message"] = error_message

    monkeypatch.setattr(graph_indexing, "claim_next_graph_indexing_job", claim)
    monkeypatch.setattr(graph_indexing, "load_graph_indexing_chunks", load)
    monkeypatch.setattr(graph_indexing, "fail_graph_indexing_job", fail)
    worker = GraphIndexingWorker(
        database=StubDatabase(),  # type: ignore[arg-type]
        indexer=StubIndexer(),  # type: ignore[arg-type]
        embeddings=StubEmbeddings(),
        extraction_model="test-model",
        index_version="1",
        embedding_batch_size=2,
        synonym_similarity_threshold=0.85,
        synonym_max_links=3,
        poll_interval_seconds=1,
    )

    processed = await worker.run_once()

    assert processed is True
    assert failure["message"] == "Document has no chunks to index"
