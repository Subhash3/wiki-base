from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from document_processing.models import DocumentChunk
from graph_rag import IndexedChunk, KnowledgeGraph, Triple, TripleProvenance

from wiki_base.database.records import GraphIndexingJobRecord
from wiki_base.workers import graph_indexing
from wiki_base.workers.graph_indexing import GraphIndexingWorker

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


class StubDatabase:
    """Provide a connection context for worker tests."""

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[object]:
        """Yield a placeholder connection."""

        yield object()


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
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A queued document is written and marked ready."""

    job = GraphIndexingJobRecord(document_id=DOCUMENT_ID)
    completed: dict[str, object] = {}

    async def claim(_connection):
        return job

    async def load(_connection, document_id):
        assert document_id == DOCUMENT_ID
        return [make_chunk()]

    async def complete(_connection, _job, **values):
        completed.update(values)

    monkeypatch.setattr(graph_indexing, "claim_next_graph_indexing_job", claim)
    monkeypatch.setattr(graph_indexing, "load_graph_indexing_chunks", load)
    monkeypatch.setattr(graph_indexing, "complete_graph_indexing_job", complete)
    worker = GraphIndexingWorker(
        database=StubDatabase(),  # type: ignore[arg-type]
        indexer=StubIndexer(),  # type: ignore[arg-type]
        output_directory=tmp_path,
        extraction_model="test-model",
        index_version="1",
        poll_interval_seconds=1,
    )

    processed = await worker.run_once()

    output_path = tmp_path / f"{DOCUMENT_ID}.json"
    assert processed is True
    assert completed["output_path"] == output_path
    assert KnowledgeGraph.from_json(output_path.read_text()).nodes == frozenset(
        {"alice", "acme"}
    )


async def test_worker_marks_failed_job(tmp_path: Path, monkeypatch) -> None:
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
        output_directory=tmp_path,
        extraction_model="test-model",
        index_version="1",
        poll_interval_seconds=1,
    )

    processed = await worker.run_once()

    assert processed is True
    assert failure["message"] == "Document has no chunks to index"
