from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from graph_rag import KnowledgeGraph, Triple, TripleProvenance

from wiki_base.graph_tools import merge_wiki_base, visualize_document

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
SECOND_DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000002")
WIKI_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")


def graph_payload(
    document_id: UUID,
    *,
    subject: str,
    object_: str,
) -> dict[str, object]:
    """Create one canonical graph mapping for utility tests."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject=subject, relation="relates to", object=object_),
        provenance=TripleProvenance(
            document_id=document_id,
            chunk_id=UUID(int=document_id.int + 100),
        ),
    )
    return graph.to_dict()


class StubConnection:
    """Return configured graphs for visualization queries."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        """Store canonical graph mappings."""

        self.payloads = payloads

    async def fetchval(self, _query: str, document_id: UUID):
        """Return the graph matching the requested document."""

        assert document_id == DOCUMENT_ID
        return self.payloads[0] if self.payloads else None

    async def fetch(self, _query: str, wiki_base_id: UUID, *arguments: object):
        """Return all configured wiki-base graphs."""

        assert wiki_base_id == WIKI_BASE_ID
        if arguments:
            assert arguments == ("test-embedding", 0.95)
            return []
        return [{"graph": payload} for payload in self.payloads]


class StubDatabase:
    """Expose one stub connection through the database context API."""

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        """Create the shared stub connection."""

        self._connection = StubConnection(payloads)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[StubConnection]:
        """Yield the configured stub connection."""

        yield self._connection


async def test_visualizes_stored_document_graph(tmp_path: Path) -> None:
    """The visualizer writes HTML from a PostgreSQL graph payload."""

    database = StubDatabase(
        [graph_payload(DOCUMENT_ID, subject="alice", object_="acme")]
    )
    output = tmp_path / "document.html"

    output_json, output_html = await visualize_document(
        database,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        output=output,
    )

    assert output_json == output.with_suffix(".json")
    assert output_html == output
    exported = KnowledgeGraph.from_json(output_json.read_text(encoding="utf-8"))
    assert exported.nodes == frozenset({"alice", "acme"})
    assert "Document 10000000" in output_html.read_text(encoding="utf-8")


async def test_merges_ready_wiki_base_graphs(tmp_path: Path) -> None:
    """The merger writes canonical JSON and a matching HTML visualization."""

    database = StubDatabase(
        [
            graph_payload(DOCUMENT_ID, subject="alice", object_="acme"),
            graph_payload(SECOND_DOCUMENT_ID, subject="acme", object_="paris"),
        ]
    )
    output = tmp_path / "wiki-base.json"

    output_json, output_html = await merge_wiki_base(
        database,  # type: ignore[arg-type]
        wiki_base_id=WIKI_BASE_ID,
        output=output,
        embedding_model="test-embedding",
    )

    merged = KnowledgeGraph.from_json(output_json.read_text(encoding="utf-8"))
    assert merged.nodes == frozenset({"alice", "acme", "paris"})
    assert output_html == output.with_suffix(".html")
    assert output_html.is_file()
