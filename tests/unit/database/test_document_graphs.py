import json
from uuid import UUID

from wiki_base.database.queries.document_graphs import (
    get_document_graph,
    list_ready_wiki_base_graphs,
    upsert_document_graph,
)

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
WIKI_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")
GRAPH = {"version": 1, "edges": []}


class StubConnection:
    """Capture document-graph database operations."""

    def __init__(self) -> None:
        """Initialize captured arguments and rows."""

        self.execute_arguments: tuple[object, ...] = ()
        self.rows: list[dict[str, object]] = []
        self.value: object | None = None

    async def execute(self, _query: str, *arguments: object) -> None:
        """Capture graph upsert arguments."""

        self.execute_arguments = arguments

    async def fetch(self, _query: str, wiki_base_id: UUID) -> list[dict[str, object]]:
        """Return configured graph rows."""

        assert wiki_base_id == WIKI_BASE_ID
        return self.rows

    async def fetchval(self, _query: str, document_id: UUID) -> object | None:
        """Return the configured document graph."""

        assert document_id == DOCUMENT_ID
        return self.value


async def test_upserts_canonical_graph_as_json() -> None:
    """Graph mappings are encoded for PostgreSQL JSONB storage."""

    connection = StubConnection()

    await upsert_document_graph(
        connection,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        graph=GRAPH,
        extraction_model="test-model",
        index_version="1",
    )

    assert connection.execute_arguments == (
        DOCUMENT_ID,
        json.dumps(GRAPH),
        "test-model",
        "1",
    )


async def test_loads_jsonb_strings_and_mappings() -> None:
    """Graph reads support asyncpg's default JSON string codec."""

    connection = StubConnection()
    connection.rows = [{"graph": json.dumps(GRAPH)}, {"graph": GRAPH}]

    graphs = await list_ready_wiki_base_graphs(
        connection,  # type: ignore[arg-type]
        WIKI_BASE_ID,
    )

    assert graphs == [GRAPH, GRAPH]


async def test_loads_one_document_graph() -> None:
    """A stored document graph is normalized into a mapping."""

    connection = StubConnection()
    connection.value = json.dumps(GRAPH)

    graph = await get_document_graph(
        connection,  # type: ignore[arg-type]
        DOCUMENT_ID,
    )

    assert graph == GRAPH
