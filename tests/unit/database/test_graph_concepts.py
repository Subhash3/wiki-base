from uuid import UUID

from graph_rag import GraphConcept, GraphConceptType

from wiki_base.database.queries.graph_concepts import (
    replace_document_graph_concepts,
    search_graph_entities,
    search_graph_relationships,
)

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
WIKI_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")


class StubConnection:
    """Capture graph concept writes and return configured search rows."""

    def __init__(self) -> None:
        """Initialize captured database operations."""

        self.executed: list[tuple[object, ...]] = []
        self.inserted: list[tuple[object, ...]] = []
        self.rows: list[dict[str, object]] = []
        self.fetch_query = ""
        self.fetch_arguments: tuple[object, ...] = ()

    async def fetchval(self, _query: str, document_id: UUID) -> UUID:
        """Return the document's wiki base."""

        assert document_id == DOCUMENT_ID
        return WIKI_BASE_ID

    async def execute(self, _query: str, *arguments: object) -> None:
        """Capture delete operations."""

        self.executed.append(arguments)

    async def executemany(
        self,
        _query: str,
        arguments: list[tuple[object, ...]],
    ) -> None:
        """Capture inserted graph concepts."""

        self.inserted = arguments

    async def fetch(self, _query: str, *arguments: object) -> list[dict[str, object]]:
        """Return configured semantic matches."""

        self.fetch_query = _query
        self.fetch_arguments = arguments
        return self.rows


async def test_replaces_document_graph_concepts() -> None:
    """Concept persistence stores scope, metadata, and vectors."""

    connection = StubConnection()
    concept = GraphConcept(
        type=GraphConceptType.ENTITY,
        key="honda unicorn",
        text="honda unicorn",
    )

    await replace_document_graph_concepts(
        connection,  # type: ignore[arg-type]
        document_id=DOCUMENT_ID,
        concepts=[concept],
        embeddings=[[1.0, 0.0]],
        embedding_model="bge-m3",
    )

    assert connection.executed == [(DOCUMENT_ID,)]
    assert connection.inserted == [
        (
            DOCUMENT_ID,
            WIKI_BASE_ID,
            "entity",
            "honda unicorn",
            "honda unicorn",
            None,
            None,
            None,
            [1.0, 0.0],
            "bge-m3",
        )
    ]


async def test_searches_entities_with_scope_threshold_and_limit() -> None:
    """Entity search returns typed semantic matches."""

    connection = StubConnection()
    connection.rows = [{"concept_text": "honda unicorn", "similarity": 0.91}]

    matches = await search_graph_entities(
        connection,  # type: ignore[arg-type]
        wiki_base_id=WIKI_BASE_ID,
        embedding_model="bge-m3",
        embedding=[1.0, 0.0],
        threshold=0.75,
        limit=2,
    )

    assert matches[0].entity == "honda unicorn"
    assert matches[0].similarity == 0.91
    assert connection.fetch_arguments == (
        WIKI_BASE_ID,
        "bge-m3",
        [1.0, 0.0],
        0.75,
        2,
    )
    assert "$4::double precision" in connection.fetch_query


async def test_searches_contextual_relationships() -> None:
    """Relationship search returns endpoints needed for PageRank seeds."""

    connection = StubConnection()
    connection.rows = [
        {
            "concept_text": "honda unicorn offers loan amount",
            "subject": "honda unicorn",
            "relationship": "offers",
            "object": "loan amount",
            "similarity": 0.67,
        }
    ]

    matches = await search_graph_relationships(
        connection,  # type: ignore[arg-type]
        wiki_base_id=WIKI_BASE_ID,
        embedding_model="bge-m3",
        embedding=[1.0, 0.0],
        threshold=0.45,
        limit=20,
    )

    assert matches[0].subject == "honda unicorn"
    assert matches[0].relationship == "offers"
    assert matches[0].object == "loan amount"
    assert "$4::double precision" in connection.fetch_query
