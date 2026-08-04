from uuid import UUID

import pytest

from wiki_base.database.queries.graph_synonyms import (
    list_wiki_base_graph_synonyms,
    replace_wiki_base_graph_synonyms,
)

WIKI_BASE_ID = UUID("20000000-0000-0000-0000-000000000001")


class StubConnection:
    """Capture synonym SQL and return configured rows."""

    def __init__(self) -> None:
        """Initialize captured operations."""

        self.executions: list[tuple[str, tuple[object, ...]]] = []
        self.rows: list[dict[str, object]] = []

    async def execute(self, query: str, *arguments: object) -> None:
        """Capture synonym rebuild statements."""

        self.executions.append((query, arguments))

    async def fetch(self, _query: str, *arguments: object) -> list[dict[str, object]]:
        """Return configured synonym rows."""

        assert arguments == (WIKI_BASE_ID, "bge-m3", 0.95)
        return self.rows


async def test_rebuilds_synonyms_with_pgvector_threshold_and_top_n() -> None:
    """Synonym generation stays scoped and delegates similarity to PostgreSQL."""

    connection = StubConnection()

    await replace_wiki_base_graph_synonyms(
        connection,  # type: ignore[arg-type]
        wiki_base_id=WIKI_BASE_ID,
        embedding_model="bge-m3",
        similarity_threshold=0.85,
        max_links_per_entity=3,
    )

    assert len(connection.executions) == 2
    insert_query, arguments = connection.executions[1]
    assert arguments == (WIKI_BASE_ID, "bge-m3", 0.85, 3)
    assert "target.embedding <=> source.embedding" in insert_query
    assert "$3::double precision" in insert_query
    assert "LIMIT $4" in insert_query


@pytest.mark.parametrize(
    ("threshold", "max_links"),
    [(1.1, 3), (0.85, 0)],
)
async def test_rejects_invalid_synonym_settings(
    threshold: float,
    max_links: int,
) -> None:
    """Invalid synonym settings fail before database writes."""

    with pytest.raises(ValueError):
        await replace_wiki_base_graph_synonyms(
            StubConnection(),  # type: ignore[arg-type]
            wiki_base_id=WIKI_BASE_ID,
            embedding_model="bge-m3",
            similarity_threshold=threshold,
            max_links_per_entity=max_links,
        )


async def test_lists_typed_synonym_edges() -> None:
    """Stored synonym rows become graph-level semantic edges."""

    connection = StubConnection()
    connection.rows = [
        {
            "first_entity": "glamour",
            "second_entity": "honda glamour",
            "similarity": 0.91,
        }
    ]

    synonyms = await list_wiki_base_graph_synonyms(
        connection,  # type: ignore[arg-type]
        wiki_base_id=WIKI_BASE_ID,
        embedding_model="bge-m3",
        similarity_threshold=0.95,
    )

    assert synonyms[0].first == "glamour"
    assert synonyms[0].second == "honda glamour"
    assert synonyms[0].similarity == 0.91
