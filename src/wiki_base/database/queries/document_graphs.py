import json
from typing import Any
from uuid import UUID

from asyncpg import Connection


async def upsert_document_graph(
    connection: Connection,
    *,
    document_id: UUID,
    graph: dict[str, Any],
    extraction_model: str,
    index_version: str,
) -> None:
    """Store the latest canonical graph for a document."""

    await connection.execute(
        """
        INSERT INTO document_graphs (
            document_id, graph, extraction_model, index_version
        )
        VALUES ($1, $2::jsonb, $3, $4)
        ON CONFLICT (document_id) DO UPDATE
        SET graph = EXCLUDED.graph,
            extraction_model = EXCLUDED.extraction_model,
            index_version = EXCLUDED.index_version
        """,
        document_id,
        json.dumps(graph),
        extraction_model,
        index_version,
    )


async def get_document_graph(
    connection: Connection,
    document_id: UUID,
) -> dict[str, Any] | None:
    """Load the canonical graph stored for one document."""

    value = await connection.fetchval(
        """
        SELECT graph
        FROM document_graphs
        WHERE document_id = $1
        """,
        document_id,
    )
    return None if value is None else _graph_payload(value)


async def list_ready_wiki_base_graphs(
    connection: Connection,
    wiki_base_id: UUID,
) -> list[dict[str, Any]]:
    """Load ready canonical document graphs for a wiki base."""

    rows = await connection.fetch(
        """
        SELECT stored_graph.graph
        FROM document_graphs AS stored_graph
        JOIN documents AS document
          ON document.id = stored_graph.document_id
        JOIN ingestion_jobs AS ingestion_job
          ON ingestion_job.document_id = document.id
        JOIN graph_indexing_jobs AS graph_job
          ON graph_job.document_id = document.id
        WHERE document.wiki_base_id = $1
          AND ingestion_job.status = 'ready'
          AND graph_job.status = 'ready'
        ORDER BY document.id
        """,
        wiki_base_id,
    )
    return [_graph_payload(row["graph"]) for row in rows]


def _graph_payload(value: Any) -> dict[str, Any]:
    """Normalize asyncpg JSONB output into a mapping."""

    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("Stored document graph is not a JSON object")
    return value
