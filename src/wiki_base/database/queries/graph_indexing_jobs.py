import json
from pathlib import Path
from typing import Any
from uuid import UUID

from asyncpg import Connection
from document_processing.models import DocumentChunk
from graph_rag import IndexedChunk

from wiki_base.database.records import GraphIndexingJobRecord


async def list_ready_wiki_base_graph_paths(
    connection: Connection,
    wiki_base_id: UUID,
) -> list[Path]:
    """List ready document graph files for a wiki base."""

    rows = await connection.fetch(
        """
        SELECT job.output_path
        FROM graph_indexing_jobs AS job
        JOIN documents AS document ON document.id = job.document_id
        JOIN ingestion_jobs AS ingestion_job
          ON ingestion_job.document_id = document.id
        WHERE document.wiki_base_id = $1
          AND ingestion_job.status = 'ready'
          AND job.status = 'ready'
          AND job.output_path IS NOT NULL
        """,
        wiki_base_id,
    )
    return [Path(row["output_path"]) for row in rows]


async def claim_next_graph_indexing_job(
    connection: Connection,
) -> GraphIndexingJobRecord | None:
    """Claim one queued document for graph indexing."""

    async with connection.transaction():
        row = await connection.fetchrow(
            """
            SELECT document_id
            FROM graph_indexing_jobs
            WHERE status = 'queued'
            LIMIT 1
            """
        )
        if row is None:
            return None
        await connection.execute(
            """
            UPDATE graph_indexing_jobs
            SET status = 'processing', started_at = now(), error_message = NULL
            WHERE document_id = $1
            """,
            row["document_id"],
        )
    return GraphIndexingJobRecord(document_id=row["document_id"])


async def load_graph_indexing_chunks(
    connection: Connection,
    document_id: UUID,
) -> list[IndexedChunk]:
    """Load all stored chunks for one document."""

    rows = await connection.fetch(
        """
        SELECT id, document_id, content, embedding_content, ordinal, token_count,
               page_number, slide_number, section, heading, caption, metadata
        FROM chunks
        WHERE document_id = $1
        """,
        document_id,
    )
    return [
        IndexedChunk(
            document_id=row["document_id"],
            chunk=DocumentChunk(
                id=row["id"],
                ordinal=row["ordinal"],
                content=row["content"],
                embedding_content=row["embedding_content"],
                token_count=row["token_count"],
                page_number=row["page_number"],
                slide_number=row["slide_number"],
                section=row["section"],
                heading=row["heading"],
                caption=row["caption"],
                metadata=_metadata(row["metadata"]),
            ),
        )
        for row in rows
    ]


async def complete_graph_indexing_job(
    connection: Connection,
    job: GraphIndexingJobRecord,
    *,
    output_path: Path,
    extraction_model: str,
    index_version: str,
) -> None:
    """Mark a graph indexing job ready."""

    await connection.execute(
        """
        UPDATE graph_indexing_jobs
        SET status = 'ready', output_path = $2, extraction_model = $3,
            index_version = $4, completed_at = now(), error_message = NULL
        WHERE document_id = $1
        """,
        job.document_id,
        str(output_path),
        extraction_model,
        index_version,
    )


async def fail_graph_indexing_job(
    connection: Connection,
    job: GraphIndexingJobRecord,
    *,
    error_message: str,
) -> None:
    """Mark a graph indexing job failed."""

    await connection.execute(
        """
        UPDATE graph_indexing_jobs
        SET status = 'failed', completed_at = now(), error_message = $2
        WHERE document_id = $1
        """,
        job.document_id,
        error_message,
    )


def _metadata(value: Any) -> dict[str, Any]:
    """Normalize a stored JSON metadata value."""

    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}
