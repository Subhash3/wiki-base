import json
from datetime import datetime
from pathlib import Path
from uuid import UUID

from asyncpg import Connection
from document_processing.models import EmbeddedChunk

from wiki_base.database.records import IngestionJobRecord


async def claim_next_ingestion_job(connection: Connection) -> IngestionJobRecord | None:
    async with connection.transaction():
        row = await connection.fetchrow(
            """
            SELECT job.id, job.wiki_base_id, job.document_id, job.staging_reference,
                   document.name, document.media_type
            FROM ingestion_jobs AS job
            JOIN documents AS document ON document.id = job.document_id
            WHERE job.status = 'queued'
            ORDER BY job.queued_at
            FOR UPDATE OF job SKIP LOCKED
            LIMIT 1
            """
        )
        if row is None:
            return None

        now = datetime.now().astimezone()
        await connection.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'processing', attempts = attempts + 1,
                started_at = COALESCE(started_at, $2), heartbeat_at = $2
            WHERE id = $1
            """,
            row["id"],
            now,
        )
        await connection.execute(
            """
            UPDATE wiki_bases
            SET started_at = COALESCE(started_at, $2)
            WHERE id = $1
            """,
            row["wiki_base_id"],
            now,
        )

    staging_reference = row["staging_reference"]
    if not staging_reference:
        raise ValueError("Queued ingestion job has no staged document")
    return IngestionJobRecord(
        id=row["id"],
        wiki_base_id=row["wiki_base_id"],
        document_id=row["document_id"],
        document_name=row["name"],
        media_type=row["media_type"],
        staging_path=Path(staging_reference),
    )


async def complete_ingestion_job(
    connection: Connection,
    job: IngestionJobRecord,
    chunks: list[EmbeddedChunk],
) -> None:
    async with connection.transaction():
        await connection.execute("DELETE FROM chunks WHERE document_id = $1", job.document_id)
        await connection.executemany(
            """
            INSERT INTO chunks (
                id, wiki_base_id, document_id, content, embedding_content, embedding,
                ordinal, token_count, page_number, slide_number, section, heading,
                caption, metadata
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                $14::jsonb
            )
            """,
            [
                (
                    item.chunk.id,
                    job.wiki_base_id,
                    job.document_id,
                    item.chunk.content,
                    item.chunk.embedding_content,
                    item.embedding,
                    item.chunk.ordinal,
                    item.chunk.token_count,
                    item.chunk.page_number,
                    item.chunk.slide_number,
                    item.chunk.section,
                    item.chunk.heading,
                    item.chunk.caption,
                    json.dumps(item.chunk.metadata),
                )
                for item in chunks
            ],
        )
        await connection.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'ready', completed_at = now(), staging_reference = NULL,
                error_code = NULL, error_message = NULL
            WHERE id = $1
            """,
            job.id,
        )
        await connection.execute(
            """
            INSERT INTO graph_indexing_jobs (document_id)
            VALUES ($1)
            ON CONFLICT (document_id) DO NOTHING
            """,
            job.document_id,
        )
        await _update_wiki_base_completion(connection, job.wiki_base_id)


async def fail_ingestion_job(
    connection: Connection,
    job: IngestionJobRecord,
    *,
    error_code: str,
    error_message: str,
) -> None:
    async with connection.transaction():
        await connection.execute(
            """
            UPDATE ingestion_jobs
            SET status = 'failed', completed_at = now(), staging_reference = NULL,
                error_code = $2, error_message = $3
            WHERE id = $1
            """,
            job.id,
            error_code,
            error_message,
        )
        await _update_wiki_base_completion(connection, job.wiki_base_id)


async def _update_wiki_base_completion(
    connection: Connection,
    wiki_base_id: UUID,
) -> None:
    """Update the completion timestamp after an ingestion job finishes."""

    pending = await connection.fetchval(
        """
        SELECT count(*)
        FROM ingestion_jobs
        WHERE wiki_base_id = $1
          AND status IN ('queued', 'processing')
        """,
        wiki_base_id,
    )

    await connection.execute(
        """
        UPDATE wiki_bases
        SET completed_at = CASE WHEN $2 = 0 THEN now() ELSE NULL END
        WHERE id = $1
        """,
        wiki_base_id,
        pending,
    )
