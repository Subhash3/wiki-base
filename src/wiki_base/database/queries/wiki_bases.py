from datetime import datetime
from uuid import UUID, uuid4

from asyncpg import Connection

from wiki_base.database.records import IngestionStatus, WikiBaseRecord
from wiki_base.ingestion.staging import StagedDocument


async def create_wiki_base_manifest(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    name: str,
    documents: list[tuple[UUID, StagedDocument]],
    embedding_model: str,
    embedding_dimensions: int,
) -> datetime:
    async with connection.transaction():
        created_at = await connection.fetchval(
            """
            INSERT INTO wiki_bases (
                id, name, status, embedding_model, embedding_dimensions
            )
            VALUES ($1, $2, 'queued', $3, $4)
            RETURNING created_at
            """,
            wiki_base_id,
            name,
            embedding_model,
            embedding_dimensions,
        )
        for document_id, document in documents:
            await connection.execute(
                """
                INSERT INTO documents (
                    id, wiki_base_id, name, media_type, content_checksum, status
                )
                VALUES ($1, $2, $3, $4, $5, 'queued')
                """,
                document_id,
                wiki_base_id,
                document.name,
                document.media_type,
                document.checksum,
            )
            await connection.execute(
                """
                INSERT INTO ingestion_jobs (
                    id, wiki_base_id, document_id, status, staging_reference
                )
                VALUES ($1, $2, $3, 'queued', $4)
                """,
                uuid4(),
                wiki_base_id,
                document_id,
                str(document.path),
            )
    return created_at


async def get_wiki_base(connection: Connection, wiki_base_id: UUID) -> WikiBaseRecord | None:
    row = await connection.fetchrow(
        """
        SELECT id, name, status, created_at, started_at, completed_at
        FROM wiki_bases
        WHERE id = $1
        """,
        wiki_base_id,
    )
    if row is None:
        return None
    return WikiBaseRecord(
        id=row["id"],
        name=row["name"],
        status=IngestionStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
    )


async def list_wiki_bases(connection: Connection) -> list[tuple[WikiBaseRecord, int]]:
    rows = await connection.fetch(
        """
        SELECT wiki_base.id, wiki_base.name, wiki_base.status,
               wiki_base.created_at, wiki_base.started_at, wiki_base.completed_at,
               count(document.id) AS document_count
        FROM wiki_bases AS wiki_base
        LEFT JOIN documents AS document ON document.wiki_base_id = wiki_base.id
        GROUP BY wiki_base.id
        ORDER BY wiki_base.created_at DESC, wiki_base.id DESC
        """
    )
    return [
        (
            WikiBaseRecord(
                id=row["id"],
                name=row["name"],
                status=IngestionStatus(row["status"]),
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
            ),
            row["document_count"],
        )
        for row in rows
    ]
