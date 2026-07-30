from uuid import UUID

from asyncpg import Connection

from wiki_base.database.records import DocumentRecord, IngestionStatus


async def list_wiki_base_documents(
    connection: Connection,
    wiki_base_id: UUID,
) -> list[DocumentRecord]:
    rows = await connection.fetch(
        """
        SELECT document.id, document.wiki_base_id, document.name,
               document.media_type, job.status, document.created_at,
               job.error_code, job.error_message
        FROM documents AS document
        JOIN ingestion_jobs AS job ON job.document_id = document.id
        WHERE document.wiki_base_id = $1
        ORDER BY document.created_at, document.id
        """,
        wiki_base_id,
    )
    return [
        DocumentRecord(
            id=row["id"],
            wiki_base_id=row["wiki_base_id"],
            name=row["name"],
            media_type=row["media_type"],
            status=IngestionStatus(row["status"]),
            created_at=row["created_at"],
            error_code=row["error_code"],
            error_message=row["error_message"],
        )
        for row in rows
    ]
