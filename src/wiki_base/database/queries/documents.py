from uuid import UUID

from asyncpg import Connection

from wiki_base.database.records import DocumentRecord, IngestionStatus


async def list_wiki_base_documents(
    connection: Connection,
    wiki_base_id: UUID,
) -> list[DocumentRecord]:
    rows = await connection.fetch(
        """
        SELECT id, wiki_base_id, name, media_type, status, created_at,
               error_code, error_message
        FROM documents
        WHERE wiki_base_id = $1
        ORDER BY created_at, id
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
