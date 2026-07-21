from dataclasses import dataclass
from uuid import UUID

from asyncpg import Connection


@dataclass(frozen=True, slots=True)
class ChunkSearchResult:
    id: UUID
    document_id: UUID
    document_name: str
    content: str
    score: float
    page_number: int | None
    slide_number: int | None
    section: str | None
    heading: str | None


async def search_chunks(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    embedding: list[float],
    limit: int,
) -> list[ChunkSearchResult]:
    rows = await connection.fetch(
        """
        SELECT chunk.id, chunk.document_id, document.name AS document_name,
               chunk.content,
               1 - (chunk.embedding <=> $2::vector) AS score,
               chunk.page_number, chunk.slide_number, chunk.section, chunk.heading
        FROM chunks AS chunk
        JOIN documents AS document ON document.id = chunk.document_id
        WHERE chunk.wiki_base_id = $1
          AND chunk.embedding IS NOT NULL
        ORDER BY chunk.embedding <=> $2::vector
        LIMIT $3
        """,
        wiki_base_id,
        embedding,
        limit,
    )
    return [
        ChunkSearchResult(
            id=row["id"],
            document_id=row["document_id"],
            document_name=row["document_name"],
            content=row["content"],
            score=float(row["score"]),
            page_number=row["page_number"],
            slide_number=row["slide_number"],
            section=row["section"],
            heading=row["heading"],
        )
        for row in rows
    ]
