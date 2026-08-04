from uuid import UUID

from asyncpg import Connection
from graph_rag import (
    EntityConceptMatch,
    GraphConcept,
    RelationshipConceptMatch,
)


async def replace_document_graph_concepts(
    connection: Connection,
    *,
    document_id: UUID,
    concepts: list[GraphConcept],
    embeddings: list[list[float]],
    embedding_model: str,
) -> UUID:
    """Replace the searchable concepts stored for one document graph."""

    if len(concepts) != len(embeddings):
        raise ValueError("Every graph concept must have one embedding")

    wiki_base_id = await connection.fetchval(
        "SELECT wiki_base_id FROM documents WHERE id = $1",
        document_id,
    )
    if wiki_base_id is None:
        raise ValueError(f"Document does not exist: {document_id}")

    await connection.execute(
        "DELETE FROM document_graph_concepts WHERE document_id = $1",
        document_id,
    )
    if not concepts:
        return wiki_base_id

    await connection.executemany(
        """
        INSERT INTO document_graph_concepts (
            document_id, wiki_base_id, concept_type, concept_key, concept_text,
            subject, relationship, object, embedding, embedding_model
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        [
            (
                document_id,
                wiki_base_id,
                concept.type.value,
                concept.key,
                concept.text,
                concept.subject,
                concept.relationship,
                concept.object,
                embedding,
                embedding_model,
            )
            for concept, embedding in zip(concepts, embeddings, strict=True)
        ],
    )
    return wiki_base_id


async def search_graph_entities(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    embedding_model: str,
    embedding: list[float],
    threshold: float,
    limit: int,
) -> list[EntityConceptMatch]:
    """Find the closest distinct entity concepts with pgvector."""

    rows = await connection.fetch(
        """
        SELECT concept.concept_text,
               1 - (concept.embedding <=> $3::vector) AS similarity
        FROM document_graph_concepts AS concept
        JOIN graph_indexing_jobs AS job
          ON job.document_id = concept.document_id
        WHERE concept.wiki_base_id = $1
          AND concept.concept_type = 'entity'
          AND concept.embedding_model = $2
          AND job.status = 'ready'
          AND (concept.embedding <=> $3::vector)
              <= (1::double precision - $4::double precision)
        ORDER BY concept.embedding <=> $3::vector
        LIMIT $5
        """,
        wiki_base_id,
        embedding_model,
        embedding,
        threshold,
        limit,
    )
    matches: list[EntityConceptMatch] = []
    seen: set[str] = set()
    for row in rows:
        entity = row["concept_text"]
        if entity not in seen:
            matches.append(
                EntityConceptMatch(
                    entity=entity,
                    similarity=float(row["similarity"]),
                )
            )
            seen.add(entity)
    return matches


async def search_graph_relationships(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    embedding_model: str,
    embedding: list[float],
    threshold: float,
    limit: int,
) -> list[RelationshipConceptMatch]:
    """Find the closest distinct relationship facts with pgvector."""

    rows = await connection.fetch(
        """
        SELECT concept.concept_text, concept.subject,
               concept.relationship, concept.object,
               1 - (concept.embedding <=> $3::vector) AS similarity
        FROM document_graph_concepts AS concept
        JOIN graph_indexing_jobs AS job
          ON job.document_id = concept.document_id
        WHERE concept.wiki_base_id = $1
          AND concept.concept_type = 'relationship'
          AND concept.embedding_model = $2
          AND job.status = 'ready'
          AND (concept.embedding <=> $3::vector)
              <= (1::double precision - $4::double precision)
        ORDER BY concept.embedding <=> $3::vector
        LIMIT $5
        """,
        wiki_base_id,
        embedding_model,
        embedding,
        threshold,
        limit,
    )
    matches: list[RelationshipConceptMatch] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["subject"], row["relationship"], row["object"])
        if key not in seen:
            matches.append(
                RelationshipConceptMatch(
                    text=row["concept_text"],
                    subject=row["subject"],
                    relationship=row["relationship"],
                    object=row["object"],
                    similarity=float(row["similarity"]),
                )
            )
            seen.add(key)
    return matches
