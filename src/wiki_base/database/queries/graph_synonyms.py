from uuid import UUID

from asyncpg import Connection
from graph_rag import SynonymEdge


async def replace_wiki_base_graph_synonyms(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    embedding_model: str,
    similarity_threshold: float,
    max_links_per_entity: int,
) -> None:
    """Rebuild high-confidence synonym edges for one wiki base."""

    if not -1 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between -1 and 1")
    if max_links_per_entity < 1:
        raise ValueError("max_links_per_entity must be positive")

    await connection.execute(
        """
        DELETE FROM wiki_base_graph_synonyms
        WHERE wiki_base_id = $1 AND embedding_model = $2
        """,
        wiki_base_id,
        embedding_model,
    )
    await connection.execute(
        """
        WITH entities AS (
            SELECT DISTINCT ON (concept.concept_text)
                   concept.concept_text AS entity,
                   concept.embedding
            FROM document_graph_concepts AS concept
            JOIN graph_indexing_jobs AS job
              ON job.document_id = concept.document_id
            WHERE concept.wiki_base_id = $1
              AND concept.concept_type = 'entity'
              AND concept.embedding_model = $2
              AND job.status = 'ready'
            ORDER BY concept.concept_text, concept.document_id
        ),
        nearest AS (
            SELECT source.entity AS source_entity,
                   candidate.entity AS candidate_entity,
                   candidate.similarity
            FROM entities AS source
            CROSS JOIN LATERAL (
                SELECT target.entity,
                       1 - (target.embedding <=> source.embedding) AS similarity
                FROM entities AS target
                WHERE target.entity <> source.entity
                  AND (target.embedding <=> source.embedding)
                      <= (1::double precision - $3::double precision)
                ORDER BY target.embedding <=> source.embedding, target.entity
                LIMIT $4
            ) AS candidate
        ),
        pairs AS (
            SELECT LEAST(source_entity, candidate_entity) AS first_entity,
                   GREATEST(source_entity, candidate_entity) AS second_entity,
                   MAX(similarity) AS similarity
            FROM nearest
            GROUP BY LEAST(source_entity, candidate_entity),
                     GREATEST(source_entity, candidate_entity)
        )
        INSERT INTO wiki_base_graph_synonyms (
            wiki_base_id, embedding_model, first_entity, second_entity, similarity
        )
        SELECT $1, $2, first_entity, second_entity, similarity
        FROM pairs
        """,
        wiki_base_id,
        embedding_model,
        similarity_threshold,
        max_links_per_entity,
    )


async def list_wiki_base_graph_synonyms(
    connection: Connection,
    *,
    wiki_base_id: UUID,
    embedding_model: str,
    similarity_threshold: float,
) -> list[SynonymEdge]:
    """Load persisted synonym edges for one wiki base."""

    rows = await connection.fetch(
        """
        SELECT first_entity, second_entity, similarity
        FROM wiki_base_graph_synonyms
        WHERE wiki_base_id = $1
          AND embedding_model = $2
          AND similarity >= $3::double precision
        ORDER BY first_entity, second_entity
        """,
        wiki_base_id,
        embedding_model,
        similarity_threshold,
    )
    return [
        SynonymEdge(
            first=row["first_entity"],
            second=row["second_entity"],
            similarity=float(row["similarity"]),
        )
        for row in rows
    ]
