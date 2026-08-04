import json
from uuid import UUID

from graph_rag.entity_linking import SemanticConceptSearch
from graph_rag.models import EntityConceptMatch, RelationshipConceptMatch

from wiki_base.database.connection import Database
from wiki_base.database.queries.graph_concepts import (
    search_graph_entities,
    search_graph_relationships,
)


class PostgresSemanticConceptSearch(SemanticConceptSearch):
    """Search one wiki base's stored graph concepts with pgvector."""

    def __init__(
        self,
        *,
        database: Database,
        wiki_base_id: UUID,
        embedding_model: str,
    ) -> None:
        """Bind semantic searches to a wiki base and embedding model."""

        self._database = database
        self._wiki_base_id = wiki_base_id
        self._embedding_model = embedding_model

    async def search_entities(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
    ) -> list[EntityConceptMatch]:
        """Return the closest stored entity concepts."""

        async with self._database.connection() as connection:
            return await search_graph_entities(
                connection,
                wiki_base_id=self._wiki_base_id,
                embedding_model=self._embedding_model,
                embedding=embedding,
                threshold=threshold,
                limit=limit,
            )

    async def search_relationships(
        self,
        embedding: list[float],
        *,
        threshold: float,
        limit: int,
        candidate_keys: frozenset[tuple[str, str, str]] | None = None,
    ) -> list[RelationshipConceptMatch]:
        """Return the closest stored relationship concepts."""

        async with self._database.connection() as connection:
            return await search_graph_relationships(
                connection,
                wiki_base_id=self._wiki_base_id,
                embedding_model=self._embedding_model,
                embedding=embedding,
                threshold=threshold,
                limit=limit,
                concept_keys=(
                    [
                        json.dumps(key, ensure_ascii=False, separators=(",", ":"))
                        for key in sorted(candidate_keys)
                    ]
                    if candidate_keys is not None
                    else None
                ),
            )
