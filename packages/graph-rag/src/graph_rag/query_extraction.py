import logging
from dataclasses import dataclass
from typing import Protocol

from llm_providers.generation.base import (
    ChatMessage,
    RecoverableGenerationError,
    StructuredGenerationProvider,
)

logger = logging.getLogger(__name__)

_MAX_QUERY_CONCEPTS = 12

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {"type": "string"},
        },
        "relationships": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["entities", "relationships"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
Extract the entities and relationships needed to retrieve an answer.
Entities are named things or key noun phrases. Relationships are actions, attributes,
comparisons, or constraints connecting those things. Use concise phrases copied from the
question and preserve meaningful modifiers, such as "loan offers" instead of "offers".
Put every named thing in entities, including every item in a comparison. Never return a
named entity as a relationship. Avoid generic standalone words such as "available" when a
meaningful phrase such as "engine options" expresses the requested relationship.
Do not answer the question or infer unstated concepts.
Return at most 12 distinct values in each list and use empty lists when appropriate.

Example:
Question: Any loan offers on Honda Unicorn?
Output: {"entities":["Honda Unicorn"],"relationships":["loan offers"]}"""


@dataclass(frozen=True, slots=True)
class QueryConcepts:
    """Entities and relationships extracted from a question."""

    entities: list[str]
    relationships: list[str]


class QueryEntityExtractor(Protocol):
    """Extract retrieval concepts from a question."""

    async def extract(self, question: str) -> QueryConcepts:
        """Return entities and relationships needed for retrieval."""

        ...


class LLMQueryEntityExtractor:
    """Extract query entities and relationships with a language model."""

    def __init__(self, *, generation: StructuredGenerationProvider) -> None:
        """Configure the structured generation provider."""

        self._generation = generation

    async def extract(self, question: str) -> QueryConcepts:
        """Extract and validate concepts from a question."""

        if not question.strip():
            return QueryConcepts(entities=[], relationships=[])

        try:
            result = await self._generation.generate_structured(
                [
                    ChatMessage(role="system", content=_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=question.strip()),
                ],
                _ENTITY_SCHEMA,
                max_tokens=256,
            )
        except RecoverableGenerationError as error:
            logger.warning("Query concept extraction failed: %s", error)
            return QueryConcepts(entities=[], relationships=[])
        raw_entities = result.get("entities")
        if not isinstance(raw_entities, list) or not all(
            isinstance(entity, str) for entity in raw_entities
        ):
            raise ValueError("LLM returned an invalid entities list")
        raw_relationships = result.get("relationships")
        if not isinstance(raw_relationships, list) or not all(
            isinstance(relationship, str) for relationship in raw_relationships
        ):
            raise ValueError("LLM returned an invalid relationships list")

        return QueryConcepts(
            entities=_clean_values(raw_entities),
            relationships=_clean_values(raw_relationships),
        )


def _clean_values(values: list[str]) -> list[str]:
    """Clean and deduplicate extracted values while preserving order."""

    cleaned = [value.strip() for value in values if value.strip()]
    return list(dict.fromkeys(cleaned))[:_MAX_QUERY_CONCEPTS]
