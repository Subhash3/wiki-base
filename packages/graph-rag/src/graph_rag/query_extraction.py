from typing import Protocol

from llm_providers.generation.base import ChatMessage, StructuredGenerationProvider

_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string"},
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """\
Extract the named entities and key noun-phrase concepts needed to retrieve an answer.
Use concise phrases copied from the question.
Do not answer the question or infer unstated entities.
Return at most 12 distinct entities.
Return an empty list when the question contains no useful entities."""


class QueryEntityExtractor(Protocol):
    """Extract retrieval entities from a question."""

    async def extract(self, question: str) -> list[str]:
        """Return the entities needed to retrieve an answer."""

        ...


class LLMQueryEntityExtractor:
    """Extract query entities with a structured language model."""

    def __init__(self, *, generation: StructuredGenerationProvider) -> None:
        """Configure the structured generation provider."""

        self._generation = generation

    async def extract(self, question: str) -> list[str]:
        """Extract and validate entities from a question."""

        if not question.strip():
            return []

        result = await self._generation.generate_structured(
            [
                ChatMessage(role="system", content=_SYSTEM_PROMPT),
                ChatMessage(role="user", content=question.strip()),
            ],
            _ENTITY_SCHEMA,
            max_tokens=256,
        )
        raw_entities = result.get("entities")
        if not isinstance(raw_entities, list) or not all(
            isinstance(entity, str) for entity in raw_entities
        ):
            raise ValueError("LLM returned an invalid entities list")

        entities = [entity.strip() for entity in raw_entities if entity.strip()]
        return list(dict.fromkeys(entities))
