from typing import Any

import pytest
from llm_providers.generation.base import ChatMessage, RecoverableGenerationError

from graph_rag.query_extraction import LLMQueryEntityExtractor, QueryConcepts


class StubStructuredGeneration:
    """Return configured structured output for extractor tests."""

    def __init__(self, result: dict[str, Any]) -> None:
        """Store the result returned by generation."""

        self.result = result
        self.messages: list[ChatMessage] = []
        self.schema: dict[str, Any] = {}
        self.max_tokens: int | None = None

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Capture the request and return the configured result."""

        self.messages = messages
        self.schema = schema
        self.max_tokens = max_tokens
        return self.result


async def test_extracts_distinct_query_concepts() -> None:
    """Concepts are cleaned and deduplicated in model order."""

    generation = StubStructuredGeneration(
        {
            "entities": [" Alice ", "Acme", "Alice", ""],
            "relationships": [" works at ", "works at", ""],
        }
    )

    concepts = await LLMQueryEntityExtractor(generation=generation).extract(
        "Where is Alice's employer Acme headquartered?"
    )

    assert concepts == QueryConcepts(
        entities=["Alice", "Acme"],
        relationships=["works at"],
    )
    assert generation.messages[-1] == ChatMessage(
        role="user",
        content="Where is Alice's employer Acme headquartered?",
    )
    assert "maxItems" not in generation.schema["properties"]["entities"]
    assert "maxItems" not in generation.schema["properties"]["relationships"]
    assert generation.max_tokens == 256
    assert "every item in a comparison" in generation.messages[0].content
    assert "engine options" in generation.messages[0].content


async def test_empty_question_skips_generation() -> None:
    """Blank questions do not call the model."""

    generation = StubStructuredGeneration(
        {"entities": ["unused"], "relationships": []}
    )

    concepts = await LLMQueryEntityExtractor(generation=generation).extract("   ")

    assert concepts == QueryConcepts(entities=[], relationships=[])
    assert generation.messages == []


async def test_recoverable_generation_failure_returns_empty_concepts() -> None:
    """A rejected structured response allows vector fallback without retrying."""

    class FailingGeneration(StubStructuredGeneration):
        async def generate_structured(
            self,
            messages: list[ChatMessage],
            schema: dict[str, Any],
            *,
            max_tokens: int = 4096,
        ) -> dict[str, Any]:
            raise RecoverableGenerationError("structured output rejected")

    concepts = await LLMQueryEntityExtractor(
        generation=FailingGeneration({}),
    ).extract("Where does Alice work?")

    assert concepts == QueryConcepts(entities=[], relationships=[])


async def test_truncates_oversized_query_concepts_locally() -> None:
    """Query concept limits do not require schema-level rejection or retry."""

    generation = StubStructuredGeneration(
        {
            "entities": [f"Entity {index}" for index in range(20)],
            "relationships": [f"Relation {index}" for index in range(20)],
        }
    )

    concepts = await LLMQueryEntityExtractor(generation=generation).extract(
        "Compare many entities"
    )

    assert len(concepts.entities) == 12
    assert concepts.entities[-1] == "Entity 11"
    assert len(concepts.relationships) == 12
    assert concepts.relationships[-1] == "Relation 11"


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"entities": "Alice", "relationships": []},
        {"entities": ["Alice", 42], "relationships": []},
    ],
)
async def test_rejects_invalid_entity_output(result: dict[str, Any]) -> None:
    """Malformed structured responses fail clearly."""

    extractor = LLMQueryEntityExtractor(
        generation=StubStructuredGeneration(result)
    )

    with pytest.raises(ValueError, match="invalid entities list"):
        await extractor.extract("Where does Alice work?")


@pytest.mark.parametrize(
    "relationships",
    [None, "works at", ["works at", 42]],
)
async def test_rejects_invalid_relationship_output(relationships: object) -> None:
    """Malformed relationship lists fail clearly."""

    extractor = LLMQueryEntityExtractor(
        generation=StubStructuredGeneration(
            {"entities": ["Alice"], "relationships": relationships}
        )
    )

    with pytest.raises(ValueError, match="invalid relationships list"):
        await extractor.extract("Where does Alice work?")
