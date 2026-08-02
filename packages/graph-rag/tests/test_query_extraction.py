from typing import Any

import pytest
from llm_providers.generation.base import ChatMessage

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
    assert generation.schema["properties"]["entities"]["maxItems"] == 12
    assert generation.schema["properties"]["relationships"]["maxItems"] == 12
    assert generation.max_tokens == 256


async def test_empty_question_skips_generation() -> None:
    """Blank questions do not call the model."""

    generation = StubStructuredGeneration(
        {"entities": ["unused"], "relationships": []}
    )

    concepts = await LLMQueryEntityExtractor(generation=generation).extract("   ")

    assert concepts == QueryConcepts(entities=[], relationships=[])
    assert generation.messages == []


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
