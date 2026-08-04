from dataclasses import replace
from typing import Any
from uuid import UUID

import pytest
from document_processing.models import DocumentChunk
from llm_providers.generation.base import ChatMessage

from graph_rag.extraction import LLMPassageEntityExtractor, LLMTripleExtractor
from graph_rag.models import Triple


def make_chunk() -> DocumentChunk:
    return DocumentChunk(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        ordinal=0,
        content="Alice works at Acme.",
        embedding_content="Alice works at Acme.",
        token_count=5,
        page_number=1,
        slide_number=None,
        section=None,
        heading=None,
        caption=None,
    )


class StubStructuredGeneration:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.messages: list[ChatMessage] = []
        self.schema: dict[str, Any] = {}
        self.max_tokens = 0

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        self.messages = messages
        self.schema = schema
        self.max_tokens = max_tokens
        return self.result


async def test_extracts_triples_using_structured_generation() -> None:
    generation = StubStructuredGeneration(
        {
            "triples": [
                {"subject": "Alice", "relation": "works at", "object": "Acme"},
            ]
        }
    )

    triples = await LLMTripleExtractor(generation=generation).extract(make_chunk())

    assert triples == [Triple(subject="Alice", relation="works at", object="Acme")]
    assert generation.messages[-1] == ChatMessage(
        role="user",
        content="Alice works at Acme.",
    )
    assert generation.schema["properties"]["triples"]["type"] == "array"
    assert "maxItems" not in generation.schema["properties"]["triples"]
    assert generation.max_tokens == 1536


async def test_supplies_section_scope_for_fragmented_passages() -> None:
    """A section can identify the omitted subject of an isolated value."""

    generation = StubStructuredGeneration(
        {
            "triples": [
                {
                    "subject": "Skoda Slavia",
                    "relation": "has price range",
                    "object": "Rs. 10.00 - 18.19 Lakh",
                }
            ]
        }
    )
    chunk = replace(
        make_chunk(),
        content="Rs. 10.00 - 18.19 Lakh",
        embedding_content="Rs. 10.00 - 18.19 Lakh",
        section="Skoda Slavia",
        heading="Skoda Slavia",
    )

    triples = await LLMTripleExtractor(generation=generation).extract(chunk)

    assert triples == [
        Triple(
            subject="Skoda Slavia",
            relation="has price range",
            object="Rs. 10.00 - 18.19 Lakh",
        )
    ]
    assert generation.messages[-1] == ChatMessage(
        role="user",
        content=(
            "SCOPE\n"
            "Section: Skoda Slavia\n\n"
            "PASSAGE\n"
            "Rs. 10.00 - 18.19 Lakh"
        ),
    )
    assert "scope metadata, not independent facts" in generation.messages[0].content


async def test_extracts_passage_entities_and_noun_phrases() -> None:
    """Passage extraction retains concepts independently of triples."""

    generation = StubStructuredGeneration(
        {
            "entities": [
                " Skoda Slavia ",
                "price range",
                "Rs. 10.00 - 18.19 Lakh",
                "Skoda Slavia",
            ]
        }
    )

    entities = await LLMPassageEntityExtractor(generation=generation).extract(
        make_chunk()
    )

    assert entities == [
        "Skoda Slavia",
        "price range",
        "Rs. 10.00 - 18.19 Lakh",
    ]
    assert "maxItems" not in generation.schema["properties"]["entities"]
    assert generation.max_tokens == 512


async def test_supplies_passage_entities_to_triple_extraction() -> None:
    """OpenIE receives likely subjects and objects from the first pass."""

    generation = StubStructuredGeneration({"triples": []})

    await LLMTripleExtractor(generation=generation).extract(
        make_chunk(),
        entities=["Alice", "Acme"],
    )

    assert generation.messages[-1].content == (
        "EXTRACTED CONCEPTS\n- Alice\n- Acme\n\n"
        "PASSAGE\nAlice works at Acme."
    )
    assert "Reuse them" in generation.messages[0].content


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"triples": "not a list"},
        {"triples": ["not an object"]},
        {"triples": [{"subject": "Alice", "relation": "works at"}]},
    ],
)
async def test_rejects_invalid_structured_output(result: dict[str, Any]) -> None:
    extractor = LLMTripleExtractor(generation=StubStructuredGeneration(result))

    with pytest.raises(ValueError, match="invalid"):
        await extractor.extract(make_chunk())


async def test_rejects_invalid_passage_entities() -> None:
    """Passage extraction requires a list of strings."""

    extractor = LLMPassageEntityExtractor(
        generation=StubStructuredGeneration({"entities": "Skoda Slavia"})
    )

    with pytest.raises(ValueError, match="passage entities"):
        await extractor.extract(make_chunk())


async def test_truncates_oversized_entity_and_triple_results_locally() -> None:
    """Provider output limits are enforced without another model request."""

    entity_generation = StubStructuredGeneration(
        {"entities": [f"Entity {index}" for index in range(40)]}
    )
    triple_generation = StubStructuredGeneration(
        {
            "triples": [
                {
                    "subject": f"Subject {index}",
                    "relation": "relates to",
                    "object": f"Object {index}",
                }
                for index in range(40)
            ]
        }
    )

    entities = await LLMPassageEntityExtractor(
        generation=entity_generation
    ).extract(make_chunk())
    triples = await LLMTripleExtractor(generation=triple_generation).extract(
        make_chunk()
    )

    assert len(entities) == 24
    assert entities[-1] == "Entity 23"
    assert len(triples) == 24
    assert triples[-1].subject == "Subject 23"
