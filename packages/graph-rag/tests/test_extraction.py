from dataclasses import replace
from typing import Any
from uuid import UUID

import pytest
from document_processing.models import DocumentChunk
from llm_providers.generation.base import ChatMessage

from graph_rag.extraction import LLMTripleExtractor
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

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        self.messages = messages
        self.schema = schema
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
