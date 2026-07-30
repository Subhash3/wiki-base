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
