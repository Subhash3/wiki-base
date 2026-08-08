import json

import httpx
import pytest

from llm_providers.generation.base import ChatMessage
from llm_providers.generation.llama_cpp import LlamaCppGenerationProvider


@pytest.mark.asyncio
async def test_generate_structured_uses_openai_compatible_json_schema() -> None:
    request_body: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        request_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"entities":["Ada"]}'}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LlamaCppGenerationProvider(
            base_url="http://llama.test/",
            model="local-model",
            timeout_seconds=30,
            client=client,
        )
        schema = {
            "type": "object",
            "properties": {"entities": {"type": "array"}},
            "required": ["entities"],
        }
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Extract entities")],
            schema,
            max_tokens=128,
        )

    assert result == {"entities": ["Ada"]}
    assert request_body["model"] == "local-model"
    assert request_body["max_tokens"] == 128
    assert request_body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "response", "strict": True, "schema": schema},
    }


@pytest.mark.asyncio
async def test_generate_returns_grounded_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "source-1" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":"Ada wrote it.","source_ids":["source-1"]}'}}
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        provider = LlamaCppGenerationProvider(
            base_url="http://llama.test",
            model="local-model",
            timeout_seconds=30,
            client=client,
        )
        answer = await provider.generate(
            [ChatMessage(role="user", content="Who wrote it?")],
            "source-1: Ada wrote it.",
        )

    assert answer.text == "Ada wrote it."
    assert answer.source_ids == ("source-1",)


@pytest.mark.asyncio
async def test_generate_structured_rejects_invalid_json() -> None:
    transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json"}}]},
        )
    )
    async with httpx.AsyncClient(transport=transport) as client:
        provider = LlamaCppGenerationProvider(
            base_url="http://llama.test",
            model="local-model",
            timeout_seconds=30,
            client=client,
        )
        with pytest.raises(ValueError, match="llama.cpp returned invalid JSON"):
            await provider.generate_structured([], {"type": "object"})
