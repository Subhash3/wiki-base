import json

import httpx

from llm_providers.generation.base import ChatMessage
from llm_providers.generation.ollama import OllamaGenerationProvider


async def test_generates_a_structured_answer() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "gemma3:270m"
        assert "[S1] policy.pdf" in body["messages"][0]["content"]
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {"answer": "Contractors are eligible.", "source_ids": ["S1"]}
                    ),
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )
        answer = await provider.generate(
            [ChatMessage(role="user", content="Does this apply to contractors?")],
            "[S1] policy.pdf\nContractors are eligible.",
        )

    assert answer.text == "Contractors are eligible."
    assert answer.source_ids == ("S1",)


async def test_generates_arbitrary_structured_content() -> None:
    schema = {
        "type": "object",
        "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
        "required": ["entities"],
    }

    async def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["format"] == schema
        assert body["options"] == {"temperature": 0}
        assert body["messages"] == [{"role": "user", "content": "Find entities."}]
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": '{"entities":["Acme"]}'}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Find entities.")],
            schema,
        )

    assert result == {"entities": ["Acme"]}
