import json

import httpx
import pytest

from llm_providers.generation.base import ChatMessage
from llm_providers.generation.ollama import OllamaGenerationProvider


async def test_generates_a_structured_answer() -> None:
    async def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "gemma3:270m"
        system_prompt = body["messages"][0]["content"]
        assert "[S1] policy.pdf" in system_prompt
        assert "ignore unrelated sources" in system_prompt
        assert "Do not interpret an unmentioned feature as absent" in system_prompt
        assert "not mentioned for the second" in system_prompt
        assert "Do not withhold supported facts" in system_prompt
        assert "only when no supplied source contains any fact relevant" in system_prompt
        assert "incomplete evidence" not in system_prompt.casefold()
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
        assert body["think"] is False
        assert body["options"] == {"temperature": 0, "num_predict": 128}
        assert body["messages"][-1] == {
            "role": "user",
            "content": "Find entities.",
        }
        assert body["messages"][0]["role"] == "system"
        assert "Return only valid JSON" in body["messages"][0]["content"]
        assert json.dumps(schema, separators=(",", ":")) in body["messages"][0][
            "content"
        ]
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
            max_tokens=128,
        )

    assert result == {"entities": ["Acme"]}


async def test_reports_invalid_structured_json() -> None:
    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "done_reason": "length",
                "message": {"role": "assistant", "content": '{"triples": ["unfinished'},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )

        with pytest.raises(
            ValueError,
            match=r"invalid JSON \(reason=length, characters=24, preview=",
        ):
            await provider.generate_structured(
                [ChatMessage(role="user", content="Extract triples.")],
                {"type": "object"},
            )


async def test_wraps_bare_array_for_single_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["entities"],
    }

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "done_reason": "stop",
                "message": {"role": "assistant", "content": '["Alice", "Acme"]'},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Extract entities.")],
            schema,
        )

    assert result == {"entities": ["Alice", "Acme"]}


async def test_decodes_json_wrapped_in_markdown_commentary() -> None:
    schema = {
        "type": "object",
        "properties": {
            "triples": {
                "type": "array",
                "items": {"type": "object"},
            }
        },
        "required": ["triples"],
    }

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": (
                        "Here is the result:\n```json\n"
                        '{"triples":[{"subject":"Alice","relation":"works at",'
                        '"object":"Acme"}]}\n```'
                    ),
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Extract triples.")],
            schema,
        )

    assert result == {
        "triples": [
            {"subject": "Alice", "relation": "works at", "object": "Acme"}
        ]
    }


async def test_decodes_python_style_structured_output() -> None:
    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["entities"],
    }

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "done_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Extracted entities: {'entities': ['Alice', 'Acme']}",
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider = OllamaGenerationProvider(
            base_url="http://ollama.test",
            model="gemma3:270m",
            timeout_seconds=10,
            client=client,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Extract entities.")],
            schema,
        )

    assert result == {"entities": ["Alice", "Acme"]}
