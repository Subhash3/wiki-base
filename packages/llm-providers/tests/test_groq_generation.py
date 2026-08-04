import json

import httpx
import pytest

from llm_providers.generation.base import (
    ChatMessage,
    GenerationRequestError,
    RecoverableGenerationError,
)
from llm_providers.generation.groq import (
    GroqGenerationProvider,
    GroqRateLimiter,
    GroqRateLimitError,
)

SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}


def make_limiter(
    *,
    requests_per_day: int = 900,
) -> GroqRateLimiter:
    """Create a limiter with roomy minute budgets for provider tests."""

    return GroqRateLimiter(
        requests_per_minute=25,
        tokens_per_minute=10000,
        requests_per_day=requests_per_day,
        tokens_per_day=100000,
    )


async def test_generates_an_answer_with_open_world_comparison_rules() -> None:
    """Groq receives the shared rules for grounded comparisons."""

    captured: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Both sources describe the feature.",
                                    "source_ids": ["S1", "S2"],
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(),
            client=client,
        )
        answer = await provider.generate(
            [ChatMessage(role="user", content="Which features differ?")],
            "[S1] Slavia evidence\n[S2] Tiago evidence",
        )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    system_prompt = payload["messages"][0]["content"]
    assert "[S1] Slavia evidence" in system_prompt
    assert "Do not interpret an unmentioned feature as absent" in system_prompt
    assert "not mentioned for the second" in system_prompt
    assert "Do not withhold supported facts" in system_prompt
    assert "only when no supplied source contains any fact relevant" in system_prompt
    assert "incomplete evidence" not in system_prompt.casefold()
    assert answer.text == "Both sources describe the feature."
    assert answer.source_ids == ("S1", "S2")


async def test_generates_strict_structured_content() -> None:
    """Groq receives strict schema settings and returns decoded JSON."""

    captured: dict[str, object] = {}

    async def respond(request: httpx.Request) -> httpx.Response:
        captured["authorization"] = request.headers["Authorization"]
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"entities":["Tiago"]}'}}],
                "usage": {"total_tokens": 42},
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(),
            client=client,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Find entities")],
            SCHEMA,
            max_tokens=100,
        )

    assert result == {"entities": ["Tiago"]}
    assert captured["authorization"] == "Bearer test-key"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "openai/gpt-oss-20b"
    assert payload["reasoning_effort"] == "low"
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_response",
            "strict": True,
            "schema": SCHEMA,
        },
    }


async def test_retries_a_rate_limited_request() -> None:
    """A Groq 429 response honors retry-after before retrying."""

    calls = 0
    waits: list[float] = []

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"entities":[]}'}}],
                "usage": {"total_tokens": 10},
            },
        )

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(),
            max_retries=1,
            client=client,
            sleeper=sleep,
        )
        result = await provider.generate_structured(
            [ChatMessage(role="user", content="Find entities")],
            SCHEMA,
            max_tokens=100,
        )

    assert result == {"entities": []}
    assert calls == 2
    assert waits == [0.0]


async def test_does_not_retry_recoverable_bad_request() -> None:
    """A structured-generation 400 is classified without another request."""

    calls = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "Failed to generate JSON matching the schema",
                    "type": "invalid_request_error",
                    "failed_generation": {"reason": "schema mismatch"},
                }
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(),
            max_retries=2,
            client=client,
        )
        with pytest.raises(RecoverableGenerationError, match="structured output"):
            await provider.generate_structured(
                [ChatMessage(role="user", content="Find entities")],
                SCHEMA,
                max_tokens=100,
            )

    assert calls == 1


async def test_reports_nonrecoverable_bad_request_without_retry() -> None:
    """Configuration-related 400 responses retain safe Groq diagnostics."""

    calls = 0

    async def respond(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "reasoning_effort is invalid",
                    "type": "invalid_request_error",
                    "code": "invalid_parameter",
                }
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(),
            max_retries=2,
            client=client,
        )
        with pytest.raises(GenerationRequestError, match="invalid_parameter"):
            await provider.generate_structured(
                [ChatMessage(role="user", content="Find entities")],
                SCHEMA,
                max_tokens=100,
            )

    assert calls == 1


async def test_stops_at_local_daily_request_budget() -> None:
    """The local daily cap prevents additional Groq requests."""

    async def respond(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"entities":[]}'}}],
                "usage": {"total_tokens": 10},
            },
        )

    async with httpx.AsyncClient(
        base_url="https://api.groq.com/openai/v1",
        transport=httpx.MockTransport(respond),
    ) as client:
        provider = GroqGenerationProvider(
            api_key="test-key",
            model="openai/gpt-oss-20b",
            timeout_seconds=10,
            rate_limiter=make_limiter(requests_per_day=1),
            client=client,
        )
        await provider.generate_structured(
            [ChatMessage(role="user", content="First")],
            SCHEMA,
            max_tokens=100,
        )
        with pytest.raises(GroqRateLimitError, match="daily request"):
            await provider.generate_structured(
                [ChatMessage(role="user", content="Second")],
                SCHEMA,
                max_tokens=100,
            )


async def test_waits_for_rolling_request_capacity() -> None:
    """The limiter spaces requests when its rolling minute is full."""

    current_time = 0.0
    waits: list[float] = []

    def clock() -> float:
        return current_time

    async def sleep(seconds: float) -> None:
        nonlocal current_time
        waits.append(seconds)
        current_time += seconds

    limiter = GroqRateLimiter(
        requests_per_minute=1,
        tokens_per_minute=100,
        requests_per_day=10,
        tokens_per_day=1000,
        clock=clock,
        sleeper=sleep,
    )

    await limiter.acquire(10)
    await limiter.acquire(10)

    assert waits == [60.0]
