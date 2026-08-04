import asyncio
import json
import logging
import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from time import monotonic
from typing import Any

import httpx

from llm_providers.generation.base import (
    ChatMessage,
    GeneratedAnswer,
    GenerationRequestError,
    RecoverableGenerationError,
    grounded_answer_prompt,
)

logger = logging.getLogger(__name__)


class GroqRateLimitError(ValueError):
    """Indicate that the configured local free-tier budget is exhausted."""


@dataclass(slots=True)
class _TokenReservation:
    timestamp: float
    tokens: int
    day: date


class GroqRateLimiter:
    """Throttle Groq requests below configured minute and daily budgets."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        tokens_per_minute: int,
        requests_per_day: int,
        tokens_per_day: int,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Configure conservative local request and token limits."""

        limits = (
            requests_per_minute,
            tokens_per_minute,
            requests_per_day,
            tokens_per_day,
        )
        if any(limit < 1 for limit in limits):
            raise ValueError("Groq rate limits must be positive")

        self._requests_per_minute = requests_per_minute
        self._tokens_per_minute = tokens_per_minute
        self._requests_per_day = requests_per_day
        self._tokens_per_day = tokens_per_day
        self._clock = clock
        self._sleeper = sleeper
        self._lock = asyncio.Lock()
        self._requests: deque[float] = deque()
        self._tokens: deque[_TokenReservation] = deque()
        self._day = datetime.now(UTC).date()
        self._daily_requests = 0
        self._daily_tokens = 0

    async def acquire(self, estimated_tokens: int) -> _TokenReservation:
        """Reserve capacity for one request, waiting for minute limits."""

        if estimated_tokens < 1:
            raise ValueError("estimated_tokens must be positive")
        if estimated_tokens > self._tokens_per_minute:
            raise GroqRateLimitError(
                "One Groq request exceeds the configured per-minute token budget"
            )

        async with self._lock:
            self._reset_daily_budget()
            if self._daily_requests >= self._requests_per_day:
                raise GroqRateLimitError("Groq daily request budget exhausted")
            if self._daily_tokens + estimated_tokens > self._tokens_per_day:
                raise GroqRateLimitError("Groq daily token budget exhausted")

            while True:
                now = self._clock()
                self._prune_minute(now)
                wait_seconds = self._wait_seconds(now, estimated_tokens)
                if wait_seconds <= 0:
                    break
                logger.info(
                    "Waiting %.1fs for configured Groq free-tier capacity",
                    wait_seconds,
                )
                await self._sleeper(wait_seconds)

            now = self._clock()
            reservation = _TokenReservation(
                timestamp=now,
                tokens=estimated_tokens,
                day=self._day,
            )
            self._requests.append(now)
            self._tokens.append(reservation)
            self._daily_requests += 1
            self._daily_tokens += estimated_tokens
            return reservation

    async def reconcile(
        self,
        reservation: _TokenReservation,
        actual_tokens: int,
    ) -> None:
        """Replace an estimated reservation with reported token usage."""

        if actual_tokens < 0:
            return
        async with self._lock:
            difference = actual_tokens - reservation.tokens
            reservation.tokens = actual_tokens
            if reservation.day == self._day:
                self._daily_tokens = max(0, self._daily_tokens + difference)

    def _reset_daily_budget(self) -> None:
        """Reset process-local daily counters at UTC midnight."""

        today = datetime.now(UTC).date()
        if today != self._day:
            self._day = today
            self._daily_requests = 0
            self._daily_tokens = 0

    def _prune_minute(self, now: float) -> None:
        """Drop request and token reservations older than one minute."""

        cutoff = now - 60
        while self._requests and self._requests[0] <= cutoff:
            self._requests.popleft()
        while self._tokens and self._tokens[0].timestamp <= cutoff:
            self._tokens.popleft()

    def _wait_seconds(self, now: float, estimated_tokens: int) -> float:
        """Calculate when both rolling minute budgets have capacity."""

        waits: list[float] = []
        if len(self._requests) >= self._requests_per_minute:
            waits.append(self._requests[0] + 60 - now)

        excess = sum(item.tokens for item in self._tokens) + estimated_tokens
        if excess > self._tokens_per_minute:
            for reservation in self._tokens:
                excess -= reservation.tokens
                if excess <= self._tokens_per_minute:
                    waits.append(reservation.timestamp + 60 - now)
                    break
        return max(waits, default=0.0)


class GroqGenerationProvider:
    """Generate grounded and schema-constrained responses through Groq."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float,
        rate_limiter: GroqRateLimiter,
        base_url: str = "https://api.groq.com/openai/v1",
        max_retries: int = 2,
        reasoning_effort: str = "low",
        client: httpx.AsyncClient | None = None,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Configure the Groq endpoint, free-tier limiter, and HTTP client."""

        if not api_key.strip():
            raise ValueError("Groq API key is required")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")

        self._model = model
        self._rate_limiter = rate_limiter
        self._max_retries = max_retries
        self._reasoning_effort = reasoning_effort
        self._sleeper = sleeper
        self._authorization = f"Bearer {api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self._owns_client = client is None

    async def generate(
        self,
        messages: list[ChatMessage],
        context: str,
    ) -> GeneratedAnswer:
        """Generate a grounded answer with source identifiers."""

        system_message = grounded_answer_prompt(context)
        result = await self.generate_structured(
            [ChatMessage(role="system", content=system_message), *messages],
            {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "source_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["answer", "source_ids"],
                "additionalProperties": False,
            },
        )
        answer = result.get("answer")
        source_ids = result.get("source_ids")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Groq returned an invalid answer")
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("Groq returned invalid source IDs")
        return GeneratedAnswer(text=answer.strip(), source_ids=tuple(source_ids))

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate strict JSON matching the supplied schema."""

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        payload = {
            "model": self._model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "strict": True,
                    "schema": schema,
                },
            },
            "temperature": 1e-8,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": self._reasoning_effort,
        }
        estimated_tokens = _estimate_tokens(messages, schema, max_tokens)
        response = await self._post(payload, estimated_tokens=estimated_tokens)
        body = response.json()
        content = _response_content(body)
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError("Groq returned invalid structured JSON") from error
        if not isinstance(result, dict):
            raise ValueError("Groq returned invalid structured content")
        return result

    async def _post(
        self,
        payload: dict[str, Any],
        *,
        estimated_tokens: int,
    ) -> httpx.Response:
        """Send one completion request with local limiting and 429 backoff."""

        for attempt in range(self._max_retries + 1):
            reservation = await self._rate_limiter.acquire(estimated_tokens)
            response = await self._client.post(
                "/chat/completions",
                headers={"Authorization": self._authorization},
                json=payload,
            )
            if response.status_code != 429:
                if response.status_code == 400:
                    _raise_bad_request(response)
                response.raise_for_status()
                if (reported_tokens := _reported_tokens(response)) is not None:
                    await self._rate_limiter.reconcile(
                        reservation,
                        reported_tokens,
                    )
                return response

            await self._rate_limiter.reconcile(reservation, 0)
            if attempt == self._max_retries:
                response.raise_for_status()
            wait_seconds = _retry_after_seconds(response)
            logger.warning(
                "Groq rate limit reached; retrying in %.1fs (%d/%d)",
                wait_seconds,
                attempt + 1,
                self._max_retries,
            )
            await self._sleeper(wait_seconds)
        raise RuntimeError("Groq request retry loop exited unexpectedly")

    async def close(self) -> None:
        """Close the owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()


def _estimate_tokens(
    messages: list[ChatMessage],
    schema: dict[str, Any],
    max_tokens: int,
) -> int:
    """Conservatively estimate input plus maximum output tokens."""

    characters = sum(len(message.content) for message in messages)
    characters += len(json.dumps(schema, separators=(",", ":")))
    return max_tokens + max(1, math.ceil(characters / 4))


def _reported_tokens(response: httpx.Response) -> int | None:
    """Read total token usage from a successful Groq response."""

    try:
        total_tokens = response.json().get("usage", {}).get("total_tokens")
    except json.JSONDecodeError:
        return None
    return total_tokens if isinstance(total_tokens, int) and total_tokens >= 0 else None


def _response_content(payload: Any) -> str:
    """Extract completion text from an OpenAI-compatible response."""

    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Groq returned no structured content") from error
    if not isinstance(content, str):
        raise ValueError("Groq returned no structured content")
    return content


def _retry_after_seconds(response: httpx.Response) -> float:
    """Return Groq's retry delay with a conservative fallback."""

    value = response.headers.get("retry-after", "60").strip().removesuffix("s")
    try:
        return max(0.0, float(value))
    except ValueError:
        return 60.0


def _raise_bad_request(response: httpx.Response) -> None:
    """Raise a sanitized, classified Groq bad-request error."""

    try:
        payload = response.json()
    except json.JSONDecodeError as error:
        raise GenerationRequestError("Groq rejected the request (HTTP 400)") from error

    error_payload = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error_payload, dict):
        raise GenerationRequestError("Groq rejected the request (HTTP 400)")

    message = _safe_error_value(error_payload.get("message"), "unknown error")
    error_type = _safe_error_value(error_payload.get("type"), "unknown")
    code = _safe_error_value(error_payload.get("code"), "unknown")
    detail = f"type={error_type}, code={code}, message={message}"
    if _is_recoverable_structured_error(error_payload, message):
        raise RecoverableGenerationError(
            f"Groq could not generate structured output ({detail})"
        )
    raise GenerationRequestError(f"Groq rejected the request ({detail})")


def _is_recoverable_structured_error(
    error_payload: dict[str, Any],
    message: str,
) -> bool:
    """Recognize content-specific structured generation failures."""

    code = str(error_payload.get("code", "")).casefold()
    normalized_message = message.casefold()
    return (
        "failed_generation" in error_payload
        or code in {"json_validate_failed", "structured_output_generation_failed"}
        or "failed to generate" in normalized_message
        or "does not match the expected schema" in normalized_message
        or "valid json" in normalized_message
    )


def _safe_error_value(value: Any, default: str) -> str:
    """Return one compact error field without dumping response payloads."""

    if not isinstance(value, str) or not value.strip():
        return default
    return " ".join(value.split())[:300]
