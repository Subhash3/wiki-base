import json
from typing import Any

import httpx

from llm_providers.generation.base import (
    ChatMessage,
    GeneratedAnswer,
    grounded_answer_prompt,
)


class LlamaCppGenerationProvider:
    """Generate schema-constrained responses with a llama.cpp HTTP server."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def generate(
        self,
        messages: list[ChatMessage],
        context: str,
    ) -> GeneratedAnswer:
        result = await self.generate_structured(
            [
                ChatMessage(role="system", content=grounded_answer_prompt(context)),
                *messages,
            ],
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
            raise ValueError("llama.cpp returned an invalid answer")
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("llama.cpp returned invalid source IDs")
        return GeneratedAnswer(text=answer.strip(), source_ids=tuple(source_ids))

    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate JSON matching the supplied schema."""

        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        response = await self._client.post(
            f"{self._base_url}/v1/chat/completions",
            json={
                "model": self._model,
                "messages": [
                    {"role": message.role, "content": message.content} for message in messages
                ],
                "temperature": 0,
                "max_tokens": max_tokens,
                "stream": False,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "response",
                        "strict": True,
                        "schema": schema,
                    },
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("llama.cpp returned no structured content") from error
        if not isinstance(content, str):
            raise ValueError("llama.cpp returned no structured content")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            preview = " ".join(content[:160].split())
            raise ValueError(
                f"llama.cpp returned invalid JSON (characters={len(content)}, preview={preview!r})"
            ) from error
        if not isinstance(result, dict):
            raise ValueError(
                f"llama.cpp returned invalid structured content ({type(result).__name__})"
            )
        return result

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
