import ast
import json
from typing import Any

import httpx

from llm_providers.generation.base import (
    ChatMessage,
    GeneratedAnswer,
    grounded_answer_prompt,
)


class OllamaGenerationProvider:
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
        system_message = grounded_answer_prompt(context)
        result = await self.generate_structured(
            [
                ChatMessage(role="system", content=system_message),
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
            raise ValueError("Ollama returned an invalid answer")
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            raise ValueError("Ollama returned invalid source IDs")
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
        schema_instruction = (
            "Return only valid JSON matching this JSON schema. Do not return "
            "Markdown, commentary, or a prose summary.\n"
            f"{json.dumps(schema, separators=(',', ':'))}"
        )
        request_messages = list(messages)
        if request_messages and request_messages[0].role == "system":
            system_message = request_messages[0]
            request_messages[0] = ChatMessage(
                role="system",
                content=f"{system_message.content}\n\n{schema_instruction}",
            )
        else:
            request_messages.insert(
                0,
                ChatMessage(role="system", content=schema_instruction),
            )
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "think": False,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request_messages
                ],
                "format": schema,
                "options": {"temperature": 0, "num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama returned no structured content")

        try:
            result = self._decode_json(content)
        except json.JSONDecodeError as error:
            done_reason = payload.get("done_reason", "unknown")
            preview = " ".join(content[:160].split())
            raise ValueError(
                f"Ollama returned invalid JSON (reason={done_reason}, "
                f"characters={len(content)}, preview={preview!r})"
            ) from error
        if isinstance(result, list):
            result = self._wrap_single_array_result(result, schema)
        if not isinstance(result, dict):
            raise ValueError(
                f"Ollama returned invalid structured content ({type(result).__name__})"
            )
        return result

    @staticmethod
    def _decode_json(content: str) -> Any:
        """Decode plain JSON or JSON wrapped in model commentary."""

        try:
            return json.loads(content)
        except json.JSONDecodeError as original_error:
            decoder = json.JSONDecoder()
            for index, character in enumerate(content):
                if character not in "[{":
                    continue
                try:
                    result, _end = decoder.raw_decode(content[index:])
                    return result
                except json.JSONDecodeError:
                    closing = "}" if character == "{" else "]"
                    end = content.rfind(closing)
                    if end <= index:
                        continue
                    try:
                        result = ast.literal_eval(content[index : end + 1])
                    except (SyntaxError, ValueError):
                        continue
                    if isinstance(result, (dict, list)):
                        return result
            raise original_error

    @staticmethod
    def _wrap_single_array_result(
        result: list[Any],
        schema: dict[str, Any],
    ) -> dict[str, Any] | list[Any]:
        """Wrap a bare array when the schema has one array property."""

        properties = schema.get("properties")
        if not isinstance(properties, dict) or len(properties) != 1:
            return result
        name, property_schema = next(iter(properties.items()))
        if (
            not isinstance(name, str)
            or not isinstance(property_schema, dict)
            or property_schema.get("type") != "array"
        ):
            return result
        return {name: result}

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
