from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    source_ids: tuple[str, ...]


class GenerationRequestError(ValueError):
    """Indicate that a generation provider rejected a request."""


class RecoverableGenerationError(GenerationRequestError):
    """Indicate that one input could not produce valid structured output."""


class GenerationProvider(Protocol):
    async def generate(
        self,
        messages: list[ChatMessage],
        context: str,
    ) -> GeneratedAnswer: ...


class StructuredGenerationProvider(Protocol):
    async def generate_structured(
        self,
        messages: list[ChatMessage],
        schema: dict[str, Any],
        *,
        max_tokens: int = 4096,
    ) -> dict[str, Any]: ...
