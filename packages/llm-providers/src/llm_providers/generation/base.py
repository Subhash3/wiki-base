from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    text: str
    source_ids: tuple[str, ...]


class GenerationProvider(Protocol):
    async def generate(
        self,
        messages: list[ChatMessage],
        context: str,
    ) -> GeneratedAnswer: ...
