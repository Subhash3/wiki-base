from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingModelInfo:
    model: str
    dimensions: int
    max_tokens: int


class EmbeddingProvider(Protocol):
    @property
    def model_info(self) -> EmbeddingModelInfo: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...
