from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RerankDocument:
    id: str
    text: str


class RerankingProvider(Protocol):
    async def rerank(self, query: str, documents: list[RerankDocument]) -> list[str]: ...
