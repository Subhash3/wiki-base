from typing import Protocol
from uuid import UUID


class IngestionQueue(Protocol):
    async def enqueue_wiki_base(self, wiki_base_id: UUID) -> None: ...
