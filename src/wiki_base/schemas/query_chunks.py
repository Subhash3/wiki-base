from uuid import UUID

from pydantic import BaseModel, ConfigDict

from wiki_base.retrieval import RetrievalMode, RetrievalStrategy


class RetrievedChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    document_id: UUID
    document_name: str
    content: str
    score: float
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    heading: str | None = None


class QueryChunksResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    wiki_base_id: UUID
    question: str
    chunks: list[RetrievedChunkResponse]
    mode: RetrievalMode
    retrieval_strategy: RetrievalStrategy
