from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from wiki_base.retrieval import RetrievalMode


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class QueryRequest(BaseModel):
    wiki_base_id: UUID
    question: str = Field(min_length=1, max_length=4000)
    history: list[ConversationMessage] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=5, ge=1, le=20)
    mode: RetrievalMode = RetrievalMode.LITE


class CitationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chunk_id: UUID
    document_id: UUID
    document_name: str
    excerpt: str
    score: float
    page: int | None = None
    slide: int | None = None
    section: str | None = None
    heading: str | None = None


class QueryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    wiki_base_id: UUID
    question: str
    answer: str
    citations: list[CitationResponse]
    mode: RetrievalMode
