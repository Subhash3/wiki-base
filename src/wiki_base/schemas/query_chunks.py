from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

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


class FactProvenanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: UUID
    chunk_id: UUID


class GraphFactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    subject: str
    relation: str
    object: str
    provenance: list[FactProvenanceResponse]
    depth: int
    seeds: list[str]


class RankedFactResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fact: GraphFactResponse
    score: float


class QueryChunksResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    wiki_base_id: UUID
    question: str
    chunks: list[RetrievedChunkResponse]
    facts: list[RankedFactResponse] = Field(default_factory=list)
    mode: RetrievalMode
    retrieval_strategy: RetrievalStrategy
