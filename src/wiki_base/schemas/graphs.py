from uuid import UUID

from pydantic import BaseModel


class GraphNodeDocumentResponse(BaseModel):
    """A source document that mentions a graph node."""

    id: UUID
    name: str
    chunk_count: int


class GraphNodeInfoResponse(BaseModel):
    """Metadata and connectivity for one graph node."""

    id: UUID
    name: str
    link_count: int
    fact_count: int
    synonym_count: int
    document_count: int
    documents: list[GraphNodeDocumentResponse]


class GraphNodeFactResponse(BaseModel):
    """One fact reachable from a graph node."""

    subject_id: UUID
    subject: str
    relation: str
    object_id: UUID
    object: str
    depth: int
    document_names: list[str]
    evidence_count: int


class GraphNodeFactsResponse(BaseModel):
    """Facts reachable from one graph node."""

    id: UUID
    name: str
    facts: list[GraphNodeFactResponse]


class GraphNodeSummaryResponse(BaseModel):
    """LLM summary of facts reachable from one graph node."""

    id: UUID
    name: str
    max_depth: int
    fact_count: int
    summary: str
