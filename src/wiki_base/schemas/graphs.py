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
    """One direct fact involving a graph node."""

    subject_id: UUID
    subject: str
    relation: str
    object_id: UUID
    object: str
    document_names: list[str]
    evidence_count: int


class GraphNodeFactsResponse(BaseModel):
    """Direct facts involving one graph node."""

    id: UUID
    name: str
    facts: list[GraphNodeFactResponse]
