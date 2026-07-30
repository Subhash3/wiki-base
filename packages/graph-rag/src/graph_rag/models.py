from dataclasses import dataclass
from uuid import UUID

from document_processing.models import DocumentChunk


@dataclass(frozen=True, slots=True)
class Triple:
    """A schemaless fact extracted from a document chunk."""

    subject: str
    relation: str
    object: str


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """A document chunk paired with its owning document."""

    document_id: UUID
    chunk: DocumentChunk


@dataclass(frozen=True, slots=True)
class TripleProvenance:
    """The document and chunk from which a triple was extracted."""

    document_id: UUID
    chunk_id: UUID


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A normalized graph edge and the chunks that support it."""

    subject: str
    relation: str
    object: str
    provenance: frozenset[TripleProvenance]
