from dataclasses import dataclass
from enum import StrEnum
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
class RankedChunk:
    """A document chunk ranked by graph relevance."""

    document_id: UUID
    chunk_id: UUID
    score: float


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """A normalized graph edge and the chunks that support it."""

    subject: str
    relation: str
    object: str
    provenance: frozenset[TripleProvenance]


@dataclass(frozen=True, slots=True)
class SynonymEdge:
    """A semantic connection between two entity nodes."""

    first: str
    second: str
    similarity: float


class GraphConceptType(StrEnum):
    """The searchable type of a graph concept."""

    ENTITY = "entity"
    RELATIONSHIP = "relationship"


@dataclass(frozen=True, slots=True)
class GraphConcept:
    """An entity or contextual relationship prepared for embedding."""

    type: GraphConceptType
    key: str
    text: str
    subject: str | None = None
    relationship: str | None = None
    object: str | None = None


@dataclass(frozen=True, slots=True)
class EntityConceptMatch:
    """An entity returned by semantic concept search."""

    entity: str
    similarity: float


@dataclass(frozen=True, slots=True)
class RelationshipConceptMatch:
    """A relationship fact returned by semantic concept search."""

    text: str
    subject: str
    relationship: str
    object: str
    similarity: float
