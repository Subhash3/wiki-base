from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DocumentSource:
    path: Path
    name: str
    media_type: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    name: str
    native_document: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    ordinal: int
    content: str
    embedding_content: str
    token_count: int
    page_number: int | None
    slide_number: int | None
    section: str | None
    heading: str | None
    caption: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]
