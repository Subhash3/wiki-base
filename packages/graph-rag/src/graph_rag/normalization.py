import re
import unicodedata

from graph_rag.models import Triple

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Return a stable key while preserving meaningful internal punctuation."""

    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    return normalized.strip(" \t\n\r.,;:!?\"'`()[]{}").casefold()


def normalize_triple(triple: Triple) -> Triple | None:
    """Normalize a triple, discarding incomplete extraction results."""

    subject = normalize_text(triple.subject)
    relation = normalize_text(triple.relation)
    object_ = normalize_text(triple.object)
    if not subject or not relation or not object_:
        return None
    return Triple(subject=subject, relation=relation, object=object_)
