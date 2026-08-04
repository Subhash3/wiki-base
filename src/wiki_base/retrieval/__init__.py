"""Retrieval and answer-generation pipeline."""

from enum import StrEnum


class RetrievalMode(StrEnum):
    """Select the chunk retrieval strategy."""

    LITE = "lite"
    PRO = "pro"
    FACTS = "facts"


class RetrievalStrategy(StrEnum):
    """Describe the strategy that supplied the retrieved chunks."""

    VECTOR = "vector"
    GRAPH = "graph"
    FACT_GRAPH = "fact_graph"
    VECTOR_FALLBACK = "vector_fallback"
