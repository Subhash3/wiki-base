"""Retrieval and answer-generation pipeline."""

from enum import StrEnum


class RetrievalMode(StrEnum):
    """Select the chunk retrieval strategy."""

    LITE = "lite"
    PRO = "pro"
