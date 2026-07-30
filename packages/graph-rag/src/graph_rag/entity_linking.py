from typing import Protocol

from graph_rag.graph import KnowledgeGraph
from graph_rag.normalization import normalize_text


class EntityLinker(Protocol):
    """Link query entities to knowledge graph nodes."""

    def link(self, entities: list[str], graph: KnowledgeGraph) -> list[str]:
        """Return graph nodes that match the query entities."""

        ...


class ExactEntityLinker:
    """Link entities through normalized exact matching."""

    def link(self, entities: list[str], graph: KnowledgeGraph) -> list[str]:
        """Return distinct matching nodes in query order."""

        linked: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            node = normalize_text(entity)
            if node in graph.nodes and node not in seen:
                linked.append(node)
                seen.add(node)
        return linked
