import json

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import GraphConcept, GraphConceptType


def graph_concepts(graph: KnowledgeGraph) -> list[GraphConcept]:
    """Return deterministic entity and relationship concepts from a graph."""

    entities = [
        GraphConcept(
            type=GraphConceptType.ENTITY,
            key=entity,
            text=entity,
        )
        for entity in sorted(graph.nodes)
    ]
    relationships = [
        GraphConcept(
            type=GraphConceptType.RELATIONSHIP,
            key=json.dumps(
                [edge.subject, edge.relation, edge.object],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            text=edge_text(edge.subject, edge.relation, edge.object),
            subject=edge.subject,
            relationship=edge.relation,
            object=edge.object,
        )
        for edge in sorted(
            graph.edges(),
            key=lambda value: (value.subject, value.relation, value.object),
        )
    ]
    return [*entities, *relationships]


def edge_text(subject: str, relationship: str, object_: str) -> str:
    """Return searchable relationship text with its entity context."""

    return f"{subject} {relationship} {object_}"
