from uuid import UUID

from graph_rag.concepts import graph_concepts
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import GraphConceptType, Triple, TripleProvenance

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def test_builds_entity_and_contextual_relationship_concepts() -> None:
    """Graph concepts include nodes and full edge context."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="honda unicorn", relation="offers", object="loan amount"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=CHUNK_ID,
        ),
    )

    concepts = graph_concepts(graph)

    assert [concept.type for concept in concepts] == [
        GraphConceptType.ENTITY,
        GraphConceptType.ENTITY,
        GraphConceptType.RELATIONSHIP,
    ]
    assert [concept.text for concept in concepts] == [
        "honda unicorn",
        "loan amount",
        "honda unicorn offers loan amount",
    ]
    assert concepts[-1].subject == "honda unicorn"
    assert concepts[-1].relationship == "offers"
    assert concepts[-1].object == "loan amount"


def test_includes_entities_without_relationships() -> None:
    """Passage-only concepts receive persisted embeddings too."""

    graph = KnowledgeGraph()
    graph.add_entity(
        "skoda slavia",
        provenance=TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID),
    )

    concepts = graph_concepts(graph)

    assert len(concepts) == 1
    assert concepts[0].type == GraphConceptType.ENTITY
    assert concepts[0].text == "skoda slavia"
