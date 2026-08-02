from uuid import UUID

from graph_rag.entity_linking import EmbeddingEntityLinker, ExactEntityLinker
from graph_rag.graph import KnowledgeGraph
from graph_rag.models import Triple, TripleProvenance
from graph_rag.query_extraction import QueryConcepts

DOCUMENT_ID = UUID("10000000-0000-0000-0000-000000000001")
CHUNK_ID = UUID("00000000-0000-0000-0000-000000000001")


def concepts(
    entities: list[str] | None = None,
    relationships: list[str] | None = None,
) -> QueryConcepts:
    """Build query concepts for linker tests."""

    return QueryConcepts(
        entities=entities or [],
        relationships=relationships or [],
    )


def make_graph() -> KnowledgeGraph:
    """Create a small normalized graph."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="alice", relation="works at", object="acme"),
        provenance=TripleProvenance(
            document_id=DOCUMENT_ID,
            chunk_id=CHUNK_ID,
        ),
    )
    return graph


class StubEmbeddings:
    """Return configured vectors and capture embedding calls."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        """Store vectors by input text."""

        self.vectors = vectors
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed and capture a document batch."""

        self.document_batches.append(texts)
        return [self.vectors[text] for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        """Embed and capture a query entity."""

        self.queries.append(text)
        return self.vectors[text]


async def test_links_normalized_exact_entities() -> None:
    """Formatting differences are normalized before matching."""

    nodes = await ExactEntityLinker().link(
        concepts([" Alice. ", "ACME"]),
        make_graph(),
    )

    assert nodes == ["alice", "acme"]


async def test_ignores_unmatched_and_duplicate_entities() -> None:
    """Unknown entities are skipped and graph nodes remain distinct."""

    nodes = await ExactEntityLinker().link(
        concepts(["Unknown", "Alice", "alice", "Another"]),
        make_graph(),
    )

    assert nodes == ["alice"]


async def test_empty_entities_return_no_nodes() -> None:
    """An empty entity list produces no graph seeds."""

    assert await ExactEntityLinker().link(concepts(), make_graph()) == []


async def test_exact_relationship_links_its_edge_endpoints() -> None:
    """An exact relationship match contributes both connected nodes."""

    nodes = await ExactEntityLinker().link(
        concepts(relationships=[" Works At. "]),
        make_graph(),
    )

    assert nodes == ["alice", "acme"]


async def test_embedding_linker_uses_exact_matches_without_embeddings() -> None:
    """Exact matches avoid unnecessary embedding calls."""

    embeddings = StubEmbeddings({})
    linker = EmbeddingEntityLinker(embeddings=embeddings)

    nodes = await linker.link(concepts(["Alice"]), make_graph())

    assert nodes == ["alice"]
    assert embeddings.document_batches == []
    assert embeddings.queries == []


async def test_embedding_linker_matches_unseen_entity_by_similarity() -> None:
    """An unmatched entity links to its closest graph node."""

    embeddings = StubEmbeddings(
        {
            "alice": [0.0, 1.0],
            "acme": [1.0, 0.0],
            "the company": [0.9, 0.1],
        }
    )
    linker = EmbeddingEntityLinker(
        embeddings=embeddings,
        similarity_threshold=0.8,
    )

    nodes = await linker.link(concepts(["the company"]), make_graph())

    assert nodes == ["acme"]
    assert embeddings.queries == ["the company"]


async def test_embedding_linker_rejects_candidates_below_threshold() -> None:
    """Weak semantic candidates do not become graph seeds."""

    embeddings = StubEmbeddings(
        {
            "alice": [0.0, 1.0],
            "acme": [0.0, 1.0],
            "unknown": [1.0, 0.0],
        }
    )
    linker = EmbeddingEntityLinker(
        embeddings=embeddings,
        similarity_threshold=0.8,
    )

    assert await linker.link(concepts(["unknown"]), make_graph()) == []


async def test_embedding_linker_matches_relationship_by_similarity() -> None:
    """A semantic relationship match contributes its edge endpoints."""

    embeddings = StubEmbeddings(
        {
            "alice works at acme": [1.0, 0.0],
            "employment": [0.9, 0.1],
        }
    )
    linker = EmbeddingEntityLinker(
        embeddings=embeddings,
        similarity_threshold=0.8,
    )

    nodes = await linker.link(
        concepts(relationships=["employment"]),
        make_graph(),
    )

    assert nodes == ["alice", "acme"]
    assert embeddings.document_batches == [["alice works at acme"]]
    assert embeddings.queries == ["employment"]


async def test_relationship_similarity_uses_edge_context() -> None:
    """Context distinguishes a relevant fact from a generic relationship label."""

    graph = KnowledgeGraph()
    graph.add_triple(
        Triple(subject="honda", relation="offers", object="abs system"),
        provenance=TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID),
    )
    graph.add_triple(
        Triple(subject="honda", relation="offers", object="long seat"),
        provenance=TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID),
    )
    graph.add_triple(
        Triple(
            subject="ex-showroom delhi",
            relation="loan amount for",
            object="total interest",
        ),
        provenance=TripleProvenance(document_id=DOCUMENT_ID, chunk_id=CHUNK_ID),
    )
    embeddings = StubEmbeddings(
        {
            "ex-showroom delhi loan amount for total interest": [0.5, 0.8660254],
            "loan offers": [1.0, 0.0],
        }
    )
    linker = EmbeddingEntityLinker(
        embeddings=embeddings,
        similarity_threshold=0.8,
        relationship_similarity_threshold=0.6,
    )

    nodes = await linker.link(
        concepts(relationships=["loan offers"]),
        graph,
    )

    assert nodes == ["ex-showroom delhi", "total interest"]
    assert embeddings.document_batches == [
        ["ex-showroom delhi loan amount for total interest"]
    ]


async def test_embedding_linker_caches_graph_node_embeddings() -> None:
    """Repeated links reuse cached graph-node vectors."""

    embeddings = StubEmbeddings(
        {
            "alice": [0.0, 1.0],
            "acme": [1.0, 0.0],
            "company": [1.0, 0.0],
            "person": [0.0, 1.0],
        }
    )
    linker = EmbeddingEntityLinker(embeddings=embeddings)

    await linker.link(concepts(["company"]), make_graph())
    await linker.link(concepts(["person"]), make_graph())

    assert embeddings.document_batches == [["acme", "alice"]]
