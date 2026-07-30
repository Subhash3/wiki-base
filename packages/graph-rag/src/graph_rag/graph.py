from collections.abc import Iterator

from graph_rag.models import GraphEdge, Triple, TripleProvenance

EdgeKey = tuple[str, str, str]


class KnowledgeGraph:
    """An in-memory entity graph with chunk-level provenance."""

    def __init__(self) -> None:
        self._node_provenance: dict[str, set[TripleProvenance]] = {}
        self._edge_provenance: dict[EdgeKey, set[TripleProvenance]] = {}
        self._provenance_triples: dict[TripleProvenance, set[Triple]] = {}

    def add_triple(self, triple: Triple, *, provenance: TripleProvenance) -> None:
        self._node_provenance.setdefault(triple.subject, set()).add(provenance)
        self._node_provenance.setdefault(triple.object, set()).add(provenance)

        edge_key = (triple.subject, triple.relation, triple.object)
        self._edge_provenance.setdefault(edge_key, set()).add(provenance)
        self._provenance_triples.setdefault(provenance, set()).add(triple)

    @staticmethod
    def merge(first: "KnowledgeGraph", second: "KnowledgeGraph") -> "KnowledgeGraph":
        """Return a new graph containing the facts and provenance of both inputs."""

        merged = KnowledgeGraph()
        for graph in (first, second):
            for edge in graph.edges():
                triple = Triple(
                    subject=edge.subject,
                    relation=edge.relation,
                    object=edge.object,
                )
                for provenance in edge.provenance:
                    merged.add_triple(triple, provenance=provenance)
        return merged

    @property
    def nodes(self) -> frozenset[str]:
        return frozenset(self._node_provenance)

    def edges(self) -> Iterator[GraphEdge]:
        for (subject, relation, object_), provenance in self._edge_provenance.items():
            yield GraphEdge(
                subject=subject,
                relation=relation,
                object=object_,
                provenance=frozenset(provenance),
            )

    def provenance_for_node(self, entity: str) -> frozenset[TripleProvenance]:
        return frozenset(self._node_provenance.get(entity, ()))

    def triples_for_provenance(self, provenance: TripleProvenance) -> frozenset[Triple]:
        return frozenset(self._provenance_triples.get(provenance, ()))
