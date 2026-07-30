import json
from collections.abc import Iterator
from typing import Any
from uuid import UUID

from graph_rag.models import GraphEdge, Triple, TripleProvenance

EdgeKey = tuple[str, str, str]


class KnowledgeGraph:
    """An in-memory entity graph with chunk-level provenance."""

    def __init__(self) -> None:
        """Create an empty knowledge graph."""

        self._node_provenance: dict[str, set[TripleProvenance]] = {}
        self._edge_provenance: dict[EdgeKey, set[TripleProvenance]] = {}
        self._provenance_triples: dict[TripleProvenance, set[Triple]] = {}

    def add_triple(self, triple: Triple, *, provenance: TripleProvenance) -> None:
        """Add a factual edge and its source provenance."""

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

    def to_json(self) -> str:
        """Serialize the canonical graph and provenance as JSON."""

        edges = []
        for edge in sorted(
            self.edges(),
            key=lambda item: (item.subject, item.relation, item.object),
        ):
            edges.append(
                {
                    "subject": edge.subject,
                    "relation": edge.relation,
                    "object": edge.object,
                    "provenance": [
                        {
                            "document_id": str(source.document_id),
                            "chunk_id": str(source.chunk_id),
                        }
                        for source in sorted(
                            edge.provenance,
                            key=lambda source: (
                                source.document_id.int,
                                source.chunk_id.int,
                            ),
                        )
                    ],
                }
            )
        return json.dumps({"version": 1, "edges": edges}, indent=2)

    @classmethod
    def from_json(cls, content: str) -> "KnowledgeGraph":
        """Load a canonical knowledge graph from JSON."""

        payload = json.loads(content)
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unsupported knowledge graph JSON")
        edges = payload.get("edges")
        if not isinstance(edges, list):
            raise ValueError("Knowledge graph JSON has invalid edges")

        graph = cls()
        for edge in edges:
            triple, provenance = cls._parse_edge(edge)
            for source in provenance:
                graph.add_triple(triple, provenance=source)
        return graph

    @staticmethod
    def _parse_edge(value: Any) -> tuple[Triple, list[TripleProvenance]]:
        """Parse one canonical JSON edge."""

        if not isinstance(value, dict):
            raise ValueError("Knowledge graph JSON has an invalid edge")
        subject = value.get("subject")
        relation = value.get("relation")
        object_ = value.get("object")
        sources = value.get("provenance")
        if (
            not isinstance(subject, str)
            or not isinstance(relation, str)
            or not isinstance(object_, str)
            or not isinstance(sources, list)
        ):
            raise ValueError("Knowledge graph JSON has an invalid edge")

        provenance: list[TripleProvenance] = []
        for source in sources:
            if not isinstance(source, dict):
                raise ValueError("Knowledge graph JSON has invalid provenance")
            try:
                provenance.append(
                    TripleProvenance(
                        document_id=UUID(source["document_id"]),
                        chunk_id=UUID(source["chunk_id"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Knowledge graph JSON has invalid provenance") from error
        return Triple(subject=subject, relation=relation, object=object_), provenance

    @property
    def nodes(self) -> frozenset[str]:
        """Return all entity nodes."""

        return frozenset(self._node_provenance)

    def edges(self) -> Iterator[GraphEdge]:
        """Iterate over factual edges."""

        for (subject, relation, object_), provenance in self._edge_provenance.items():
            yield GraphEdge(
                subject=subject,
                relation=relation,
                object=object_,
                provenance=frozenset(provenance),
            )

    def provenance_for_node(self, entity: str) -> frozenset[TripleProvenance]:
        """Return the sources that mention an entity."""

        return frozenset(self._node_provenance.get(entity, ()))

    def triples_for_provenance(self, provenance: TripleProvenance) -> frozenset[Triple]:
        """Return facts extracted from one document chunk."""

        return frozenset(self._provenance_triples.get(provenance, ()))
