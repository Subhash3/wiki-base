import json
from collections.abc import Iterator
from typing import Any
from uuid import UUID

from graph_rag.models import GraphEdge, SynonymEdge, Triple, TripleProvenance

EdgeKey = tuple[str, str, str]


class KnowledgeGraph:
    """An in-memory entity graph with chunk-level provenance."""

    def __init__(self) -> None:
        """Create an empty knowledge graph."""

        self._entity_provenance: dict[str, set[TripleProvenance]] = {}
        self._triple_node_provenance: dict[str, set[TripleProvenance]] = {}
        self._edge_provenance: dict[EdgeKey, set[TripleProvenance]] = {}
        self._provenance_triples: dict[TripleProvenance, set[Triple]] = {}
        self._synonyms: dict[tuple[str, str], float] = {}

    def add_entity(self, entity: str, *, provenance: TripleProvenance) -> None:
        """Associate an extracted entity with its source chunk."""

        self._entity_provenance.setdefault(entity, set()).add(provenance)

    def add_triple(self, triple: Triple, *, provenance: TripleProvenance) -> None:
        """Add a factual edge and its source provenance."""

        self._triple_node_provenance.setdefault(triple.subject, set()).add(provenance)
        self._triple_node_provenance.setdefault(triple.object, set()).add(provenance)

        edge_key = (triple.subject, triple.relation, triple.object)
        self._edge_provenance.setdefault(edge_key, set()).add(provenance)
        self._provenance_triples.setdefault(provenance, set()).add(triple)

    def add_synonym(self, first: str, second: str, *, similarity: float) -> bool:
        """Connect two existing entity nodes as semantic synonyms."""

        if not -1 <= similarity <= 1:
            raise ValueError("synonym similarity must be between -1 and 1")
        if first == second or first not in self.nodes or second not in self.nodes:
            return False
        key = (first, second) if first < second else (second, first)
        self._synonyms[key] = max(similarity, self._synonyms.get(key, -1.0))
        return True

    @staticmethod
    def merge(first: "KnowledgeGraph", second: "KnowledgeGraph") -> "KnowledgeGraph":
        """Return a new graph containing the facts and provenance of both inputs."""

        merged = KnowledgeGraph()
        for graph in (first, second):
            for entity in graph.nodes:
                for provenance in graph.entity_provenance_for_node(entity):
                    merged.add_entity(entity, provenance=provenance)
            for edge in graph.edges():
                triple = Triple(
                    subject=edge.subject,
                    relation=edge.relation,
                    object=edge.object,
                )
                for provenance in edge.provenance:
                    merged.add_triple(triple, provenance=provenance)
        for graph in (first, second):
            for synonym in graph.synonyms():
                merged.add_synonym(
                    synonym.first,
                    synonym.second,
                    similarity=synonym.similarity,
                )
        return merged

    def to_json(self) -> str:
        """Serialize the canonical graph and provenance as JSON."""

        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical graph as a JSON-compatible mapping."""

        entities = [
            {
                "entity": entity,
                "provenance": self._serialize_provenance(provenance),
            }
            for entity, provenance in sorted(self._entity_provenance.items())
        ]
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
                    "provenance": self._serialize_provenance(edge.provenance),
                }
            )
        synonyms = [
            {
                "first": synonym.first,
                "second": synonym.second,
                "similarity": synonym.similarity,
            }
            for synonym in sorted(
                self.synonyms(),
                key=lambda item: (item.first, item.second),
            )
        ]
        return {
            "version": 2,
            "entities": entities,
            "edges": edges,
            "synonyms": synonyms,
        }

    @classmethod
    def from_json(cls, content: str) -> "KnowledgeGraph":
        """Load a canonical knowledge graph from JSON."""

        return cls.from_dict(json.loads(content))

    @classmethod
    def from_dict(cls, payload: Any) -> "KnowledgeGraph":
        """Load a canonical graph from a JSON-compatible mapping."""

        if not isinstance(payload, dict) or payload.get("version") not in {1, 2}:
            raise ValueError("Unsupported knowledge graph JSON")
        edges = payload.get("edges")
        if not isinstance(edges, list):
            raise ValueError("Knowledge graph JSON has invalid edges")

        graph = cls()
        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            raise ValueError("Knowledge graph JSON has invalid entities")
        for entity in entities:
            name, provenance = cls._parse_entity(entity)
            for source in provenance:
                graph.add_entity(name, provenance=source)
        for edge in edges:
            triple, provenance = cls._parse_edge(edge)
            for source in provenance:
                graph.add_triple(triple, provenance=source)
        synonyms = payload.get("synonyms", [])
        if not isinstance(synonyms, list):
            raise ValueError("Knowledge graph JSON has invalid synonyms")
        for synonym in synonyms:
            first, second, similarity = cls._parse_synonym(synonym)
            graph.add_synonym(first, second, similarity=similarity)
        return graph

    @staticmethod
    def _parse_entity(value: Any) -> tuple[str, list[TripleProvenance]]:
        """Parse one canonical JSON entity association."""

        if not isinstance(value, dict) or not isinstance(value.get("entity"), str):
            raise ValueError("Knowledge graph JSON has an invalid entity")
        return value["entity"], KnowledgeGraph._parse_provenance(
            value.get("provenance"),
        )

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

        return (
            Triple(subject=subject, relation=relation, object=object_),
            KnowledgeGraph._parse_provenance(sources),
        )

    @staticmethod
    def _parse_provenance(value: Any) -> list[TripleProvenance]:
        """Parse canonical chunk provenance records."""

        if not isinstance(value, list):
            raise ValueError("Knowledge graph JSON has invalid provenance")
        provenance: list[TripleProvenance] = []
        for source in value:
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
        return provenance

    @staticmethod
    def _parse_synonym(value: Any) -> tuple[str, str, float]:
        """Parse one canonical semantic synonym edge."""

        if not isinstance(value, dict):
            raise ValueError("Knowledge graph JSON has an invalid synonym")
        first = value.get("first")
        second = value.get("second")
        similarity = value.get("similarity")
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or not isinstance(similarity, int | float)
        ):
            raise ValueError("Knowledge graph JSON has an invalid synonym")
        return first, second, float(similarity)

    @staticmethod
    def _serialize_provenance(
        provenance: set[TripleProvenance] | frozenset[TripleProvenance],
    ) -> list[dict[str, str]]:
        """Return deterministic JSON-compatible chunk provenance."""

        return [
            {
                "document_id": str(source.document_id),
                "chunk_id": str(source.chunk_id),
            }
            for source in sorted(
                provenance,
                key=lambda source: (source.document_id.int, source.chunk_id.int),
            )
        ]

    @property
    def nodes(self) -> frozenset[str]:
        """Return all entity nodes."""

        return frozenset(self._entity_provenance | self._triple_node_provenance)

    def edges(self) -> Iterator[GraphEdge]:
        """Iterate over factual edges."""

        for (subject, relation, object_), provenance in self._edge_provenance.items():
            yield GraphEdge(
                subject=subject,
                relation=relation,
                object=object_,
                provenance=frozenset(provenance),
            )

    def synonyms(self) -> Iterator[SynonymEdge]:
        """Iterate over semantic synonym edges."""

        for (first, second), similarity in self._synonyms.items():
            yield SynonymEdge(first=first, second=second, similarity=similarity)

    def provenance_for_node(self, entity: str) -> frozenset[TripleProvenance]:
        """Return the sources that mention an entity."""

        return frozenset(
            self._entity_provenance.get(entity, set())
            | self._triple_node_provenance.get(entity, set())
        )

    def entity_provenance_for_node(
        self,
        entity: str,
    ) -> frozenset[TripleProvenance]:
        """Return chunks where passage extraction found the entity."""

        return frozenset(self._entity_provenance.get(entity, ()))

    def triple_provenance_for_node(
        self,
        entity: str,
    ) -> frozenset[TripleProvenance]:
        """Return chunks where the entity participates in a triple."""

        return frozenset(self._triple_node_provenance.get(entity, ()))

    def triples_for_provenance(self, provenance: TripleProvenance) -> frozenset[Triple]:
        """Return facts extracted from one document chunk."""

        return frozenset(self._provenance_triples.get(provenance, ()))
