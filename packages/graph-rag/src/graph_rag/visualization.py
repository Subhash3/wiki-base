import json
from uuid import UUID

import networkx as nx
from pyvis.network import Network

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import TripleProvenance


class GraphVisualizer:
    """Project a knowledge graph into NetworkX and render it with PyVis."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph

    def build(self, *, document_id: UUID | None = None) -> nx.MultiDiGraph:
        network = nx.MultiDiGraph()
        for entity in sorted(self._graph.nodes):
            provenance = self._filter_provenance(
                self._graph.provenance_for_node(entity),
                document_id=document_id,
            )
            if provenance:
                network.add_node(
                    entity,
                    label=entity,
                    title=self._provenance_title(provenance),
                    provenance=self._serialize_provenance(provenance),
                    group="entity",
                )
                for source_document_id in sorted(
                    {source.document_id for source in provenance},
                    key=lambda value: value.int,
                ):
                    document_node = self._document_node_id(source_document_id)
                    network.add_node(
                        document_node,
                        label=f"Document {str(source_document_id)[:8]}",
                        title=f"Document: {source_document_id}",
                        document_id=str(source_document_id),
                        group="document",
                        shape="box",
                    )
                    network.add_edge(
                        document_node,
                        entity,
                        label="contains",
                        title="Visualization-only provenance link",
                        arrows="to",
                        dashes=True,
                        visualization_only=True,
                    )

        graph_edges = sorted(
            self._graph.edges(),
            key=lambda edge: (edge.subject, edge.relation, edge.object),
        )
        for edge in graph_edges:
            provenance = self._filter_provenance(
                edge.provenance,
                document_id=document_id,
            )
            if provenance:
                network.add_edge(
                    edge.subject,
                    edge.object,
                    label=edge.relation,
                    title=self._provenance_title(provenance),
                    provenance=self._serialize_provenance(provenance),
                    arrows="to",
                )
        return network

    @staticmethod
    def _document_node_id(document_id: UUID) -> str:
        return f"document:{document_id}"

    @staticmethod
    def _add_document_nodes(network: nx.MultiDiGraph) -> None:
        for entity, attributes in list(network.nodes(data=True)):
            if attributes.get("group") == "document":
                continue
            provenance = attributes.get("provenance", [])
            if not isinstance(provenance, list):
                continue
            for source in provenance:
                if not isinstance(source, dict):
                    continue
                document_id = source.get("document_id")
                if not isinstance(document_id, str):
                    continue
                document_node = f"document:{document_id}"
                network.add_node(
                    document_node,
                    label=f"Document {document_id[:8]}",
                    title=f"Document: {document_id}",
                    document_id=document_id,
                    group="document",
                    shape="box",
                )
                edge_data = network.get_edge_data(
                    document_node,
                    entity,
                    default={},
                )
                already_linked = any(
                    edge_attributes.get("visualization_only")
                    for edge_attributes in edge_data.values()
                )
                if not already_linked:
                    network.add_edge(
                        document_node,
                        entity,
                        label="contains",
                        title="Visualization-only provenance link",
                        arrows="to",
                        dashes=True,
                        visualization_only=True,
                    )

    @staticmethod
    def to_json(network: nx.MultiDiGraph) -> str:
        payload = nx.node_link_data(network, edges="edges")
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(content: str) -> nx.MultiDiGraph:
        payload = json.loads(content)
        network = nx.node_link_graph(
            payload,
            directed=True,
            multigraph=True,
            edges="edges",
        )
        if not isinstance(network, nx.MultiDiGraph):
            raise ValueError("JSON did not contain a directed multigraph")
        GraphVisualizer._add_document_nodes(network)
        return network

    @staticmethod
    def merge(networks: list[nx.MultiDiGraph]) -> nx.MultiDiGraph:
        """Merge visualization networks while preserving parallel edges."""

        merged = nx.MultiDiGraph()
        for network in networks:
            for node, attributes in network.nodes(data=True):
                existing = merged.nodes.get(node, {})
                combined = {**existing, **attributes}
                provenance = GraphVisualizer._merge_provenance(
                    existing.get("provenance"),
                    attributes.get("provenance"),
                )
                if provenance:
                    combined["provenance"] = provenance
                else:
                    combined.pop("provenance", None)
                merged.add_node(node, **combined)

            for source, target, attributes in network.edges(data=True):
                duplicate = any(
                    existing == attributes
                    for existing in merged.get_edge_data(source, target, default={}).values()
                )
                if not duplicate:
                    merged.add_edge(source, target, **attributes)
        return merged

    @staticmethod
    def to_html(network: nx.MultiDiGraph, *, title: str = "Graph RAG") -> str:
        visualization = Network(
            height="100vh",
            width="100%",
            directed=True,
            heading=title,
            bgcolor="#f8fafc",
            font_color="#0f172a",
            cdn_resources="in_line",
        )
        visualization.from_nx(network)
        visualization.set_options(
            """
            {
              "nodes": {
                "shape": "dot",
                "color": {
                  "background": "#2563eb",
                  "border": "#1e3a8a",
                  "highlight": {"background": "#60a5fa", "border": "#1e40af"}
                },
                "font": {"color": "#0f172a"}
              },
              "groups": {
                "document": {
                  "shape": "box",
                  "color": {"background": "#f59e0b", "border": "#92400e"},
                  "font": {"color": "#451a03"}
                },
                "entity": {
                  "shape": "dot",
                  "color": {"background": "#2563eb", "border": "#1e3a8a"}
                }
              },
              "edges": {
                "color": {"color": "#94a3b8", "highlight": "#475569"},
                "font": {"align": "middle", "color": "#475569"},
                "smooth": {"type": "dynamic"}
              },
              "interaction": {"hover": true, "navigationButtons": true},
              "physics": {
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 200}
              }
            }
            """
        )
        return visualization.generate_html(notebook=False)

    @staticmethod
    def _filter_provenance(
        provenance: frozenset[TripleProvenance],
        *,
        document_id: UUID | None,
    ) -> tuple[TripleProvenance, ...]:
        return tuple(
            sorted(
                (
                    source
                    for source in provenance
                    if document_id is None or source.document_id == document_id
                ),
                key=lambda source: (source.document_id.int, source.chunk_id.int),
            )
        )

    @staticmethod
    def _serialize_provenance(
        provenance: tuple[TripleProvenance, ...],
    ) -> list[dict[str, str]]:
        return [
            {
                "document_id": str(source.document_id),
                "chunk_id": str(source.chunk_id),
            }
            for source in provenance
        ]

    @staticmethod
    def _merge_provenance(*values: object) -> list[dict[str, str]]:
        provenance: dict[tuple[str, str], dict[str, str]] = {}
        for value in values:
            if not isinstance(value, list):
                continue
            for source in value:
                if not isinstance(source, dict):
                    continue
                document_id = source.get("document_id")
                chunk_id = source.get("chunk_id")
                if isinstance(document_id, str) and isinstance(chunk_id, str):
                    provenance[(document_id, chunk_id)] = {
                        "document_id": document_id,
                        "chunk_id": chunk_id,
                    }
        return [provenance[key] for key in sorted(provenance)]

    @staticmethod
    def _provenance_title(provenance: tuple[TripleProvenance, ...]) -> str:
        return "<br>".join(
            f"Document: {source.document_id}<br>Chunk: {source.chunk_id}"
            for source in provenance
        )
