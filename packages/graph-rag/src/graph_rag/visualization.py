import json
import math
from html import escape
from typing import Any
from uuid import UUID

import networkx as nx
from pyvis.network import Network

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import TripleProvenance

_BASE_ENTITY_SIZE = 12.0
_MAX_ENTITY_SIZE = 48.0
_NETWORK_INITIALIZATION = "network = new vis.Network(container, data, options);"
_AFTER_NETWORK_INITIALIZATION = """
                  const enableHtmlTooltips = function (dataSet) {
                    const updates = [];
                    dataSet.forEach(function (item) {
                      if (typeof item.title !== "string" || !item.title.includes("<br>")) {
                        return;
                      }
                      const tooltip = document.createElement("div");
                      tooltip.innerHTML = item.title;
                      updates.push({id: item.id, title: tooltip});
                    });
                    dataSet.update(updates);
                  };
                  enableHtmlTooltips(nodes);
                  enableHtmlTooltips(edges);

                  network.once("stabilizationIterationsDone", function () {
                    network.stopSimulation();
                    network.storePositions();
                    network.setOptions({physics: {enabled: false}});
                  });
"""
_CONNECTIVITY_COLORS = {
    "hub": {"background": "#7c3aed", "border": "#4c1d95"},
    "connected": {"background": "#2563eb", "border": "#1e3a8a"},
    "leaf": {"background": "#0d9488", "border": "#134e4a"},
    "isolated": {"background": "#64748b", "border": "#334155"},
}
_LEGEND_HTML = """
<div style="position:fixed;top:12px;right:12px;z-index:10;background:#ffffffee;
padding:10px 12px;border:1px solid #cbd5e1;border-radius:8px;font:13px sans-serif;
color:#0f172a;box-shadow:0 2px 8px #0f172a22">
  <strong>Connectivity</strong><br>
  <span style="color:#7c3aed">●</span> Hub&nbsp;
  <span style="color:#2563eb">●</span> Connected&nbsp;
  <span style="color:#0d9488">●</span> Leaf&nbsp;
  <span style="color:#64748b">●</span> Isolated&nbsp;
  <span style="color:#f59e0b">■</span> Document
</div>
"""


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
                        size=24,
                    )
                    network.add_edge(
                        document_node,
                        entity,
                        label="contains",
                        title="Visualization-only provenance link",
                        arrows="to",
                        dashes=True,
                        physics=False,
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
                    title=(
                        f"Relationship: {escape(edge.relation)}<br>"
                        f"{self._provenance_title(provenance)}"
                    ),
                    provenance=self._serialize_provenance(provenance),
                    arrows="to",
                )
        for synonym in self._graph.synonyms():
            if synonym.first in network and synonym.second in network:
                network.add_edge(
                    synonym.first,
                    synonym.second,
                    label="synonym",
                    title=f"Semantic similarity: {synonym.similarity:.3f}",
                    similarity=synonym.similarity,
                    arrows="",
                    dashes=True,
                    color="#7c3aed",
                    synonym=True,
                )
        self._style_entity_nodes(network)
        self._annotate_document_nodes(network)
        return network

    @staticmethod
    def _document_node_id(document_id: UUID) -> str:
        return f"document:{document_id}"

    @staticmethod
    def to_html(network: nx.MultiDiGraph, *, title: str = "Graph RAG") -> str:
        """Render an interactive HTML visualization."""

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
        GraphVisualizer._apply_entity_colors(visualization)
        visualization.set_options(json.dumps(GraphVisualizer._options()))
        html = visualization.generate_html(notebook=False)
        html = html.replace("<body>", f"<body>{_LEGEND_HTML}", 1)
        return html.replace(
            _NETWORK_INITIALIZATION,
            f"{_NETWORK_INITIALIZATION}{_AFTER_NETWORK_INITIALIZATION}",
            1,
        )

    @staticmethod
    def _style_entity_nodes(network: nx.MultiDiGraph) -> None:
        """Scale and color entities by their unique entity neighbors."""

        connection_counts: dict[str, int] = {}
        for node, attributes in network.nodes(data=True):
            if attributes.get("group") != "entity":
                continue
            neighbors = set(network.predecessors(node)) | set(network.successors(node))
            connection_counts[node] = sum(
                network.nodes[neighbor].get("group") == "entity" for neighbor in neighbors
            )
        sorted_counts = sorted(connection_counts.values())
        hub_threshold = max(
            3,
            sorted_counts[math.ceil(len(sorted_counts) * 0.9) - 1]
            if sorted_counts
            else 3,
        )
        for node, connection_count in connection_counts.items():
            attributes = network.nodes[node]
            connectivity = GraphVisualizer._connectivity_group(
                connection_count,
                hub_threshold=hub_threshold,
            )
            attributes["connections"] = connection_count
            attributes["connectivity"] = connectivity
            attributes["color"] = _CONNECTIVITY_COLORS[connectivity]
            attributes["size"] = min(
                _MAX_ENTITY_SIZE,
                _BASE_ENTITY_SIZE + 6.0 * math.sqrt(connection_count),
            )
            attributes["title"] = (
                f"Entity: {escape(str(node))}<br>"
                f"Connected entities: {connection_count}<br>{attributes['title']}"
            )

    @staticmethod
    def _connectivity_group(connection_count: int, *, hub_threshold: int) -> str:
        """Classify one entity using domain-independent graph structure."""

        if connection_count == 0:
            return "isolated"
        if connection_count == 1:
            return "leaf"
        if connection_count >= hub_threshold:
            return "hub"
        return "connected"

    @staticmethod
    def _annotate_document_nodes(network: nx.MultiDiGraph) -> None:
        """Add document entity counts to their visualization tooltips."""

        for node, attributes in network.nodes(data=True):
            if attributes.get("group") != "document":
                continue
            entity_count = sum(
                data.get("visualization_only")
                for _source, _target, data in network.out_edges(node, data=True)
            )
            attributes["title"] = f"{attributes['title']}<br>Entities: {entity_count}"

    @staticmethod
    def _apply_entity_colors(visualization: Network) -> None:
        """Restore node colors discarded by PyVis group conversion."""

        for node in visualization.nodes:
            connectivity = node.get("connectivity")
            if connectivity in _CONNECTIVITY_COLORS:
                node["color"] = _CONNECTIVITY_COLORS[connectivity]
                node.pop("group", None)

    @staticmethod
    def _options() -> dict[str, Any]:
        """Return options for a force-directed layout that freezes once ready."""

        return {
            "nodes": {
                "shape": "dot",
                "color": {
                    "background": "#2563eb",
                    "border": "#1e3a8a",
                    "highlight": {
                        "background": "#60a5fa",
                        "border": "#1e40af",
                    },
                },
                "font": {"color": "#0f172a"},
            },
            "groups": {
                "document": {
                    "shape": "box",
                    "color": {"background": "#f59e0b", "border": "#92400e"},
                    "font": {"color": "#451a03"},
                },
            },
            "edges": {
                "color": {"color": "#94a3b8", "highlight": "#475569"},
                "font": {"align": "middle", "color": "#475569"},
                "smooth": {"type": "dynamic"},
            },
            "interaction": {
                "hover": True,
                "navigationButtons": True,
            },
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "stabilization": {
                    "enabled": True,
                    "iterations": 200,
                    "fit": True,
                },
            },
        }

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
    def _provenance_title(provenance: tuple[TripleProvenance, ...]) -> str:
        chunks_by_document: dict[UUID, set[UUID]] = {}
        for source in provenance:
            chunks_by_document.setdefault(source.document_id, set()).add(source.chunk_id)

        sections = []
        for document_id, chunk_ids in sorted(
            chunks_by_document.items(),
            key=lambda item: item[0].int,
        ):
            chunks = "<br>".join(
                f"&nbsp;&nbsp;{chunk_id}"
                for chunk_id in sorted(chunk_ids, key=lambda value: value.int)
            )
            sections.append(f"Document: {document_id}<br>Chunks:<br>{chunks}")
        return "<br><br>".join(sections)
