from graph_rag.graph import KnowledgeGraph
from graph_rag.models import GraphEdge, GraphFact

FactKey = tuple[str, str, str]


class GraphFactTraverser:
    """Collect canonical facts near a set of graph nodes."""

    def __init__(self, *, max_depth: int = 2, max_candidates: int = 500) -> None:
        """Configure traversal depth and its candidate bound."""

        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        self._max_depth = max_depth
        self._max_candidates = max_candidates

    @property
    def max_depth(self) -> int:
        """Return the configured traversal depth."""

        return self._max_depth

    @property
    def max_candidates(self) -> int:
        """Return the configured candidate limit."""

        return self._max_candidates

    def traverse(self, graph: KnowledgeGraph, nodes: list[str]) -> list[GraphFact]:
        """Return distinct facts reachable from the supplied graph nodes."""

        seeds = list(dict.fromkeys(node for node in nodes if node in graph.nodes))
        if not seeds:
            return []

        edges = sorted(
            graph.edges(),
            key=lambda edge: (edge.subject, edge.relation, edge.object),
        )
        subject_nodes = {edge.subject for edge in edges}
        reached: dict[FactKey, tuple[GraphEdge, int, set[str]]] = {}

        for seed in seeds:
            frontier = {seed}
            visited_nodes = {seed}
            for depth in range(1, self._max_depth + 1):
                next_frontier: set[str] = set()
                for edge in edges:
                    if edge.subject not in frontier and edge.object not in frontier:
                        continue
                    key = (edge.subject, edge.relation, edge.object)
                    existing = reached.get(key)
                    if existing is None:
                        reached[key] = (edge, depth, {seed})
                    else:
                        existing[2].add(seed)
                        if depth < existing[1]:
                            reached[key] = (existing[0], depth, existing[2])

                    for endpoint in (edge.subject, edge.object):
                        if endpoint in subject_nodes and endpoint not in visited_nodes:
                            next_frontier.add(endpoint)
                    if len(reached) >= self._max_candidates:
                        break
                if len(reached) >= self._max_candidates or not next_frontier:
                    break
                visited_nodes.update(next_frontier)
                frontier = next_frontier

        return [
            GraphFact(
                subject=edge.subject,
                relation=edge.relation,
                object=edge.object,
                provenance=edge.provenance,
                depth=depth,
                seeds=frozenset(fact_seeds),
            )
            for edge, depth, fact_seeds in sorted(
                reached.values(),
                key=lambda item: (
                    item[1],
                    item[0].subject,
                    item[0].relation,
                    item[0].object,
                ),
            )
        ]
