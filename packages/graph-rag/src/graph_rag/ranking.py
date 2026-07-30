import networkx as nx

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RankedChunk, TripleProvenance


def build_ranking_graph(graph: KnowledgeGraph) -> nx.Graph:
    """Build an undirected entity graph for PageRank."""

    ranking_graph = nx.Graph()
    ranking_graph.add_nodes_from(graph.nodes)
    ranking_graph.add_edges_from(
        (edge.subject, edge.object, {"weight": 1.0})
        for edge in graph.edges()
    )
    return ranking_graph


def personalized_page_rank(
    graph: nx.Graph,
    seeds: list[str],
    *,
    alpha: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Rank graph nodes using equal personalization over valid seeds."""

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    valid_seeds = list(dict.fromkeys(seed for seed in seeds if seed in graph))
    if not graph or not valid_seeds:
        return {}

    seed_weight = 1.0 / len(valid_seeds)
    personalization = {node: 0.0 for node in graph}
    personalization.update({seed: seed_weight for seed in valid_seeds})
    scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        dangling=personalization,
        nstart=personalization,
        max_iter=max_iterations,
        tol=tolerance,
        weight="weight",
    )
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))


def aggregate_chunk_scores(
    graph: KnowledgeGraph,
    node_scores: dict[str, float],
) -> list[RankedChunk]:
    """Sum positive node scores across their source chunks."""

    chunk_scores: dict[TripleProvenance, float] = {}
    for node, score in node_scores.items():
        if score <= 0:
            continue
        for provenance in graph.provenance_for_node(node):
            chunk_scores[provenance] = chunk_scores.get(provenance, 0.0) + score

    ranked = [
        RankedChunk(
            document_id=provenance.document_id,
            chunk_id=provenance.chunk_id,
            score=score,
        )
        for provenance, score in chunk_scores.items()
    ]
    return sorted(
        ranked,
        key=lambda chunk: (-chunk.score, chunk.document_id.int, chunk.chunk_id.int),
    )
