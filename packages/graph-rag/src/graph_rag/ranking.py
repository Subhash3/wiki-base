import networkx as nx

from graph_rag.graph import KnowledgeGraph
from graph_rag.models import RankedChunk, TripleProvenance


def build_ranking_graph(graph: KnowledgeGraph) -> nx.Graph:
    """Build an undirected entity graph for PageRank."""

    ranking_graph = nx.Graph()
    ranking_graph.add_nodes_from(graph.nodes)
    for edge in graph.edges():
        ranking_graph.add_edge(edge.subject, edge.object, weight=1.0, kind="fact")
    for synonym in graph.synonyms():
        existing_weight = (
            ranking_graph.edges[synonym.first, synonym.second].get("weight", 0.0)
            if ranking_graph.has_edge(synonym.first, synonym.second)
            else 0.0
        )
        ranking_graph.add_edge(
            synonym.first,
            synonym.second,
            weight=max(existing_weight, synonym.similarity),
            kind="synonym" if not existing_weight else "fact",
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
    *,
    entity_weight: float = 1.0,
    triple_weight: float = 1.0,
) -> list[RankedChunk]:
    """Project node scores through entity mentions and triple provenance."""

    if entity_weight < 0 or triple_weight < 0:
        raise ValueError("provenance weights cannot be negative")

    chunk_scores: dict[TripleProvenance, float] = {}
    for node, score in node_scores.items():
        if score <= 0:
            continue
        provenance_weights: dict[TripleProvenance, float] = {}
        for provenance in graph.entity_provenance_for_node(node):
            provenance_weights[provenance] = entity_weight
        for provenance in graph.triple_provenance_for_node(node):
            provenance_weights[provenance] = max(
                triple_weight,
                provenance_weights.get(provenance, 0.0),
            )
        for provenance, weight in provenance_weights.items():
            chunk_scores[provenance] = (
                chunk_scores.get(provenance, 0.0) + score * weight
            )

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
