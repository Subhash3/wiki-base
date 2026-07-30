import argparse
from pathlib import Path

from graph_rag.graph import KnowledgeGraph
from graph_rag.visualization import GraphVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a canonical Graph RAG JSON file as interactive HTML."
    )
    parser.add_argument("json_file", type=Path, help="Graph JSON file to visualize")
    return parser.parse_args()


def visualize_file(json_file: Path) -> Path:
    if not json_file.is_file():
        raise ValueError(f"JSON file does not exist: {json_file}")
    if json_file.suffix.lower() != ".json":
        raise ValueError(f"Expected a .json file: {json_file}")

    graph = KnowledgeGraph.from_json(json_file.read_text(encoding="utf-8"))
    visualizer = GraphVisualizer(graph)
    network = visualizer.build()
    html_file = json_file.with_suffix(".html")
    html_file.write_text(
        visualizer.to_html(network, title=f"Graph RAG · {json_file.stem}"),
        encoding="utf-8",
    )
    return html_file


def run() -> None:
    arguments = parse_args()
    output = visualize_file(arguments.json_file)
    print(output)
