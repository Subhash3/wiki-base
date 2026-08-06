import argparse
from pathlib import Path

from graph_rag.graph import KnowledgeGraph
from graph_rag.visualization import GraphVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a canonical Graph RAG JSON file as interactive HTML."
    )
    parser.add_argument("json_file", type=Path, help="Graph JSON file to visualize")
    parser.add_argument("--3d", dest="export_3d", action="store_true", help="Also export 3D HTML")
    return parser.parse_args()


def visualize_file(json_file: Path, *, export_3d: bool = False) -> Path:
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
    if export_3d:
        html_3d_file = json_file.with_name(f"{json_file.stem}-3d.html")
        html_3d_file.write_text(
            visualizer.to_3d_html(network, title=f"Graph RAG · {json_file.stem} · 3D"),
            encoding="utf-8",
        )
    return html_file


def run() -> None:
    arguments = parse_args()
    output = visualize_file(arguments.json_file, export_3d=arguments.export_3d)
    print(output)
    if arguments.export_3d:
        print(arguments.json_file.with_name(f"{arguments.json_file.stem}-3d.html"))
