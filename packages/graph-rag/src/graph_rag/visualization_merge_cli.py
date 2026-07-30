import argparse
from pathlib import Path

from graph_rag.visualization import GraphVisualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Graph RAG JSON files and render an interactive HTML graph."
    )
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="Graph JSON files or directories containing graph JSON files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Merged JSON path (default: merged.json beside the first source)",
    )
    return parser.parse_args()


def find_json_files(inputs: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            files.update(
                path
                for path in input_path.glob("*.json")
                if path.name != "merged.json"
            )
        elif input_path.is_file() and input_path.suffix.lower() == ".json":
            files.add(input_path)
        else:
            raise ValueError(f"Expected a JSON file or directory: {input_path}")
    if not files:
        raise ValueError("No graph JSON files found")
    return sorted(files)


def merge_files(inputs: list[Path], *, output: Path | None = None) -> tuple[Path, Path]:
    json_files = find_json_files(inputs)
    output_json = output or json_files[0].parent / "merged.json"
    if output_json.suffix.lower() != ".json":
        raise ValueError(f"Output must be a .json file: {output_json}")

    networks = [
        GraphVisualizer.from_json(path.read_text(encoding="utf-8"))
        for path in json_files
        if path.resolve() != output_json.resolve()
    ]
    if not networks:
        raise ValueError("No source graph JSON files remain after excluding the output")

    merged = GraphVisualizer.merge(networks)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(GraphVisualizer.to_json(merged), encoding="utf-8")
    output_html = output_json.with_suffix(".html")
    output_html.write_text(
        GraphVisualizer.to_html(merged, title="Graph RAG · Merged"),
        encoding="utf-8",
    )
    return output_json, output_html


def run() -> None:
    arguments = parse_args()
    output_json, output_html = merge_files(arguments.inputs, output=arguments.output)
    print(output_json)
    print(output_html)
