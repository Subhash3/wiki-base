import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
from pydantic import BaseModel, Field, ValidationError, model_validator

from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.schemas.query_chunks import QueryChunksResponse, RetrievedChunkResponse


class RelevantChunk(BaseModel):
    """Identify one chunk that should be retrieved for a benchmark case."""

    id: UUID | None = None
    document_name: str | None = None
    content_contains: str | None = None

    @model_validator(mode="after")
    def validate_identifier(self) -> "RelevantChunk":
        """Require at least one stable way to identify the chunk."""

        if not any((self.id, self.document_name, self.content_contains)):
            raise ValueError("A relevant chunk needs an id, document name, or content marker")
        return self

    def matches(self, chunk: RetrievedChunkResponse) -> bool:
        """Return whether a retrieved chunk satisfies this relevance label."""

        if self.id is not None and chunk.id != self.id:
            return False
        if self.document_name is not None and chunk.document_name != self.document_name:
            return False
        return not (
            self.content_contains is not None
            and self.content_contains.casefold() not in chunk.content.casefold()
        )


class BenchmarkCase(BaseModel):
    """Describe one retrieval question and its expected evidence."""

    id: str = Field(min_length=1)
    wiki_base_id: UUID
    question: str = Field(min_length=1)
    relevant_chunks: list[RelevantChunk] = Field(min_length=1)
    expected_facts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class BenchmarkDataset(BaseModel):
    """Contain a versioned collection of retrieval benchmark cases."""

    version: int = 1
    name: str = Field(min_length=1)
    cases: list[BenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_case_ids(self) -> "BenchmarkDataset":
        """Reject duplicate case identifiers."""

        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Benchmark case ids must be unique")
        return self


class CaseMetrics(BaseModel):
    """Store retrieval metrics for one benchmark case."""

    expected: int
    matched: int
    hit: bool
    recall: float
    reciprocal_rank: float


class RankedChunkResult(BaseModel):
    """Store a compact retrieved chunk in a benchmark report."""

    rank: int
    id: UUID
    document_id: UUID
    document_name: str
    score: float
    content_preview: str


class BenchmarkCaseResult(BaseModel):
    """Record one mode's result for one benchmark case."""

    case_id: str
    wiki_base_id: UUID
    question: str
    mode: RetrievalMode
    retrieval_strategy: RetrievalStrategy | None = None
    duration_ms: float
    chunks: list[RankedChunkResult] = Field(default_factory=list)
    metrics: CaseMetrics | None = None
    error: str | None = None


class ModeSummary(BaseModel):
    """Summarize retrieval quality for one mode."""

    mode: RetrievalMode
    cases: int
    successful: int
    errors: int
    hits: int
    hit_rate: float
    mean_recall: float
    mean_reciprocal_rank: float
    graph_responses: int
    vector_fallbacks: int


class BenchmarkReport(BaseModel):
    """Store one reproducible retrieval benchmark run."""

    run_name: str
    dataset_name: str
    dataset_path: str
    created_at: datetime
    base_url: str
    limit: int
    results: list[BenchmarkCaseResult]
    summaries: list[ModeSummary]


class BenchmarkRunner:
    """Run retrieval benchmark cases against the Wiki Base API."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        """Use the supplied client for API requests."""

        self._client = client

    async def run(
        self,
        dataset: BenchmarkDataset,
        *,
        dataset_path: Path,
        run_name: str,
        modes: list[RetrievalMode],
        limit: int,
    ) -> BenchmarkReport:
        """Run every case in every selected retrieval mode."""

        results: list[BenchmarkCaseResult] = []
        for case in dataset.cases:
            for mode in modes:
                results.append(await self._run_case(case, mode=mode, limit=limit))

        return BenchmarkReport(
            run_name=run_name,
            dataset_name=dataset.name,
            dataset_path=str(dataset_path),
            created_at=datetime.now(UTC),
            base_url=str(self._client.base_url),
            limit=limit,
            results=results,
            summaries=[_summarize_mode(results, mode) for mode in modes],
        )

    async def _run_case(
        self,
        case: BenchmarkCase,
        *,
        mode: RetrievalMode,
        limit: int,
    ) -> BenchmarkCaseResult:
        """Run one case and preserve request failures in the report."""

        started_at = asyncio.get_running_loop().time()
        try:
            response = await self._client.get(
                "/querychunks",
                params={
                    "wiki_base_id": str(case.wiki_base_id),
                    "question": case.question,
                    "limit": limit,
                    "mode": mode.value,
                },
            )
            response.raise_for_status()
            payload = QueryChunksResponse.model_validate(response.json())
        except httpx.HTTPStatusError as error:
            message = _http_error_message(error.response)
            return _failed_result(case, mode, started_at, message)
        except (httpx.HTTPError, ValidationError, ValueError) as error:
            return _failed_result(case, mode, started_at, str(error))

        duration_ms = _elapsed_ms(started_at)
        return BenchmarkCaseResult(
            case_id=case.id,
            wiki_base_id=case.wiki_base_id,
            question=case.question,
            mode=mode,
            retrieval_strategy=payload.retrieval_strategy,
            duration_ms=duration_ms,
            chunks=[
                RankedChunkResult(
                    rank=rank,
                    id=chunk.id,
                    document_id=chunk.document_id,
                    document_name=chunk.document_name,
                    score=chunk.score,
                    content_preview=" ".join(chunk.content.split())[:240],
                )
                for rank, chunk in enumerate(payload.chunks, start=1)
            ],
            metrics=evaluate_retrieval(case.relevant_chunks, payload.chunks),
        )


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load and validate a benchmark dataset from JSON."""

    if not path.is_file():
        raise ValueError(f"Benchmark dataset does not exist: {path}")
    return BenchmarkDataset.model_validate_json(path.read_text(encoding="utf-8"))


def evaluate_retrieval(
    relevant_chunks: list[RelevantChunk],
    retrieved_chunks: list[RetrievedChunkResponse],
) -> CaseMetrics:
    """Calculate hit, recall, and reciprocal rank for retrieved chunks."""

    matched_ranks = [
        rank
        for relevant in relevant_chunks
        if (
            rank := next(
                (
                    index
                    for index, chunk in enumerate(retrieved_chunks, start=1)
                    if relevant.matches(chunk)
                ),
                None,
            )
        )
        is not None
    ]
    matched = len(matched_ranks)
    return CaseMetrics(
        expected=len(relevant_chunks),
        matched=matched,
        hit=matched > 0,
        recall=matched / len(relevant_chunks),
        reciprocal_rank=1 / min(matched_ranks) if matched_ranks else 0.0,
    )


def write_report(report: BenchmarkReport, output: Path) -> Path:
    """Write a benchmark report as formatted JSON."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return output


def run() -> None:
    """Run the benchmark CLI."""

    parser = argparse.ArgumentParser(
        description="Compare retrieval modes against a labeled dataset."
    )
    parser.add_argument("dataset", type=Path, help="Benchmark dataset JSON")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Wiki Base API URL (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--limit", type=int, default=5, help="Chunks per query")
    parser.add_argument(
        "--mode",
        action="append",
        choices=[mode.value for mode in RetrievalMode],
        help="Mode to run; repeat to select both (default: both)",
    )
    parser.add_argument("--run-name", default="baseline", help="Name stored in the report")
    parser.add_argument("--output", type=Path, help="Output report JSON")
    parser.add_argument("--timeout", type=float, default=180, help="Request timeout in seconds")
    arguments = parser.parse_args()

    if arguments.limit < 1 or arguments.limit > 20:
        parser.error("--limit must be between 1 and 20")
    if arguments.timeout <= 0:
        parser.error("--timeout must be positive")

    dataset = load_dataset(arguments.dataset)
    modes = _selected_modes(arguments.mode)
    output = arguments.output or arguments.dataset.with_suffix(".results.json")
    report = asyncio.run(
        _run_from_cli(
            dataset,
            dataset_path=arguments.dataset,
            run_name=arguments.run_name,
            modes=modes,
            limit=arguments.limit,
            base_url=arguments.base_url,
            timeout_seconds=arguments.timeout,
        )
    )
    write_report(report, output)
    _print_report(report, output)


async def _run_from_cli(
    dataset: BenchmarkDataset,
    *,
    dataset_path: Path,
    run_name: str,
    modes: list[RetrievalMode],
    limit: int,
    base_url: str,
    timeout_seconds: float,
) -> BenchmarkReport:
    """Create an HTTP client and run a CLI benchmark."""

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        timeout=timeout_seconds,
    ) as client:
        return await BenchmarkRunner(client).run(
            dataset,
            dataset_path=dataset_path,
            run_name=run_name,
            modes=modes,
            limit=limit,
        )


def _failed_result(
    case: BenchmarkCase,
    mode: RetrievalMode,
    started_at: float,
    message: str,
) -> BenchmarkCaseResult:
    """Create a failed case result without aborting the full run."""

    return BenchmarkCaseResult(
        case_id=case.id,
        wiki_base_id=case.wiki_base_id,
        question=case.question,
        mode=mode,
        duration_ms=_elapsed_ms(started_at),
        error=message,
    )


def _elapsed_ms(started_at: float) -> float:
    """Return rounded elapsed milliseconds from an event-loop timestamp."""

    return round((asyncio.get_running_loop().time() - started_at) * 1000, 2)


def _summarize_mode(
    results: list[BenchmarkCaseResult],
    mode: RetrievalMode,
) -> ModeSummary:
    """Aggregate case results for one retrieval mode."""

    selected = [result for result in results if result.mode == mode]
    metrics = [result.metrics for result in selected if result.metrics is not None]
    total = len(selected)
    hits = sum(metric.hit for metric in metrics)
    return ModeSummary(
        mode=mode,
        cases=total,
        successful=len(metrics),
        errors=total - len(metrics),
        hits=hits,
        hit_rate=hits / total if total else 0.0,
        mean_recall=sum(metric.recall for metric in metrics) / total if total else 0.0,
        mean_reciprocal_rank=(
            sum(metric.reciprocal_rank for metric in metrics) / total if total else 0.0
        ),
        graph_responses=sum(
            result.retrieval_strategy
            in {RetrievalStrategy.GRAPH, RetrievalStrategy.FACT_GRAPH}
            for result in selected
        ),
        vector_fallbacks=sum(
            result.retrieval_strategy == RetrievalStrategy.VECTOR_FALLBACK
            for result in selected
        ),
    )


def _selected_modes(values: list[str] | None) -> list[RetrievalMode]:
    """Return distinct selected modes in command-line order."""

    if not values:
        return list(RetrievalMode)
    return list(dict.fromkeys(RetrievalMode(value) for value in values))


def _http_error_message(response: httpx.Response) -> str:
    """Return a concise API error for a failed benchmark request."""

    try:
        payload = response.json()
        detail = payload.get("detail") or payload.get("message") or json.dumps(payload)
    except (ValueError, AttributeError):
        detail = response.text.strip()
    return f"HTTP {response.status_code}: {detail or response.reason_phrase}"


def _print_report(report: BenchmarkReport, output: Path) -> None:
    """Print case and mode summaries for a completed run."""

    for result in report.results:
        if result.error:
            print(f"{result.case_id} [{result.mode.value}] ERROR {result.error}")
            continue
        assert result.metrics is not None
        print(
            f"{result.case_id} [{result.mode.value}/{result.retrieval_strategy.value}] "
            f"hit={result.metrics.hit} recall={result.metrics.recall:.3f} "
            f"rr={result.metrics.reciprocal_rank:.3f} time={result.duration_ms:.0f}ms"
        )
    for summary in report.summaries:
        print(
            f"{summary.mode.value}: hit_rate={summary.hit_rate:.3f} "
            f"mean_recall={summary.mean_recall:.3f} "
            f"mrr={summary.mean_reciprocal_rank:.3f} "
            f"errors={summary.errors} fallbacks={summary.vector_fallbacks}"
        )
    print(output)


if __name__ == "__main__":
    run()
