import json
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from pydantic import ValidationError

from wiki_base.benchmarking import (
    BenchmarkCase,
    BenchmarkDataset,
    BenchmarkRunner,
    RelevantChunk,
    evaluate_retrieval,
    load_dataset,
    write_report,
)
from wiki_base.retrieval import RetrievalMode, RetrievalStrategy
from wiki_base.schemas.query_chunks import RetrievedChunkResponse

WIKI_BASE_ID = UUID("10000000-0000-0000-0000-000000000001")
DOCUMENT_ID = UUID("20000000-0000-0000-0000-000000000001")
RELEVANT_CHUNK_ID = UUID("30000000-0000-0000-0000-000000000001")
OTHER_CHUNK_ID = UUID("30000000-0000-0000-0000-000000000002")


def make_chunk(chunk_id: UUID, content: str) -> RetrievedChunkResponse:
    """Create one API chunk response for metric tests."""

    return RetrievedChunkResponse(
        id=chunk_id,
        document_id=DOCUMENT_ID,
        document_name="vehicles.pdf",
        content=content,
        score=0.8,
    )


def make_dataset() -> BenchmarkDataset:
    """Create one labeled benchmark case."""

    return BenchmarkDataset(
        name="test dataset",
        cases=[
            BenchmarkCase(
                id="vehicle-price",
                wiki_base_id=WIKI_BASE_ID,
                question="What does it cost?",
                relevant_chunks=[
                    RelevantChunk(
                        id=RELEVANT_CHUNK_ID,
                        content_contains="price range",
                    )
                ],
            )
        ],
    )


def test_evaluates_retrieval_against_multiple_relevance_labels() -> None:
    """Recall counts expected chunks while rank uses the first relevant result."""

    metrics = evaluate_retrieval(
        [
            RelevantChunk(id=RELEVANT_CHUNK_ID),
            RelevantChunk(content_contains="missing evidence"),
        ],
        [
            make_chunk(OTHER_CHUNK_ID, "Unrelated text"),
            make_chunk(RELEVANT_CHUNK_ID, "The price range is ten to twelve lakh."),
        ],
    )

    assert metrics.expected == 2
    assert metrics.matched == 1
    assert metrics.hit is True
    assert metrics.recall == 0.5
    assert metrics.reciprocal_rank == 0.5


def test_relevant_chunk_requires_an_identifier() -> None:
    """Empty relevance labels cannot silently match every retrieved chunk."""

    with pytest.raises(ValidationError, match="needs an id"):
        RelevantChunk()


def test_loads_and_validates_dataset(tmp_path: Path) -> None:
    """Dataset loading rejects duplicate case identifiers."""

    payload = make_dataset().model_dump(mode="json")
    payload["cases"].append(payload["cases"][0])
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="must be unique"):
        load_dataset(path)


def test_development_dataset_is_valid() -> None:
    """The checked-in development benchmark follows the supported schema."""

    path = Path(__file__).parents[2] / "benchmarks" / "graphrag.json"

    dataset = load_dataset(path)

    assert dataset.version == 1
    assert len(dataset.cases) == 4


async def test_runner_compares_modes_and_records_vector_fallback(tmp_path: Path) -> None:
    """The runner records ranked metrics and the actual retrieval strategy."""

    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        mode = RetrievalMode(request.url.params["mode"])
        strategy = (
            RetrievalStrategy.VECTOR
            if mode == RetrievalMode.LITE
            else RetrievalStrategy.VECTOR_FALLBACK
        )
        return httpx.Response(
            200,
            json={
                "wiki_base_id": str(WIKI_BASE_ID),
                "question": "What does it cost?",
                "chunks": [
                    make_chunk(OTHER_CHUNK_ID, "Unrelated text").model_dump(mode="json"),
                    make_chunk(
                        RELEVANT_CHUNK_ID,
                        "The price range is ten to twelve lakh.",
                    ).model_dump(mode="json"),
                ],
                "mode": mode.value,
                "retrieval_strategy": strategy.value,
            },
        )

    async with httpx.AsyncClient(
        base_url="http://benchmark.test",
        transport=httpx.MockTransport(handle),
    ) as client:
        report = await BenchmarkRunner(client).run(
            make_dataset(),
            dataset_path=Path("benchmarks/test.json"),
            run_name="before-openie",
            modes=[RetrievalMode.LITE, RetrievalMode.PRO],
            limit=2,
        )

    assert len(requests) == 2
    assert report.results[0].metrics is not None
    assert report.results[0].metrics.reciprocal_rank == 0.5
    assert report.results[1].retrieval_strategy == RetrievalStrategy.VECTOR_FALLBACK
    assert report.summaries[0].hit_rate == 1.0
    assert report.summaries[1].vector_fallbacks == 1

    output = write_report(report, tmp_path / "result.json")
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["run_name"] == "before-openie"
    assert stored["results"][0]["chunks"][1]["id"] == str(RELEVANT_CHUNK_ID)


async def test_runner_preserves_api_errors() -> None:
    """One failed request remains visible without aborting the benchmark."""

    def handle(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "retrieval unavailable"})

    async with httpx.AsyncClient(
        base_url="http://benchmark.test",
        transport=httpx.MockTransport(handle),
    ) as client:
        report = await BenchmarkRunner(client).run(
            make_dataset(),
            dataset_path=Path("benchmarks/test.json"),
            run_name="failing-run",
            modes=[RetrievalMode.PRO],
            limit=5,
        )

    assert report.results[0].error == "HTTP 503: retrieval unavailable"
    assert report.summaries[0].errors == 1
    assert report.summaries[0].hit_rate == 0.0
