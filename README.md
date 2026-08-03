# Wiki Base

A self-hosted service for creating immutable, vectorized document collections and
answering questions from them with citations.

The architecture and implementation roadmap are documented in [PLAN.md](PLAN.md).

## Development

Requirements:

- Python 3.12+
- `uv`
- Docker with Compose

Create local configuration and start PostgreSQL:

```bash
cp .env.example .env
docker compose up -d postgres
uv sync
uv run wiki-base-init-db
uv run uvicorn wiki_base.main:app --reload
```

In a second terminal, start the single ingestion worker:

```bash
uv run wiki-base-worker
```

Start the graph indexing worker in another terminal:

```bash
uv run wiki-base-graph-worker
```

Structured extraction and final answers can use different Ollama models:

```env
WIKI_BASE_EXTRACTION_MODEL=qwen3.5:0.8b
WIKI_BASE_ANSWER_GENERATION_MODEL=qwen3.5:9b
```

The extraction model handles graph indexing and query-concept extraction. The answer model
is used only after retrieval has selected supporting chunks.

The graph worker stores one canonical JSONB graph per document in PostgreSQL. Graph-indexing
jobs contain processing status and errors only; Pro retrieval loads ready document graphs
from the database and merges them in memory.

The API is then available at `http://localhost:8000`. Useful initial endpoints are:

- `GET /health`
- `GET /ready`
- `GET /capabilities`
- `GET /wiki-bases`
- `GET /querychunks`
- `POST /query`

Run checks with:

```bash
uv run ruff check .
uv run pytest
```

## Workspace projects

- [`document-processing`](packages/document-processing/README.md) — reusable document parsing and chunking
- [`graph-rag`](packages/graph-rag/README.md) — graph-based retrieval-augmented generation
- [`llm-providers`](packages/llm-providers/README.md) — shared model provider interfaces and implementations
