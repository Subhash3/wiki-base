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

Structured extraction and final answers can use different providers and models:

```env
WIKI_BASE_EXTRACTION_PROVIDER=groq
WIKI_BASE_EXTRACTION_MODEL=openai/gpt-oss-20b
WIKI_BASE_GROQ_API_KEY=gsk_your_key_here
WIKI_BASE_ANSWER_GENERATION_PROVIDER=ollama
WIKI_BASE_ANSWER_GENERATION_MODEL=gemma3:270m
```

The extraction model handles graph indexing and query-concept extraction. The answer model
is used only after retrieval has selected supporting chunks. Groq defaults stay below the
documented GPT-OSS free-tier minute and daily limits and stop locally when the daily budget
is exhausted.

The graph worker uses passage entity extraction followed by entity-guided OpenIE. It stores
one mention-aware canonical JSONB graph and its pgvector entity and relationship index per
document. It also builds conservative wiki-base synonym edges from persisted embeddings.
Pro retrieval merges ready document graphs and synonym edges in memory before PageRank.
Facts retrieval follows bounded canonical triples, ranks them semantically, and supplies
the selected facts with their provenance passages to answer generation.

Render one stored document graph, or export and visualize the merged graph for a wiki base:

```bash
uv run graph-rag-visualize <document-id>
uv run graph-rag-visualize-merge <wiki-base-id>
```

Both commands write canonical JSON and interactive HTML files using the supplied ID as
the filename.

The API is then available at `http://localhost:8000`. Useful initial endpoints are:

- `GET /health`
- `GET /ready`
- `GET /capabilities`
- `GET /wiki-bases`
- `GET /querychunks`
- `POST /query`

Persistent debug logs are written to `logs/` by default:

- `logs/api.log` records retrieval, ranked facts and chunks, complete answer context,
  generated answers, and citation resolution.
- `logs/ingestion-worker.log` records document parsing and chunk-ingestion activity.
- `logs/graph-indexing-worker.log` records chunk-level entity and triple extraction,
  graph construction, concept embedding, and indexing failures.

Each file rotates at 20 MB and keeps five backups. Change the location with
`WIKI_BASE_LOG_DIRECTORY`; set `WIKI_BASE_LOG_LEVEL=INFO` when full debug traces are not
needed. Because debug logs contain document passages and model inputs, treat the directory
as application data rather than publishing it.

Run checks with:

```bash
uv run ruff check .
uv run pytest
```

Run the Lite, Pro, and Facts retrieval benchmark against the development dataset:

```bash
uv run wiki-base-benchmark benchmarks/graphrag.json
```

See [`benchmarks/README.md`](benchmarks/README.md) for the dataset format, metrics, and
baseline report options.

## Workspace projects

- [`document-processing`](packages/document-processing/README.md) — reusable document parsing and chunking
- [`graph-rag`](packages/graph-rag/README.md) — graph-based retrieval-augmented generation
- [`llm-providers`](packages/llm-providers/README.md) — shared model provider interfaces and implementations
