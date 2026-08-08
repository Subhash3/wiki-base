# LLM Providers

Provider interfaces and implementations shared by projects in this workspace.

The package supports Ollama embeddings and Ollama, llama.cpp, or Groq structured,
citation-aware generation. The llama.cpp provider uses the server's OpenAI-compatible
chat completions API with JSON-schema constrained output. Groq generation uses strict
JSON schemas, conservative free-tier throttling, and at most one retry for rate-limit
responses. Structured generation `400` responses are classified without retrying. A
provider-neutral reranking interface is also included.

## Development

From the repository root:

```bash
uv sync --package llm-providers
uv run --package llm-providers python -c "import llm_providers"
```
