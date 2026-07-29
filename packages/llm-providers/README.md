# LLM Providers

Provider interfaces and implementations shared by projects in this workspace.

The initial implementation supports Ollama for embeddings and structured,
citation-aware generation. A provider-neutral reranking interface is also
included.

## Development

From the repository root:

```bash
uv sync --package llm-providers
uv run --package llm-providers python -c "import llm_providers"
```
