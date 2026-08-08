from typing import Literal

from llm_providers.embeddings.base import EmbeddingProvider
from llm_providers.embeddings.ollama import OllamaEmbeddingProvider
from llm_providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)

from wiki_base.config.settings import Settings

EmbeddingProviderName = Literal["ollama", "llama-cpp", "openai-compatible"]


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Create the configured embedding provider."""

    if settings.embedding_provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=settings.ollama_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    if settings.embedding_provider == "llama-cpp":
        return OpenAICompatibleEmbeddingProvider(
            base_url=settings.llama_cpp_url,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            max_tokens=settings.embedding_max_tokens,
            timeout_seconds=settings.llama_cpp_timeout_seconds,
        )

    if not settings.embedding_base_url:
        raise ValueError(
            "WIKI_BASE_EMBEDDING_BASE_URL is required for openai-compatible embeddings"
        )
    return OpenAICompatibleEmbeddingProvider(
        base_url=settings.embedding_base_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        max_tokens=settings.embedding_max_tokens,
        timeout_seconds=settings.embedding_timeout_seconds,
        api_key=(
            settings.embedding_api_key.get_secret_value()
            if settings.embedding_api_key is not None
            else None
        ),
    )
