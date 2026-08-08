import pytest
from llm_providers.embeddings.ollama import OllamaEmbeddingProvider
from llm_providers.embeddings.openai_compatible import (
    OpenAICompatibleEmbeddingProvider,
)

from wiki_base.config.settings import Settings
from wiki_base.embeddings import create_embedding_provider


def test_creates_ollama_embedding_provider_by_default() -> None:
    provider = create_embedding_provider(Settings())

    assert isinstance(provider, OllamaEmbeddingProvider)


def test_creates_llama_cpp_embedding_provider() -> None:
    provider = create_embedding_provider(
        Settings(
            embedding_provider="llama-cpp",
            embedding_model="local-embedding-model",
            embedding_dimensions=768,
            llama_cpp_url="http://llama.test",
        )
    )

    assert isinstance(provider, OpenAICompatibleEmbeddingProvider)
    assert provider.model_info.model == "local-embedding-model"
    assert provider.model_info.dimensions == 768
    assert provider._base_url == "http://llama.test"


def test_openai_compatible_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="WIKI_BASE_EMBEDDING_BASE_URL"):
        create_embedding_provider(Settings(embedding_provider="openai-compatible"))
