from llm_providers.generation.llama_cpp import LlamaCppGenerationProvider

from wiki_base.config.settings import Settings
from wiki_base.generation import create_generation_provider, create_groq_rate_limiter


def test_create_llama_cpp_generation_provider() -> None:
    settings = Settings(
        llama_cpp_url="http://llama.test/v1/..",
        llama_cpp_timeout_seconds=45,
    )

    provider = create_generation_provider(
        settings,
        provider="llama-cpp",
        model="local-model",
        groq_rate_limiter=create_groq_rate_limiter(settings),
    )

    assert isinstance(provider, LlamaCppGenerationProvider)
    assert provider._base_url == "http://llama.test/v1/.."
    assert provider._model == "local-model"
