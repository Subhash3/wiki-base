from typing import Literal

from llm_providers.generation.groq import (
    GroqGenerationProvider,
    GroqRateLimiter,
)
from llm_providers.generation.ollama import OllamaGenerationProvider

from wiki_base.config.settings import Settings

GenerationProviderName = Literal["ollama", "groq"]
GenerationProvider = OllamaGenerationProvider | GroqGenerationProvider


def create_groq_rate_limiter(settings: Settings) -> GroqRateLimiter:
    """Create the shared process-local Groq free-tier limiter."""

    return GroqRateLimiter(
        requests_per_minute=settings.groq_requests_per_minute,
        tokens_per_minute=settings.groq_tokens_per_minute,
        requests_per_day=settings.groq_requests_per_day,
        tokens_per_day=settings.groq_tokens_per_day,
    )


def create_generation_provider(
    settings: Settings,
    *,
    provider: GenerationProviderName,
    model: str,
    groq_rate_limiter: GroqRateLimiter,
) -> GenerationProvider:
    """Create an Ollama or Groq generation provider."""

    if provider == "ollama":
        return OllamaGenerationProvider(
            base_url=settings.ollama_url,
            model=model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )

    if settings.groq_api_key is None:
        raise ValueError("WIKI_BASE_GROQ_API_KEY is required for the Groq provider")
    return GroqGenerationProvider(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url=settings.groq_url,
        model=model,
        timeout_seconds=settings.groq_timeout_seconds,
        rate_limiter=groq_rate_limiter,
        max_retries=settings.groq_max_retries,
        reasoning_effort=settings.groq_reasoning_effort,
    )
