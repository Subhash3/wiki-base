from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="WIKI_BASE_",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "DEBUG"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)

    database_url: str = "postgresql://wiki_base:wiki_base@localhost:5432/wiki_base"
    database_min_pool_size: int = Field(default=1, ge=1)
    database_max_pool_size: int = Field(default=10, ge=1)

    staging_directory: Path = Path(".wiki-base-staging")
    max_documents_per_request: int = Field(default=100, ge=1)
    max_document_size_bytes: int = Field(default=50 * 1024 * 1024, ge=1)
    max_request_size_bytes: int = Field(default=500 * 1024 * 1024, ge=1)

    embedding_model: str = "bge-m3:latest"
    embedding_dimensions: int = Field(default=1024, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1)
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = Field(default=120, gt=0)
    extraction_model: str = "gemma3:270m"
    answer_generation_model: str = "gemma3:270m"
    ocr_languages: str = "english"
    ocr_force_full_page: bool = False
    chunk_max_tokens: int = Field(default=700, ge=50)
    chunk_tokenizer_model: str = "BAAI/bge-m3"
    worker_poll_interval_seconds: float = Field(default=1, gt=0)
    graph_index_version: str = "1"
    graph_entity_similarity_threshold: float = Field(default=0.75, ge=-1, le=1)
    graph_relationship_similarity_threshold: float = Field(default=0.6, ge=-1, le=1)
    graph_entity_max_links: int = Field(default=1, ge=1)
    graph_entity_embedding_batch_size: int = Field(default=128, ge=1)
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def parsed_ocr_languages(self) -> list[str]:
        """Return configured RapidOCR languages."""

        return [language.strip() for language in self.ocr_languages.split(",") if language.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
