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
    log_level: str = "INFO"
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
    generation_model: str = "qwen3.5:0.8b"
    chunk_max_tokens: int = Field(default=700, ge=50)
    worker_poll_interval_seconds: float = Field(default=1, gt=0)
    graph_directory: Path = Path(".wiki-base-graphs")
    graph_index_version: str = "1"
    cors_origins: str = "http://localhost:8080,http://127.0.0.1:8080"

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
