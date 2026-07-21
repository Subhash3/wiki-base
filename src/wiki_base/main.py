from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wiki_base.api.errors import ServiceError, service_error_handler
from wiki_base.api.routes import router
from wiki_base.config.logging import configure_logging
from wiki_base.config.settings import get_settings
from wiki_base.database.connection import Database
from wiki_base.providers.embeddings.ollama import OllamaEmbeddingProvider
from wiki_base.providers.generation.ollama import OllamaGenerationProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    database = Database(settings.database_url)
    await database.connect(
        min_size=settings.database_min_pool_size,
        max_size=settings.database_max_pool_size,
    )
    app.state.database = database
    embedding_provider = OllamaEmbeddingProvider(
        base_url=settings.ollama_url,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    app.state.embedding_provider = embedding_provider
    generation_provider = OllamaGenerationProvider(
        base_url=settings.ollama_url,
        model=settings.generation_model,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    app.state.generation_provider = generation_provider

    try:
        yield
    finally:
        await generation_provider.close()
        await embedding_provider.close()
        await database.disconnect()


def create_app(*, use_lifespan: bool = True) -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Wiki Base",
        version="0.1.0",
        description="Create immutable knowledge bases and query them with citations.",
        lifespan=lifespan if use_lifespan else None,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.add_exception_handler(ServiceError, service_error_handler)
    app.include_router(router)
    return app


app = create_app()


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "wiki_base.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
    )
