from typing import Annotated

from fastapi import Depends, Request
from graph_rag import ExactEntityLinker, HippoRAGRetriever, LLMQueryEntityExtractor
from llm_providers.embeddings.base import EmbeddingProvider
from llm_providers.generation.base import GenerationProvider, StructuredGenerationProvider

from wiki_base.config.settings import Settings, get_settings
from wiki_base.database.connection import Database
from wiki_base.ingestion.staging import DocumentStaging
from wiki_base.services.query_chunks import QueryChunksService
from wiki_base.services.querying import QueryService
from wiki_base.services.wiki_bases import WikiBaseService


def get_database(request: Request) -> Database:
    return request.app.state.database


DatabaseDependency = Annotated[Database, Depends(get_database)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider


EmbeddingProviderDependency = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]


def get_generation_provider(request: Request) -> GenerationProvider:
    return request.app.state.generation_provider


GenerationProviderDependency = Annotated[GenerationProvider, Depends(get_generation_provider)]


def get_structured_generation_provider(request: Request) -> StructuredGenerationProvider:
    """Return the application's structured generation provider."""

    return request.app.state.generation_provider


StructuredGenerationProviderDependency = Annotated[
    StructuredGenerationProvider,
    Depends(get_structured_generation_provider),
]


def get_create_document_staging(settings: SettingsDependency) -> DocumentStaging:
    return DocumentStaging(
        directory=settings.staging_directory,
        max_document_size_bytes=settings.max_document_size_bytes,
        max_request_size_bytes=settings.max_request_size_bytes,
    )


DocumentStagingDependency = Annotated[DocumentStaging, Depends(get_create_document_staging)]


def get_wiki_base_service(
    database: DatabaseDependency, settings: SettingsDependency, staging: DocumentStagingDependency
) -> WikiBaseService:
    return WikiBaseService(
        database=database,
        staging=staging,
        max_documents=settings.max_documents_per_request,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )


WikiBaseServiceDependency = Annotated[
    WikiBaseService,
    Depends(get_wiki_base_service),
]


def get_query_chunks_service(
    database: DatabaseDependency,
    embeddings: EmbeddingProviderDependency,
    generation: StructuredGenerationProviderDependency,
) -> QueryChunksService:
    return QueryChunksService(
        database=database,
        embeddings=embeddings,
        graph_retriever=HippoRAGRetriever(
            entity_extractor=LLMQueryEntityExtractor(generation=generation),
            entity_linker=ExactEntityLinker(),
        ),
    )


QueryChunksServiceDependency = Annotated[
    QueryChunksService,
    Depends(get_query_chunks_service),
]


def get_query_service(
    chunks: QueryChunksServiceDependency,
    generation: GenerationProviderDependency,
) -> QueryService:
    return QueryService(chunks=chunks, generation=generation)


QueryServiceDependency = Annotated[QueryService, Depends(get_query_service)]
