from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, File, Form, UploadFile, status

from wiki_base.api.dependencies import (
    DatabaseDependency,
    SettingsDependency,
    WikiBaseServiceDependency,
)
from wiki_base.api.errors import ServiceError
from wiki_base.database.queries.documents import list_wiki_base_documents
from wiki_base.graph_tools import load_merged_wiki_base_graph
from wiki_base.schemas.graphs import (
    GraphNodeDocumentResponse,
    GraphNodeFactResponse,
    GraphNodeFactsResponse,
    GraphNodeInfoResponse,
)
from wiki_base.schemas.wiki_bases import (
    WikiBaseQueuedResponse,
    WikiBaseStatusResponse,
    WikiBaseSummaryResponse,
)

router = APIRouter(prefix="/wiki-bases", tags=["wiki bases"])


@router.post("", response_model=WikiBaseQueuedResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_wiki_base(
    service: WikiBaseServiceDependency,
    name: Annotated[str, Form(min_length=1, max_length=200)],
    documents: Annotated[list[UploadFile], File()],
) -> WikiBaseQueuedResponse:
    result = await service.create(name=name, uploads=documents)
    return WikiBaseQueuedResponse.model_validate(result)


@router.get("", response_model=list[WikiBaseSummaryResponse])
async def list_wiki_bases(
    service: WikiBaseServiceDependency,
) -> list[WikiBaseSummaryResponse]:
    results = await service.list()
    return [WikiBaseSummaryResponse.model_validate(result) for result in results]


@router.get("/{wiki_base_id}/status", response_model=WikiBaseStatusResponse)
async def get_wiki_base_status(
    wiki_base_id: UUID,
    service: WikiBaseServiceDependency,
) -> WikiBaseStatusResponse:
    result = await service.get_status(wiki_base_id)
    return WikiBaseStatusResponse.model_validate(result)


@router.get("/{wiki_base_id}/graph", response_model=dict[str, Any])
async def get_wiki_base_graph(
    wiki_base_id: UUID,
    database: DatabaseDependency,
    settings: SettingsDependency,
) -> dict[str, Any]:
    """Return one canonical graph merged from all ready documents."""

    try:
        graph = await load_merged_wiki_base_graph(
            database,
            wiki_base_id=wiki_base_id,
            embedding_model=settings.embedding_model,
            synonym_similarity_threshold=settings.graph_synonym_similarity_threshold,
        )
    except ValueError as error:
        raise ServiceError(
            code="graph_not_ready",
            message=str(error),
            status_code=status.HTTP_404_NOT_FOUND,
        ) from error
    return graph.to_dict()


@router.get("/{wiki_base_id}/graph/nodes/{node_id}", response_model=GraphNodeInfoResponse)
async def get_wiki_base_graph_node(
    wiki_base_id: UUID,
    node_id: UUID,
    database: DatabaseDependency,
    settings: SettingsDependency,
) -> GraphNodeInfoResponse:
    """Return connectivity and source-document metadata for one graph node."""

    graph = await _load_graph(database, settings, wiki_base_id)
    name = _require_graph_node(graph, node_id)
    provenance = graph.provenance_for_node(name)
    chunk_counts: dict[UUID, int] = {}
    for source in provenance:
        chunk_counts[source.document_id] = chunk_counts.get(source.document_id, 0) + 1
    async with database.connection() as connection:
        documents = await list_wiki_base_documents(connection, wiki_base_id)
    documents_by_id = {document.id: document for document in documents}
    node_documents = [
        GraphNodeDocumentResponse(
            id=document_id,
            name=documents_by_id[document_id].name,
            chunk_count=chunk_count,
        )
        for document_id, chunk_count in sorted(chunk_counts.items(), key=lambda item: item[0].int)
        if document_id in documents_by_id
    ]
    facts = [edge for edge in graph.edges() if name in {edge.subject, edge.object}]
    synonyms = [edge for edge in graph.synonyms() if name in {edge.first, edge.second}]
    return GraphNodeInfoResponse(
        id=node_id,
        name=name,
        link_count=len(facts) + len(synonyms),
        fact_count=len(facts),
        synonym_count=len(synonyms),
        document_count=len(node_documents),
        documents=node_documents,
    )


@router.get(
    "/{wiki_base_id}/graph/nodes/{node_id}/facts",
    response_model=GraphNodeFactsResponse,
)
async def get_wiki_base_graph_node_facts(
    wiki_base_id: UUID,
    node_id: UUID,
    database: DatabaseDependency,
    settings: SettingsDependency,
) -> GraphNodeFactsResponse:
    """Return direct canonical facts involving one graph node."""

    graph = await _load_graph(database, settings, wiki_base_id)
    name = _require_graph_node(graph, node_id)
    async with database.connection() as connection:
        documents = await list_wiki_base_documents(connection, wiki_base_id)
    document_names = {document.id: document.name for document in documents}
    facts = [
        GraphNodeFactResponse(
            subject_id=UUID(graph.node_id(edge.subject)),
            subject=edge.subject,
            relation=edge.relation,
            object_id=UUID(graph.node_id(edge.object)),
            object=edge.object,
            document_names=sorted(
                {
                    document_names[source.document_id]
                    for source in edge.provenance
                    if source.document_id in document_names
                }
            ),
            evidence_count=len(edge.provenance),
        )
        for edge in sorted(
            (edge for edge in graph.edges() if name in {edge.subject, edge.object}),
            key=lambda edge: (edge.subject, edge.relation, edge.object),
        )
    ]
    return GraphNodeFactsResponse(id=node_id, name=name, facts=facts)


async def _load_graph(database, settings, wiki_base_id: UUID):
    try:
        return await load_merged_wiki_base_graph(
            database,
            wiki_base_id=wiki_base_id,
            embedding_model=settings.embedding_model,
            synonym_similarity_threshold=settings.graph_synonym_similarity_threshold,
        )
    except ValueError as error:
        raise ServiceError(
            code="graph_not_ready",
            message=str(error),
            status_code=status.HTTP_404_NOT_FOUND,
        ) from error


def _require_graph_node(graph, node_id: UUID) -> str:
    name = graph.node_name(str(node_id))
    if name is None:
        raise ServiceError(
            code="graph_node_not_found",
            message=f"Graph node {node_id} was not found.",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return name
