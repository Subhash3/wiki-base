from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from wiki_base.api.dependencies import DatabaseDependency

router = APIRouter(tags=["operations"])


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    database: Literal["up", "down"]


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def readiness(database: DatabaseDependency, response: Response) -> ReadinessResponse:
    database_is_ready = await database.is_ready()
    if not database_is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="not_ready", database="down")
    return ReadinessResponse(status="ready", database="up")
