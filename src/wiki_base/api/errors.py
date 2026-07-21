from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse


@dataclass(slots=True)
class ServiceError(Exception):
    code: str
    message: str
    status_code: int


async def service_error_handler(_request: Request, error: ServiceError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={"error": {"code": error.code, "message": error.message}},
    )
