from fastapi import APIRouter

from wiki_base.api.routes.capabilities import router as capabilities_router
from wiki_base.api.routes.health import router as health_router
from wiki_base.api.routes.queries import router as queries_router
from wiki_base.api.routes.query_chunks import router as query_chunks_router
from wiki_base.api.routes.wiki_bases import router as wiki_bases_router

router = APIRouter()
router.include_router(health_router)
router.include_router(capabilities_router)
router.include_router(wiki_bases_router)
router.include_router(query_chunks_router)
router.include_router(queries_router)
