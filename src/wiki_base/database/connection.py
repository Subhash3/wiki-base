import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

import asyncpg
from asyncpg import Connection, Pool
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)


async def _initialize_connection(connection: Connection) -> None:
    await register_vector(connection)


class Database:
    def __init__(self, url: str) -> None:
        self._url = url
        self._pool: Pool | None = None

    async def connect(self, *, min_size: int, max_size: int) -> None:
        logger.info("connecting to db")
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            self._url,
            min_size=min_size,
            max_size=max_size,
            init=_initialize_connection,
        )

    async def disconnect(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[Connection]:
        if self._pool is None:
            raise RuntimeError("Database is not connected")
        async with self._pool.acquire() as connection:
            yield cast(Connection, connection)

    async def is_ready(self) -> bool:
        if self._pool is None:
            return False
        try:
            async with self._pool.acquire() as connection:
                return await connection.fetchval("SELECT TRUE") is True
        except (asyncpg.PostgresError, OSError):
            logger.exception("Database readiness check failed")
            return False
