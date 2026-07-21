import asyncio
from pathlib import Path

import asyncpg

from wiki_base.config.settings import get_settings


def load_schema() -> str:
    schema_path = Path(__file__).resolve().parents[3] / "sql" / "schema.sql"
    return schema_path.read_text(encoding="utf-8")


async def initialize_database() -> None:
    settings = get_settings()
    schema = load_schema()
    connection = await asyncpg.connect(settings.database_url)
    try:
        await connection.execute(schema)
    finally:
        await connection.close()


def run() -> None:
    asyncio.run(initialize_database())
