import asyncio
import os
from typing import Iterable, Optional

import asyncpg
from loguru import logger


TIMESCALE_URL = os.getenv("TIMESCALE_URL", "postgres://user:password@localhost:5432/apex")
CANDIDATE_TABLES = ("market_data", "raw_trades")


async def purge_stale_market_data(
    pg_pool: Optional[asyncpg.Pool] = None,
    *,
    candidate_tables: Iterable[str] = CANDIDATE_TABLES,
) -> dict:
    """
    Performs a full market data reset by truncating the historical trade tables.
    """
    created_pool = None
    summary = {
        "operation": "TRUNCATE",
        "truncated_rows": 0,
        "tables": {},
    }

    try:
        if pg_pool is None:
            created_pool = await asyncpg.create_pool(dsn=TIMESCALE_URL)
            pg_pool = created_pool

        async with pg_pool.acquire() as conn:
            for table_name in candidate_tables:
                exists = await conn.fetchval("SELECT to_regclass($1)", f"public.{table_name}")
                if exists is None:
                    summary["tables"][table_name] = 0
                    continue

                row_count = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {table_name}"
                )
                await conn.execute(f"TRUNCATE TABLE {table_name}")

                row_count = int(row_count or 0)
                summary["tables"][table_name] = row_count
                summary["truncated_rows"] += row_count

        logger.info(
            "TimescaleDB nuclear reset complete | truncated_rows={} | tables={}",
            summary["truncated_rows"],
            summary["tables"],
        )
        return summary
    finally:
        if created_pool is not None:
            await created_pool.close()


async def main():
    try:
        await purge_stale_market_data()
    except Exception as exc:
        logger.error(f"Failed to purge stale market data: {exc}")
        raise


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
