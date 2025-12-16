import os
import asyncio
from typing import Optional

import asyncpg

_POOL: Optional[asyncpg.pool.Pool] = None
_POOL_LOCK = asyncio.Lock()


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    return url


async def get_pool() -> asyncpg.pool.Pool:
    global _POOL
    if _POOL is not None:
        return _POOL
    async with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        url = _database_url()
        if not url:
            raise RuntimeError("DATABASE_URL not set")
        _POOL = await asyncpg.create_pool(
            dsn=url,
            min_size=int(os.environ.get("PGPOOL_MIN", "1") or 1),
            max_size=int(os.environ.get("PGPOOL_MAX", "5") or 5),
            command_timeout=float(os.environ.get("PG_COMMAND_TIMEOUT_SEC", "10") or 10),
        )
        return _POOL


async def close_pool() -> None:
    global _POOL
    if _POOL is None:
        return
    await _POOL.close()
    _POOL = None


async def init_db() -> None:
    url = _database_url()
    if not url:
        return

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sim_accounts (
              anon_id TEXT PRIMARY KEY,
              balance_usd DOUBLE PRECISION NOT NULL,
              created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE TABLE IF NOT EXISTS sim_positions (
              id UUID PRIMARY KEY,
              anon_id TEXT NOT NULL REFERENCES sim_accounts(anon_id) ON DELETE CASCADE,
              symbol TEXT NOT NULL,
              side TEXT NOT NULL,
              leverage INTEGER NOT NULL,
              entry_price DOUBLE PRECISION NOT NULL,
              qty DOUBLE PRECISION NOT NULL,
              margin_usd DOUBLE PRECISION NOT NULL,
              tp_price DOUBLE PRECISION,
              sl_price DOUBLE PRECISION,
              liq_price DOUBLE PRECISION,
              is_open BOOLEAN NOT NULL DEFAULT TRUE,
              opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              closed_at TIMESTAMPTZ,
              close_price DOUBLE PRECISION,
              close_reason TEXT
            );

            CREATE INDEX IF NOT EXISTS sim_positions_open_idx ON sim_positions(is_open);
            CREATE INDEX IF NOT EXISTS sim_positions_anon_open_idx ON sim_positions(anon_id, is_open);

            CREATE TABLE IF NOT EXISTS sim_events (
              id UUID PRIMARY KEY,
              anon_id TEXT NOT NULL,
              position_id UUID,
              type TEXT NOT NULL,
              payload JSONB NOT NULL DEFAULT '{}'::jsonb,
              ts TIMESTAMPTZ NOT NULL DEFAULT now()
            );

            CREATE INDEX IF NOT EXISTS sim_events_anon_ts_idx ON sim_events(anon_id, ts DESC);
            """
        )
