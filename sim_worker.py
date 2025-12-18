import os
import time
import uuid
import json
from typing import Any, Dict, Optional, Tuple

import aiohttp
import asyncio

from sim_db import get_pool, init_db


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)) or default)
    except Exception:
        return default


def _env_flag(name: str, default: str = "0") -> bool:
    v = os.environ.get(name)
    if v is None:
        v = default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


SIM_WORKER_ENABLED = _env_flag("SIM_WORKER_ENABLED", "1")
SIM_WORKER_INTERVAL_MS = _env_int("SIM_WORKER_INTERVAL_MS", 1000)
PRICE_CACHE_TTL_SEC = _env_float("SIM_PRICE_CACHE_TTL_SEC", 0.5)

TAKER_FEE_RATE = 0.0004
MAINT_MARGIN_RATE = 0.005


class PriceCache:
    def __init__(self):
        self._cache: Dict[str, Tuple[float, float]] = {}

    def get(self, symbol: str) -> Optional[float]:
        sym = str(symbol).upper()
        item = self._cache.get(sym)
        if not item:
            return None
        ts, price = item
        if time.time() - ts > PRICE_CACHE_TTL_SEC:
            return None
        return price

    def set(self, symbol: str, price: float) -> None:
        self._cache[str(symbol).upper()] = (time.time(), float(price))


PRICE_CACHE = PriceCache()


async def binance_futures_price(session: aiohttp.ClientSession, symbol: str) -> Optional[float]:
    sym = str(symbol).upper()
    cached = PRICE_CACHE.get(sym)
    if cached is not None:
        return cached

    try:
        async with session.get(
            "https://fapi.binance.com/fapi/v1/ticker/price",
            params={"symbol": sym},
        ) as resp:
            if resp.status < 200 or resp.status >= 300:
                return None
            data = await resp.json(content_type=None)
            p = float(data.get("price", 0.0) or 0.0)
            if p <= 0:
                return None
            PRICE_CACHE.set(sym, p)
            return p
    except Exception:
        return None


def should_close(
    *,
    side: str,
    price: float,
    tp: Optional[float],
    sl: Optional[float],
    liq: Optional[float],
) -> Optional[str]:
    s = str(side).lower()
    p = float(price)

    if s == "long":
        if tp is not None and p >= float(tp):
            return "tp"
        if sl is not None and p <= float(sl):
            return "sl"
        if liq is not None and p <= float(liq):
            return "liq"
        return None

    # short
    if tp is not None and p <= float(tp):
        return "tp"
    if sl is not None and p >= float(sl):
        return "sl"
    if liq is not None and p >= float(liq):
        return "liq"
    return None


def pnl_for_close(*, side: str, entry: float, price: float, qty: float) -> float:
    s = str(side).lower()
    if s == "short":
        return (float(entry) - float(price)) * float(qty)
    return (float(price) - float(entry)) * float(qty)


async def close_position_if_triggered(conn, *, pos: Dict[str, Any], price: float, reason: str) -> bool:
    pos_id = pos["id"]
    anon_id = pos["anon_id"]

    # Re-lock row to avoid races.
    row = await conn.fetchrow(
        """
        SELECT id, anon_id, symbol, side, entry_price, qty, margin_usd, margin_mode
        FROM sim_positions
        WHERE id=$1 AND is_open=TRUE
        FOR UPDATE
        """,
        pos_id,
    )
    if row is None:
        return False

    entry = float(row["entry_price"])
    qty = float(row["qty"])
    margin = float(row["margin_usd"])
    symbol = str(row["symbol"]).upper()
    side = str(row["side"]).lower()
    margin_mode = str(row["margin_mode"] or "isolated").lower()

    pnl = pnl_for_close(side=side, entry=entry, price=price, qty=qty)
    exit_fee = (float(price) * float(qty)) * TAKER_FEE_RATE
    raw_credit = float(margin) + float(pnl) - float(exit_fee)

    acct = await conn.fetchrow(
        "SELECT anon_id, balance_usd FROM sim_accounts WHERE anon_id=$1 FOR UPDATE",
        anon_id,
    )
    if acct is None:
        return False

    bal = float(acct["balance_usd"])
    if margin_mode == "cross":
        credit = raw_credit
        new_bal = max(0.0, bal + credit)
    else:
        credit = max(0.0, raw_credit)
        new_bal = bal + credit

    await conn.execute(
        "UPDATE sim_accounts SET balance_usd=$2, updated_at=now() WHERE anon_id=$1",
        anon_id,
        float(new_bal),
    )

    await conn.execute(
        """
        UPDATE sim_positions
        SET is_open=FALSE, closed_at=now(), close_price=$3, close_reason=$4
        WHERE id=$1 AND anon_id=$2
        """,
        pos_id,
        anon_id,
        float(price),
        str(reason),
    )

    await conn.execute(
        "INSERT INTO sim_events(id, anon_id, position_id, type, payload) VALUES ($1,$2,$3,$4,$5::jsonb)",
        uuid.uuid4(),
        anon_id,
        pos_id,
        "position_closed",
        json.dumps(
            {
                "symbol": symbol,
                "side": side,
                "close_price": float(price),
                "pnl": float(pnl),
                "reason": str(reason),
                "credit": float(credit),
                "exit_fee_usd": float(exit_fee),
                "fee_rate": float(TAKER_FEE_RATE),
                "ts_ms": int(time.time() * 1000),
            },
            ensure_ascii=False,
        ),
    )

    return True


async def tick_once(session: aiohttp.ClientSession) -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, anon_id, symbol, side, entry_price, qty, margin_usd, margin_mode, tp_price, sl_price, liq_price
            FROM sim_positions
            WHERE is_open=TRUE
            ORDER BY opened_at ASC
            LIMIT 500
            """
        )

        if not rows:
            return

        prices: Dict[str, float] = {}
        for r in rows:
            sym = str(r["symbol"]).upper()
            if sym in prices:
                continue
            p = await binance_futures_price(session, sym)
            if p is not None and p > 0:
                prices[sym] = float(p)

        if not prices:
            return

        by_anon: Dict[str, list] = {}
        for r in rows:
            anon = str(r["anon_id"])
            by_anon.setdefault(anon, []).append(r)

        balances: Dict[str, float] = {}
        for anon_id in by_anon.keys():
            acct = await conn.fetchrow("SELECT balance_usd FROM sim_accounts WHERE anon_id=$1", anon_id)
            if acct is None:
                balances[anon_id] = 0.0
            else:
                balances[anon_id] = float(acct["balance_usd"])

        for anon_id, plist in by_anon.items():
            cross_positions = [p for p in plist if str(p["margin_mode"] or "isolated").lower() == "cross"]
            cross_total_notional = 0.0
            for p in cross_positions:
                sym = str(p["symbol"]).upper()
                mark = prices.get(sym)
                if mark is None:
                    continue
                q = float(p["qty"])
                cross_total_notional += max(0.0, float(mark) * q)

            for r in plist:
                symbol = str(r["symbol"]).upper()
                price = prices.get(symbol)
                if price is None:
                    continue

                tp = float(r["tp_price"]) if r["tp_price"] is not None else None
                sl = float(r["sl_price"]) if r["sl_price"] is not None else None
                liq = float(r["liq_price"]) if r["liq_price"] is not None else None

                margin_mode = str(r["margin_mode"] or "isolated").lower()

                liq_for_close = liq
                if margin_mode == "cross":
                    liq_for_close = None

                reason = should_close(
                    side=str(r["side"]),
                    price=float(price),
                    tp=tp,
                    sl=sl,
                    liq=liq_for_close,
                )

                if reason is None and margin_mode == "cross":
                    q = float(r["qty"])
                    entry = float(r["entry_price"])
                    margin = float(r["margin_usd"])
                    notional = max(0.0, float(price) * q)
                    mm = notional * MAINT_MARGIN_RATE
                    pnl = pnl_for_close(side=str(r["side"]), entry=entry, price=float(price), qty=q)
                    share = 0.0
                    if cross_total_notional > 0:
                        share = max(0.0, balances.get(anon_id, 0.0)) * (notional / cross_total_notional)
                    effective_margin = margin + share
                    if (effective_margin + pnl) <= mm:
                        reason = "liq"

                if reason is None:
                    continue

                async with conn.transaction():
                    await close_position_if_triggered(conn, pos=dict(r), price=float(price), reason=reason)


async def main() -> None:
    if not SIM_WORKER_ENABLED:
        return

    if not str(os.environ.get("DATABASE_URL", "")).strip():
        return

    await init_db()

    timeout = aiohttp.ClientTimeout(total=4)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                await tick_once(session)
            except Exception:
                pass

            await asyncio.sleep(max(0.2, SIM_WORKER_INTERVAL_MS / 1000.0))


if __name__ == "__main__":
    asyncio.run(main())
