import os
import uuid
import time
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sim_db import get_pool


router = APIRouter(prefix="/simserver")


def _enabled() -> bool:
    return str(os.environ.get("SIM_SERVER_ENABLED", "1")).strip().lower() in {"1", "true", "yes", "on"}


async def _binance_futures_price(symbol: str) -> Optional[float]:
    sym = str(symbol).upper()
    try:
        timeout = aiohttp.ClientTimeout(total=4)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                "https://fapi.binance.com/fapi/v1/ticker/price",
                params={"symbol": sym},
            ) as resp:
                if resp.status < 200 or resp.status >= 300:
                    return None
                data = await resp.json(content_type=None)
        p = float(data.get("price", 0.0) or 0.0)
        return p if p > 0 else None
    except Exception:
        return None


def _liq_price(entry_price: float, leverage: int, side: str) -> float:
    lev = max(1, int(leverage or 1))
    s = str(side).lower()
    if s == "short":
        return float(entry_price) * (1.0 + 1.0 / lev)
    return float(entry_price) * (1.0 - 1.0 / lev)


class EnsureAccountIn(BaseModel):
    anon_id: str
    initial_balance_usd: float = 10_000.0


class OpenPositionIn(BaseModel):
    anon_id: str
    symbol: str
    side: str
    leverage: int
    margin_usd: float
    entry_price: Optional[float] = None
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


class UpdateRiskIn(BaseModel):
    anon_id: str
    position_id: str
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None


class ClosePositionIn(BaseModel):
    anon_id: str
    position_id: str


@router.get("/health")
async def simserver_health():
    return {"ok": True, "enabled": _enabled()}


@router.post("/account/ensure")
async def ensure_account(payload: EnsureAccountIn):
    if not _enabled():
        return JSONResponse({"error": "disabled"}, status_code=503)

    anon = str(payload.anon_id or "").strip()
    if not anon:
        return JSONResponse({"error": "missing_anon_id"}, status_code=400)

    try:
        pool = await get_pool()
    except Exception:
        return JSONResponse({"error": "db_unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow("SELECT anon_id, balance_usd FROM sim_accounts WHERE anon_id=$1", anon)
            if row is None:
                bal = float(payload.initial_balance_usd or 0.0)
                if bal <= 0:
                    bal = 10_000.0
                await conn.execute(
                    "INSERT INTO sim_accounts(anon_id,balance_usd) VALUES($1,$2)",
                    anon,
                    bal,
                )
                return {"anon_id": anon, "balance_usd": bal}
            return {"anon_id": str(row["anon_id"]), "balance_usd": float(row["balance_usd"])}


@router.get("/state")
async def get_state(anon_id: str = ""):
    if not _enabled():
        return JSONResponse({"error": "disabled"}, status_code=503)

    anon = str(anon_id or "").strip()
    if not anon:
        return JSONResponse({"error": "missing_anon_id"}, status_code=400)

    try:
        pool = await get_pool()
    except Exception:
        return JSONResponse({"error": "db_unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        acct = await conn.fetchrow("SELECT anon_id, balance_usd FROM sim_accounts WHERE anon_id=$1", anon)
        if acct is None:
            return JSONResponse({"error": "account_not_found"}, status_code=404)

        rows = await conn.fetch(
            """
            SELECT id, symbol, side, leverage, entry_price, qty, margin_usd, tp_price, sl_price, liq_price,
                   opened_at, closed_at, is_open, close_price, close_reason
            FROM sim_positions
            WHERE anon_id=$1
            ORDER BY opened_at DESC
            LIMIT 200
            """,
            anon,
        )

    positions: List[Dict[str, Any]] = []
    for r in rows:
        positions.append({
            "id": str(r["id"]),
            "symbol": r["symbol"],
            "side": r["side"],
            "leverage": int(r["leverage"]),
            "entry_price": float(r["entry_price"]),
            "qty": float(r["qty"]),
            "margin_usd": float(r["margin_usd"]),
            "tp_price": float(r["tp_price"]) if r["tp_price"] is not None else None,
            "sl_price": float(r["sl_price"]) if r["sl_price"] is not None else None,
            "liq_price": float(r["liq_price"]) if r["liq_price"] is not None else None,
            "is_open": bool(r["is_open"]),
            "opened_at": r["opened_at"].isoformat() if r["opened_at"] is not None else None,
            "closed_at": r["closed_at"].isoformat() if r["closed_at"] is not None else None,
            "close_price": float(r["close_price"]) if r["close_price"] is not None else None,
            "close_reason": r["close_reason"],
        })

    return {
        "account": {"anon_id": str(acct["anon_id"]), "balance_usd": float(acct["balance_usd"])},
        "positions": positions,
    }


@router.post("/position/open")
async def open_position(payload: OpenPositionIn = Body(...)):
    if not _enabled():
        return JSONResponse({"error": "disabled"}, status_code=503)

    anon = str(payload.anon_id or "").strip()
    if not anon:
        return JSONResponse({"error": "missing_anon_id"}, status_code=400)

    symbol = str(payload.symbol or "").strip().upper()
    if not symbol:
        return JSONResponse({"error": "missing_symbol"}, status_code=400)

    side = str(payload.side or "").strip().lower()
    if side not in {"long", "short"}:
        return JSONResponse({"error": "invalid_side"}, status_code=400)

    leverage = max(1, min(200, int(payload.leverage or 1)))
    margin = float(payload.margin_usd or 0.0)
    if margin <= 0:
        return JSONResponse({"error": "invalid_margin"}, status_code=400)

    entry = float(payload.entry_price) if payload.entry_price is not None else None
    if entry is None or entry <= 0:
        entry = await _binance_futures_price(symbol)
    if entry is None or entry <= 0:
        return JSONResponse({"error": "price_unavailable"}, status_code=503)

    notional = margin * leverage
    qty = notional / entry
    liq = _liq_price(entry, leverage, side)

    pid = uuid.uuid4()
    try:
        pool = await get_pool()
    except Exception:
        return JSONResponse({"error": "db_unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        async with conn.transaction():
            acct = await conn.fetchrow("SELECT balance_usd FROM sim_accounts WHERE anon_id=$1 FOR UPDATE", anon)
            if acct is None:
                return JSONResponse({"error": "account_not_found"}, status_code=404)

            bal = float(acct["balance_usd"])
            if bal + 1e-9 < margin:
                return JSONResponse({"error": "insufficient_balance", "balance_usd": bal}, status_code=400)

            new_bal = bal - margin
            await conn.execute(
                "UPDATE sim_accounts SET balance_usd=$2, updated_at=now() WHERE anon_id=$1",
                anon,
                new_bal,
            )

            await conn.execute(
                """
                INSERT INTO sim_positions(
                  id, anon_id, symbol, side, leverage, entry_price, qty, margin_usd,
                  tp_price, sl_price, liq_price
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                """,
                pid,
                anon,
                symbol,
                side,
                leverage,
                float(entry),
                float(qty),
                float(margin),
                float(payload.tp_price) if payload.tp_price is not None else None,
                float(payload.sl_price) if payload.sl_price is not None else None,
                float(liq),
            )

            await conn.execute(
                "INSERT INTO sim_events(id, anon_id, position_id, type, payload) VALUES ($1,$2,$3,$4,$5)",
                uuid.uuid4(),
                anon,
                pid,
                "position_opened",
                {"symbol": symbol, "side": side, "entry_price": float(entry), "margin_usd": margin, "leverage": leverage},
            )

    return {"ok": True, "position_id": str(pid), "entry_price": float(entry), "liq_price": float(liq)}


@router.post("/position/risk")
async def update_risk(payload: UpdateRiskIn = Body(...)):
    if not _enabled():
        return JSONResponse({"error": "disabled"}, status_code=503)

    anon = str(payload.anon_id or "").strip()
    if not anon:
        return JSONResponse({"error": "missing_anon_id"}, status_code=400)

    try:
        pid = uuid.UUID(str(payload.position_id))
    except Exception:
        return JSONResponse({"error": "invalid_position_id"}, status_code=400)

    try:
        pool = await get_pool()
    except Exception:
        return JSONResponse({"error": "db_unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                "SELECT id FROM sim_positions WHERE id=$1 AND anon_id=$2 AND is_open=TRUE FOR UPDATE",
                pid,
                anon,
            )
            if row is None:
                return JSONResponse({"error": "position_not_found"}, status_code=404)

            await conn.execute(
                "UPDATE sim_positions SET tp_price=$3, sl_price=$4 WHERE id=$1 AND anon_id=$2",
                pid,
                anon,
                float(payload.tp_price) if payload.tp_price is not None else None,
                float(payload.sl_price) if payload.sl_price is not None else None,
            )

            await conn.execute(
                "INSERT INTO sim_events(id, anon_id, position_id, type, payload) VALUES ($1,$2,$3,$4,$5)",
                uuid.uuid4(),
                anon,
                pid,
                "risk_updated",
                {"tp_price": payload.tp_price, "sl_price": payload.sl_price},
            )

    return {"ok": True}


@router.post("/position/close")
async def close_position(payload: ClosePositionIn = Body(...)):
    if not _enabled():
        return JSONResponse({"error": "disabled"}, status_code=503)

    anon = str(payload.anon_id or "").strip()
    if not anon:
        return JSONResponse({"error": "missing_anon_id"}, status_code=400)

    try:
        pid = uuid.UUID(str(payload.position_id))
    except Exception:
        return JSONResponse({"error": "invalid_position_id"}, status_code=400)

    try:
        pool = await get_pool()
    except Exception:
        return JSONResponse({"error": "db_unavailable"}, status_code=503)
    async with pool.acquire() as conn:
        async with conn.transaction():
            pos = await conn.fetchrow(
                """
                SELECT id, symbol, side, entry_price, qty, margin_usd
                FROM sim_positions
                WHERE id=$1 AND anon_id=$2 AND is_open=TRUE
                FOR UPDATE
                """,
                pid,
                anon,
            )
            if pos is None:
                return JSONResponse({"error": "position_not_found"}, status_code=404)

            price = await _binance_futures_price(str(pos["symbol"]))
            if price is None or price <= 0:
                return JSONResponse({"error": "price_unavailable"}, status_code=503)

            entry = float(pos["entry_price"])
            qty = float(pos["qty"])
            margin = float(pos["margin_usd"])
            side = str(pos["side"]).lower()
            pnl = (price - entry) * qty if side == "long" else (entry - price) * qty

            acct = await conn.fetchrow("SELECT balance_usd FROM sim_accounts WHERE anon_id=$1 FOR UPDATE", anon)
            if acct is None:
                return JSONResponse({"error": "account_not_found"}, status_code=404)

            bal = float(acct["balance_usd"])
            new_bal = bal + margin + pnl
            await conn.execute(
                "UPDATE sim_accounts SET balance_usd=$2, updated_at=now() WHERE anon_id=$1",
                anon,
                float(new_bal),
            )

            await conn.execute(
                """
                UPDATE sim_positions
                SET is_open=FALSE, closed_at=now(), close_price=$3, close_reason=$4
                WHERE id=$1 AND anon_id=$2
                """,
                pid,
                anon,
                float(price),
                "manual_close",
            )

            await conn.execute(
                "INSERT INTO sim_events(id, anon_id, position_id, type, payload) VALUES ($1,$2,$3,$4,$5)",
                uuid.uuid4(),
                anon,
                pid,
                "position_closed",
                {"close_price": float(price), "pnl": float(pnl), "reason": "manual_close", "ts_ms": int(time.time() * 1000)},
            )

    return {"ok": True}
