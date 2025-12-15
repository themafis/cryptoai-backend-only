from __future__ import annotations

import os
import math
import time
import json
import asyncio
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from market_registry import REGISTRY
import pandas as pd
import sys
from pathlib import Path

# Ensure vendored third_party packages (e.g., pandas_ta) are importable BEFORE importing them
_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

import pandas_ta as ta
import requests
from fastapi import FastAPI, Body, WebSocket
from starlette.websockets import WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from exchange_adapters import (
    BinanceSpotAdapter,
    MEXCAdapter,
    OKXAdapter,
    CoinbaseAdapter,
    GateAdapter,
    KuCoinAdapter,
    HTXAdapter,
)

# WhaleTrack Binance+MEXC analyzer (separate module, orchestrated from main.py)
import binance_futures_analyzer
try:
    import websockets  # type: ignore
except Exception:
    websockets = None  # Optional dependency; REST fallback will remain

# Optional: ccxt as the last fallback
try:
    import ccxt
    _HAS_CCXT = True
except Exception:
    ccxt = None
    _HAS_CCXT = False

app = FastAPI(title="CryptoAI Backend v2")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def upsert_simulator_profile(
    anon_id: str,
    nickname: Optional[str],
    contact_info: Optional[str],
    equity: Optional[float] = None,
    successful_trades: Optional[int] = None,
    roe: Optional[float] = None,
) -> None:
    """Best-effort upsert of simulator profile into Supabase.

    Stores anon_id (CryptoAI anonymous UUID), optional nickname and contact
    information so that leaderboard rewards can be associated with a way to
    contact the user. This is intentionally fire-and-forget: failures are
    swallowed so they do not break the main app flow.
    """

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[SIM] Supabase env missing, skipping profile upsert")
        return

    try:
        url = f"{SUPABASE_URL}/rest/v1/simulator_profiles"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        payload: Dict[str, Any] = {"anon_id": str(anon_id)}
        if nickname is not None:
            payload["nickname"] = nickname
        if contact_info is not None:
            payload["contact_info"] = contact_info
        if equity is not None:
            try:
                payload["equity"] = float(equity)
            except Exception:
                pass
        if successful_trades is not None:
            try:
                payload["successful_trades"] = int(successful_trades)
            except Exception:
                pass
        if roe is not None:
            try:
                payload["roe"] = float(roe)
            except Exception:
                pass

        print("[SIM] Supabase profile upsert anon_id=", anon_id)
        resp = requests.post(
            url,
            params={"on_conflict": "anon_id"},
            json=payload,
            headers=headers,
            timeout=5,
        )
        if not (200 <= resp.status_code < 300) and ("successful_trades" in payload or "roe" in payload):
            payload.pop("successful_trades", None)
            payload.pop("roe", None)
            resp = requests.post(
                url,
                params={"on_conflict": "anon_id"},
                json=payload,
                headers=headers,
                timeout=5,
            )
        print("[SIM] Supabase profile status=", resp.status_code, "body=", resp.text[:200])
    except Exception:
        # Best-effort only; do not propagate errors to clients.
        return


def log_simulator_topup(anon_id: str, pack_usdt: int, amount_try: float, apple_transaction_id: Optional[str]) -> None:
    """Best-effort insert into simulator_topups table in Supabase.

    This is a V1 stub: it only records the top-up intent and does not yet
    adjust server-side simulator account balances or verify Apple receipts.
    """

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return

    try:
        url = f"{SUPABASE_URL}/rest/v1/simulator_topups"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        payload: Dict[str, Any] = {
            "anon_id": str(anon_id),
            "pack_usdt": int(pack_usdt),
            "amount_try": float(amount_try),
        }
        if apple_transaction_id:
            payload["apple_transaction_id"] = str(apple_transaction_id)

        print("[SIM] Supabase topup insert anon_id=", anon_id, "pack_usdt=", pack_usdt, "amount_try=", amount_try)
        resp = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=5,
        )
        print("[SIM] Supabase topup status=", resp.status_code, "body=", resp.text[:200])
    except Exception:
        return


class WebSocketBroadcaster:
    def __init__(self) -> None:
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    async def broadcast_json(self, payload: Dict[str, Any]) -> None:
        try:
            message = json.dumps(payload, ensure_ascii=False)
        except Exception:
            message = json.dumps({"type": "error", "message": "serialization_failed"})

        async with self._lock:
            targets = list(self._clients)
        if not targets:
            return

        coros = []
        for ws in targets:
            coros.append(self._safe_send(ws, message))
        await asyncio.gather(*coros, return_exceptions=True)

    async def _safe_send(self, ws: WebSocket, message: str) -> None:
        try:
            await ws.send_text(message)
        except Exception:
            await self.unregister(ws)


PRICE_ALERTS_BROADCASTER = WebSocketBroadcaster()
PRICE_ALERTS_EVENT_QUEUE: asyncio.Queue = asyncio.Queue(maxsize=5000)


@app.websocket("/ws/price_alerts")
async def ws_price_alerts(ws: WebSocket):
    await PRICE_ALERTS_BROADCASTER.register(ws)
    try:
        await ws.send_text(json.dumps({"type": "connected", "channel": "price_alerts"}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await PRICE_ALERTS_BROADCASTER.unregister(ws)
    except Exception:
        await PRICE_ALERTS_BROADCASTER.unregister(ws)

# -----------------------------
# Simple in-memory TTL Cache
# -----------------------------
class TTLCache:
    def __init__(self):
        self._store: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if exp < time.time():
            # expired
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: Any, ttl_sec: float):
        self._store[key] = (time.time() + ttl_sec, value)

CACHE = TTLCache()

# -----------------------------
# Utilities: Missing value policy
# -----------------------------
MISSING = "-"
MISSING_OBJ = {"error": "veri çekilemedi"}

EXCHANGE_CONTEXT_MAX_CANDLES = 30
EXCHANGE_CONTEXT_MAX_ORDERBOOK_LEVELS = 20

_SYMBOL_VARIANTS: Dict[str, List[Tuple[str, float]]] = {
    "PEPEUSDT": [("1000PEPEUSDT", 0.001)],
    "1000PEPEUSDT": [("PEPEUSDT", 1000.0)],
    "BONKUSDT": [("1000BONKUSDT", 0.001)],
    "1000BONKUSDT": [("BONKUSDT", 1000.0)],
}


def _symbol_variants(symbol: str) -> List[Tuple[str, float]]:
    s = str(symbol).upper()
    out: List[Tuple[str, float]] = [(s, 1.0)]
    for alt, scale in _SYMBOL_VARIANTS.get(s, []):
        if alt != s:
            out.append((str(alt).upper(), float(scale)))
    return out


def _scale_orderbook_levels(levels: Any, scale: float) -> List[List[float]]:
    try:
        sc = float(scale)
        if sc == 1.0:
            return [[float(p), float(q)] for p, q in (levels or [])]
        return [[float(p) * sc, float(q)] for p, q in (levels or [])]
    except Exception:
        return []


_EXCHANGE_ADAPTERS = {
    "binance": BinanceSpotAdapter(),
    "mexc": MEXCAdapter(),
    "okx": OKXAdapter(),
    "coinbase": CoinbaseAdapter(),
    "gate": GateAdapter(),
    "kucoin": KuCoinAdapter(),
    "htx": HTXAdapter(),
}


_INDICATOR_EXCHANGE_FALLBACK: List[str] = [
    "binance",
    "mexc",
    "okx",
    "coinbase",
    "gate",
    "kucoin",
    "htx",
]


_EXCHANGE_COOLDOWN_UNTIL: Dict[str, float] = {}
_EXCHANGE_SNAPSHOT_ERRORS: Dict[str, int] = {}
_EXCHANGE_SYMBOL_COOLDOWN_UNTIL: Dict[str, float] = {}
_EXCHANGE_SYMBOL_SNAPSHOT_ERRORS: Dict[str, int] = {}


def _is_exchange_on_cooldown(exchange_id: str) -> bool:
    until = _EXCHANGE_COOLDOWN_UNTIL.get(exchange_id)
    return bool(until and until > time.time())


def _set_exchange_cooldown(exchange_id: str, seconds: float) -> None:
    _EXCHANGE_COOLDOWN_UNTIL[exchange_id] = time.time() + max(1.0, seconds)


def _inc_exchange_snapshot_error(exchange_id: str) -> None:
    _EXCHANGE_SNAPSHOT_ERRORS[exchange_id] = int(_EXCHANGE_SNAPSHOT_ERRORS.get(exchange_id, 0)) + 1


def _ex_sym_key(exchange_id: str, symbol: str) -> str:
    return f"{str(exchange_id).lower()}:{str(symbol).upper()}"


def _is_exchange_symbol_on_cooldown(exchange_id: str, symbol: str) -> bool:
    until = _EXCHANGE_SYMBOL_COOLDOWN_UNTIL.get(_ex_sym_key(exchange_id, symbol))
    return bool(until and until > time.time())


def _set_exchange_symbol_cooldown(exchange_id: str, symbol: str, seconds: float) -> None:
    _EXCHANGE_SYMBOL_COOLDOWN_UNTIL[_ex_sym_key(exchange_id, symbol)] = time.time() + max(1.0, seconds)


def _inc_exchange_symbol_snapshot_error(exchange_id: str, symbol: str) -> None:
    k = _ex_sym_key(exchange_id, symbol)
    _EXCHANGE_SYMBOL_SNAPSHOT_ERRORS[k] = int(_EXCHANGE_SYMBOL_SNAPSHOT_ERRORS.get(k, 0)) + 1


def build_exchange_snapshots(symbol: str) -> Dict[str, Any]:
    cache_key = f"exchange_snapshots:{symbol.upper()}"
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    out: Dict[str, Any] = {}
    sym = symbol.upper()
    variants = _symbol_variants(sym)
    for ex_id, adapter in _EXCHANGE_ADAPTERS.items():
        if _is_exchange_on_cooldown(ex_id) or _is_exchange_symbol_on_cooldown(ex_id, sym):
            out[ex_id] = {"price": None, "volume24h": None, "bestBid": None, "bestAsk": None, "status": "cooldown"}
            continue
        t = None
        ob = None
        chosen_scale = 1.0
        for cs, scale in variants:
            try:
                t = adapter.get_ticker_24h(cs)
            except Exception:
                t = None
            try:
                ob = adapter.get_orderbook_top(cs)
            except Exception:
                ob = None
            if t is not None or ob is not None:
                chosen_scale = float(scale)
                break

        if t is None and ob is None:
            _inc_exchange_snapshot_error(ex_id)
            _inc_exchange_symbol_snapshot_error(ex_id, sym)
            _set_exchange_symbol_cooldown(ex_id, sym, 15.0)

        status = "missing" if (t is None and ob is None) else "ok"

        out[ex_id] = {
            "price": (float(t.price) * chosen_scale) if t and t.price is not None else None,
            "volume24h": float(t.volume_24h) if t and t.volume_24h is not None else None,
            "bestBid": (float(ob.best_bid) * chosen_scale) if ob and ob.best_bid is not None else None,
            "bestAsk": (float(ob.best_ask) * chosen_scale) if ob and ob.best_ask is not None else None,
            "status": status,
        }

    CACHE.set(cache_key, out, ttl_sec=5.0)
    return out


def _sum_qty(levels: Any, top_n: int = 10) -> float:
    try:
        total = 0.0
        for row in (levels or [])[: int(top_n)]:
            total += float(row[1])
        return float(total)
    except Exception:
        return 0.0


def _build_microstructure(orderbook: Any) -> Dict[str, Any]:
    try:
        bids = getattr(orderbook, "bids", None)
        asks = getattr(orderbook, "asks", None)
        if bids is None and isinstance(orderbook, dict):
            bids = orderbook.get("bids")
        if asks is None and isinstance(orderbook, dict):
            asks = orderbook.get("asks")
        bids = bids or []
        asks = asks or []

        best_bid = float(bids[0][0]) if bids else None
        best_ask = float(asks[0][0]) if asks else None

        if best_bid is None or best_ask is None:
            return {
                "bestBid": best_bid,
                "bestAsk": best_ask,
                "mid": None,
                "spread": None,
                "spreadPct": None,
                "bidQtyTop10": _sum_qty(bids, 10) if bids else 0.0,
                "askQtyTop10": _sum_qty(asks, 10) if asks else 0.0,
                "imbalanceTop10": None,
            }

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        spread_pct = (spread / mid * 100.0) if mid > 0 else None
        bid_qty_10 = _sum_qty(bids, 10)
        ask_qty_10 = _sum_qty(asks, 10)
        denom = bid_qty_10 + ask_qty_10
        imbalance_10 = ((bid_qty_10 - ask_qty_10) / denom) if denom > 0 else None

        return {
            "bestBid": best_bid,
            "bestAsk": best_ask,
            "mid": float(mid),
            "spread": float(spread),
            "spreadPct": float(spread_pct) if spread_pct is not None else None,
            "bidQtyTop10": float(bid_qty_10),
            "askQtyTop10": float(ask_qty_10),
            "imbalanceTop10": float(imbalance_10) if imbalance_10 is not None else None,
        }
    except Exception:
        return {
            "bestBid": None,
            "bestAsk": None,
            "mid": None,
            "spread": None,
            "spreadPct": None,
            "bidQtyTop10": 0.0,
            "askQtyTop10": 0.0,
            "imbalanceTop10": None,
        }


def _interval_to_seconds(interval: str) -> Optional[int]:
    try:
        s = str(interval).strip().lower()
        if not s:
            return None
        unit = s[-1]
        num = int(s[:-1])
        if num <= 0:
            return None
        if unit == "m":
            return num * 60
        if unit == "h":
            return num * 3600
        if unit == "d":
            return num * 86400
        if unit == "w":
            return num * 7 * 86400
        return None
    except Exception:
        return None


def _iso_utc_now_ms() -> str:
    try:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except Exception:
        return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _median_float(values: List[float]) -> Optional[float]:
    try:
        cleaned = [float(v) for v in (values or []) if v is not None and np.isfinite(v)]
        if not cleaned:
            return None
        return float(np.median(np.array(cleaned, dtype=float)))
    except Exception:
        return None


def build_exchange_context(
    symbol: str,
    candle_interval: str,
    candle_limit: int = 30,
    orderbook_limit: int = 20,
) -> Dict[str, Any]:
    candle_limit = min(int(candle_limit), int(EXCHANGE_CONTEXT_MAX_CANDLES))
    orderbook_limit = min(int(orderbook_limit), int(EXCHANGE_CONTEXT_MAX_ORDERBOOK_LEVELS))
    cache_key = f"exchange_context:{symbol.upper()}:{candle_interval}:{int(candle_limit)}:{int(orderbook_limit)}"
    cached = CACHE.get(cache_key)
    if cached is not None:
        return cached

    interval_sec = _interval_to_seconds(candle_interval)
    now_ts = time.time()

    out: Dict[str, Any] = {}
    prices_for_median: Dict[str, float] = {}
    last_candle_open_for_median: Dict[str, int] = {}
    sym = symbol.upper()
    variants = _symbol_variants(sym)
    for ex_id, adapter in _EXCHANGE_ADAPTERS.items():
        fetch_start = time.time()
        fetched_at_utc = _iso_utc_now_ms()

        if _is_exchange_on_cooldown(ex_id) or _is_exchange_symbol_on_cooldown(ex_id, sym):
            out[ex_id] = {
                "status": "cooldown",
                "ticker24h": None,
                "orderbook": None,
                "candles": None,
                "microstructure": None,
                "meta": {
                    "fetchedAtUtc": fetched_at_utc,
                    "fetchMs": 0.0,
                    "intervalSec": interval_sec,
                },
                "quality": {
                    "spreadPct": None,
                    "spreadOutlier": None,
                    "priceDeviationPctVsMedian": None,
                    "priceOutlierVsMedian": None,
                    "candleGap": None,
                    "lastCandleOpenTime": None,
                    "lastCandleAgeSec": None,
                    "isStale": True,
                },
            }
            continue

        t = None
        ob = None
        candles = None
        chosen_scale = 1.0
        for cs, scale in variants:
            try:
                t = adapter.get_ticker_24h(cs)
            except Exception:
                t = None
            try:
                ob = adapter.get_orderbook(cs, limit=int(orderbook_limit))
            except Exception:
                ob = None
            try:
                candles = adapter.get_candles(cs, interval=str(candle_interval), limit=int(candle_limit))
            except Exception:
                candles = None
            if t is not None or ob is not None or candles:
                chosen_scale = float(scale)
                break

        fetch_ms = float((time.time() - fetch_start) * 1000.0)

        if t is None and ob is None and not candles:
            _inc_exchange_snapshot_error(ex_id)
            _inc_exchange_symbol_snapshot_error(ex_id, sym)
            _set_exchange_symbol_cooldown(ex_id, sym, 15.0)

        status = "missing" if (t is None and ob is None and not candles) else "ok"

        ticker_obj = None
        if t is not None:
            try:
                ticker_obj = {
                    "symbol": getattr(t, "symbol", sym),
                    "price": float(getattr(t, "price", 0.0) or 0.0) * chosen_scale,
                    "volume24h": float(getattr(t, "volume_24h", 0.0) or 0.0),
                    "high24h": (float(getattr(t, "high_24h")) * chosen_scale) if getattr(t, "high_24h", None) not in (None, "") else None,
                    "low24h": (float(getattr(t, "low_24h")) * chosen_scale) if getattr(t, "low_24h", None) not in (None, "") else None,
                    "priceChange24h": (float(getattr(t, "price_change")) * chosen_scale) if getattr(t, "price_change", None) not in (None, "") else None,
                    "priceChangePercent24h": float(getattr(t, "price_change_pct")) if getattr(t, "price_change_pct", None) not in (None, "") else None,
                }
            except Exception:
                ticker_obj = None

        orderbook_obj = None
        bids_scaled = None
        asks_scaled = None
        if ob is not None:
            try:
                bids_raw = getattr(ob, "bids", None) or []
                asks_raw = getattr(ob, "asks", None) or []
                bids_scaled = [[float(p) * chosen_scale, float(q)] for p, q in bids_raw]
                asks_scaled = [[float(p) * chosen_scale, float(q)] for p, q in asks_raw]
                orderbook_obj = {
                    "bids": bids_scaled,
                    "asks": asks_scaled,
                }
            except Exception:
                orderbook_obj = None

        candles_obj = None
        if candles:
            try:
                candles_obj = [
                    {
                        "t": int(getattr(c, "t")),
                        "open": float(getattr(c, "open")) * chosen_scale,
                        "high": float(getattr(c, "high")) * chosen_scale,
                        "low": float(getattr(c, "low")) * chosen_scale,
                        "close": float(getattr(c, "close")) * chosen_scale,
                        "volume": float(getattr(c, "volume")),
                        "quoteVolume": float(getattr(c, "quote_volume")) if getattr(c, "quote_volume", None) not in (None, "") else None,
                    }
                    for c in candles
                ]
            except Exception:
                candles_obj = None

        micro_raw = _build_microstructure({"bids": bids_scaled or [], "asks": asks_scaled or []}) if ob is not None else None

        spread_pct = None
        spread_outlier = None
        if micro_raw is not None:
            try:
                spread_pct = micro_raw.get("spreadPct")
                if spread_pct is not None and np.isfinite(spread_pct):
                    spread_pct = float(spread_pct)
                    spread_outlier = bool(spread_pct > 0.67)
            except Exception:
                spread_pct = None
                spread_outlier = None

        last_candle_open_time = None
        last_candle_age_sec = None
        candle_gap = None
        if candles_obj:
            try:
                ts = sorted([int(row.get("t")) for row in candles_obj if row.get("t") is not None])
                if ts:
                    last_candle_open_time = int(ts[-1])
                    last_candle_age_sec = float(max(0.0, now_ts - float(last_candle_open_time)))
                    last_candle_open_for_median[ex_id] = last_candle_open_time
                if interval_sec is not None and len(ts) >= 2:
                    gap_sec = int(ts[-1]) - int(ts[-2])
                    candle_gap = bool(abs(float(gap_sec) - float(interval_sec)) > float(interval_sec) * 0.5)
            except Exception:
                last_candle_open_time = None
                last_candle_age_sec = None
                candle_gap = None

        price_for_median = None
        try:
            if ticker_obj is not None and ticker_obj.get("price") not in (None, 0.0):
                price_for_median = float(ticker_obj.get("price"))
            elif micro_raw is not None and micro_raw.get("mid") not in (None, 0.0):
                price_for_median = float(micro_raw.get("mid"))
        except Exception:
            price_for_median = None
        if price_for_median is not None and np.isfinite(price_for_median) and price_for_median > 0:
            prices_for_median[ex_id] = float(price_for_median)

        micro = None if spread_outlier else micro_raw
        if spread_outlier:
            orderbook_obj = None

        out[ex_id] = {
            "status": status,
            "ticker24h": ticker_obj,
            "orderbook": orderbook_obj,
            "candles": candles_obj,
            "microstructure": micro,
            "meta": {
                "fetchedAtUtc": fetched_at_utc,
                "fetchMs": fetch_ms,
                "intervalSec": interval_sec,
            },
            "quality": {
                "spreadPct": spread_pct,
                "spreadOutlier": spread_outlier,
                "priceDeviationPctVsMedian": None,
                "priceOutlierVsMedian": None,
                "candleGap": candle_gap,
                "lastCandleOpenTime": last_candle_open_time,
                "lastCandleAgeSec": last_candle_age_sec,
                "isStale": None,
            },
        }

    median_price = _median_float(list(prices_for_median.values()))
    median_last_candle_open = _median_float([float(v) for v in last_candle_open_for_median.values()])

    for ex_id, entry in out.items():
        q = entry.get("quality") or {}
        ex_price = prices_for_median.get(ex_id)
        if median_price is not None and ex_price is not None and median_price > 0:
            try:
                dev = (float(ex_price) - float(median_price)) / float(median_price) * 100.0
                q["priceDeviationPctVsMedian"] = float(dev)
                q["priceOutlierVsMedian"] = bool(abs(float(dev)) > 0.67)
            except Exception:
                q["priceDeviationPctVsMedian"] = None
                q["priceOutlierVsMedian"] = None

        last_t = q.get("lastCandleOpenTime")
        is_stale = None
        try:
            if interval_sec is None:
                if last_t is None:
                    is_stale = True
            else:
                if last_t is None:
                    is_stale = True
                else:
                    age_sec = float(max(0.0, now_ts - float(last_t)))
                    stale_by_age = bool(age_sec > float(2 * interval_sec))
                    stale_by_median = False
                    if median_last_candle_open is not None:
                        stale_by_median = bool(abs(float(last_t) - float(median_last_candle_open)) > float(2 * interval_sec))
                    is_stale = bool(stale_by_age or stale_by_median)
        except Exception:
            is_stale = None

        q["isStale"] = is_stale
        entry["quality"] = q

    CACHE.set(cache_key, out, ttl_sec=5.0)
    return out


@app.get("/debug/exchange_snapshots")
def debug_exchange_snapshots(symbol: str = "BTCUSDT"):
    """Debug endpoint: return current multi-exchange snapshots and internal cooldown/error counters."""

    return {
        "symbol": symbol.upper(),
        "snapshots": build_exchange_snapshots(symbol),
        "cooldowns": _EXCHANGE_COOLDOWN_UNTIL,
        "errors": _EXCHANGE_SNAPSHOT_ERRORS,
        "symbol_cooldowns": _EXCHANGE_SYMBOL_COOLDOWN_UNTIL,
        "symbol_errors": _EXCHANGE_SYMBOL_SNAPSHOT_ERRORS,
    }


@app.get("/debug/exchange_context")
def debug_exchange_context(symbol: str = "BTCUSDT", candle_interval: str = "5m"):
    return {
        "symbol": symbol.upper(),
        "candle_interval": candle_interval,
        "context": build_exchange_context(symbol, candle_interval=candle_interval),
        "cooldowns": _EXCHANGE_COOLDOWN_UNTIL,
        "errors": _EXCHANGE_SNAPSHOT_ERRORS,
        "symbol_cooldowns": _EXCHANGE_SYMBOL_COOLDOWN_UNTIL,
        "symbol_errors": _EXCHANGE_SYMBOL_SNAPSHOT_ERRORS,
    }


def safe_float(value: Any) -> Any:
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return MISSING
        return float(value)
    return MISSING


def safe_list(values: Optional[List[Any]]) -> Any:
    if values is None:
        return MISSING
    cleaned: List[float] = []
    for v in values:
        if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
            cleaned.append(float(v))
    return cleaned if cleaned else MISSING


def safe_dict(data_dict: Optional[Dict[str, Any]]) -> Any:
    if data_dict is None:
        return MISSING_OBJ
    result: Dict[str, Any] = {}
    for k, v in data_dict.items():
        if isinstance(v, (int, float)):
            result[k] = safe_float(v)
        elif isinstance(v, dict):
            result[k] = safe_dict(v)
        else:
            result[k] = v
    return result if result else MISSING_OBJ


def clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def resolve_reference_symbol(symbol: str) -> str:
    """Return the best symbol to use for indicators based on the registry.

    Today this effectively returns the same Binance symbol (e.g. "BTCUSDT"),
    but it goes through REGISTRY so that in future phases, if a coin is not
    on Binance, we can transparently fall back to MEXC/KuCoin/etc.
    """

    try:
        s = str(symbol).upper()
        # For now we only normalize USDT pairs via Coin base symbol.
        if not s.endswith("USDT"):
            return s

        base = s.replace("USDT", "")

        # Lazy-sync if registry is empty to avoid relying on debug endpoint.
        if not REGISTRY.coins:
            REGISTRY.sync_from_binance()

        coin = REGISTRY.get_coin(base)
        if not coin or not coin.reference_market_id_for_indicators:
            return s
        market = REGISTRY.get_market(coin.reference_market_id_for_indicators)
        if not market:
            return s
        return market.symbol_raw.upper() or s
    except Exception:
        # On any failure, fall back to original symbol to keep behaviour.
        return str(symbol).upper()

# -----------------------------
# Alert Engine core (Rule model, indexes, symbol lists)
# -----------------------------


class Metric(str, Enum):
    PRICE = "price"
    VOLUME = "volume"


class ConditionType(str, Enum):
    PRICE_CHANGE_PCT = "price_change_pct"
    ABSOLUTE_PRICE = "absolute_price"
    VOLUME_CHANGE_PCT = "volume_change_pct"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    BOTH = "both"
    CROSS_UP = "cross_up"
    CROSS_DOWN = "cross_down"


@dataclass
class Rule:
    """In-memory representation of a single alert rule.

    Phase 1: memory-only; later phases can persist this in DB.
    """

    id: str
    user_id: str
    symbol: str  # canonical symbol, e.g. "BTCUSDT" (for now Binance-focused)
    condition_type: ConditionType
    metric: Metric
    direction: Direction
    threshold: float
    timeframe: str
    is_active: bool = True
    cooldown_seconds: Optional[int] = None

    # Phase 2+: resolved market identifier, e.g. "binance_spot:BTCUSDT".
    # Optional so existing clients can omit it; we infer from `symbol` via REGISTRY.
    market_id: Optional[str] = None

    _last_trigger_ts: float = field(default=0.0, repr=False)

    def can_trigger_now(self, now_ts: Optional[float] = None) -> bool:
        if not self.is_active:
            return False
        if self.cooldown_seconds is None:
            return True
        if now_ts is None:
            now_ts = time.time()
        return (now_ts - self._last_trigger_ts) >= self.cooldown_seconds

    def mark_triggered(self, now_ts: Optional[float] = None) -> None:
        if now_ts is None:
            now_ts = time.time()
        self._last_trigger_ts = now_ts


rules_by_symbol: Dict[str, List[Rule]] = {}
rules_by_symbol_and_tf: Dict[Tuple[str, str], List[Rule]] = {}
rules_by_id: Dict[str, Rule] = {}


def _index_key(symbol: str, timeframe: str) -> Tuple[str, str]:
    return symbol.upper(), timeframe


def add_rule(rule: Rule) -> None:
    rules_by_id[rule.id] = rule
    sym = rule.symbol.upper()
    rules_by_symbol.setdefault(sym, []).append(rule)
    key = _index_key(sym, rule.timeframe)
    rules_by_symbol_and_tf.setdefault(key, []).append(rule)


def remove_rule(rule_id: str) -> None:
    rule = rules_by_id.pop(rule_id, None)
    if rule is None:
        return
    sym = rule.symbol.upper()
    if sym in rules_by_symbol:
        rules_by_symbol[sym] = [r for r in rules_by_symbol[sym] if r.id != rule_id]
        if not rules_by_symbol[sym]:
            del rules_by_symbol[sym]
    key = _index_key(sym, rule.timeframe)
    if key in rules_by_symbol_and_tf:
        rules_by_symbol_and_tf[key] = [r for r in rules_by_symbol_and_tf[key] if r.id != rule_id]
        if not rules_by_symbol_and_tf[key]:
            del rules_by_symbol_and_tf[key]


def update_rule(updated: Rule) -> None:
    if updated.id not in rules_by_id:
        add_rule(updated)
        return
    remove_rule(updated.id)
    add_rule(updated)


class SymbolRefCounter:
    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}

    def count(self, symbol: str) -> int:
        return self._counts.get(symbol.upper(), 0)

    def increment(self, symbol: str) -> bool:
        sym = symbol.upper()
        old = self._counts.get(sym, 0)
        new = old + 1
        self._counts[sym] = new
        return old == 0 and new == 1

    def decrement(self, symbol: str) -> bool:
        sym = symbol.upper()
        old = self._counts.get(sym, 0)
        if old <= 1:
            self._counts.pop(sym, None)
            return old == 1
        self._counts[sym] = old - 1
        return False


symbol_refcounter = SymbolRefCounter()


# Curated base universe for WhaleTrack (Binance + MEXC)
WHALE_TRACK_COIN_LIST: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "BNBUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "TONUSDT",
    "LINKUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "LTCUSDT",
    "OPUSDT",
    "ARBUSDT",
    "APTUSDT",
    "SUIUSDT",
    "NEARUSDT",
    "INJUSDT",
    "MATICUSDT",
    "PEPEUSDT",
    "AAVEUSDT",
    "UNIUSDT",
    "DOTUSDT",
    "SEIUSDT",
    "ORDIUSDT",
    "RUNEUSDT",
    "TURBOUSDT",
    "WIFUSDT",
    "LDOUSDT",
]


# Minimum 24h volume for a symbol to be considered in Price Alerts universe
PRICE_ALERTS_MIN_24H_VOLUME_USD: float = 5_000_000.0


@dataclass
class PriceState:
    symbol: str
    last_price: float
    last_volume: float
    ts_ms: int


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_time_ms: int
    close_time_ms: int


@dataclass
class RuleTriggeredEvent:
    rule_id: str
    user_id: str
    symbol: str
    timeframe: str
    reason: str
    ts_ms: int


def on_price_update(symbol: str, state: PriceState) -> List[RuleTriggeredEvent]:
    """Evaluate realtime absolute-price rules for the given symbol.

    Analyzer is expected to call this on each relevant price tick.
    """

    sym = symbol.upper()
    now_ms = state.ts_ms
    events: List[RuleTriggeredEvent] = []

    candidates = [
        r
        for r in rules_by_symbol.get(sym, [])
        if r.timeframe == "realtime" and r.is_active
    ]

    for rule in candidates:
        if rule.condition_type is not ConditionType.ABSOLUTE_PRICE:
            continue
        if not rule.can_trigger_now(now_ms / 1000.0):
            continue

        price = state.last_price
        hit = False

        if rule.direction is Direction.UP:
            hit = price >= rule.threshold
        elif rule.direction is Direction.DOWN:
            hit = price <= rule.threshold
        elif rule.direction is Direction.CROSS_UP:
            hit = price >= rule.threshold
        elif rule.direction is Direction.CROSS_DOWN:
            hit = price <= rule.threshold
        elif rule.direction is Direction.BOTH:
            hit = price >= rule.threshold or price <= rule.threshold

        if hit:
            rule.mark_triggered(now_ms / 1000.0)
            events.append(
                RuleTriggeredEvent(
                    rule_id=rule.id,
                    user_id=rule.user_id,
                    symbol=sym,
                    timeframe="realtime",
                    reason=f"price {price:.4f} crossed level {rule.threshold}",
                    ts_ms=now_ms,
                )
            )

    return events


def on_candle_close(
    symbol: str,
    timeframe: str,
    candle,
    *,
    price_pct_change: Optional[float] = None,
    volume_pct_change: Optional[float] = None,
    pct_change_lookback_close: Optional[float] = None,
    pct_change: Optional[float] = None,
) -> List[RuleTriggeredEvent]:
    """Evaluate candle-based rules when a timeframe candle closes.

    Analyzer is expected to compute percentage changes and pass them in.
    """

    # Backward compatibility: older callers may pass `pct_change` for price change.
    if price_pct_change is None and pct_change is not None:
        price_pct_change = pct_change

    sym = symbol.upper()
    key = _index_key(sym, timeframe)
    rules = rules_by_symbol_and_tf.get(key, [])
    if not rules:
        return []

    now_ms = getattr(candle, "close_time_ms", int(time.time() * 1000))
    events: List[RuleTriggeredEvent] = []

    for rule in rules:
        if not rule.is_active:
            continue
        if not rule.can_trigger_now(now_ms / 1000.0):
            continue

        hit = False

        if rule.condition_type is ConditionType.PRICE_CHANGE_PCT and price_pct_change is not None:
            if rule.direction is Direction.UP:
                hit = price_pct_change >= rule.threshold
            elif rule.direction is Direction.DOWN:
                hit = price_pct_change <= -abs(rule.threshold)
            elif rule.direction is Direction.BOTH:
                hit = abs(price_pct_change) >= abs(rule.threshold)
        elif rule.condition_type is ConditionType.VOLUME_CHANGE_PCT and volume_pct_change is not None:
            if rule.direction is Direction.UP:
                hit = volume_pct_change >= rule.threshold
            elif rule.direction is Direction.DOWN:
                hit = volume_pct_change <= -abs(rule.threshold)
            elif rule.direction is Direction.BOTH:
                hit = abs(volume_pct_change) >= abs(rule.threshold)

        if hit:
            rule.mark_triggered(now_ms / 1000.0)
            val = price_pct_change if rule.condition_type is ConditionType.PRICE_CHANGE_PCT else volume_pct_change
            try:
                val_str = f"{float(val):.2f}%" if val is not None else "-"
            except Exception:
                val_str = "-"
            events.append(
                RuleTriggeredEvent(
                    rule_id=rule.id,
                    user_id=rule.user_id,
                    symbol=sym,
                    timeframe=timeframe,
                    reason=f"{rule.condition_type.value} reached {val_str} on {timeframe}",
                    ts_ms=now_ms,
                )
            )

    return events


# -----------------------------
# In-memory /rules HTTP API (Phase 1.2 – no DB yet)
# -----------------------------


class RuleIn(BaseModel):
    user_id: str
    symbol: str
    condition_type: ConditionType
    metric: Metric
    direction: Direction
    threshold: float
    timeframe: str = "1m"
    cooldown_seconds: Optional[int] = None
    # Optional resolved market identifier. If not provided, the backend
    # will infer a suitable Binance market from the registry.
    market_id: Optional[str] = None


def _rule_to_dict(rule: Rule) -> Dict[str, Any]:
    return {
        "id": rule.id,
        "user_id": rule.user_id,
        "symbol": rule.symbol,
        "market_id": rule.market_id,
        "condition_type": rule.condition_type.value,
        "metric": rule.metric.value,
        "direction": rule.direction.value,
        "threshold": rule.threshold,
        "timeframe": rule.timeframe,
        "is_active": rule.is_active,
        "cooldown_seconds": rule.cooldown_seconds,
    }


@app.post("/rules")
def create_rule(rule_in: RuleIn):
    """Create a rule in memory.

    Phase 1: rules disappear when process restarts; this is fine for dev/tests.
    """

    rule_id = f"r_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    # Phase 2 (step 1): resolve market_id from registry if not provided.
    market_id: Optional[str] = rule_in.market_id
    if market_id is None:
        sym = rule_in.symbol.upper()
        # Prefer Binance Futures USDT, then Binance Spot USDT
        from market_registry import REGISTRY  # local import to avoid cycles at startup

        coin = REGISTRY.get_coin(sym.replace("USDT", "")) if sym.endswith("USDT") else None
        if coin and coin.primary_market_id:
            market_id = coin.primary_market_id

    rule = Rule(
        id=rule_id,
        user_id=rule_in.user_id,
        symbol=rule_in.symbol,
        condition_type=rule_in.condition_type,
        metric=rule_in.metric,
        direction=rule_in.direction,
        threshold=rule_in.threshold,
        timeframe=rule_in.timeframe,
        cooldown_seconds=rule_in.cooldown_seconds,
        market_id=market_id,
    )
    add_rule(rule)
    # Price Alerts için dinamik sembol takibi: ilk kural eklenince subscribe edilebilsin
    symbol_refcounter.increment(rule.symbol)
    return {"id": rule_id, "rule": _rule_to_dict(rule)}


@app.delete("/rules/{rule_id}")
def delete_rule(rule_id: str):
    rule = rules_by_id.get(rule_id)
    if not rule:
        return {"ok": False, "error": "not_found"}
    symbol = rule.symbol
    remove_rule(rule_id)
    symbol_refcounter.decrement(symbol)
    return {"ok": True}


@app.get("/rules")
def list_rules(user_id: Optional[str] = None):
    items = list(rules_by_id.values())
    if user_id is not None:
        items = [r for r in items if r.user_id == user_id]
    return {"items": [_rule_to_dict(r) for r in items]}


@app.get("/price_alerts/universe")
def price_alerts_universe():
    """Return the current Binance USDT universe suitable for Price Alerts.

    This is derived from Binance 24h ticker data using a minimum quote
    volume filter so that only reasonably liquid pairs appear in the list.
    """

    symbols = build_price_alerts_universe()
    return {"symbols": symbols, "min_24h_volume_usd": PRICE_ALERTS_MIN_24H_VOLUME_USD}


@app.get("/simulator/markets")
def simulator_markets(limit: int = 200, min_quote_volume_usd: float = 0.0):
    tickers = get_binance_24h_tickers() or []
    items: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            sym = str(t.get("symbol", "")).upper()
            if not sym.endswith("USDT"):
                continue
            qv = float(t.get("quoteVolume", 0.0) or 0.0)
            if qv < float(min_quote_volume_usd):
                continue
            price = float(t.get("lastPrice", 0.0) or 0.0)
            pct = float(t.get("priceChangePercent", 0.0) or 0.0)
            items.append({
                "symbol": sym,
                "price": price,
                "priceChangePercent24h": pct,
                "quoteVolume24h": qv,
            })
        except Exception:
            continue
    items.sort(key=lambda x: float(x.get("quoteVolume24h", 0.0) or 0.0), reverse=True)
    items = items[: max(1, int(limit))]
    return {"items": items}


@app.get("/simulator/tickers")
def simulator_tickers(symbols: str):
    wanted = [s.strip().upper() for s in str(symbols).split(",") if s.strip()]
    wanted = wanted[:200]
    tickers = get_binance_24h_tickers() or []
    out: Dict[str, Any] = {}
    by_sym: Dict[str, Dict[str, Any]] = {}
    for t in tickers:
        try:
            sym = str(t.get("symbol", "")).upper()
            if sym:
                by_sym[sym] = t
        except Exception:
            continue
    for sym in wanted:
        try:
            t = by_sym.get(sym) or {}
            ws_mid = _ws_mid_price(sym)
            price = ws_mid if ws_mid is not None else float(t.get("lastPrice", 0.0) or 0.0)
            pct = float(t.get("priceChangePercent", 0.0) or 0.0)
            out[sym] = {
                "price": float(price) if price else None,
                "priceChangePercent24h": float(pct),
            }
        except Exception:
            out[sym] = {"price": None, "priceChangePercent24h": 0.0}
    missing = [s for s in wanted if s not in by_sym]
    return {"items": out, "missing": missing}


@app.get("/simulator/price/{symbol}")
def simulator_price(symbol: str):
    sym = str(symbol).upper()
    cache_k = f"simulator:price:{sym}"
    cached = CACHE.get(cache_k)
    if isinstance(cached, dict) and cached.get("symbol") == sym:
        return cached
    ws_mid = _ws_mid_price(sym)
    tickers = get_binance_24h_tickers() or []
    t = None
    for row in tickers:
        try:
            if str(row.get("symbol", "")).upper() == sym:
                t = row
                break
        except Exception:
            continue
    price = None
    pct = 0.0
    try:
        if ws_mid is not None:
            price = float(ws_mid)
        elif t is not None:
            price = float(t.get("lastPrice", 0.0) or 0.0)
        if t is not None:
            pct = float(t.get("priceChangePercent", 0.0) or 0.0)
    except Exception:
        price = None
        pct = 0.0
    out = {"symbol": sym, "price": price, "priceChangePercent24h": pct}
    CACHE.set(cache_k, out, ttl_sec=1.5)
    return out


@app.get("/simulator/klines")
def simulator_klines(symbol: str, interval: str = "1m", limit: int = 120):
    sym = str(symbol).upper()
    interval = str(interval)
    limit = max(10, min(int(limit or 120), 500))

    cache_k = f"simulator:klines:{sym}:{interval}:{limit}"
    cached = CACHE.get(cache_k)
    if cached is not None:
        return cached

    try:
        df = get_klines_cached(sym, interval=interval, limit=limit)
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return {"symbol": sym, "interval": interval, "candles": []}

        candles = []
        for _, row in df.iterrows():
            try:
                candles.append([
                    int(row["timestamp"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row.get("volume", 0.0) or 0.0),
                ])
            except Exception:
                continue

        out = {"symbol": sym, "interval": interval, "candles": candles}
        CACHE.set(cache_k, out, ttl_sec=2.0)
        return out
    except Exception:
        return {"symbol": sym, "interval": interval, "candles": []}


@app.post("/simulator/topup/verify")
def simulator_topup_verify(payload: Dict[str, Any] = Body(...)):
    anon = str(payload.get("anon_id", "") or "").strip()
    pack_usdt = int(payload.get("pack_usdt", 0) or 0)
    amount_try = float(payload.get("amount_try", 0) or 0)

    if not anon or pack_usdt <= 0 or amount_try <= 0:
        return JSONResponse({"status": "error", "reason": "invalid_payload"}, status_code=400)

    try:
        log_simulator_topup(
            anon_id=anon,
            pack_usdt=pack_usdt,
            amount_try=amount_try,
            apple_transaction_id=payload.get("apple_transaction_id"),
        )
        return {"status": "ok"}
    except Exception:
        return JSONResponse({"status": "error", "reason": "internal"}, status_code=500)


@app.post("/simulator/profile")
def simulator_profile(payload: Dict[str, Any] = Body(...)):
    anon = str(payload.get("anon_id", "") or "").strip()
    if not anon:
        return JSONResponse({"status": "error", "reason": "missing_anon_id"}, status_code=400)

    try:
        nickname = payload.get("nickname")
        contact_info = payload.get("contact_info")
        equity = payload.get("equity")
        successful_trades = payload.get("successful_trades")
        roe = payload.get("roe")
        upsert_simulator_profile(
            anon_id=anon,
            nickname=nickname,
            contact_info=contact_info,
            equity=equity,
            successful_trades=successful_trades,
            roe=roe,
        )
        return {"status": "ok"}
    except Exception:
        return JSONResponse({"status": "error", "reason": "internal"}, status_code=500)


@app.get("/simulator/profile/status")
def simulator_profile_status(anon_id: str = ""):
    anon = str(anon_id or "").strip()
    if not anon:
        return JSONResponse({"status": "error", "reason": "missing_anon_id"}, status_code=400)

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"exists": False}

    try:
        url = f"{SUPABASE_URL}/rest/v1/simulator_profiles"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            url,
            params={
                "select": "anon_id",
                "anon_id": f"eq.{anon}",
                "limit": 1,
            },
            headers=headers,
            timeout=5,
        )
        if not (200 <= resp.status_code < 300):
            return {"exists": False}
        rows = resp.json() if resp.text else []
        return {"exists": bool(rows)}
    except Exception:
        return {"exists": False}


@app.get("/simulator/leaderboard/top")
def simulator_leaderboard_top(limit: int = 10):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"items": []}
    lim = max(1, min(100, int(limit or 10)))
    try:
        url = f"{SUPABASE_URL}/rest/v1/simulator_profiles"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            url,
            params={
                "select": "anon_id,nickname,equity,successful_trades,roe",
                "order": "equity.desc.nullslast",
                "limit": lim,
            },
            headers=headers,
            timeout=5,
        )
        if not (200 <= resp.status_code < 300):
            resp = requests.get(
                url,
                params={
                    "select": "anon_id,nickname,equity",
                    "order": "equity.desc.nullslast",
                    "limit": lim,
                },
                headers=headers,
                timeout=5,
            )
        if not (200 <= resp.status_code < 300):
            return {"items": []}
        rows = resp.json() if resp.text else []
        return {"items": rows}
    except Exception:
        return {"items": []}


@app.get("/simulator/leaderboard/slice")
def simulator_leaderboard_slice(limit: int = 400):
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"items": []}
    lim = max(1, min(1000, int(limit or 400)))
    try:
        url = f"{SUPABASE_URL}/rest/v1/simulator_profiles"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }
        resp = requests.get(
            url,
            params={
                "select": "anon_id,nickname,equity,successful_trades,roe",
                "order": "equity.desc.nullslast",
                "limit": lim,
            },
            headers=headers,
            timeout=5,
        )
        if not (200 <= resp.status_code < 300):
            resp = requests.get(
                url,
                params={
                    "select": "anon_id,nickname,equity",
                    "order": "equity.desc.nullslast",
                    "limit": lim,
                },
                headers=headers,
                timeout=5,
            )
        if not (200 <= resp.status_code < 300):
            return {"items": []}
        rows = resp.json() if resp.text else []
        return {"items": rows}
    except Exception:
        return {"items": []}


@app.get("/simulator/leaderboard/me")
def simulator_leaderboard_me(anon_id: str = ""):
    anon = str(anon_id or "").strip()
    if not anon:
        return JSONResponse({"status": "error", "reason": "missing_anon_id"}, status_code=400)

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return {"found": False, "rank": None}

    try:
        base_url = f"{SUPABASE_URL}/rest/v1/simulator_profiles"
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        me_resp = requests.get(
            base_url,
            params={
                "select": "anon_id,nickname,equity,successful_trades,roe",
                "anon_id": f"eq.{anon}",
                "limit": 1,
            },
            headers=headers,
            timeout=5,
        )
        if not (200 <= me_resp.status_code < 300):
            me_resp = requests.get(
                base_url,
                params={
                    "select": "anon_id,nickname,equity",
                    "anon_id": f"eq.{anon}",
                    "limit": 1,
                },
                headers=headers,
                timeout=5,
            )
        if not (200 <= me_resp.status_code < 300):
            return {"found": False, "rank": None}

        rows = me_resp.json() if me_resp.text else []
        if not rows:
            return {"found": False, "rank": None}
        me = rows[0]
        equity = me.get("equity")
        try:
            equity_f = float(equity) if equity is not None else 0.0
        except Exception:
            equity_f = 0.0

        count_headers = dict(headers)
        count_headers["Prefer"] = "count=exact"
        count_headers["Range"] = "0-0"

        count_resp = requests.get(
            base_url,
            params={
                "select": "anon_id",
                "equity": f"gt.{equity_f}",
            },
            headers=count_headers,
            timeout=5,
        )
        rank = None
        if 200 <= count_resp.status_code < 300:
            cr = count_resp.headers.get("content-range") or count_resp.headers.get("Content-Range")
            if cr and "/" in cr:
                try:
                    total = int(cr.split("/")[-1])
                    rank = total + 1
                except Exception:
                    rank = None

        result = {"found": True, "rank": rank}
        result.update(me)
        return result
    except Exception:
        return {"found": False, "rank": None}

# -----------------------------
# Binance Futures WebSocket manager (shared cache)
# -----------------------------
FSTREAM_WS = "wss://fstream.binance.com/stream"

class WSSymbolCache:
    def __init__(self):
        self.klines_1m: List[List[float]] = []  # [ts,o,h,l,c,v]
        self.klines_5m: List[List[float]] = []
        self.book_ticker: Dict[str, float] = {}
        self.depth_bids: List[List[float]] = []
        self.depth_asks: List[List[float]] = []

class FuturesWSManager:
    def __init__(self):
        self.cache: Dict[str, WSSymbolCache] = {}
        self._running = False

    def ensure_symbol(self, symbol: str):
        s = symbol.upper()
        if s not in self.cache:
            self.cache[s] = WSSymbolCache()

    def get_klines(self, symbol: str, interval: str) -> List[List[float]]:
        s = symbol.upper()
        sc = self.cache.get(s)
        if not sc:
            return []
        if interval == '1m':
            return sc.klines_1m
        if interval == '5m':
            return sc.klines_5m
        return []

    def get_orderbook(self, symbol: str) -> Dict[str, List[List[float]]]:
        s = symbol.upper()
        sc = self.cache.get(s)
        if not sc:
            return {"bids": [], "asks": []}
        return {"bids": sc.depth_bids, "asks": sc.depth_asks}

    async def run(self, preload: List[str]):
        if websockets is None:
            return
        for s in preload:
            self.ensure_symbol(s)
        backoff = 1
        while True:
            try:
                streams: List[str] = []
                for s in list(self.cache.keys()):
                    low = s.lower()
                    streams.extend([
                        f"{low}@kline_1m",
                        f"{low}@kline_5m",
                        f"{low}@bookTicker",
                        f"{low}@depth20@100ms",
                    ])
                if not streams:
                    await asyncio.sleep(1.0)
                    continue
                url = FSTREAM_WS + "?streams=" + "/".join(streams)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
                    backoff = 1
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        data = msg.get("data") or {}
                        stream = msg.get("stream", "")
                        if not data or not stream:
                            continue
                        sym = (data.get('s') or data.get('ps') or '').upper()
                        if not sym:
                            try:
                                sym = stream.split('@')[0].upper()
                            except Exception:
                                continue
                        self.ensure_symbol(sym)
                        sc = self.cache[sym]
                        if '@kline_1m' in stream:
                            k = data.get('k') or {}
                            try:
                                row = [int(k.get('t')), float(k.get('o')), float(k.get('h')), float(k.get('l')), float(k.get('c')), float(k.get('v'))]
                                if sc.klines_1m and sc.klines_1m[-1][0] == row[0]:
                                    sc.klines_1m[-1] = row
                                else:
                                    sc.klines_1m.append(row)
                                    if len(sc.klines_1m) > 360:
                                        sc.klines_1m = sc.klines_1m[-360:]
                            except Exception:
                                pass
                        elif '@kline_5m' in stream:
                            k = data.get('k') or {}
                            try:
                                row = [int(k.get('t')), float(k.get('o')), float(k.get('h')), float(k.get('l')), float(k.get('c')), float(k.get('v'))]
                                if sc.klines_5m and sc.klines_5m[-1][0] == row[0]:
                                    sc.klines_5m[-1] = row
                                else:
                                    sc.klines_5m.append(row)
                                    if len(sc.klines_5m) > 360:
                                        sc.klines_5m = sc.klines_5m[-360:]
                            except Exception:
                                pass
                        elif '@bookTicker' in stream:
                            try:
                                sc.book_ticker = {
                                    'b': float(data.get('b', 0.0) or 0.0),
                                    'B': float(data.get('B', 0.0) or 0.0),
                                    'a': float(data.get('a', 0.0) or 0.0),
                                    'A': float(data.get('A', 0.0) or 0.0),
                                }
                            except Exception:
                                pass
                        elif '@depth' in stream:
                            try:
                                bids = data.get('b', [])
                                asks = data.get('a', [])
                                sc.depth_bids = [[float(p), float(q)] for p, q in bids[:20]]
                                sc.depth_asks = [[float(p), float(q)] for p, q in asks[:20]]
                            except Exception:
                                pass
            except Exception:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

WS_MANAGER = FuturesWSManager()
# -----------------------------
# Binance REST helpers (Futures first, Spot fallback)
# -----------------------------
BINANCE_SPOT = "https://api.binance.com"
BINANCE_FUT = "https://fapi.binance.com"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CryptoAI/2.0"})


def http_get(url: str, params: Dict[str, Any] = None, timeout: int = 10) -> Any:
    r = SESSION.get(url, params=params or {}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_klines_rest(symbol: str, interval: str, limit: int = 100, futures_first: bool = True) -> Optional[pd.DataFrame]:
    base_orders = [BINANCE_FUT, BINANCE_SPOT] if futures_first else [BINANCE_SPOT, BINANCE_FUT]
    for base in base_orders:
        try:
            url = f"{base}/api/v3/klines" if base == BINANCE_SPOT else f"{base}/fapi/v1/klines"
            data = http_get(url, {"symbol": symbol, "interval": interval, "limit": limit})
            df = pd.DataFrame(data, columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "qav", "num_trades", "tbbav", "tbqav", "ignore"
            ])
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            return df
        except Exception:
            continue
    return None


def get_orderbook_rest(symbol: str, limit: int = 50, futures_first: bool = True) -> Optional[Dict[str, Any]]:
    base_orders = [BINANCE_FUT, BINANCE_SPOT] if futures_first else [BINANCE_SPOT, BINANCE_FUT]
    for base in base_orders:
        try:
            if base == BINANCE_SPOT:
                url = f"{base}/api/v3/depth"
            else:
                url = f"{base}/fapi/v1/depth"
            data = http_get(url, {"symbol": symbol, "limit": limit})
            return data
        except Exception:
            continue
    return None


def get_binance_24h_tickers() -> Optional[List[Dict[str, Any]]]:
    """Fetch 24h statistics for all Binance symbols (spot).

    Used to derive the Price Alerts universe by filtering USDT pairs with
    sufficient 24h quote volume.
    """

    cache_k = "binance:ticker24h:all"
    cached = CACHE.get(cache_k)
    if isinstance(cached, list) and cached:
        return cached  # type: ignore[return-value]

    try:
        url = f"{BINANCE_SPOT}/api/v3/ticker/24hr"
        data = http_get(url, timeout=10)
        if isinstance(data, list):
            CACHE.set(cache_k, data, ttl_sec=8.0)
            return data  # type: ignore[return-value]
    except Exception:
        return None
    return None


def _ws_mid_price(symbol: str) -> Optional[float]:
    try:
        s = str(symbol).upper()
        sc = WS_MANAGER.cache.get(s)
        if not sc:
            return None
        bt = sc.book_ticker or {}
        b = float(bt.get("b", 0.0) or 0.0)
        a = float(bt.get("a", 0.0) or 0.0)
        if b > 0 and a > 0:
            return float((b + a) / 2.0)
        if a > 0:
            return float(a)
        if b > 0:
            return float(b)
        return None
    except Exception:
        return None


def build_price_alerts_universe() -> List[str]:
    """Return a filtered list of USDT symbols suitable for Price Alerts.

    Criteria (can be tightened later):
    - Quote asset is USDT (symbol endswith "USDT").
    - 24h quote volume in USDT >= PRICE_ALERTS_MIN_24H_VOLUME_USD.
    """

    tickers = get_binance_24h_tickers()
    if not tickers:
        return []

    universe: List[str] = []
    for t in tickers:
        try:
            symbol = str(t.get("symbol", ""))
            if not symbol.endswith("USDT"):
                continue
            quote_volume = float(t.get("quoteVolume", 0.0) or 0.0)
            if quote_volume < PRICE_ALERTS_MIN_24H_VOLUME_USD:
                continue
            universe.append(symbol)
        except Exception:
            continue

    # Deterministic order for UI
    universe = sorted(set(universe))
    return universe


def get_funding_rate_rest(symbol: str) -> Dict[str, Any]:
    for cs, _scale in _symbol_variants(str(symbol).upper()):
        try:
            prem = http_get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", {"symbol": cs})
            return {
                "fundingRate": float(prem.get("lastFundingRate", 0.0)),
                "nextFundingTime": prem.get("nextFundingTime", 0),
                "openInterest": float(prem.get("openInterest", 0) or 0),
                "liquidations": {
                    "longLiquidations": 0.0,
                    "shortLiquidations": 0.0,
                    "liquidationPrice": 0.0,
                },
            }
        except Exception:
            continue
    return {
        "fundingRate": 0.0001,
        "nextFundingTime": 0,
        "openInterest": 0.0,
        "liquidations": {
            "longLiquidations": 0.0,
            "shortLiquidations": 0.0,
            "liquidationPrice": 0.0,
        },
    }


class SimulatorProfilePayload(BaseModel):
    anon_id: str
    nickname: Optional[str] = None
    contact_info: Optional[str] = None


class SimulatorTopupVerifyPayload(BaseModel):
    anon_id: str
    pack_usdt: int = 1000
    amount_try: float
    apple_transaction_id: Optional[str] = None


# -----------------------------
# CCXT fallback
# -----------------------------
_ccxt_binance = None
if _HAS_CCXT:
    try:
        _ccxt_binance = ccxt.binance({"enableRateLimit": True})
    except Exception:
        _ccxt_binance = None


def get_klines_ccxt(symbol: str, interval: str, limit: int = 100) -> Optional[pd.DataFrame]:
    if not _ccxt_binance:
        return None
    try:
        ohlcv = _ccxt_binance.fetch_ohlcv(symbol, interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        return df
    except Exception:
        return None

# -----------------------------
# High-level fetch with cache and backoff
# -----------------------------
INTERVAL_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d"
}


def cache_key(symbol: str, interval: str, limit: int) -> str:
    return f"klines:{symbol}:{interval}:{limit}"


def get_ohlcv(symbol: str, interval: str, limit: int = 100, ttl: float = 2.0) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV with smart preference:
    - Use WS cache if it is sufficiently warm (enough rows) to compute indicators.
    - Otherwise, fallback to REST to avoid empty indicator arrays.
    - For 15m, derive from 5m WS only if we have at least limit*3 5m rows.
    """
    MIN_ROWS = max(30, int(limit * 0.5))  # minimum bars required from WS to trust it

    if interval in ("1m", "5m"):
        rows = WS_MANAGER.get_klines(symbol, interval)
        if rows and len(rows) >= MIN_ROWS:
            df_ws = pd.DataFrame(rows[-limit:], columns=["timestamp", "open", "high", "low", "close", "volume"])
            return df_ws
        # WS not warm enough -> try REST first
        df = get_klines_rest(symbol, interval, limit, futures_first=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            CACHE.set(cache_key(symbol, interval, limit), df, ttl)
            return df
        # last resort: return whatever WS has
        if rows:
            return pd.DataFrame(rows[-limit:], columns=["timestamp", "open", "high", "low", "close", "volume"])

    if interval == "15m":
        rows5 = WS_MANAGER.get_klines(symbol, "5m")
        if rows5 and len(rows5) >= max(3 * limit, 90):  # need enough 5m bars to resample
            d5 = pd.DataFrame(rows5, columns=["timestamp", "open", "high", "low", "close", "volume"])
            d5["timestamp"] = pd.to_datetime(d5["timestamp"], unit="ms")
            d5 = d5.set_index("timestamp")
            agg = d5.resample('15T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
            if not agg.empty:
                out = agg.tail(limit).reset_index()
                # keep standard column names for downstream TA
                out.rename(columns={"timestamp":"open_time"}, inplace=False)
                return out.rename(columns={"open_time":"timestamp"})
        # Not enough WS data -> REST 15m
        df = get_klines_rest(symbol, interval, limit, futures_first=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            CACHE.set(cache_key(symbol, interval, limit), df, ttl)
            return df
    key = cache_key(symbol, interval, limit)
    cached = CACHE.get(key)
    if isinstance(cached, pd.DataFrame):
        return cached

    # Try REST (futures -> spot)
    df = get_klines_rest(symbol, interval, limit, futures_first=True)
    if df is None:
        # CCXT last resort
        df = get_klines_ccxt(symbol, interval, limit)

    if isinstance(df, pd.DataFrame) and not df.empty:
        CACHE.set(key, df, ttl)
        return df
    return None


def _candles_to_df(candles: Any) -> Optional[pd.DataFrame]:
    try:
        if not candles:
            return None
        rows = []
        for c in candles:
            try:
                ts_ms = int(getattr(c, "t")) * 1000
                rows.append(
                    {
                        "timestamp": ts_ms,
                        "open": float(getattr(c, "open")),
                        "high": float(getattr(c, "high")),
                        "low": float(getattr(c, "low")),
                        "close": float(getattr(c, "close")),
                        "volume": float(getattr(c, "volume")),
                    }
                )
            except Exception:
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        return df[["timestamp", "open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def get_indicator_ohlcv_with_fallback(
    symbol: str,
    interval: str,
    limit: int = 100,
    ttl: float = 2.0,
    preferred_exchange: Optional[str] = None,
) -> Tuple[Optional[str], Optional[pd.DataFrame]]:
    sym = str(symbol).upper()
    itv = str(interval)
    lim = int(limit)

    candidates: List[str] = []
    if preferred_exchange:
        pe = str(preferred_exchange).lower()
        if pe in _EXCHANGE_ADAPTERS:
            candidates.append(pe)
    for ex in _INDICATOR_EXCHANGE_FALLBACK:
        if ex not in candidates:
            candidates.append(ex)

    for ex_id in candidates:
        cache_k = f"indicator_ohlcv:{ex_id}:{sym}:{itv}:{lim}"
        cached = CACHE.get(cache_k)
        if isinstance(cached, pd.DataFrame) and not cached.empty:
            return ex_id, cached

        if ex_id == "binance":
            for cs, scale in _symbol_variants(sym):
                df = get_ohlcv(cs, itv, lim, ttl)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    df = ensure_numeric_ohlcv(df)
                    try:
                        if float(scale) != 1.0:
                            for col in ["open", "high", "low", "close"]:
                                if col in df.columns:
                                    df[col] = pd.to_numeric(df[col], errors='coerce') * float(scale)
                    except Exception:
                        pass
                    CACHE.set(cache_k, df, ttl)
                    return ex_id, df
            continue

        adapter = _EXCHANGE_ADAPTERS.get(ex_id)
        if adapter is None:
            continue
        for cs, scale in _symbol_variants(sym):
            try:
                candles = adapter.get_candles(cs, interval=itv, limit=lim)
            except Exception:
                candles = None
            df = _candles_to_df(candles)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df = ensure_numeric_ohlcv(df)
                try:
                    if float(scale) != 1.0:
                        for col in ["open", "high", "low", "close"]:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce') * float(scale)
                except Exception:
                    pass
                CACHE.set(cache_k, df, ttl)
                return ex_id, df

    return None, None


def get_orderbook(symbol: str, limit: int = 20, ttl: float = 2.0) -> Optional[Dict[str, Any]]:
    sym = str(symbol).upper()
    variants = _symbol_variants(sym)

    for cs, scale in variants:
        ob_ws = WS_MANAGER.get_orderbook(cs)
        if ob_ws.get('bids') or ob_ws.get('asks'):
            if float(scale) == 1.0:
                return ob_ws
            return {
                "bids": _scale_orderbook_levels(ob_ws.get("bids") or [], float(scale)),
                "asks": _scale_orderbook_levels(ob_ws.get("asks") or [], float(scale)),
            }

    key = f"orderbook:{sym}:{int(limit)}"
    cached = CACHE.get(key)
    if isinstance(cached, dict):
        return cached

    for cs, scale in variants:
        data = get_orderbook_rest(cs, limit, futures_first=True)
        if data and isinstance(data, dict):
            bids = _scale_orderbook_levels(data.get("bids") or [], float(scale))
            asks = _scale_orderbook_levels(data.get("asks") or [], float(scale))
            out = {"bids": bids[: int(limit)], "asks": asks[: int(limit)]}
            CACHE.set(key, out, ttl)
            return out

    return None

# -----------------------------
# Indicators helpers
# -----------------------------

def manual_vwap(df: pd.DataFrame) -> pd.Series:
    try:
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        vol = df['volume'].replace(0.0, np.nan)
        vwap = (tp * vol).cumsum() / vol.cumsum()
        return vwap.fillna(method='ffill').fillna(df['close'])
    except Exception:
        return pd.Series([df['close'].mean()] * len(df), index=df.index)


def ensure_numeric_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce OHLCV numeric, drop fully empty rows, forward-fill minimal, keep last 'limit' rows."""
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # drop rows where close is nan
    df = df.dropna(subset=[col for col in ["open","high","low","close"] if col in df.columns])
    return df


def robust_bbands(close: pd.Series, win: int = 20) -> pd.DataFrame:
    try:
        bb = ta.bbands(close, length=win)
        if isinstance(bb, pd.DataFrame) and not bb.empty:
            return bb
    except Exception:
        pass
    # Fallback: simple rolling
    mid = close.rolling(win).mean()
    std = close.rolling(win).std(ddof=0)
    upper = mid + 2 * std
    lower = mid - 2 * std
    return pd.DataFrame({
        f"BBL_{win}_2.0_2.0": lower,
        f"BBM_{win}_2.0_2.0": mid,
        f"BBU_{win}_2.0_2.0": upper
    })

# -----------------------------
# Indicator utilities
# -----------------------------
def last_rsi_value_for(
    symbol: str,
    interval: str,
    length: int = 14,
    limit: int = 120,
    ttl: float = 2.0,
    preferred_exchange: Optional[str] = None,
):
    _, df = get_indicator_ohlcv_with_fallback(symbol, interval, limit, ttl, preferred_exchange=preferred_exchange)
    if df is None or df.empty:
        return 0.0
    close = pd.to_numeric(df.get('close', pd.Series(dtype='float64')), errors='coerce')
    rsi_series = ta.rsi(close, length=length)
    rsi_series = rsi_series.dropna() if isinstance(rsi_series, pd.Series) else pd.Series(dtype='float64')
    if rsi_series.empty:
        return 0.0
    return float(rsi_series.iloc[-1])

def series_tail_floats(series: Optional[pd.Series], n: int) -> List[float]:
    if not isinstance(series, pd.Series):
        return []
    arr = series.dropna().tail(n).astype(float).tolist()
    return arr

def row_to_float_dict(df_like: Optional[pd.DataFrame]) -> Dict[str, float]:
    if not isinstance(df_like, pd.DataFrame) or df_like.empty:
        return {}
    # Prefer last fully-populated row to avoid NaNs at the very tail
    clean_df = df_like.dropna()
    if clean_df.empty:
        return {}
    row = clean_df.iloc[-1]
    try:
        return {str(k): float(v) for k, v in row.items() if np.isfinite(v)}
    except Exception:
        # Fallback: attempt to coerce all
        out: Dict[str, float] = {}
        for k, v in row.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out

def build_bollinger_dict(close: pd.Series) -> Dict[str, float]:
    """Return a combined dict for Bollinger with 5 and 20 period keys that iOS expects.
    Includes keys like BBU_5_2.0, BBM_5_2.0, BBL_5_2.0 and same for 20; plus lower/middle/upper.
    """
    out: Dict[str, float] = {}
    for length in (5, 20):
        try:
            bb_df = ta.bbands(close, length=length)
        except Exception:
            bb_df = None
        if not isinstance(bb_df, pd.DataFrame) or bb_df.dropna().empty:
            # fallback manual
            mid = close.rolling(length).mean()
            std = close.rolling(length).std(ddof=0)
            upper = mid + 2 * std
            lower = mid - 2 * std
            tmp = pd.DataFrame({
                f"BBL_{length}_2.0": lower,
                f"BBM_{length}_2.0": mid,
                f"BBU_{length}_2.0": upper,
            })
        else:
            # normalize only duplicated 2.0 suffix; keep original column names from pandas_ta
            # expected keys like BBL_5_2.0, BBM_5_2.0, BBU_5_2.0 (and 20 period variants)
            tmp = bb_df.rename(columns=lambda k: k.replace("_2.0_2.0", "_2.0"))
        row = tmp.dropna().iloc[-1] if not tmp.dropna().empty else None
        if row is not None:
            for k, v in row.items():
                try:
                    out[str(k)] = float(v)
                except Exception:
                    pass
            # Provide compatibility aliases if library returned duplicated length tokens like 'BBU_5_5_2.0'
            # Map 'X_5_5_2.0' -> 'X_5_2.0' and 'X_20_20_2.0' -> 'X_20_2.0'
            for base in ("BBL", "BBM", "BBU"):
                dup_key = f"{base}_{length}_{length}_2.0"
                norm_key = f"{base}_{length}_2.0"
                if dup_key in out and norm_key not in out:
                    out[norm_key] = out[dup_key]
            # also provide generic aliases for current length (last computed)
            out.setdefault("lower", float(row.iloc[0]))
            out.setdefault("middle", float(row.iloc[1]))
            out.setdefault("upper", float(row.iloc[2]))
    return out

def bollinger_with_alias(bb_dict: Dict[str, float]) -> Dict[str, float]:
    # Provide generic keys expected by some UIs
    out = dict(bb_dict)
    # try to map by searching keys
    lower_key = next((k for k in bb_dict.keys() if k.startswith("BBL_") or k.lower().endswith("lower")), None)
    mid_key = next((k for k in bb_dict.keys() if k.startswith("BBM_") or k.lower().endswith("mid") or k.lower().endswith("middle")), None)
    upper_key = next((k for k in bb_dict.keys() if k.startswith("BBU_") or k.lower().endswith("upper")), None)
    if lower_key and "lower" not in out:
        out["lower"] = float(bb_dict[lower_key])
    if mid_key and "middle" not in out:
        out["middle"] = float(bb_dict[mid_key])
    if upper_key and "upper" not in out:
        out["upper"] = float(bb_dict[upper_key])
    return out

# -----------------------------
# Background refresh (simulate WS impact)
# -----------------------------
POPULAR = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "TRXUSDT"]
REFRESH_INTERVALS = ["1m", "5m", "15m", "1h"]


async def background_refresher():
    while True:
        try:
            for sym in POPULAR:
                for itv in REFRESH_INTERVALS:
                    _ = get_ohlcv(sym, itv, 100, ttl=2.0)
            await asyncio.sleep(1.5)
        except Exception:
            await asyncio.sleep(2.0)


async def price_alerts_worker():
    """Background worker for Price Alerts (Binance only).

    - Every ~30 seconds, fetches Binance 24h tickers (spot).
    - Builds a symbol -> lastPrice map.
    - For all realtime absolute-price rules (metric=price), calls
      `on_price_update` so the Alert Engine can decide which rules
      triggered. For now we just log the count of triggered events;
      later this can be wired into a notification pipeline.
    """

    while True:
        try:
            # No rules defined: back off quickly
            if not rules_by_id:
                await asyncio.sleep(10.0)
                continue

            tickers = get_binance_24h_tickers()
            if not tickers:
                await asyncio.sleep(10.0)
                continue

            price_by_symbol: Dict[str, float] = {}
            for t in tickers:
                try:
                    symbol = str(t.get("symbol", "")).upper()
                    price = float(t.get("lastPrice", 0.0) or 0.0)
                    if symbol and price > 0:
                        price_by_symbol[symbol] = price
                except Exception:
                    continue

            now_ms = int(time.time() * 1000)
            total_events = 0

            # Evaluate only relevant rules (Price, ABSOLUTE_PRICE, realtime)
            for rule in list(rules_by_id.values()):
                if (
                    rule.metric is not Metric.PRICE
                    or rule.condition_type is not ConditionType.ABSOLUTE_PRICE
                    or rule.timeframe != "realtime"
                ):
                    continue

                sym = rule.symbol.upper()
                price = price_by_symbol.get(sym)
                if price is None:
                    continue

                state = PriceState(
                    symbol=sym,
                    last_price=price,
                    last_volume=0.0,
                    ts_ms=now_ms,
                )
                events = on_price_update(sym, state)
                total_events += len(events)

                for ev in events:
                    try:
                        PRICE_ALERTS_EVENT_QUEUE.put_nowait(ev)
                    except Exception:
                        pass

            if total_events:
                print(f"[PriceAlerts] triggered {total_events} rule(s) in last cycle", flush=True)

        except Exception as exc:
            print(f"[PriceAlerts] worker error: {exc}", flush=True)

        # Run at most twice per minute to stay well within Binance limits
        await asyncio.sleep(30.0)


@app.on_event("startup")
async def on_startup():
    # Start background refresher
    asyncio.create_task(background_refresher())
    # Start WS manager with a small preload; any new symbols will be added on demand
    try:
        asyncio.create_task(WS_MANAGER.run(["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
    except Exception:
        pass
    # Start Price Alerts worker (Binance-only, low frequency)
    try:
        asyncio.create_task(price_alerts_worker())
    except Exception:
        pass
    async def notification_worker():
        while True:
            try:
                ev = await PRICE_ALERTS_EVENT_QUEUE.get()
                payload = {"type": "rule_triggered", "event": asdict(ev)}
                await PRICE_ALERTS_BROADCASTER.broadcast_json(payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(0.2)

    try:
        asyncio.create_task(notification_worker())
    except Exception:
        pass

    # Start WhaleTrack analyzer (Binance+MEXC) in the same event loop
    try:
        def _hook_on_candle_close(
            symbol: str,
            timeframe: str,
            ts_ms: int,
            price_pct_change: Optional[float],
            volume_pct_change: Optional[float],
        ) -> None:
            try:
                candle = Candle(
                    open=0.0,
                    high=0.0,
                    low=0.0,
                    close=0.0,
                    volume=0.0,
                    open_time_ms=int(ts_ms) - 60_000,
                    close_time_ms=int(ts_ms),
                )
                events = on_candle_close(
                    symbol,
                    timeframe,
                    candle,
                    price_pct_change=price_pct_change,
                    volume_pct_change=volume_pct_change,
                )
                if events:
                    for ev in events:
                        try:
                            PRICE_ALERTS_EVENT_QUEUE.put_nowait(ev)
                        except Exception:
                            pass
            except Exception as exc:
                print(f"[PriceAlerts] on_candle_close hook error: {exc}", flush=True)

        hooks = binance_futures_analyzer.AnalyzerHooks(on_candle_close=_hook_on_candle_close)
        asyncio.create_task(binance_futures_analyzer.main(hooks=hooks))
    except Exception:
        pass

# -----------------------------
# Whale AI helpers
# -----------------------------

def whale_system_instructions(lang: str) -> str:
    """System prompt for Whale AI Engine (Turkish, senior microstructure analyst)."""

    return (
        "You are **WASKA Whale AI Engine**, a senior-level crypto market microstructure analyst specialized in\n"
        "orderflow, whale tracking, short-term liquidity dynamics, volatility regimes and leverage flow interpretation.\n\n"
        "Your job is to transform a single whale alert + a full metrics block into a **premium, data-rich,\n"
        "high-value intelligence report** that helps the user understand context, patterns and risks —\n"
        "WITHOUT giving investment advice.\n\n"
        "Your analysis MUST feel like a professional desk note prepared for an experienced trader.\n"
        "It must make the user feel that they have unlocked something valuable, exclusive and meaningful.\n\n"
        "========================================================\n"
        "= HARD RULES (YOU MUST FOLLOW THEM PERFECTLY)\n"
        "========================================================\n\n"
        "1. **Use ONLY the data provided.**\n"
        "   - If any metric is null → treat as 'bilgi eksik' and move on.\n"
        "   - NEVER invent numbers, predictions or false certainty.\n\n"
        "2. **NEVER give trading instructions.**\n"
        "   Forbidden:\n"
        "   - 'Al', 'sat', 'long aç', 'short aç'\n"
        "   - entry/exit/SL/TP\n"
        "   - % hedef, kesin yön, kesin tahmin\n\n"
        "3. **Express scenarios, not instructions.**\n"
        "   - You may describe potential reactions, liquidity sweeps, squeeze risk,\n"
        "     continuation probability, but only as possibilities — NOT directives.\n\n"
        "4. **Tone = professional, net, sakin, veri odaklı.**\n"
        "   - No motivational fluff.\n"
        "   - No generic advice.\n"
        "   - No repeated warnings.\n\n"
        "5. **NEVER echo raw field names or internal data structure words.**\n"
        "   Forbidden:\n"
        "   - volume_5m_change_pct\n"
        "   - oi_5m_change_pct\n"
        "   - avg_reaction_15m_pct\n"
        "   - reaction_consistency_pct\n"
        "   - any other snake_case field name from the metrics block\n"
        "   - the words 'JSON', 'metric', 'metrics', 'Whale AI Engine metrics' in your answer\n"
        "   Instead convert all of them into human-readable phrasing like:\n"
        "   - 'son 5 dakikadaki hacim artışı'\n"
        "   - 'kısa vadeli açık pozisyon değişimi'\n"
        "   - 'benzer büyüklükteki hareketlerin geçmiş ortalaması'\n\n"
        "6. **Each bullet MUST add new value. No repetition.**\n"
        "   Every bullet MUST include:\n"
        "   - a metric,\n"
        "   - a relationship between metrics,\n"
        "   - a structural comment (orderbook/liquidity),\n"
        "   - or a short-term scenario description.\n\n"
        "7. **NEVER add disclaimers such as 'yatırım tavsiyesi değildir.'**\n"
        "   The app handles legal language. You do not.\n\n"
        "8. **Language:** Respond ONLY in **Turkish**, but keep standard technical terms in English\n"
        "   (funding rate, liquidity, open interest, orderbook, volatility, leverage).\n\n"
        "========================================================\n"
        "= STYLE RULES (STRICT)\n"
        "========================================================\n\n"
        "- Write in **clear bullet points**, not paragraphs.\n"
        "- Be **specific**, not vague.\n"
        "- Use **approximate ranges** if meaningful (örn: 'yaklaşık %3–5 aralığında geçmiş tepki').\n"
        "- Always convert raw numbers into human‑friendly form: round to at most one decimal, never show many digits or scientific notation, and prefer prose such as 'yaklaşık %4 civarında' instead of '3.14e‑05' or '50.96'.\n"
        "- Do NOT repeat mechanical phrases like 'düşündürüyor', 'olabileceğini düşündürüyor' over and over; vary your wording while keeping a professional tone.\n"
        "- Do NOT reuse the same leading sentence template across bullets (for example, do not start several bullets with 'Geçmişte, benzer büyüklükteki olaylar...' or 'Bu durum, fiyatın kısa vadede...'). Each bullet's first sentence must be structurally different from the others and introduce a clearly distinct idea.\n"
        "- If a metric is extreme, say it clearly: 'olağan dışı yüksek', 'üst banda yakın', 'normalin belirgin şekilde altında'.\n"
        "- If multiple metrics align (örn funding + OI), highlight the combined meaning.\n\n"
        "========================================================\n"
        "= OUTPUT FORMAT (STRICT – DO NOT BREAK)\n"
        "========================================================\n\n"
        "You MUST produce exactly **three sections**, each marked with the following tags:\n\n"
        "[WHAT_YOU_CAN_DO]\n"
        "• bullet point…\n"
        "• bullet point…\n"
        "• bullet point…\n\n"
        "[HISTORY]\n"
        "• bullet point…\n"
        "• bullet point…\n"
        "• bullet point…\n\n"
        "[RISKS]\n"
        "• bullet point…\n"
        "• bullet point…\n"
        "• bullet point…\n\n"
        "- WHAT_YOU_CAN_DO → 3–5 bullet points\n"
        "- HISTORY → 3–5 bullet points\n"
        "- RISKS → 3–6 bullet points\n\n"
        "DO NOT add any sections beyond these.\n"
        "DO NOT add a conclusion or final line. Stop after the last risk bullet.\n\n"
        "========================================================\n"
        "= SECTION DEFINITIONS\n"
        "========================================================\n\n"
        "1) [WHAT_YOU_CAN_DO]\n"
        "- Provide **contextual interpretation**, NOT trading instructions.\n"
        "- Use actual metrics to give **useful behavioural insights**:\n"
        "  Examples:\n"
        "  - Strong volume + rising OI → leverage build-up\n"
        "  - High volatility_index + high stress_index → environment unstable\n"
        "  - Strong liquidity gap → stop-hunt risk\n"
        "- Explain **how this event fits into a broader short-term behaviour pattern**.\n"
        "- Emphasize that whale flow is a **tamamlayıcı sinyal**, not standalone.\n\n"
        "2) [HISTORY]\n"
        "- Based on:\n"
        "  - similar_events_last_90d\n"
        "  - avg_reaction_15m_pct\n"
        "  - avg_reaction_1h_pct\n"
        "  - reaction_consistency_pct\n"
        "- Describe **typical past reactions** for similar-size events.\n"
        "- Give approximate ranges (örn: 'geçmişte ilk 15 dakikada çoğunlukla %2–4 arasında tepki gelmiş').\n"
        "- Clearly state that these are **historical tendencies**, not predictions.\n"
        "- If sample size is small or weak, say it once: 'örnek sayısı düşük, net bir desen oluşmamış'.\n\n"
        "3) [RISKS]\n"
        "- Base each bullet on clear structural factors:\n"
        "  Examples:\n"
        "  - High volatility + high volume → noise risk\n"
        "  - Orderbook imbalance + thin liquidity → wick/stop-hunt risk\n"
        "  - Funding + OI alignment → squeeze risk\n"
        "  - Strong trend + whale counter-flow → trap risk\n"
        "- Include:\n"
        "  - 1 risk about leverage behaviour\n"
        "  - 1 risk about liquidity structure\n"
        "  - 1 risk about emotional, short-horizon reactions\n"
        "  - 1 risk about exchange-specific flow dominance\n\n"
        "========================================================\n"
        "= PROVIDED INPUT BLOCKS (ALWAYS)\n"
        "========================================================\n\n"
        "You will receive three inputs:\n\n"
        "1) [ALERT]\n"
        "   - Symbol\n"
        "   - Exchange\n"
        "   - USD size\n"
        "   - Time\n"
        "   - Optional raw payload\n\n"
        "2) [SHORT-TERM MARKET DATA]\n"
        "   - Last price\n"
        "   - 1-minute change\n"
        "   - 5-minute change\n"
        "   - Realized volatility (annualized approximation)\n\n"
        "3) [WHALE AI METRICS JSON]\n"
        "   28-field metrics block including:\n"
        "   - whale inflow/outflow\n"
        "   - orderbook imbalance\n"
        "   - liquidity levels\n"
        "   - OI changes\n"
        "   - funding\n"
        "   - sentiment\n"
        "   - volume & volatility\n"
        "   - historical patterns\n"
        "   - impact score\n\n"
        "========================================================\n"
        "= YOUR TASK\n"
        "========================================================\n\n"
        "Using ONLY the data provided:\n\n"
        "❗ Produce:\n"
        "- 3–5 bullets under [WHAT_YOU_CAN_DO]\n"
        "- 3–5 bullets under [HISTORY]\n"
        "- 3–6 bullets under [RISKS]\n\n"
        "❗ Each bullet MUST:\n"
        "- contain real metric interpretation,\n"
        "- add NEW unique insight,\n"
        "- avoid repetition,\n"
        "- avoid generic language,\n"
        "- avoid trading instructions.\n\n"
        "❗ Finish cleanly after the last bullet in [RISKS].\n"
    )


def build_whale_prompt(
    symbol: str,
    exchange: str,
    amount_usd: float,
    timestamp_iso: str,
    raw_json: str | None,
) -> str:
    raw_part = f"\nRaw payload JSON (may be partial):\n{raw_json}" if raw_json else ""
    return f"""
We have observed a large whale-related event on a crypto exchange.

[ALERT]
- Symbol: {symbol}
- Exchange: {exchange}
- Notional size (USD): {amount_usd:,.0f}
- Time (UTC): {timestamp_iso}
{raw_part}

You will also receive:
- A [SHORT-TERM MARKET DATA] block with 1-minute and 5-minute price behaviour and realized
  volatility around the time of the alert.
- A [WHALE AI METRICS JSON] block with 28 fields describing orderbook, open interest, funding,
  volume/volatility, historical patterns and an impact score.

Your job is to use these inputs to produce three sections:
1) WHAT_YOU_CAN_DO
2) HISTORY
3) RISKS

1) WHAT_YOU_CAN_DO:
- Write 3–5 concise bullet points.
- Explain how this event can be used as an additional data point in the user's own decision
  process. Emphasize that it is NOT a standalone buy/sell signal.
- Refer explicitly to metrics when helpful. Examples:
  • If volume_5m_change_pct is strongly positive, mention that volume spiked relative to recent
    baseline.
  • If volatility_index is high and market_stress_index is elevated, note that the environment is
    already stressed.
  • If funding_rate and oi_5m_change_pct point in the same direction, explain what that says about
    leverage build-up.
- Use descriptive language like "kısa vadede hareketin önemli bir kısmı zaten fiyatlanmış olabilir"
  instead of direct instructions such as "alış yap" or "short aç".

2) HISTORY:
- Write 3–5 bullet points.
- Use metrics such as similar_events_last_90d, avg_reaction_15m_pct, avg_reaction_1h_pct and
  reaction_consistency_pct when they are available.
- Describe, in general terms, how similar-size events have behaved in the past (for example,
  whether short squeezes or continuation moves were more common, and typical short-term reaction
  ranges like ±3–5%).
- Make it clear that these are historical patterns, NOT predictions.
- If historical metrics are missing (null), say briefly that there is no strong pattern sample and
  rely more on current metrics (volume, volatility, positioning) for softer guidance.

3) RISKS:
- Write 3–6 bullet points.
- Base each risk on specific metrics or structural aspects. Examples:
  • High volume_5m_change_pct and high volatility_index can mean that single flows are harder to
    interpret because of noise.
  • When funding_rate and oi_5m_change_pct are aligned, there may be squeeze risk on the crowded
    side.
  • Strong orderbook_imbalance and liquidity_comment indicating thin liquidity can increase the
    chance of stop-hunting moves.
- Include at least one risk about using leverage in already stressed markets, one about focusing on
  a single exchange or timeframe, and one about emotional trading after large moves.
- End this section with a clear reminder that the analysis is informational only and NOT financial
  advice; the user must make their own decisions and define acceptable loss levels.

Output format (VERY IMPORTANT):
Return the answer as three sections, clearly marked with these tags exactly:

[WHAT_YOU_CAN_DO]
...bullet points...

[HISTORY]
...bullet points...

[RISKS]
...bullet points...
""".strip()


def _extract_tagged_section(content: str, tag: str) -> str:
    """Extract text under [TAG] ... until next [OTHER] or end."""
    if not content:
        return ""
    marker = f"[{tag}]"
    start = content.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    end = len(content)
    for other in ("WHAT_YOU_CAN_DO", "HISTORY", "RISKS"):
        if other == tag:
            continue
        idx = content.find(f"[{other}]", start)
        if idx != -1:
            end = min(end, idx)
    return content[start:end].strip()


def split_whale_sections(content: str) -> tuple[str, str, str]:
    """Split Groq content into (what_you_can_do, history, risks)."""
    what = _extract_tagged_section(content, "WHAT_YOU_CAN_DO")
    hist = _extract_tagged_section(content, "HISTORY")
    risks = _extract_tagged_section(content, "RISKS")
    return what, hist, risks


# -----------------------------
# Whale AI metrics builder
# -----------------------------

_OI_HISTORY: Dict[str, list[tuple[float, float]]] = {}


def _oi_push(symbol: str, oi_value: float, now_ts: float) -> None:
    """Store OI history per symbol (timestamp, value), keep ~20 minutes."""
    if not np.isfinite(oi_value):
        return
    s = symbol.upper()
    hist = _OI_HISTORY.get(s, [])
    hist.append((now_ts, float(oi_value)))
    # keep last 20 minutes
    cutoff = now_ts - 20 * 60
    hist = [row for row in hist if row[0] >= cutoff]
    _OI_HISTORY[s] = hist


def _oi_change_pct(symbol: str, window_sec: int, now_ts: float) -> float:
    s = symbol.upper()
    hist = _OI_HISTORY.get(s) or []
    if not hist:
        return 0.0
    current = hist[-1][1]
    target_ts = now_ts - window_sec
    past_vals = [v for (t, v) in hist if t <= target_ts]
    if not past_vals:
        return 0.0
    base = past_vals[-1]
    if not base:
        return 0.0
    return float((current - base) / base * 100.0)


def build_whale_metrics(symbol: str) -> Dict[str, Any]:
    """Compute as many Whale AI Engine metrics as possible from Binance data.

    Returns a dict with all 28 expected keys so that the LLM sees a stable schema.
    Fields we cannot compute yet are set to None; the prompt tells the model not
    to invent numbers when a field is null.
    """

    sym = symbol.upper()
    now_ts = time.time()

    # Defaults
    metrics: Dict[str, Any] = {
        # 1) ALERT temel verileri (bunlar whale_ai içinde ayrıca geçilecek)
        # 2) Whale Trend (placeholder - gerçek whale engine'den gelmeli)
        "whale_inflows_24h": None,
        "whale_outflows_24h": None,
        "whale_trend": None,
        "whale_deviation_vs_7d": None,
        # 3) Order book microstructure
        "orderbook_imbalance": None,
        "nearest_liquidity_support": None,
        "nearest_liquidity_resistance": None,
        "liquidity_comment": None,
        # 4) Open Interest changes
        "oi_1m_change_pct": 0.0,
        "oi_5m_change_pct": 0.0,
        "oi_15m_change_pct": 0.0,
        # 5) Funding & positioning
        "funding_rate": 0.0,
        "positioning": None,
        "sentiment_aggregated": None,
        "sentiment_score": None,
        # 6) Volume & volatility
        "volume_5m_change_pct": 0.0,
        "volatility_index": 0.0,
        "market_stress_index": 0.0,
        # 7) Historical pattern (placeholder)
        "similar_events_last_90d": None,
        "avg_reaction_15m_pct": None,
        "avg_reaction_1h_pct": None,
        "reaction_consistency_pct": None,
        # 8) Impact score (derived placeholder)
        "impact_score": None,
    }

    # --- Orderbook microstructure ---
    try:
        ob = get_orderbook(sym, limit=20, ttl=2.0) or {"bids": [], "asks": []}
        bids = ob.get("bids") or []
        asks = ob.get("asks") or []
        bids_f = [[float(p), float(q)] for p, q in bids]
        asks_f = [[float(p), float(q)] for p, q in asks]
        bid_vol = float(sum(q for _, q in bids_f)) if bids_f else 0.0
        ask_vol = float(sum(q for _, q in asks_f)) if asks_f else 0.0
        total = bid_vol + ask_vol
        if total > 0:
            imbalance = (bid_vol - ask_vol) / total * 100.0
            metrics["orderbook_imbalance"] = float(imbalance)
        # nearest liquidity levels: en yakın büyük bid/ask
        def _nearest_liq(levels: list[list[float]], side: str) -> float | None:
            if not levels:
                return None
            # en yüksek hacimli ilk 5 seviye içinden fiyata en yakın olanı al
            sorted_by_vol = sorted(levels, key=lambda x: x[1], reverse=True)[:5]
            # zaten fiyata göre sıralı geldiği varsayımı zayıf; en büyük hacimli seviye yeterli
            return float(sorted_by_vol[0][0]) if sorted_by_vol else None

        support = _nearest_liq(bids_f, "bid")
        resistance = _nearest_liq(asks_f, "ask")
        metrics["nearest_liquidity_support"] = support
        metrics["nearest_liquidity_resistance"] = resistance

        # liquidity_comment: çok kabaca duvar yoğunluğuna göre
        if total > 0:
            ratio = bid_vol / total
            if ratio > 0.65:
                metrics["liquidity_comment"] = "buy_wall"
            elif ratio < 0.35:
                metrics["liquidity_comment"] = "sell_wall"
            else:
                metrics["liquidity_comment"] = "balanced"
    except Exception:
        pass

    # --- Funding & OI ---
    try:
        funding = get_funding_rate_rest(sym)
        fr = float(funding.get("fundingRate", 0.0) or 0.0)
        oi_val = float(funding.get("openInterest", 0.0) or 0.0)
        metrics["funding_rate"] = fr
        _oi_push(sym, oi_val, now_ts)
        # compute short-term OI pct changes
        metrics["oi_1m_change_pct"] = _oi_change_pct(sym, 60, now_ts)
        metrics["oi_5m_change_pct"] = _oi_change_pct(sym, 5 * 60, now_ts)
        metrics["oi_15m_change_pct"] = _oi_change_pct(sym, 15 * 60, now_ts)

        # positioning: basit sınıflama
        oi5 = metrics["oi_5m_change_pct"] or 0.0
        if fr > 0 and oi5 > 0:
            metrics["positioning"] = "longs_aggressive"
        elif fr < 0 and oi5 > 0:
            metrics["positioning"] = "shorts_aggressive"
        else:
            metrics["positioning"] = "balanced"
    except Exception:
        pass

    # --- Volume & Volatility from 1m OHLCV ---
    try:
        df = get_ohlcv(sym, "1m", 60, ttl=2.0)
        if df is not None and not df.empty:
            df = ensure_numeric_ohlcv(df)
            close = df.get("close")
            vol = df.get("volume")
            if close is not None and not close.empty:
                tail = close.tail(30)
                ret = tail.pct_change().dropna()
                if not ret.empty:
                    vol_annual = float(ret.std(ddof=0) * np.sqrt(60 * 24 * 365))
                    # normalize to 0-100 band roughly, clip
                    metrics["volatility_index"] = float(max(0.0, min(100.0, vol_annual * 100.0)))
            if vol is not None and len(vol) >= 10:
                last5 = float(vol.tail(5).sum())
                prev5 = float(vol.tail(10).head(5).sum()) or 0.0
                if prev5 > 0:
                    metrics["volume_5m_change_pct"] = float((last5 - prev5) / prev5 * 100.0)
            # market_stress_index: volatilite + abs(oi_5m_change_pct)
            v_idx = float(metrics.get("volatility_index") or 0.0)
            oi5_abs = abs(float(metrics.get("oi_5m_change_pct") or 0.0))
            stress_raw = v_idx * 0.6 + min(oi5_abs, 50.0) * 0.4
            metrics["market_stress_index"] = float(max(0.0, min(100.0, stress_raw)))
    except Exception:
        pass

    # impact_score: basit birleştirilmiş gösterge (placeholder)
    try:
        vol_idx = float(metrics.get("volatility_index") or 0.0)
        stress = float(metrics.get("market_stress_index") or 0.0)
        ob_imb = abs(float(metrics.get("orderbook_imbalance") or 0.0))
        base = 0.4 * vol_idx + 0.4 * stress + 0.2 * min(ob_imb, 100.0)
        metrics["impact_score"] = float(max(0.0, min(100.0, base)))
    except Exception:
        pass

    return metrics

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "API v2 çalışıyor!"}


@app.get("/debug/markets/binance")
def debug_markets_binance():
    """Debug endpoint: sync Binance markets into the in-memory registry and
    return a small sample. This is for development/inspection only and is not
    used by the mobile app in production flows yet.

    It is safe to call multiple times; the registry keeps only one copy of
    each market/coin in memory.
    """

    try:
        REGISTRY.sync_from_binance()
    except Exception as e:
        return {"error": f"sync_failed: {e}"}

    # Build a small summary to keep payload reasonable
    exchanges = [
        {"id": ex.id, "name": ex.name, "type": ex.type}
        for ex in REGISTRY.exchanges.values()
    ]

    coins_sample = []
    for i, coin in enumerate(sorted(REGISTRY.coins.values(), key=lambda c: c.id)[:20]):
        coins_sample.append(
            {
                "id": coin.id,
                "name": coin.name,
                "primary_market_id": coin.primary_market_id,
                "reference_market_id_for_indicators": coin.reference_market_id_for_indicators,
                "markets_count": len(coin.markets),
            }
        )

    markets_sample = []
    for i, m in enumerate(sorted(REGISTRY.markets.values(), key=lambda m: m.id)[:50]):
        markets_sample.append(
            {
                "id": m.id,
                "exchange_id": m.exchange_id,
                "symbol": m.symbol_raw,
                "base": m.base_asset,
                "quote": m.quote_asset,
                "type": m.type,
                "status": m.status,
            }
        )

    return {
        "exchanges": exchanges,
        "coins_sample": coins_sample,
        "markets_sample": markets_sample,
        "stats": {
            "total_exchanges": len(REGISTRY.exchanges),
            "total_coins": len(REGISTRY.coins),
            "total_markets": len(REGISTRY.markets),
        },
    }


@app.get("/scalping/{symbol}")
def scalping(symbol: str = "BTCUSDT"):
    try:
        exchange_snapshots = build_exchange_snapshots(symbol)
        exchange_context = build_exchange_context(symbol, candle_interval="15m")
        indicator_source_exchange, df = get_indicator_ohlcv_with_fallback(symbol, "15m", 100, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}

        # Ensure numeric types
        df = ensure_numeric_ohlcv(df)

        # Pivots
        high = df['high'].max()
        low = df['low'].min()
        close = df['close'].iloc[-1]
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)

        # Fibo
        swing_high = high
        swing_low = low
        diff = swing_high - swing_low
        fibonacci_levels = {
            "0.236": swing_high - 0.236 * diff,
            "0.382": swing_high - 0.382 * diff,
            "0.500": swing_high - 0.500 * diff,
            "0.618": swing_high - 0.618 * diff,
            "0.786": swing_high - 0.786 * diff,
        }

        # VWAP (manual)
        vwap = manual_vwap(df)

        # Keltner & Bollinger (robust)
        keltner = ta.kc(df['high'], df['low'], df['close'])
        bb = robust_bbands(df['close'])

        # ATR
        atr_raw = None
        try:
            atr_raw = ta.atr(df['high'], df['low'], df['close'])
        except Exception:
            atr_raw = None
        atr_vals = (atr_raw.dropna() if isinstance(atr_raw, pd.Series) else pd.Series(dtype='float64'))
        current_atr = float(atr_vals.iloc[-1]) if not atr_vals.empty else 0.0

        # Return
        return {
            "exchange_snapshots": exchange_snapshots,
            "exchange_context": exchange_context,
            "indicator_source_exchange": indicator_source_exchange,
            "priceData": {
                "currentPrice": safe_float(df['close'].iloc[-1]),
                "high24h": safe_float(df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()),
                "low24h": safe_float(df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()),
                "atr": safe_float(current_atr)
            },
            "pivotPoints": {
                "pivot": safe_float(pp),
                "resistance1": safe_float(r1),
                "support1": safe_float(s1),
                "resistance2": safe_float(r2),
                "support2": safe_float(s2)
            },
            "fibonacciLevels": {k: safe_float(v) for k, v in fibonacci_levels.items()},
            "vwap": safe_list(vwap.dropna().tail(3).tolist()),
            "technicalIndicators": {
                "RSI": series_tail_floats(ta.rsi(df['close'], length=14), 3),
                "MACD": row_to_float_dict(ta.macd(df['close'])),
                "WilliamsR": series_tail_floats(ta.willr(df['high'], df['low'], df['close']), 3),
                "CCI": series_tail_floats(ta.cci(df['high'], df['low'], df['close']), 3),
                "ATR": series_tail_floats(atr_vals if isinstance(atr_vals, pd.Series) else pd.Series(dtype='float64'), 3),
                "Keltner": row_to_float_dict(keltner),
                "BollingerBands": build_bollinger_dict(df['close']),
                "Bollinger": build_bollinger_dict(df['close']),
                "Volume": series_tail_floats(df['volume'], 3),
            },
            "microMetrics": {
                "RSI_1m": float(last_rsi_value_for(symbol, '1m', 14, 120, 2.0, preferred_exchange=indicator_source_exchange)),
                "RSI_5m": float(last_rsi_value_for(symbol, '5m', 14, 120, 2.0, preferred_exchange=indicator_source_exchange)),
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/news")
def news(symbol: Optional[str] = None, max_age_days: int = 7):
    try:
        # For now, serve a small mock list compatible with APIService.NewsItem / NewsResponse
        # Fields: id, title, url, source, description, created_at, kind, votes, currencies
        now_iso = datetime.utcnow().isoformat() + "Z"
        base = symbol or "BTC"
        items = [
            {
                "id": f"{base}-{int(time.time())}",
                "title": f"{base} market update",
                "url": f"https://news.example.com/{base.lower()}",
                "source": "CryptoAI",
                "description": f"{base} için piyasa özeti ve volatilite görünümü.",
                "created_at": now_iso,
                "kind": "article",
                "votes": {"up": 10, "down": 1},
                "currencies": [base]
            }
        ]
        return {"items": items}
    except Exception as e:
        return {"items": [], "error": str(e)}


@app.get("/miniscalping/{symbol}")
def miniscalping(symbol: str = "BTCUSDT"):
    try:
        exchange_snapshots = build_exchange_snapshots(symbol)
        exchange_context = build_exchange_context(symbol, candle_interval="5m")
        indicator_source_exchange, df = get_indicator_ohlcv_with_fallback(symbol, "5m", 100, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}
        df = ensure_numeric_ohlcv(df)

        # Technicals
        rsi = ta.rsi(df['close'], length=14)
        macd_df = ta.macd(df['close'])
        stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        atr = ta.atr(df['high'], df['low'], df['close'])
        ema_12 = ta.ema(df['close'], length=12)
        ema_26 = ta.ema(df['close'], length=26)
        sma_20 = ta.sma(df['close'], length=20)
        adx_df = ta.adx(df['high'], df['low'], df['close'])
        bb_df = robust_bbands(df['close'])
        stochrsi_df = ta.stochrsi(df['close'])
        cci = ta.cci(df['high'], df['low'], df['close'])
        obv = ta.obv(df['close'], df['volume'])
        vwap = manual_vwap(df)

        # Pivots
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])

        # Sentiment mock
        fear_greed_index = int(np.random.randint(20, 80))
        social_sentiment = float(np.random.uniform(-1, 1))
        news_sentiment = float(np.random.uniform(-1, 1))

        # Orderbook / Funding
        ob = get_orderbook(symbol, limit=20, ttl=2.0) or {"bids": [], "asks": []}
        # Convert string bids/asks -> float pairs for iOS decoding [[Double]]
        bids_f = [[float(p), float(q)] for p, q in ob.get('bids', [])[:10]] if ob.get('bids') else []
        asks_f = [[float(p), float(q)] for p, q in ob.get('asks', [])[:10]] if ob.get('asks') else []
        funding = get_funding_rate_rest(symbol)

        # Response
        return {
            "exchange_snapshots": exchange_snapshots,
            "exchange_context": exchange_context,
            "indicator_source_exchange": indicator_source_exchange,
            "priceData": {
                "currentPrice": float(df['close'].iloc[-1]),
                "priceChange24h": None,
                "priceChangePercent24h": None,
                "volume24h": None,
                "high24h": float(df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()),
                "low24h": float(df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()),
                "volatility": float((df['close'].pct_change().dropna().std() or 0.0) * np.sqrt(24 * 365)),
                "atr": float(atr.dropna().iloc[-1] if isinstance(atr, pd.Series) and not atr.dropna().empty else 0.0),
            },
            "pivotPoints": {
                "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
            },
            "marketSentiment": {
                "fearGreedIndex": fear_greed_index,
                "socialSentiment": safe_float(social_sentiment),
                "newsSentiment": safe_float(news_sentiment),
                "institutionalFlow": safe_float(np.random.uniform(1e7, 5e7))
            },
            "orderBook": {
                "bids": bids_f,
                "asks": asks_f,
                "spread": (lambda a,b: safe_float(a-b))(asks_f[0][0], bids_f[0][0]) if asks_f and bids_f else MISSING,
                "bidVolume": safe_float(sum([lvl[1] for lvl in bids_f])) if bids_f else MISSING,
                "askVolume": safe_float(sum([lvl[1] for lvl in asks_f])) if asks_f else MISSING,
            },
            "fundingRate": {
                "symbol": symbol,
                "fundingRate": safe_float(funding.get('fundingRate', 0.0001)),
                "nextFundingTime": funding.get('nextFundingTime', 0),
                "openInterest": funding.get('openInterest', 0)
            },
            "correlationData": {
                "btcCorrelation": safe_float(np.random.uniform(0.5, 0.95)),
                "ethCorrelation": safe_float(np.random.uniform(0.3, 0.8)),
                "marketCapRank": 1
            },
            "onChainMetrics": {
                "activeAddresses": int(np.random.randint(5e5, 1e6)),
                "transactionCount": int(np.random.randint(2e5, 5e5)),
                "networkHashRate": safe_float(np.random.uniform(3e8, 6e8)),
                "stakingRatio": safe_float(np.random.uniform(0.1, 0.3)),
                "whaleTransactions": int(np.random.randint(20, 100))
            },
            "technicalIndicators": {
                "RSI": series_tail_floats(rsi, 3),
                "MACD": row_to_float_dict(macd_df),
                "Stochastic": row_to_float_dict(stoch_df),
                "ATR": series_tail_floats(atr, 3),
                "Volume": series_tail_floats(df['volume'], 3),
                "EMA_12": series_tail_floats(ema_12, 3),
                "EMA_26": series_tail_floats(ema_26, 3),
                "SMA_20": series_tail_floats(sma_20, 3),
                "ADX": row_to_float_dict(adx_df),
                "BollingerBands": build_bollinger_dict(df['close']),
                "Bollinger": build_bollinger_dict(df['close']),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
                },
            },
            "microMetrics": {
                "RSI_1m": float(last_rsi_value_for(symbol, '1m', 14, 120, 2.0, preferred_exchange=indicator_source_exchange)),
                "RSI_5m": float(last_rsi_value_for(symbol, '5m', 14, 120, 2.0, preferred_exchange=indicator_source_exchange)),
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/dailytrading/{symbol}")
def dailytrading(symbol: str = "BTCUSDT"):
    try:
        exchange_snapshots = build_exchange_snapshots(symbol)
        exchange_context = build_exchange_context(symbol, candle_interval="1h")
        indicator_source_exchange, df = get_indicator_ohlcv_with_fallback(symbol, "1h", 100, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])

        rsi_values = ta.rsi(df['close'], length=14).dropna()
        macd_values = ta.macd(df['close'])
        atr_values = ta.atr(df['high'], df['low'], df['close']).dropna()

        bb = robust_bbands(df['close'])

        return {
            "exchange_snapshots": exchange_snapshots,
            "exchange_context": exchange_context,
            "indicator_source_exchange": indicator_source_exchange,
            "priceData": {
                "currentPrice": safe_float(float(df['close'].iloc[-1])),
                "high24h": safe_float(float(df['high'].tail(24).max())),
                "low24h": safe_float(float(df['low'].tail(24).min())),
                "atr": safe_float(float(atr_values.iloc[-1]) if not atr_values.empty else 0.0),
            },
            "pivotPoints": {
                "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0),
            },
            "technicalIndicators": {
                "RSI": series_tail_floats(rsi_values, 3),
                "MACD": row_to_float_dict(macd_values),
                "ADX": row_to_float_dict(ta.adx(df['high'], df['low'], df['close'])),
                "BollingerBands": build_bollinger_dict(df['close']),
                "Bollinger": build_bollinger_dict(df['close']),
                "ATR": series_tail_floats(atr_values, 3),
                "Volume": series_tail_floats(df['volume'], 10),
                "PivotPoints": {
                    "pivot": float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": float(s2.iloc[-1] if not s2.empty else 0.0),
                },
            },
            "marketSentiment": {
                "fearGreedIndex": int(np.random.randint(20, 80)),
                "socialSentiment": safe_float(float(np.random.uniform(-1, 1))),
                "newsSentiment": safe_float(float(np.random.uniform(-1, 1))),
                "institutionalFlow": safe_float(float(np.random.uniform(1e7, 5e7)))
            },
            "correlationData": {
                "btcCorrelation": safe_float(float(np.random.uniform(0.5, 0.95))),
                "ethCorrelation": safe_float(float(np.random.uniform(0.3, 0.8))),
                "marketCapRank": 1
            },
            "onChain": {
                "activeAddresses": int(np.random.randint(5e5, 1e6)),
                "transactionCount": int(np.random.randint(2e5, 5e5)),
                "networkHashRate": safe_float(float(np.random.uniform(3e8, 6e8))),
                "stakingRatio": safe_float(float(np.random.uniform(0.1, 0.3))),
                "whaleTransactions": int(np.random.randint(20, 100))
            },
            "multiTimeframe": {
                "1h": {"trend": "bullish" if df['close'].iloc[-1] > df['close'].iloc[-2] else "bearish", "strength": safe_float(float(np.random.uniform(0.3, 0.9)))},
                "4h": {"trend": "bullish", "strength": safe_float(float(np.random.uniform(0.3, 0.9)))},
                "1d": {"trend": "bearish", "strength": safe_float(float(np.random.uniform(0.3, 0.9)))},
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/orderbook/{symbol}")
def orderbook(symbol: str = "BTCUSDT", limit: int = 20):
    try:
        ob = get_orderbook(symbol, limit=limit, ttl=2.0)
        if not ob:
            return {"error": "Orderbook alınamadı"}
        return {
            "lastUpdateId": int(datetime.utcnow().timestamp() * 1000),
            "bids": [[float(p), float(q)] for p, q in ob.get('bids', [])[:limit]] if ob.get('bids') else [],
            "asks": [[float(p), float(q)] for p, q in ob.get('asks', [])[:limit]] if ob.get('asks') else [],
            "spread": (float(ob['asks'][0][0]) - float(ob['bids'][0][0])) if ob.get('asks') and ob.get('bids') else 0.0,
            "bidVolume": float(sum([float(b[1]) for b in ob.get('bids', [])[:limit]])) if ob.get('bids') else 0.0,
            "askVolume": float(sum([float(a[1]) for a in ob.get('asks', [])[:limit]])) if ob.get('asks') else 0.0,
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/sentiment/{symbol}")
def sentiment(symbol: str = "BTCUSDT"):
    try:
        return {
            "fearGreedIndex": int(np.random.randint(20, 80)),
            "socialSentiment": float(np.random.uniform(-1, 1)),
            "newsSentiment": float(np.random.uniform(-1, 1)),
            "institutionalFlow": float(np.random.uniform(1e7, 5e7)),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/indicators_binance/{symbol}")
def indicators_binance(symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100):
    try:
        ref_symbol = resolve_reference_symbol(symbol)
        df = get_ohlcv(ref_symbol, interval, limit, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        out = {
            "RSI": series_tail_floats(ta.rsi(df['close'], length=14), 100),
            "MACD": row_to_float_dict(ta.macd(df['close'])),
            "EMA_12": series_tail_floats(ta.ema(df['close'], length=12), 100),
            "EMA_26": series_tail_floats(ta.ema(df['close'], length=26), 100),
            "SMA_20": series_tail_floats(ta.sma(df['close'], length=20), 100),
            "ADX": row_to_float_dict(ta.adx(df['high'], df['low'], df['close'])),
            "BollingerBands": row_to_float_dict(robust_bbands(df['close'])),
            "StochRSI": row_to_float_dict(ta.stochrsi(df['close'])),
            "CCI": series_tail_floats(ta.cci(df['high'], df['low'], df['close']), 100),
            "ATR": series_tail_floats(ta.atr(df['high'], df['low'], df['close']), 100),
            "OBV": series_tail_floats(ta.obv(df['close'], df['volume']), 100),
            "VWAP": series_tail_floats(manual_vwap(df), 100),
            "Keltner": row_to_float_dict(ta.kc(df['high'], df['low'], df['close'])),
            "Volume": series_tail_floats(df['volume'], 100),
        }
        # Pivot points (single
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        out["PivotPoints"] = {
            "pivot": float(pivot.iloc[-1] if not pivot.empty else 0.0),
            "resistance1": float(r1.iloc[-1] if not r1.empty else 0.0),
            "support1": float(s1.iloc[-1] if not s1.empty else 0.0),
            "resistance2": float(r2.iloc[-1] if not r2.empty else 0.0),
            "support2": float(s2.iloc[-1] if not s2.empty else 0.0),
        }
        return out
    except Exception as e:
        return {"error": str(e)}


@app.get("/swingtrading/{symbol}")
def swingtrading(symbol: str = "BTCUSDT"):
    try:
        exchange_snapshots = build_exchange_snapshots(symbol)
        # Use 1d data for swing context
        ref_symbol = resolve_reference_symbol(symbol)
        df = get_ohlcv(ref_symbol, "1d", 100, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])

        rsi_values = ta.rsi(df['close'], length=14).dropna()
        macd_values = ta.macd(df['close'])
        bb = robust_bbands(df['close'])

        return {
            "exchange_snapshots": exchange_snapshots,
            "priceData": {
                "currentPrice": float(df['close'].iloc[-1]),
                "high24h": float(df['high'].iloc[-1]),
                "low24h": float(df['low'].iloc[-1]),
                "atr": float((ta.atr(df['high'], df['low'], df['close']).dropna().iloc[-1]) if not ta.atr(df['high'], df['low'], df['close']).dropna().empty else 0.0),
            },
            "pivotPoints": {
                "pivot": float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": float(s2.iloc[-1] if not s2.empty else 0.0),
                "nearestSupport": float(s1.iloc[-1] if not s1.empty else 0.0),
                "nearestResistance": float(r1.iloc[-1] if not r1.empty else 0.0)
            },
            "fundamental": {
                "marketCap": float(np.random.uniform(5e11, 1e12)),
                "circulatingSupply": float(np.random.uniform(1.8e7, 2.1e7)),
                "developerActivity": float(np.random.uniform(0.6, 0.9)),
                "githubCommits": int(np.random.randint(800, 2000)),
                "roadmapProgress": float(np.random.uniform(0.6, 0.9))
            },
            "macro": {
                "fedRate": float(np.random.uniform(4.5, 6.0)),
                "inflation": float(np.random.uniform(2.5, 4.0)),
                "dollarIndex": float(np.random.uniform(100, 105)),
                "goldPrice": float(np.random.uniform(1900, 2200)),
                "oilPrice": float(np.random.uniform(70, 85))
            },
            "regulatory": {
                "secStatus": "pending",
                "euRegulation": "compliant",
                "asiaRegulation": "partial",
                "regulatoryRisk": float(np.random.uniform(0.1, 0.5))
            },
            "technicalIndicators": {
                "RSI": series_tail_floats(rsi_values, 3),
                "MACD": row_to_float_dict(macd_values),
                "ADX": row_to_float_dict(ta.adx(df['high'], df['low'], df['close'])),
                "BollingerBands": row_to_float_dict(bb)
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/ai/mini/{symbol}")
def ai_mini(symbol: str = "BTCUSDT"):
    try:
        ref_symbol = resolve_reference_symbol(symbol)
        # Fetch base data
        df = get_ohlcv(ref_symbol, "5m", 120, ttl=2.0)
        if df is None or df.empty:
            return {"commentary": "Veri alınamadı: OHLCV boş."}

        # Ensure numeric
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

        # Indicators
        rsi_series = ta.rsi(df['close'], length=14).dropna()
        macd_df = ta.macd(df['close'])
        bb = robust_bbands(df['close'])
        bb_dict = bollinger_with_alias(row_to_float_dict(bb))
        bbp = bb_dict.get("BBP_20_2.0_2.0") or bb_dict.get("bbp") or None

        atr = ta.atr(df['high'], df['low'], df['close']).dropna()
        atr_val = float(atr.iloc[-1]) if not atr.empty else 0.0

        # Orderbook spread (same reference symbol)
        ob = get_orderbook(ref_symbol, limit=10, ttl=2.0) or {"bids": [], "asks": []}
        spread = 0.0
        if ob.get('asks') and ob.get('bids'):
            spread = float(ob['asks'][0][0]) - float(ob['bids'][0][0])

        # Extract last values
        rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.empty else 0.0
        macd_hist = 0.0
        if isinstance(macd_df, pd.DataFrame) and not macd_df.dropna().empty:
            row = macd_df.dropna().iloc[-1]
            macd_hist = float(row.get('MACDh_12_26_9', 0.0))

        # Simple rules
        bias = "nötr"
        if rsi_val >= 60 and macd_hist > 0:
            bias = "yükseliş"
        elif rsi_val <= 40 and macd_hist < 0:
            bias = "düşüş"

        vol_note = "yüksek volatilite" if atr_val > 0 and (atr_val / max(1e-9, float(df['close'].iloc[-1])) > 0.002) else "sakin volatilite"
        bb_note = "orta bant çevresi" if bbp is None else ("üst banda yakın" if bbp > 0.6 else ("alt banda yakın" if bbp < 0.4 else "orta bant çevresi"))
        liq_note = "spread dar" if spread < (float(df['close'].iloc[-1]) * 0.0005) else "spread geniş"

        text = f"AI: {symbol} için {bias}. {bb_note}, {vol_note}, {liq_note}. RSI={rsi_val:.1f}, MACDh={macd_hist:.3f}."
        return {"commentary": text}
    except Exception as e:
        return {"commentary": f"Veri alınamadı: {str(e)}"}


@app.post("/whale_ai")
def whale_ai(payload: dict = Body(...)):
    """Whale alert AI analysis endpoint used by the iOS app.

    Expects JSON:
    {
      "symbol": "BTCUSDT",
      "exchange": "Binance Futures",
      "amount_usd": 123456.0,
      "timestamp_ms": 1733839200000,
      "language": "tr",
      "raw_json": "{...}"
    }

    Returns JSON:
    {
      "what_you_can_do": "...",
      "history": "...",
      "risks": "..."
    }
    """

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY not configured"})

    symbol = str(payload.get("symbol") or "BTCUSDT").upper()
    exchange = str(payload.get("exchange") or "Unknown")
    try:
        amount_usd = float(payload.get("amount_usd") or 0.0)
    except Exception:
        amount_usd = 0.0
    try:
        ts_ms_raw = int(payload.get("timestamp_ms") or 0)
    except Exception:
        ts_ms_raw = 0
    language = str(payload.get("language") or "en")
    raw_json = payload.get("raw_json")
    if isinstance(raw_json, dict):
        try:
            raw_json = json.dumps(raw_json)  # type: ignore[assignment]
        except Exception:
            raw_json = None

    # Fallback to now if timestamp is invalid
    try:
        ts_iso = datetime.utcfromtimestamp(ts_ms_raw / 1000.0).isoformat() + "Z"
    except Exception:
        ts_iso = datetime.utcnow().isoformat() + "Z"

    system_prompt = whale_system_instructions(language)

    # Build metrics block (28-field schema) from market data; some fields may remain None
    metrics = build_whale_metrics(symbol)

    user_prompt = build_whale_prompt(
        symbol,
        exchange,
        amount_usd,
        ts_iso,
        raw_json if isinstance(raw_json, str) else None,
    )

    # Attach structured metrics JSON so the model sees concrete numbers but does not invent missing ones
    try:
        metrics_json = json.dumps(metrics, ensure_ascii=False)
        user_prompt = f"{user_prompt}\n\nWhale AI Engine metrics for this symbol (JSON, some fields may be null – never invent numbers):\n{metrics_json}"
    except Exception:
        pass

    # Enrich prompt with real short-term price behaviour (1m candles)
    recent_block = ""
    try:
        df = get_ohlcv(symbol, "1m", 60, ttl=2.0)
        if df is not None and not df.empty:
            df = ensure_numeric_ohlcv(df)
            close_series = df.get("close")
            if close_series is not None and not close_series.empty:
                last_price = float(close_series.iloc[-1])
                # 1 dakika önceki ve 5 dakika önceki kapanışlar
                price_1m_ago = float(close_series.iloc[-2]) if len(close_series) >= 2 else last_price
                price_5m_ago = float(close_series.iloc[-6]) if len(close_series) >= 6 else price_1m_ago

                change_1m = ((last_price - price_1m_ago) / price_1m_ago * 100.0) if price_1m_ago else 0.0
                change_5m = ((last_price - price_5m_ago) / price_5m_ago * 100.0) if price_5m_ago else 0.0

                # Basit realized volatilite (son 30 bar)
                tail = close_series.tail(30)
                ret = tail.pct_change().dropna()
                vol_annual = float(ret.std(ddof=0) * np.sqrt(60 * 24 * 365)) if not ret.empty else 0.0

                recent_block = (
                    "\n\nRecent 1-minute market data (for context, not a signal):\n"
                    f"- Last price: {last_price:.4f} USDT\n"
                    f"- Change over last 1 minute: {change_1m:+.2f}%\n"
                    f"- Change over last 5 minutes: {change_5m:+.2f}%\n"
                    f"- Realized volatility (approx, last 30×1m bars, annualized): {vol_annual:.1f}%\n"
                    "Use these numbers only as context when describing what happened after this flow; do not treat them as a trading signal."
                )
    except Exception:
        recent_block = ""

    if recent_block:
        user_prompt = f"{user_prompt}{recent_block}"

    groq_payload = {
        "model": os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.4,
        "max_tokens": 800,
        "top_p": 0.95,
    }

    try:
        resp = SESSION.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CryptoAI/2.0",
            },
            json=groq_payload,
            timeout=30,
        )
    except Exception as e:
        return JSONResponse(status_code=502, content={"error": f"ai_call_failed: {str(e)}"})

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

    try:
        data = resp.json()
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"invalid_ai_response: {str(e)}"})

    content = (
        (data.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    what, hist, risks = split_whale_sections(str(content))

    return {
        "what_you_can_do": what or "",
        "history": hist or "",
        "risks": risks or "",
        "metrics": metrics,
    }

@app.post("/ai/chat")
def ai_chat(payload: dict = Body(...)):
    """
    Proxy endpoint to Groq Chat Completions API.
    Expects a JSON body with fields: model, messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty.
    Uses GROQ_API_KEY from environment variables.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "GROQ_API_KEY not configured"})
    try:
        resp = SESSION.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "CryptoAI/2.0"
            },
            json=payload,
            timeout=30,
        )
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"error": resp.text})
        data = resp.json()
        # Return raw Groq response to the client; iOS will parse `choices[0].message.content`.
        return data
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
