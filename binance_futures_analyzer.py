#!/usr/bin/env python3
"""Binance Futures real-time analyzer using only free WebSocket and REST data.

This module is designed to be production-minded despite living in a single file:

* Uses asyncio for concurrency, separating each Binance stream into its own task
* Provides resilient reconnect logic with exponential backoff and REST catch-up
* Builds synthetic 1-second candles from aggTrades (Binance does not offer 1s klines)
* Computes directional pressure, spoofing / manipulation risk, pump/dump alerts,
  liquidation clusters, and a heuristic open-interest trend indicator
* Applies adaptive thresholds (z-scores, moving averages) with per-symbol overrides
* Keeps all rolling windows bounded to avoid unbounded memory growth

Prerequisites
-------------
Python 3.10+
```
pip install aiohttp websockets
```

Run
---
```
python binance_futures_analyzer.py
```
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import random
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import aiohttp
import websockets
from websockets.server import serve as ws_serve

BINANCE_WS_BASE = "wss://fstream.binance.com/stream?streams="
MEXC_WS_BASE = "wss://contract.mexc.com/edge"
BINANCE_REST_BASE = "https://fapi.binance.com"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Curated universe of major USDT perpetuals. This is a shared backend, so
# even if thousands of mobile users connect, Binance load depends only on
# how many symbols we subscribe to here, not user count.
#
# We deliberately keep this to a few dozen of the most liquid contracts to
# stay well within WebSocket stream limits and avoid unnecessary noise.
DEFAULT_USDT_SYMBOLS: List[str] = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
    "TONUSDT",
    "TRXUSDT",
    "MATICUSDT",
    "OPUSDT",
    "ARBUSDT",
    "LTCUSDT",
    "APTUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "INJUSDT",
    "ATOMUSDT",
    "SUIUSDT",
    "PEPEUSDT",
    "RUNEUSDT",
    "SEIUSDT",
    "AAVEUSDT",
    "UNIUSDT",
    "ETCUSDT",
    "STXUSDT",
    "PYTHUSDT",
    "FTMUSDT",
    "TIAUSDT",
    "IMXUSDT",
    "1000BONKUSDT",
]


def _default_symbol_list() -> Dict[str, "SymbolConfig"]:
    base: Dict[str, SymbolConfig] = {
        # BTC: biraz daha agresif eşikler
        "BTCUSDT": SymbolConfig(
            long_pressure_threshold=0.55,
            short_pressure_threshold=-0.55,
            imbalance_threshold=0.7,
            wall_size_factor=3.0,
            volume_zscore_threshold=3.0,
            liquidation_notional_threshold=75_000,
            oi_positive_threshold=120_000,
            oi_negative_threshold=-120_000,
        ),
        # ETH: default kalibrasyon yeterli
        "ETHUSDT": SymbolConfig(),
    }

    # Diğer semboller için default SymbolConfig kullan.
    for sym in DEFAULT_USDT_SYMBOLS:
        base.setdefault(sym, SymbolConfig())

    return base


@dataclass
class SymbolConfig:
    """Per-symbol calibration knobs."""

    long_pressure_threshold: float = 0.6
    short_pressure_threshold: float = -0.6
    imbalance_threshold: float = 0.7
    wall_size_factor: float = 3.0
    volume_zscore_threshold: float = 3.0
    liquidation_notional_threshold: float = 50_000.0
    oi_positive_threshold: float = 80_000.0
    oi_negative_threshold: float = -80_000.0
    top_depth_levels: int = 20
    candle_history: int = 120
    volume_stats_window: int = 90  # seconds


@dataclass
class Config:
    """Global tuning parameters."""

    symbols: Dict[str, SymbolConfig] = field(default_factory=_default_symbol_list)
    aggtrade_rest_limit: int = 1000
    reconnect_backoff_initial: float = 1.0
    reconnect_backoff_max: float = 60.0
    liquidation_window_seconds: int = 30
    liquidation_cluster_window_ms: int = 2_000
    liquidation_cluster_count: int = 3
    trade_pressure_windows_ms: Tuple[int, int] = (1_000, 5_000)
    trade_pressure_long_window_ms: int = 60_000
    event_queue_size: int = 20_000
    orderbook_history: int = 60  # store ~6 seconds of depth@100ms

    def for_symbol(self, symbol: str) -> SymbolConfig:
        return self.symbols.get(symbol, SymbolConfig())


CONFIG = Config()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Alert Engine (Phase 1.1) - In-memory rules (no DB)
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    id: str
    user_id: str
    symbol: str
    metric: str
    direction: str
    threshold: float
    timeframe: str
    created_at_ms: int = field(default_factory=now_ms)
    enabled: bool = True


rules_by_id: Dict[str, Rule] = {}
rules_by_symbol: Dict[str, List[Rule]] = defaultdict(list)
rules_by_symbol_and_tf: Dict[Tuple[str, str], List[Rule]] = defaultdict(list)


def _norm_symbol(symbol: str) -> str:
    return (symbol or "").upper().strip()


def _norm_timeframe(timeframe: str) -> str:
    return (timeframe or "").lower().strip()


def _remove_rule_from_indexes(rule: Rule) -> None:
    sym = _norm_symbol(rule.symbol)
    tf = _norm_timeframe(rule.timeframe)

    if sym in rules_by_symbol:
        rules_by_symbol[sym] = [r for r in rules_by_symbol[sym] if r.id != rule.id]
        if not rules_by_symbol[sym]:
            del rules_by_symbol[sym]

    key = (sym, tf)
    if key in rules_by_symbol_and_tf:
        rules_by_symbol_and_tf[key] = [r for r in rules_by_symbol_and_tf[key] if r.id != rule.id]
        if not rules_by_symbol_and_tf[key]:
            del rules_by_symbol_and_tf[key]


def _add_rule_to_indexes(rule: Rule) -> None:
    sym = _norm_symbol(rule.symbol)
    tf = _norm_timeframe(rule.timeframe)

    rule.symbol = sym
    rule.timeframe = tf

    rules_by_symbol[sym].append(rule)
    rules_by_symbol_and_tf[(sym, tf)].append(rule)


def create_rule(
    *,
    user_id: str,
    symbol: str,
    metric: str,
    direction: str,
    threshold: float,
    timeframe: str,
    enabled: bool = True,
) -> Rule:
    return Rule(
        id=str(uuid.uuid4()),
        user_id=user_id,
        symbol=_norm_symbol(symbol),
        metric=(metric or "").strip(),
        direction=(direction or "").lower().strip(),
        threshold=float(threshold),
        timeframe=_norm_timeframe(timeframe),
        enabled=bool(enabled),
    )


def add_rule(rule: Rule) -> Rule:
    if not rule.id:
        rule.id = str(uuid.uuid4())
    if rule.id in rules_by_id:
        raise ValueError(f"Rule id already exists: {rule.id}")

    rule.symbol = _norm_symbol(rule.symbol)
    rule.timeframe = _norm_timeframe(rule.timeframe)

    rules_by_id[rule.id] = rule
    _add_rule_to_indexes(rule)
    return rule


def update_rule(rule: Rule) -> Rule:
    if not rule.id or rule.id not in rules_by_id:
        raise KeyError(f"Rule id not found: {rule.id}")

    existing = rules_by_id[rule.id]
    _remove_rule_from_indexes(existing)

    rule.symbol = _norm_symbol(rule.symbol)
    rule.timeframe = _norm_timeframe(rule.timeframe)

    rules_by_id[rule.id] = rule
    _add_rule_to_indexes(rule)
    return rule


def remove_rule(rule_id: str) -> bool:
    rid = (rule_id or "").strip()
    existing = rules_by_id.get(rid)
    if existing is None:
        return False

    _remove_rule_from_indexes(existing)
    del rules_by_id[rid]
    return True


async def exponential_backoff(base: float, max_delay: float) -> Iterable[float]:
    delay = base
    while True:
        jitter = random.uniform(0, delay * 0.25)
        yield min(delay + jitter, max_delay)
        delay = min(delay * 2, max_delay)


def prune_deque(dq: Deque, threshold_ms: int, current_ms: int, key=lambda x: x[0]):
    """Remove items older than threshold_ms relative to current_ms."""
    limit = current_ms - threshold_ms
    while dq and key(dq[0]) < limit:
        dq.popleft()


# ---------------------------------------------------------------------------
# Candle builder from tradestream
# ---------------------------------------------------------------------------


class CandleBuilder:
    """Aggregates agg-trade ticks into synthetic 1-second OHLCV candles."""

    def __init__(self, history: int):
        self.history: Deque[Dict[str, Any]] = deque(maxlen=history)
        self._bucket_start: Optional[int] = None
        self._current: Optional[Dict[str, Any]] = None

    def add_trade(self, ts_ms: int, price: float, qty: float) -> Optional[Dict[str, Any]]:
        bucket_start = (ts_ms // 1000) * 1000
        if self._bucket_start is None:
            self._bucket_start = bucket_start
            self._current = {
                "timestamp": bucket_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
            }
            return None

        if bucket_start != self._bucket_start:
            closed = self._current
            if closed:
                self.history.append(closed)
            self._bucket_start = bucket_start
            self._current = {
                "timestamp": bucket_start,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": qty,
            }
            return closed

        candle = self._current
        assert candle is not None
        candle["high"] = max(candle["high"], price)
        candle["low"] = min(candle["low"], price)
        candle["close"] = price
        candle["volume"] += qty
        return None

    def latest_candle(self) -> Optional[Dict[str, Any]]:
        return self._current


# ---------------------------------------------------------------------------
# MEXC trade tracker (trades only, no orderbook/liquidations yet)
# ---------------------------------------------------------------------------


class MexcTradeTracker:
    """Connects to MEXC futures deal stream and normalizes trades.

    Uses the public `sub.deal` channel for a small set of symbols and forwards
    normalized `agg_trade` and `synthetic_candle` events into the shared
    analyzer queue. Analyzer takes care of pressure / volume spike logic.
    """

    def __init__(self, session: aiohttp.ClientSession, queue: asyncio.Queue):
        # MEXC futures symbols use underscore form like BTC_USDT; Analyzer will
        # see normalized symbols BTCUSDT/ETHUSDT via a simple mapping.
        self.symbols: List[str] = [
            "BTC_USDT",
            "ETH_USDT",
        ]
        self.session = session
        self.queue = queue
        self.candles: Dict[str, CandleBuilder] = {
            symbol: CandleBuilder(history=120) for symbol in self.symbols
        }
        # contractSize per symbol (e.g. BTC_USDT: 0.001 BTC per contract).
        self.contract_sizes: Dict[str, float] = {symbol: 1.0 for symbol in self.symbols}
        self._backoff = exponential_backoff(1.0, 60.0)

    async def run(self) -> None:
        async for delay in self._backoff:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("MEXC trade stream crashed; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)

    async def _connect(self) -> None:
        logging.info("Connecting MEXC WS: %s", MEXC_WS_BASE)
        await self._load_contract_sizes()
        async with websockets.connect(MEXC_WS_BASE, ping_interval=None) as ws:
            # subscribe to deal streams for configured symbols
            for symbol in self.symbols:
                sub = {"method": "sub.deal", "param": {"symbol": symbol}, "gzip": False}
                await ws.send(json.dumps(sub))
            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                async for message in ws:
                    await self._handle_message(message)
            finally:
                ping_task.cancel()

    async def _ping_loop(self, ws: websockets.WebSocketClientProtocol) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                try:
                    await ws.send(json.dumps({"method": "ping"}))
                except Exception:
                    logging.warning("MEXC ping failed; connection will be reset")
                    return
        except asyncio.CancelledError:
            return

    async def _load_contract_sizes(self) -> None:
        """Fetch contractSize for each symbol via public REST and cache it.

        Endpoint: GET /api/v1/contract/detail?symbol=BTC_USDT
        Docs: MEXC futures market endpoints.
        """
        for symbol in self.symbols:
            url = f"https://contract.mexc.com/api/v1/contract/detail?symbol={symbol}"
            try:
                async with self.session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        logging.warning("MEXC contract detail %s status=%s", symbol, resp.status)
                        continue
                    data = await resp.json()
            except Exception as exc:
                logging.warning("MEXC contract detail fetch failed for %s: %s", symbol, exc)
                continue

            try:
                info = data.get("data") or {}
                size = float(info.get("contractSize", 1.0))
                if size <= 0:
                    raise ValueError("contractSize<=0")
                self.contract_sizes[symbol] = size
                logging.info("MEXC contractSize %s=%s", symbol, size)
            except Exception as exc:
                logging.warning("MEXC contract detail parse failed for %s: %s", symbol, exc)

    async def _handle_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except Exception:
            logging.debug("Invalid JSON from MEXC: %s", raw[:200])
            return

        if payload.get("channel") != "push.deal":
            return

        symbol = payload.get("symbol")
        if symbol not in self.symbols:
            return

        trades = payload.get("data", [])
        if not isinstance(trades, list):
            return

        for trade in trades:
            price = float(trade.get("p", 0.0))
            qty = float(trade.get("v", 0.0))
            ts = int(trade.get("t", now_ms()))
            # M flag is maker side; treat M==1 as maker==seller similar to Binance
            is_buyer_maker = bool(trade.get("M", 0))
            await self._handle_trade(symbol, price, qty, ts, is_buyer_maker)

    async def _handle_trade(self, symbol: str, price: float, qty: float, ts: int, is_buyer_maker: bool) -> None:
        # Normalize quantity using contractSize so that Analyzer sees a base
        # asset amount similar to Binance (qty * contractSize).
        contract_size = self.contract_sizes.get(symbol, 1.0)
        effective_qty = qty * contract_size

        # 1s candles use effective quantity so synthetic_1s_candle.volume is
        # consistent with notional calculations.
        closed = self.candles[symbol].add_trade(ts, price, effective_qty)

        display_symbol = symbol.replace("_", "")  # e.g. BTC_USDT -> BTCUSDT

        trade_event = {
            "type": "agg_trade",
            "exchange": "MEXC Futures",
            "symbol": display_symbol,
            "price": price,
            "qty": effective_qty,
            "timestamp": ts,
            "is_buyer_maker": is_buyer_maker,
        }
        await self.queue.put(trade_event)

        if closed:
            await self.queue.put(
                {
                    "type": "synthetic_candle",
                    "exchange": "MEXC Futures",
                    "symbol": display_symbol,
                    "candle": closed,
                }
            )


# ---------------------------------------------------------------------------
# Trackers
# ---------------------------------------------------------------------------


class AggTradeTracker:
    """Consumes aggTrade stream, builds candles and forwards trade events."""

    def __init__(self, config: Config, session: aiohttp.ClientSession, queue: asyncio.Queue):
        self.config = config
        self.session = session
        self.queue = queue
        self.symbols = list(config.symbols.keys())
        self.candles: Dict[str, CandleBuilder] = {
            symbol: CandleBuilder(history=config.for_symbol(symbol).candle_history)
            for symbol in self.symbols
        }
        self.last_trade_id: Dict[str, Optional[int]] = {symbol: None for symbol in self.symbols}
        self._backoff = exponential_backoff(config.reconnect_backoff_initial, config.reconnect_backoff_max)

    async def run(self) -> None:
        async for delay in self._backoff:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("aggTrade stream crashed; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            else:
                # Successful completion (shouldn't happen under normal operation)
                await asyncio.sleep(delay)

    async def _connect(self) -> None:
        streams = "/".join(f"{symbol.lower()}@aggTrade" for symbol in self.symbols)
        ws_url = BINANCE_WS_BASE + streams
        logging.info("Connecting aggTrade stream: %s", ws_url)
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            await self._bootstrap_trades()
            async for message in ws:
                payload = json.loads(message)
                data = payload.get("data", {})
                if data.get("e") != "aggTrade":
                    continue
                symbol = data["s"]
                trade_id = int(data["a"])
                price = float(data["p"])
                qty = float(data["q"])
                ts = int(data["T"])
                is_buyer_maker = bool(data["m"])
                await self._handle_trade(symbol, trade_id, price, qty, ts, is_buyer_maker)

    async def _bootstrap_trades(self) -> None:
        """Fetch missing trades via REST to keep candles continuous."""
        tasks = [self._catch_up_symbol(symbol) for symbol in self.symbols]
        await asyncio.gather(*tasks)

    async def _catch_up_symbol(self, symbol: str) -> None:
        last_id = self.last_trade_id.get(symbol)
        params = {"symbol": symbol, "limit": self.config.aggtrade_rest_limit}
        if last_id is not None:
            params["fromId"] = last_id + 1
        url = f"{BINANCE_REST_BASE}/fapi/v1/aggTrades"
        try:
            async with self.session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logging.warning("aggTrade catch-up %s failed: %s", symbol, resp.status)
                    return
                trades = await resp.json()
        except asyncio.TimeoutError:
            logging.warning("aggTrade catch-up timeout for %s", symbol)
            return
        except Exception:
            logging.exception("aggTrade catch-up error for %s", symbol)
            return

        if not isinstance(trades, list):
            return
        for trade in trades:
            trade_id = int(trade["a"])
            price = float(trade["p"])
            qty = float(trade["q"])
            ts = int(trade["T"])
            is_buyer_maker = bool(trade["m"])
            await self._handle_trade(symbol, trade_id, price, qty, ts, is_buyer_maker)

    async def _handle_trade(
        self,
        symbol: str,
        trade_id: int,
        price: float,
        qty: float,
        ts: int,
        is_buyer_maker: bool,
    ) -> None:
        last_id = self.last_trade_id.get(symbol)
        if last_id is not None and trade_id <= last_id:
            return  # duplicate
        self.last_trade_id[symbol] = trade_id

        candle_builder = self.candles[symbol]
        closed_candle = candle_builder.add_trade(ts, price, qty)

        trade_event = {
            "type": "agg_trade",
            "exchange": "Binance Futures",
            "symbol": symbol,
            "trade_id": trade_id,
            "price": price,
            "qty": qty,
            "timestamp": ts,
            "is_buyer_maker": is_buyer_maker,
        }
        await self.queue.put(trade_event)

        if closed_candle:
            # Provide completed candle to analyzer.
            await self.queue.put(
                {
                    "type": "synthetic_candle",
                    "symbol": symbol,
                    "candle": closed_candle,
                }
            )


class OrderBookState:
    """Maintains order book snapshot and derives manipulation heuristics."""

    def __init__(self, symbol: str, depth_levels: int, history: int, config: SymbolConfig):
        self.symbol = symbol
        self.depth_levels = depth_levels
        self.config = config
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        self.last_update_id: Optional[int] = None
        self.metrics_history: Deque[Dict[str, Any]] = deque(maxlen=history)
        self.top_level_history: Deque[Dict[str, float]] = deque(maxlen=history)
        self.last_large_wall: Optional[Dict[str, Any]] = None

    def set_snapshot(self, bids: List[List[str]], asks: List[List[str]], last_update_id: int) -> None:
        self.bids = {float(price): float(qty) for price, qty in bids[: self.depth_levels]}
        self.asks = {float(price): float(qty) for price, qty in asks[: self.depth_levels]}
        self.last_update_id = last_update_id
        logging.debug("%s depth snapshot loaded", self.symbol)

    def apply_delta(self, bids: List[List[str]], asks: List[List[str]]) -> None:
        for price_str, qty_str in bids:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
        for price_str, qty_str in asks:
            price = float(price_str)
            qty = float(qty_str)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

        # Trim to depth_levels
        self.bids = dict(sorted(self.bids.items(), key=lambda kv: kv[0], reverse=True)[: self.depth_levels])
        self.asks = dict(sorted(self.asks.items(), key=lambda kv: kv[0])[: self.depth_levels])

    def compute_metrics(self, timestamp: int) -> Dict[str, Any]:
        bid_total = sum(price * qty for price, qty in self.bids.items())
        ask_total = sum(price * qty for price, qty in self.asks.items())
        imbalance = 0.0
        if bid_total + ask_total > 0:
            imbalance = (bid_total - ask_total) / (bid_total + ask_total)

        top_bid_notional = 0.0
        top_ask_notional = 0.0
        if self.bids:
            top_bid_price, top_bid_qty = max(self.bids.items())
            top_bid_notional = top_bid_price * top_bid_qty
        if self.asks:
            top_ask_price, top_ask_qty = min(self.asks.items())
            top_ask_notional = top_ask_price * top_ask_qty

        self.metrics_history.append(
            {
                "timestamp": timestamp,
                "bid_total": bid_total,
                "ask_total": ask_total,
                "imbalance": imbalance,
            }
        )
        self.top_level_history.append(
            {
                "timestamp": timestamp,
                "bid": top_bid_notional,
                "ask": top_ask_notional,
            }
        )

        wall_event = self._detect_wall(top_bid_notional, top_ask_notional, timestamp)

        risk = "LOW"
        if abs(imbalance) >= self.config.imbalance_threshold and wall_event.get("significant"):
            risk = "HIGH"
        elif abs(imbalance) >= (self.config.imbalance_threshold * 0.8):
            risk = "MEDIUM"

        return {
            "timestamp": timestamp,
            "bid_total": bid_total,
            "ask_total": ask_total,
            "imbalance": imbalance,
            "manipulation_risk": risk,
            "wall_event": wall_event,
        }

    def _detect_wall(self, top_bid: float, top_ask: float, timestamp: int) -> Dict[str, Any]:
        """Detect suspicious walls appearing/disappearing quickly."""
        window = [entry for entry in self.top_level_history if timestamp - entry["timestamp"] <= 1_500]
        avg_bid = sum(entry["bid"] for entry in window) / len(window) if window else 0.0
        avg_ask = sum(entry["ask"] for entry in window) / len(window) if window else 0.0

        significant = False
        wall_side = None
        wall_notional = 0.0
        removed = False

        if avg_bid > 0 and top_bid > avg_bid * self.config.wall_size_factor:
            significant = True
            wall_side = "BID"
            wall_notional = top_bid
            self.last_large_wall = {"side": "BID", "timestamp": timestamp, "size": top_bid}

        if avg_ask > 0 and top_ask > avg_ask * self.config.wall_size_factor:
            significant = True
            wall_side = "ASK"
            wall_notional = top_ask
            self.last_large_wall = {"side": "ASK", "timestamp": timestamp, "size": top_ask}

        if self.last_large_wall:
            elapsed = timestamp - self.last_large_wall["timestamp"]
            if elapsed <= 150:
                if self.last_large_wall["side"] == "BID" and top_bid < self.last_large_wall["size"] / self.config.wall_size_factor:
                    removed = True
                    significant = True
                    wall_side = "BID_REMOVED"
                if self.last_large_wall["side"] == "ASK" and top_ask < self.last_large_wall["size"] / self.config.wall_size_factor:
                    removed = True
                    significant = True
                    wall_side = "ASK_REMOVED"
            if elapsed > 500:
                self.last_large_wall = None

        return {
            "significant": significant,
            "side": wall_side,
            "notional": wall_notional,
            "removed": removed,
        }


class OrderBookTracker:
    """Tracks depth@100ms stream and emits manipulation metrics."""

    def __init__(self, config: Config, session: aiohttp.ClientSession, queue: asyncio.Queue):
        self.config = config
        self.session = session
        self.queue = queue
        self.symbols = list(config.symbols.keys())
        self.states: Dict[str, OrderBookState] = {
            symbol: OrderBookState(
                symbol=symbol,
                depth_levels=config.for_symbol(symbol).top_depth_levels,
                history=config.orderbook_history,
                config=config.for_symbol(symbol),
            )
            for symbol in self.symbols
        }
        self._backoff = exponential_backoff(config.reconnect_backoff_initial, config.reconnect_backoff_max)

    async def run(self) -> None:
        async for delay in self._backoff:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("depth stream crashed; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)

    async def _connect(self) -> None:
        streams = "/".join(f"{symbol.lower()}@depth@100ms" for symbol in self.symbols)
        ws_url = BINANCE_WS_BASE + streams
        logging.info("Connecting depth stream: %s", ws_url)
        await self._bootstrap_books()
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            async for message in ws:
                payload = json.loads(message)
                data = payload.get("data", {})
                if data.get("e") != "depthUpdate":
                    continue
                symbol = data["s"]
                state = self.states[symbol]
                last_update_id = state.last_update_id
                first_id = int(data["U"])
                final_id = int(data["u"])
                if last_update_id is None:
                    continue
                if final_id <= last_update_id:
                    continue
                if first_id > last_update_id + 1:
                    logging.warning("depth gap detected for %s; rebootstrap", symbol)
                    await self._bootstrap_symbol(symbol)
                    continue
                state.last_update_id = final_id
                state.apply_delta(data.get("b", []), data.get("a", []))
                metrics = state.compute_metrics(int(data["E"]))
                await self.queue.put(
                    {
                        "type": "orderbook_metrics",
                        "exchange": "Binance Futures",
                        "symbol": symbol,
                        "metrics": metrics,
                    }
                )

    async def _bootstrap_books(self) -> None:
        await asyncio.gather(*(self._bootstrap_symbol(symbol) for symbol in self.symbols))

    async def _bootstrap_symbol(self, symbol: str) -> None:
        url = f"{BINANCE_REST_BASE}/fapi/v1/depth"
        top_levels = int(self.config.for_symbol(symbol).top_depth_levels)
        desired = min(max(top_levels, top_levels * 2), 1000)
        allowed_limits = (5, 10, 20, 50, 100, 500, 1000)
        limit = max((x for x in allowed_limits if x <= desired), default=100)
        if limit < top_levels:
            limit = min((x for x in allowed_limits if x >= top_levels), default=1000)
        params = {"symbol": symbol, "limit": int(limit)}
        try:
            async with self.session.get(url, params=params, timeout=10) as resp:
                if resp.status != 200:
                    logging.warning("depth snapshot %s failed: %s", symbol, resp.status)
                    return
                snapshot = await resp.json()
        except asyncio.TimeoutError:
            logging.warning("depth snapshot timeout for %s", symbol)
            return
        except Exception:
            logging.exception("depth snapshot error for %s", symbol)
            return

        state = self.states[symbol]
        state.set_snapshot(snapshot.get("bids", []), snapshot.get("asks", []), int(snapshot.get("lastUpdateId", 0)))


class LiquidationTracker:
    """Consumes liquidation stream and summarizes recent activity."""

    def __init__(self, config: Config, queue: asyncio.Queue):
        self.config = config
        self.queue = queue
        self.symbols = list(config.symbols.keys())
        self.histories: Dict[str, Deque[Dict[str, Any]]] = {
            symbol: deque()
            for symbol in self.symbols
        }
        self._backoff = exponential_backoff(config.reconnect_backoff_initial, config.reconnect_backoff_max)

    async def run(self) -> None:
        async for delay in self._backoff:
            try:
                await self._connect()
            except asyncio.CancelledError:
                raise
            except Exception:
                logging.exception("liquidation stream crashed; retrying in %.2fs", delay)
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(delay)

    async def _connect(self) -> None:
        ws_url = BINANCE_WS_BASE + "!forceOrder@arr"
        logging.info("Connecting liquidation stream: %s", ws_url)
        async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
            async for message in ws:
                payload = json.loads(message)
                data = payload.get("data", {})
                order = data.get("o", {})
                symbol = order.get("s")
                if symbol not in self.histories:
                    continue
                side = order.get("S")  # BUY = long forced cover, SELL = short forced cover
                qty = float(order.get("q", 0.0))
                price = float(order.get("p", 0.0))
                notional = qty * price
                ts = int(order.get("T", payload.get("E", now_ms())))
                event = {
                    "type": "liquidation",
                    "exchange": "Binance Futures",
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "notional": notional,
                    "timestamp": ts,
                }
                await self.queue.put(event)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class Analyzer:
    def __init__(
        self,
        config: Config,
        queue: asyncio.Queue,
        broadcaster: Optional[AlertBroadcaster] = None,
        hooks: Optional["AnalyzerHooks"] = None,
    ):
        self.config = config
        self.queue = queue
        self.broadcaster = broadcaster
        self.hooks = hooks
        self.stdout_alerts = os.environ.get("ANALYZER_STDOUT_ALERTS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.stdout_signals_only = os.environ.get("ANALYZER_STDOUT_SIGNALS_ONLY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        # All state is keyed by composite key "{exchange}:{symbol}" to support
        # multiple exchanges using the same analyzer pipeline.
        self.trade_windows: Dict[str, Deque[Tuple[int, float, str]]] = {}
        self.oi_window: Dict[str, Deque[Tuple[int, float]]] = {}
        self.candle_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self.volume_stats: Dict[str, Dict[str, float]] = {}
        self.orderbook_metrics: Dict[str, Dict[str, Any]] = {}
        self.liquidation_history: Dict[str, Deque[Dict[str, Any]]] = {}
        self.last_alert_ts: Dict[str, int] = defaultdict(lambda: 0)

    @staticmethod
    def _key(exchange: str, symbol: str) -> str:
        return f"{exchange}:{symbol}"

    async def run(self) -> None:
        while True:
            event = await self.queue.get()
            event_type = event.get("type")
            symbol = event.get("symbol")
            exchange = event.get("exchange", "Binance Futures")
            if symbol is None or exchange is None:
                continue

            key = self._key(exchange, symbol)

            # For now, config is still symbol-based for Binance symbols; for
            # other exchanges we fall back to default SymbolConfig via
            # Config.for_symbol, so we only filter here for Binance.
            if exchange == "Binance Futures" and symbol not in self.config.symbols:
                continue

            if event_type == "agg_trade":
                self._process_trade(key, symbol, event)
            elif event_type == "synthetic_candle":
                self._process_candle(key, symbol, event)
            elif event_type == "orderbook_metrics":
                self.orderbook_metrics[key] = event["metrics"]
            elif event_type == "liquidation":
                self._process_liquidation(key, symbol, event)

            if symbol:
                alert = self._maybe_emit_alert(key, symbol, exchange)
                if alert:
                    if self.stdout_alerts:
                        if (
                            (not self.stdout_signals_only)
                            or alert.get("signal")
                            or alert.get("orderbook_wall")
                            or (alert.get("liquidations") or {}).get("cluster_signal")
                        ):
                            print(json.dumps(alert), flush=True)
                    if self.broadcaster:
                        await self.broadcaster.broadcast(alert)
                    if self.hooks and self.hooks.on_alert:
                        try:
                            self.hooks.on_alert(alert)
                        except Exception:
                            pass

    def _process_trade(self, key: str, symbol: str, event: Dict[str, Any]) -> None:
        ts = event["timestamp"]
        qty = event["qty"]
        price = event.get("price")
        side = "sell" if event["is_buyer_maker"] else "buy"
        trade = (ts, qty, side)
        trades = self.trade_windows.setdefault(key, deque())
        trades.append(trade)
        prune_deque(trades, self.config.trade_pressure_long_window_ms, ts)

        # Maintain OI proxy window (1 minute default)
        oi_window = self.oi_window.setdefault(key, deque())
        net = qty if side == "buy" else -qty
        oi_window.append((ts, net))
        prune_deque(oi_window, self.config.trade_pressure_long_window_ms, ts)

        if self.hooks and self.hooks.on_price_update and price is not None:
            try:
                self.hooks.on_price_update(symbol, float(price), float(qty), int(ts))
            except Exception:
                pass

    def _process_candle(self, key: str, symbol: str, event: Dict[str, Any]) -> None:
        candle = event["candle"]
        history = self.candle_history.get(key)
        if history is None:
            history = self.candle_history[key] = deque(maxlen=self.config.for_symbol(symbol).candle_history)
        history.append(candle)
        stats_window = self.config.for_symbol(symbol).volume_stats_window
        recent = [c for c in history if candle["timestamp"] - c["timestamp"] <= stats_window * 1000]
        volumes = [c["volume"] for c in recent]
        if volumes:
            mean = sum(volumes) / len(volumes)
            variance = sum((v - mean) ** 2 for v in volumes) / max(len(volumes) - 1, 1)
            std = math.sqrt(max(variance, 1e-9))
            self.volume_stats[key] = {"mean": mean, "std": std}

        if self.hooks and self.hooks.on_candle_close:
            try:
                ts_ms = int(candle.get("timestamp"))
                if ts_ms % 60_000 == 59_000:
                    prev_ts = ts_ms - 60_000
                    prev_close = None
                    for c in reversed(history):
                        if int(c.get("timestamp")) == prev_ts:
                            prev_close = float(c.get("close"))
                            break

                    price_pct_change: Optional[float] = None
                    if prev_close is not None and prev_close > 0:
                        price_pct_change = ((float(candle.get("close")) - prev_close) / prev_close) * 100.0

                    cur_start = ts_ms - 59_000
                    prev_start = ts_ms - 119_000
                    cur_end = ts_ms
                    prev_end = ts_ms - 60_000

                    cur_vol = 0.0
                    prev_vol = 0.0
                    for c in history:
                        c_ts = int(c.get("timestamp"))
                        v = float(c.get("volume", 0.0) or 0.0)
                        if cur_start <= c_ts <= cur_end:
                            cur_vol += v
                        elif prev_start <= c_ts <= prev_end:
                            prev_vol += v

                    volume_pct_change: Optional[float] = None
                    if prev_vol > 0:
                        volume_pct_change = ((cur_vol - prev_vol) / prev_vol) * 100.0

                    self.hooks.on_candle_close(
                        symbol,
                        "1m",
                        ts_ms,
                        price_pct_change,
                        volume_pct_change,
                    )
            except Exception:
                pass

    def _process_liquidation(self, key: str, symbol: str, event: Dict[str, Any]) -> None:
        history = self.liquidation_history.setdefault(key, deque())
        history.append(event)
        prune_deque(
            history,
            self.config.liquidation_window_seconds * 1000,
            event["timestamp"],
            key=lambda e: e["timestamp"],
        )

    def _maybe_emit_alert(self, key: str, symbol: str, exchange: str) -> Optional[Dict[str, Any]]:
        cfg = self.config.for_symbol(symbol)
        trades = self.trade_windows.get(key)
        if not trades:
            return None
        latest_ts = trades[-1][0]

        def window_sum(ms: int, side_filter: Optional[str] = None) -> float:
            total = 0.0
            threshold = latest_ts - ms
            for ts, qty, side in reversed(trades):
                if ts < threshold:
                    break
                if side_filter is None or side == side_filter:
                    total += qty
            return total

        buy_1s = window_sum(self.config.trade_pressure_windows_ms[0], "buy")
        sell_1s = window_sum(self.config.trade_pressure_windows_ms[0], "sell")
        buy_5s = window_sum(self.config.trade_pressure_windows_ms[1], "buy")
        sell_5s = window_sum(self.config.trade_pressure_windows_ms[1], "sell")
        total = buy_5s + sell_5s
        pressure = 0.0
        if total > 0:
            pressure = (buy_5s - sell_5s) / total

        oi_window = self.oi_window.get(key, deque())
        oi_estimated_change = sum(net for _, net in oi_window)
        if oi_estimated_change > cfg.oi_positive_threshold:
            oi_trend = "LONG_DOMINANT"
        elif oi_estimated_change < cfg.oi_negative_threshold:
            oi_trend = "SHORT_DOMINANT"
        else:
            oi_trend = "NEUTRAL"

        orderbook_metrics = self.orderbook_metrics.get(key, {})
        manipulation_risk = orderbook_metrics.get("manipulation_risk", "UNKNOWN")
        imbalance = orderbook_metrics.get("imbalance")

        latest_candle = self.candle_history.get(key, deque())[-1] if self.candle_history.get(key) else None
        stats = self.volume_stats.get(key, {"mean": 0.0, "std": 0.0})
        volume_z = 0.0
        if latest_candle and stats["std"] > 0:
            volume_z = (latest_candle["volume"] - stats["mean"]) / stats["std"]

        liquidation_summary = self._summarize_liquidations(key, symbol, latest_ts)

        signal = None
        if volume_z >= cfg.volume_zscore_threshold:
            if pressure >= cfg.long_pressure_threshold:
                signal = "PUMP_ALERT"
            elif pressure <= cfg.short_pressure_threshold:
                signal = "DUMP_ALERT"

        high_pressure = abs(pressure) >= abs(cfg.long_pressure_threshold)
        high_manip = manipulation_risk == "HIGH"
        liquidations_present = liquidation_summary["cluster_signal"] is not None

        should_alert = any([signal, high_pressure, high_manip, liquidations_present])
        if not should_alert:
            return None

        last_ts = self.last_alert_ts[key]
        if latest_ts - last_ts < 500:  # throttle alerts per symbol to 0.5s
            return None
        self.last_alert_ts[key] = latest_ts

        alert = {
            "exchange": exchange,
            "symbol": symbol,
            "timestamp": latest_ts,
            "long_pressure": round(max(pressure, 0.0), 4),
            "short_pressure": round(max(-pressure, 0.0), 4),
            "pressure_raw": round(pressure, 4),
            "pressure_buy_volume_5s": buy_5s,
            "pressure_sell_volume_5s": sell_5s,
            "manipulation_risk": manipulation_risk,
            "imbalance": imbalance,
            "signal": signal,
            "volume_spike_zscore": round(volume_z, 2),
            "oi_estimated_change": oi_estimated_change,
            "oi_trend": oi_trend,
            "liquidations": liquidation_summary,
        }
        if latest_candle:
            alert["synthetic_1s_candle"] = latest_candle
        if orderbook_metrics.get("wall_event", {}).get("significant"):
            alert["orderbook_wall"] = orderbook_metrics["wall_event"]
        return alert

    def _summarize_liquidations(self, key: str, symbol: str, now_timestamp: int) -> Dict[str, Any]:
        history = self.liquidation_history.get(key, deque())
        prune_deque(
            history,
            self.config.liquidation_window_seconds * 1000,
            now_timestamp,
            key=lambda e: e["timestamp"],
        )
        long_count = sum(1 for event in history if event["side"] == "BUY")
        short_count = sum(1 for event in history if event["side"] == "SELL")
        long_notional = sum(event["notional"] for event in history if event["side"] == "BUY")
        short_notional = sum(event["notional"] for event in history if event["side"] == "SELL")

        cluster_signal = None
        cluster_threshold = self.config.liquidation_cluster_count
        window_ms = self.config.liquidation_cluster_window_ms
        cluster_buy = 0
        cluster_sell = 0
        for event in history:
            window_events = [e for e in history if 0 <= event["timestamp"] - e["timestamp"] <= window_ms]
            buy_in_window = sum(1 for e in window_events if e["side"] == "BUY")
            sell_in_window = sum(1 for e in window_events if e["side"] == "SELL")
            cluster_buy = max(cluster_buy, buy_in_window)
            cluster_sell = max(cluster_sell, sell_in_window)

        if cluster_buy >= cluster_threshold:
            cluster_signal = "SHORT_SQUEEZE"
        elif cluster_sell >= cluster_threshold:
            cluster_signal = "LONG_SQUEEZE"

        large_liquidations = [
            e for e in history if e["notional"] >= self.config.for_symbol(symbol).liquidation_notional_threshold
        ]

        return {
            "long_count": long_count,
            "short_count": short_count,
            "long_notional": long_notional,
            "short_notional": short_notional,
            "cluster_signal": cluster_signal,
            "large_liquidations": large_liquidations,
        }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


class AlertBroadcaster:
    def __init__(self) -> None:
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._lock = asyncio.Lock()

    async def register(self, websocket: websockets.WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.add(websocket)

    async def unregister(self, websocket: websockets.WebSocketServerProtocol) -> None:
        async with self._lock:
            self._clients.discard(websocket)

    async def broadcast(self, alert: Dict[str, Any]) -> None:
        message = json.dumps(alert)
        async with self._lock:
            targets = list(self._clients)
        if not targets:
            return
        coros = []
        for ws in targets:
            coros.append(self._safe_send(ws, message))
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    async def _safe_send(self, websocket: websockets.WebSocketServerProtocol, message: str) -> None:
        try:
            await websocket.send(message)
        except Exception:
            await self.unregister(websocket)

    async def handler(self, websocket: websockets.WebSocketServerProtocol) -> None:
        await self.register(websocket)
        try:
            await websocket.send(json.dumps({"status": "connected"}))
            async for _ in websocket:
                # Keep the connection alive; clients do not need to send data.
                pass
        finally:
            await self.unregister(websocket)


async def main(hooks: Optional["AnalyzerHooks"] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=CONFIG.event_queue_size)

    broadcaster = AlertBroadcaster()

    async with aiohttp.ClientSession() as session:
        agg = AggTradeTracker(CONFIG, session, queue)
        depth = OrderBookTracker(CONFIG, session, queue)
        liq = LiquidationTracker(CONFIG, queue)
        mexc_trades = MexcTradeTracker(session, queue)
        analyzer = Analyzer(CONFIG, queue, broadcaster=broadcaster, hooks=hooks)

        ws_server = None
        ws_enabled = os.environ.get("ANALYZER_WS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
        if ws_enabled:
            ws_server = ws_serve(
                broadcaster.handler,
                host="0.0.0.0",
                port=8765,
                ping_interval=20,
                ping_timeout=20,
            )

        tasks = [
            asyncio.create_task(agg.run(), name="aggTrade"),
            asyncio.create_task(depth.run(), name="depth"),
            asyncio.create_task(liq.run(), name="liquidation"),
            asyncio.create_task(mexc_trades.run(), name="mexc_trades"),
            asyncio.create_task(analyzer.run(), name="analyzer"),
        ]

        try:
            if ws_server is not None:
                try:
                    async with ws_server:
                        await asyncio.gather(*tasks)
                except OSError as exc:
                    logging.warning("Analyzer WS server disabled: %s", exc)
                    await asyncio.gather(*tasks)
            else:
                await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except Exception:
            logging.exception("Analyzer crashed")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Shutting down analyzer")


@dataclass
class AnalyzerHooks:
    on_price_update: Optional[Callable[[str, float, float, int], None]] = None
    on_candle_close: Optional[Callable[[str, str, int, Optional[float], Optional[float]], None]] = None
    on_alert: Optional[Callable[[Dict[str, Any]], None]] = None

