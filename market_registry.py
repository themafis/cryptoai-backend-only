"""Unified registry for exchanges, markets and coins.

Phase 1: provide in-memory models + basic Binance sync, without touching
existing endpoints. Later phases can import and use this registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import threading
import time

import requests


# -----------------------------
# Data models
# -----------------------------


@dataclass
class Exchange:
    id: str  # e.g. "binance_spot", "binance_futures"
    name: str
    type: str  # "spot" / "futures" / "perp" / "cex"
    base_url: str


@dataclass
class Market:
    id: str  # e.g. "binance_spot:BTCUSDT"
    exchange_id: str
    symbol_raw: str  # e.g. "BTCUSDT"
    base_asset: str
    quote_asset: str
    status: str  # TRADING / BREAK / HALT / DELISTED
    type: str  # spot/futures
    volume_24h: float = 0.0


@dataclass
class Coin:
    id: str  # canonical symbol, e.g. "BTC"
    name: str
    markets: List[str] = field(default_factory=list)  # list of Market.id
    primary_market_id: Optional[str] = None
    reference_market_id_for_indicators: Optional[str] = None


# -----------------------------
# Registry (in-memory, singleton)
# -----------------------------


class MarketRegistry:
    """Simple in-memory registry for exchanges / markets / coins.

    For now this is process-local and rebuilt on startup. Later we can
    persist to a DB if needed.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.exchanges: Dict[str, Exchange] = {}
        self.markets: Dict[str, Market] = {}
        self.coins: Dict[str, Coin] = {}
        self.last_sync_ts: Optional[float] = None

    # -------- Public helpers --------

    def get_coin(self, coin_id: str) -> Optional[Coin]:
        with self._lock:
            return self.coins.get(coin_id.upper())

    def get_market(self, market_id: str) -> Optional[Market]:
        with self._lock:
            return self.markets.get(market_id)

    def list_markets_for_coin(self, coin_id: str) -> List[Market]:
        with self._lock:
            coin = self.coins.get(coin_id.upper())
            if not coin:
                return []
            return [self.markets[mid] for mid in coin.markets if mid in self.markets]

    # -------- Sync from Binance (Phase 1) --------

    def sync_from_binance(self) -> None:
        """Fetch spot + futures USDT markets from Binance and populate registry.

        This is a conservative first step: we only care about USDT pairs
        and keep everything in-memory. It does not affect existing code
        until other modules explicitly import and consume this registry.
        """

        session = requests.Session()

        # Spot markets
        spot = Exchange(
            id="binance_spot",
            name="Binance Spot",
            type="spot",
            base_url="https://api.binance.com",
        )

        futures = Exchange(
            id="binance_futures",
            name="Binance Futures",
            type="futures",
            base_url="https://fapi.binance.com",
        )

        with self._lock:
            self.exchanges[spot.id] = spot
            self.exchanges[futures.id] = futures

        # Fetch spot exchangeInfo
        try:
            spot_info = session.get(f"{spot.base_url}/api/v3/exchangeInfo", timeout=10)
            spot_info.raise_for_status()
            spot_json = spot_info.json()
            spot_symbols = spot_json.get("symbols", [])
        except Exception:
            spot_symbols = []

        # Fetch futures exchangeInfo
        try:
            fut_info = session.get(f"{futures.base_url}/fapi/v1/exchangeInfo", timeout=10)
            fut_info.raise_for_status()
            fut_json = fut_info.json()
            fut_symbols = fut_json.get("symbols", [])
        except Exception:
            fut_symbols = []

        with self._lock:
            # Reset current state for Binance markets/coins
            self.markets = {
                mid: m
                for mid, m in self.markets.items()
                if not (m.exchange_id.startswith("binance_"))
            }

            # Coins may be shared across many exchanges; we will upsert.
            for entry in spot_symbols:
                try:
                    if entry.get("quoteAsset") != "USDT":
                        continue
                    if entry.get("status") != "TRADING":
                        continue
                    symbol = entry["symbol"]
                    base = entry["baseAsset"]
                    quote = entry["quoteAsset"]
                except Exception:
                    continue

                market_id = f"{spot.id}:{symbol}"
                market = Market(
                    id=market_id,
                    exchange_id=spot.id,
                    symbol_raw=symbol,
                    base_asset=base,
                    quote_asset=quote,
                    status=entry.get("status", "UNKNOWN"),
                    type="spot",
                )
                self.markets[market_id] = market
                self._upsert_coin_for_market(market)

            for entry in fut_symbols:
                try:
                    if entry.get("quoteAsset") != "USDT":
                        continue
                    if entry.get("status") != "TRADING":
                        continue
                    symbol = entry["symbol"]
                    base = entry["baseAsset"]
                    quote = entry["quoteAsset"]
                except Exception:
                    continue

                market_id = f"{futures.id}:{symbol}"
                market = Market(
                    id=market_id,
                    exchange_id=futures.id,
                    symbol_raw=symbol,
                    base_asset=base,
                    quote_asset=quote,
                    status=entry.get("status", "UNKNOWN"),
                    type="futures",
                )
                self.markets[market_id] = market
                self._upsert_coin_for_market(market)

            # After all markets are inserted, compute primary/reference markets.
            self._recompute_primary_and_reference_markets()
            self.last_sync_ts = time.time()

    # -------- Internal helpers --------

    def _upsert_coin_for_market(self, market: Market) -> None:
        coin_id = market.base_asset.upper()
        coin = self.coins.get(coin_id)
        if not coin:
            coin = Coin(id=coin_id, name=coin_id)
            self.coins[coin_id] = coin
        if market.id not in coin.markets:
            coin.markets.append(market.id)

    def _recompute_primary_and_reference_markets(self) -> None:
        """Very simple heuristic for now.

        - primary_market_id: prefer Binance Futures USDT, else Binance Spot USDT.
        - reference_market_id_for_indicators: same logic.

        Later we can add 24h volume-based ranking here.
        """

        for coin in self.coins.values():
            spot_market = None
            futures_market = None
            for mid in coin.markets:
                m = self.markets.get(mid)
                if not m:
                    continue
                if m.exchange_id == "binance_futures" and m.quote_asset == "USDT":
                    futures_market = futures_market or m
                elif m.exchange_id == "binance_spot" and m.quote_asset == "USDT":
                    spot_market = spot_market or m

            # Prefer futures as primary if available, else spot
            primary = futures_market or spot_market
            if primary:
                coin.primary_market_id = primary.id
                coin.reference_market_id_for_indicators = primary.id


# Global singleton instance to be imported by other modules later
REGISTRY = MarketRegistry()
