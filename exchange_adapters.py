"""Exchange adapters for multi-exchange market data.

Phase 1: define a common interface and provide minimal, safe implementations.
Concrete per-exchange details (exact REST paths, params) can be refined
incrementally without touching the rest of the system.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import requests


@dataclass
class Ticker24h:
    symbol: str
    price: float
    volume_24h: float
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    price_change: Optional[float] = None
    price_change_pct: Optional[float] = None


@dataclass
class OrderBookTop:
    symbol: str
    best_bid: Optional[float]
    best_ask: Optional[float]


@dataclass
class OrderBookDepth:
    symbol: str
    bids: List[List[float]]
    asks: List[List[float]]


@dataclass
class Candle:
    t: int  # unix seconds
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None


class BaseAdapter:
    """Base class with small HTTP helpers.

    This is intentionally very small; we only implement what we actually need
    (24h ticker + top-of-book). More complex endpoints can be added later.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None, timeout: float = 5.0) -> Any:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # Public interface
    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:  # pragma: no cover - to be implemented per exchange
        raise NotImplementedError

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:  # pragma: no cover - to be implemented per exchange
        raise NotImplementedError

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:  # pragma: no cover - to be implemented per exchange
        raise NotImplementedError

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:  # pragma: no cover - to be implemented per exchange
        raise NotImplementedError


class BinanceSpotAdapter(BaseAdapter):
    """Binance Spot public data adapter (USDT pairs).

    NOTE: This is a thin wrapper over endpoints we already use elsewhere in
    the codebase; keeping it here makes multi-exchange logic easier.
    """

    def __init__(self) -> None:
        super().__init__("https://api.binance.com")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        try:
            data = self._get("/api/v3/ticker/24hr", params={"symbol": symbol.upper()})
        except Exception:
            return None
        try:
            price = float(data.get("lastPrice", 0.0) or 0.0)
            vol = float(data.get("quoteVolume", 0.0) or 0.0)
            high_24h = float(data.get("highPrice")) if data.get("highPrice") not in (None, "") else None
            low_24h = float(data.get("lowPrice")) if data.get("lowPrice") not in (None, "") else None
            price_change = float(data.get("priceChange")) if data.get("priceChange") not in (None, "") else None
            price_change_pct = float(data.get("priceChangePercent")) if data.get("priceChangePercent") not in (None, "") else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=vol,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change=price_change,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        try:
            data = self._get("/api/v3/depth", params={"symbol": symbol.upper(), "limit": 5})
        except Exception:
            return None
        try:
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        try:
            data = self._get("/api/v3/depth", params={"symbol": symbol.upper(), "limit": int(limit)})
        except Exception:
            return None
        try:
            bids = [[float(p), float(q)] for p, q in (data.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q in (data.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        try:
            data = self._get("/api/v3/klines", params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)})
        except Exception:
            return None

        if not isinstance(data, list) or not data:
            return None

        out: List[Candle] = []
        for row in data:
            try:
                ts_ms = int(row[0])
                out.append(Candle(
                    t=int(ts_ms // 1000),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[7]) if len(row) > 7 else None,
                ))
            except Exception:
                continue
        return out


# Placeholders for other exchanges. These can be implemented incrementally
# using their public REST APIs (all free for market data):


class KuCoinAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__("https://api.kucoin.com")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        """24h stats for a KuCoin symbol.

        Uses GET /api/v1/market/stats?symbol=BTC-USDT
        """

        inst = self._to_symbol(symbol)
        try:
            data = self._get("/api/v1/market/stats", params={"symbol": inst})
        except Exception:
            return None

        try:
            detail = data.get("data") or {}
            price = float(detail.get("last", 0.0) or 0.0)
            # volValue is quote volume, vol is base volume.
            vol_quote = detail.get("volValue") or detail.get("vol") or 0.0
            volume_24h = float(vol_quote or 0.0)
            high_24h = float(detail.get("high")) if detail.get("high") not in (None, "") else None
            low_24h = float(detail.get("low")) if detail.get("low") not in (None, "") else None
            change_rate = detail.get("changeRate")
            price_change_pct = float(change_rate) * 100.0 if change_rate not in (None, "") else None
            price_change = float(detail.get("changePrice")) if detail.get("changePrice") not in (None, "") else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change=price_change,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        """Best bid/ask from KuCoin level1 order book.

        Uses GET /api/v1/market/orderbook/level1?symbol=BTC-USDT
        """

        inst = self._to_symbol(symbol)
        try:
            data = self._get("/api/v1/market/orderbook/level1", params={"symbol": inst})
        except Exception:
            return None

        try:
            detail = data.get("data") or {}
            best_bid_raw = detail.get("bestBid")
            best_ask_raw = detail.get("bestAsk")
            best_bid = float(best_bid_raw) if best_bid_raw not in (None, "") else None
            best_ask = float(best_ask_raw) if best_ask_raw not in (None, "") else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        inst = self._to_symbol(symbol)
        try:
            path = "/api/v1/market/orderbook/level2_20" if int(limit) <= 20 else "/api/v1/market/orderbook/level2_100"
            data = self._get(path, params={"symbol": inst})
        except Exception:
            return None

        try:
            detail = data.get("data") or {}
            bids = [[float(p), float(q)] for p, q in (detail.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q in (detail.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        inst = self._to_symbol(symbol)
        type_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "1hour",
            "4h": "4hour",
            "1d": "1day",
        }
        ktype = type_map.get(interval)
        if not ktype:
            return None

        try:
            data = self._get("/api/v1/market/candles", params={"symbol": inst, "type": ktype})
        except Exception:
            return None

        try:
            rows = (data.get("data") or [])
            out: List[Candle] = []
            for row in reversed(rows[: int(limit)]):
                out.append(Candle(
                    t=int(float(row[0])),
                    open=float(row[1]),
                    close=float(row[2]),
                    high=float(row[3]),
                    low=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[6]) if len(row) > 6 else None,
                ))
            return out
        except Exception:
            return None

    @staticmethod
    def _to_symbol(symbol: str) -> str:
        s = symbol.upper()
        if s.endswith("USDT"):
            return f"{s[:-4]}-USDT"
        if s.endswith("USD"):
            return f"{s[:-3]}-USD"
        return s


class MEXCAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__("https://api.mexc.com")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        """24hr ticker statistics for a single symbol.

        Uses GET /api/v3/ticker/24hr?symbol=SYMBOL
        Response example (single object):
        {
          "symbol": "BTCUSDT",
          "lastPrice": "46263.71",
          "quoteVolume": "12345.6",
          ...
        }
        """

        try:
            data = self._get("/api/v3/ticker/24hr", params={"symbol": symbol.upper()})
        except Exception:
            return None

        # Endpoint may return either an object or a list; normalize to object.
        if isinstance(data, list) and data:
            data = data[0]

        try:
            price = float(data.get("lastPrice", 0.0) or 0.0)
            # Prefer quoteVolume if present; fall back to base volume otherwise.
            vol_raw = data.get("quoteVolume")
            if vol_raw in (None, ""):
                vol_raw = data.get("volume", 0.0)
            volume_24h = float(vol_raw or 0.0)
            high_24h = float(data.get("highPrice")) if data.get("highPrice") not in (None, "") else None
            low_24h = float(data.get("lowPrice")) if data.get("lowPrice") not in (None, "") else None
            price_change = float(data.get("priceChange")) if data.get("priceChange") not in (None, "") else None
            price_change_pct = float(data.get("priceChangePercent")) if data.get("priceChangePercent") not in (None, "") else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change=price_change,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        """Best bid/ask from the MEXC order book ticker.

        Uses GET /api/v3/ticker/bookTicker?symbol=SYMBOL
        Example response:
        {
          "symbol": "AEUSDT",
          "bidPrice": "0.11001",
          "bidQty": "115.59",
          "askPrice": "0.11127",
          "askQty": "215.48"
        }
        or a list of such objects.
        """

        try:
            data = self._get("/api/v3/ticker/bookTicker", params={"symbol": symbol.upper()})
        except Exception:
            return None

        if isinstance(data, list) and data:
            data = data[0]

        try:
            bid_raw = data.get("bidPrice")
            ask_raw = data.get("askPrice")
            best_bid = float(bid_raw) if bid_raw not in (None, "") else None
            best_ask = float(ask_raw) if ask_raw not in (None, "") else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        try:
            data = self._get("/api/v3/depth", params={"symbol": symbol.upper(), "limit": int(limit)})
        except Exception:
            return None
        try:
            bids = [[float(p), float(q)] for p, q in (data.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q in (data.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        try:
            data = self._get("/api/v3/klines", params={"symbol": symbol.upper(), "interval": interval, "limit": int(limit)})
        except Exception:
            return None
        if not isinstance(data, list) or not data:
            return None
        out: List[Candle] = []
        for row in data:
            try:
                ts_ms = int(row[0])
                out.append(Candle(
                    t=int(ts_ms // 1000),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[7]) if len(row) > 7 else None,
                ))
            except Exception:
                continue
        return out


class OKXAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__("https://www.okx.com")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        """24h ticker for a single spot instrument.

        Uses GET /api/v5/market/ticker?instId=BTC-USDT style symbols.
        We accept Binance-style "BTCUSDT" and convert to "BTC-USDT" for OKX.
        """

        inst_id = self._to_inst_id(symbol)
        try:
            data = self._get("/api/v5/market/ticker", params={"instId": inst_id})
        except Exception:
            return None

        try:
            items = data.get("data") or []
            if not items:
                return None
            item = items[0]
            # "last" is last traded price, volCcy24h is 24h quote volume.
            price = float(item.get("last", 0.0) or 0.0)
            vol_quote_raw = item.get("volCcy24h") or item.get("vol24h") or 0.0
            volume_24h = float(vol_quote_raw or 0.0)
            high_24h = float(item.get("high24h")) if item.get("high24h") not in (None, "") else None
            low_24h = float(item.get("low24h")) if item.get("low24h") not in (None, "") else None
            open_24h = float(item.get("open24h")) if item.get("open24h") not in (None, "") else None
            price_change = (price - open_24h) if open_24h and open_24h > 0 else None
            price_change_pct = ((price - open_24h) / open_24h * 100.0) if open_24h and open_24h > 0 else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change=price_change,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        """Best bid/ask from OKX order book.

        Uses GET /api/v5/market/books?instId=...&sz=1
        """

        inst_id = self._to_inst_id(symbol)
        try:
            data = self._get("/api/v5/market/books", params={"instId": inst_id, "sz": 1})
        except Exception:
            return None

        try:
            items = data.get("data") or []
            if not items:
                return None
            item = items[0]
            bids = item.get("bids") or []
            asks = item.get("asks") or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        inst_id = self._to_inst_id(symbol)
        try:
            data = self._get("/api/v5/market/books", params={"instId": inst_id, "sz": int(limit)})
        except Exception:
            return None
        try:
            items = data.get("data") or []
            if not items:
                return None
            item = items[0]
            bids = [[float(p), float(q)] for p, q, *_ in (item.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q, *_ in (item.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        inst_id = self._to_inst_id(symbol)
        bar_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        bar = bar_map.get(interval)
        if not bar:
            return None
        try:
            data = self._get("/api/v5/market/candles", params={"instId": inst_id, "bar": bar, "limit": int(limit)})
        except Exception:
            return None
        try:
            rows = data.get("data") or []
            out: List[Candle] = []
            for row in reversed(rows[: int(limit)]):
                ts_ms = int(float(row[0]))
                out.append(Candle(
                    t=int(ts_ms // 1000),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    quote_volume=float(row[6]) if len(row) > 6 else None,
                ))
            return out
        except Exception:
            return None

    @staticmethod
    def _to_inst_id(symbol: str) -> str:
        """Convert a Binance-style symbol (BTCUSDT) to OKX instId (BTC-USDT)."""

        s = symbol.upper()
        if s.endswith("USDT"):
            return s[:-4] + "-USDT"
        return s


class CoinbaseAdapter(BaseAdapter):
    def __init__(self) -> None:
        # Coinbase Advanced / Exchange API base URL
        super().__init__("https://api.exchange.coinbase.com")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        """24h ticker for a Coinbase product.

        Uses GET /products/{product_id}/stats (includes high/low/open/last/volume)
        We accept Binance-style "BTCUSDT" and convert to "BTC-USDT".
        """

        product_id = self._to_product_id(symbol)
        try:
            data = self._get(f"/products/{product_id}/stats")
        except Exception:
            return None

        try:
            price = float(data.get("last", 0.0) or 0.0)
            vol_raw = data.get("volume", 0.0)  # base volume over 24h
            volume_24h = float(vol_raw or 0.0)
            high_24h = float(data.get("high")) if data.get("high") not in (None, "") else None
            low_24h = float(data.get("low")) if data.get("low") not in (None, "") else None
            open_24h = float(data.get("open")) if data.get("open") not in (None, "") else None
            price_change = (price - open_24h) if open_24h and open_24h > 0 else None
            price_change_pct = ((price - open_24h) / open_24h * 100.0) if open_24h and open_24h > 0 else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change=price_change,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        """Best bid/ask from Coinbase order book.

        Uses GET /products/{product_id}/book?level=1
        """

        product_id = self._to_product_id(symbol)
        try:
            data = self._get(f"/products/{product_id}/book", params={"level": 1})
        except Exception:
            return None

        try:
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        product_id = self._to_product_id(symbol)
        try:
            data = self._get(f"/products/{product_id}/book", params={"level": 2})
        except Exception:
            return None
        try:
            bids = [[float(p), float(q)] for p, q, *_ in (data.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q, *_ in (data.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        product_id = self._to_product_id(symbol)
        gran_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "1d": 86400,
        }
        gran = gran_map.get(interval)
        if not gran:
            return None

        end = datetime.now(timezone.utc)
        start = end - timedelta(seconds=int(gran) * int(limit))
        try:
            data = self._get(
                f"/products/{product_id}/candles",
                params={
                    "granularity": int(gran),
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
            )
        except Exception:
            return None
        if not isinstance(data, list) or not data:
            return None
        out: List[Candle] = []
        for row in reversed(data[: int(limit)]):
            try:
                out.append(Candle(
                    t=int(row[0]),
                    low=float(row[1]),
                    high=float(row[2]),
                    open=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                ))
            except Exception:
                continue
        return out

    @staticmethod
    def _to_product_id(symbol: str) -> str:
        """Convert Binance-style symbol (BTCUSDT) to Coinbase product id (BTC-USDT)."""

        s = symbol.upper()
        if s.endswith("USDT"):
            return s[:-4] + "-USDT"
        if s.endswith("USD"):
            return s[:-3] + "-USD"
        return s


class GateAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__("https://api.gateio.ws/api/v4")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        pair = self._to_currency_pair(symbol)
        try:
            data = self._get("/spot/tickers", params={"currency_pair": pair})
        except Exception:
            return None

        if isinstance(data, list) and data:
            data = data[0]

        try:
            price = float(data.get("last", 0.0) or 0.0)
            # Gate.io returns both base_volume and quote_volume; prefer quote_volume if present.
            vol_quote = data.get("quote_volume") or data.get("base_volume") or 0.0
            volume_24h = float(vol_quote or 0.0)
            high_24h = float(data.get("high_24h")) if data.get("high_24h") not in (None, "") else None
            low_24h = float(data.get("low_24h")) if data.get("low_24h") not in (None, "") else None
            price_change_pct = float(data.get("change_percentage")) if data.get("change_percentage") not in (None, "") else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
                price_change_pct=price_change_pct,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        pair = self._to_currency_pair(symbol)
        try:
            data = self._get("/spot/order_book", params={"currency_pair": pair, "limit": 1})
        except Exception:
            return None

        try:
            bids = data.get("bids") or []
            asks = data.get("asks") or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        pair = self._to_currency_pair(symbol)
        try:
            data = self._get("/spot/order_book", params={"currency_pair": pair, "limit": int(limit)})
        except Exception:
            return None
        try:
            bids = [[float(p), float(q)] for p, q, *_ in (data.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q, *_ in (data.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        pair = self._to_currency_pair(symbol)
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }
        gate_interval = interval_map.get(interval)
        if not gate_interval:
            return None

        try:
            data = self._get(
                "/spot/candlesticks",
                params={"currency_pair": pair, "interval": gate_interval, "limit": int(limit)},
            )
        except Exception:
            return None

        if not isinstance(data, list) or not data:
            return None

        def _parse_row(row: Any) -> Optional[Candle]:
            try:
                if isinstance(row, dict):
                    ts = int(float(row.get("t")))
                    o = float(row.get("o"))
                    h = float(row.get("h"))
                    l = float(row.get("l"))
                    c = float(row.get("c"))
                    v = float(row.get("v"))
                    return Candle(t=ts, open=o, high=h, low=l, close=c, volume=v)
                if not isinstance(row, (list, tuple)) or len(row) < 6:
                    return None

                ts = int(float(row[0]))

                # Common Gate.io ordering: [t, volume, close, high, low, open]
                try_a = {
                    "volume": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "open": float(row[5]),
                }

                # Alternative ordering: [t, open, close, high, low, volume]
                try_b = {
                    "open": float(row[1]),
                    "close": float(row[2]),
                    "high": float(row[3]),
                    "low": float(row[4]),
                    "volume": float(row[5]),
                }

                def _valid(o: float, h: float, l: float, c: float) -> bool:
                    return (h >= max(o, c, l)) and (l <= min(o, c, h))

                if _valid(try_a["open"], try_a["high"], try_a["low"], try_a["close"]):
                    return Candle(
                        t=ts,
                        open=try_a["open"],
                        high=try_a["high"],
                        low=try_a["low"],
                        close=try_a["close"],
                        volume=try_a["volume"],
                    )

                return Candle(
                    t=ts,
                    open=try_b["open"],
                    high=try_b["high"],
                    low=try_b["low"],
                    close=try_b["close"],
                    volume=try_b["volume"],
                )
            except Exception:
                return None

        out: List[Candle] = []
        for row in data[: int(limit)]:
            c = _parse_row(row)
            if c is not None:
                out.append(c)

        out.sort(key=lambda x: x.t)
        return out

    @staticmethod
    def _to_currency_pair(symbol: str) -> str:
        s = symbol.upper()
        if s.endswith("USDT"):
            return f"{s[:-4]}_USDT"
        if s.endswith("USD"):
            return f"{s[:-3]}_USD"
        return s


class HTXAdapter(BaseAdapter):
    def __init__(self) -> None:
        # Huobi/HTX public market data base URL
        super().__init__("https://api.huobi.pro")

    def get_ticker_24h(self, symbol: str) -> Optional[Ticker24h]:
        inst = self._to_symbol(symbol)
        try:
            data = self._get("/market/detail", params={"symbol": inst})
        except Exception:
            return None

        try:
            tick = data.get("tick") or {}
            price = float(tick.get("close", 0.0) or 0.0)
            # HTX: tick.get("vol") is quote volume, tick.get("amount") is base.
            volume_24h = float(tick.get("vol", 0.0) or tick.get("amount", 0.0) or 0.0)
            high_24h = float(tick.get("high")) if tick.get("high") not in (None, "") else None
            low_24h = float(tick.get("low")) if tick.get("low") not in (None, "") else None
            if price <= 0:
                return None
            return Ticker24h(
                symbol=symbol.upper(),
                price=price,
                volume_24h=volume_24h,
                high_24h=high_24h,
                low_24h=low_24h,
            )
        except Exception:
            return None

    def get_orderbook_top(self, symbol: str) -> Optional[OrderBookTop]:
        inst = self._to_symbol(symbol)
        try:
            data = self._get("/market/depth", params={"symbol": inst, "type": "step0"})
        except Exception:
            return None

        try:
            tick = data.get("tick") or {}
            bids = tick.get("bids") or []
            asks = tick.get("asks") or []
            best_bid = float(bids[0][0]) if bids else None
            best_ask = float(asks[0][0]) if asks else None
            return OrderBookTop(symbol=symbol.upper(), best_bid=best_bid, best_ask=best_ask)
        except Exception:
            return None

    def get_orderbook(self, symbol: str, limit: int = 20) -> Optional[OrderBookDepth]:
        inst = self._to_symbol(symbol)
        try:
            data = self._get("/market/depth", params={"symbol": inst, "type": "step0"})
        except Exception:
            return None
        try:
            tick = data.get("tick") or {}
            bids = [[float(p), float(q)] for p, q in (tick.get("bids") or [])][:limit]
            asks = [[float(p), float(q)] for p, q in (tick.get("asks") or [])][:limit]
            return OrderBookDepth(symbol=symbol.upper(), bids=bids, asks=asks)
        except Exception:
            return None

    def get_candles(self, symbol: str, interval: str, limit: int = 100) -> Optional[List[Candle]]:
        inst = self._to_symbol(symbol)
        period_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "1h": "60min",
            "4h": "4hour",
            "1d": "1day",
        }
        period = period_map.get(interval)
        if not period:
            return None
        try:
            data = self._get("/market/history/kline", params={"symbol": inst, "period": period, "size": int(limit)})
        except Exception:
            return None
        try:
            rows = data.get("data") or []
            out: List[Candle] = []
            for row in reversed(rows[: int(limit)]):
                out.append(Candle(
                    t=int(row.get("id")),
                    open=float(row.get("open")),
                    high=float(row.get("high")),
                    low=float(row.get("low")),
                    close=float(row.get("close")),
                    volume=float(row.get("amount")),
                    quote_volume=float(row.get("vol")) if row.get("vol") is not None else None,
                ))
            return out
        except Exception:
            return None

    @staticmethod
    def _to_symbol(symbol: str) -> str:
        # HTX expects lowercase symbols like "btcusdt"
        return symbol.lower()
