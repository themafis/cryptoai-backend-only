import os
import math
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Ensure vendored third_party packages (e.g., pandas_ta) are importable BEFORE importing them
_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY))

import pandas_ta as ta
import requests
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
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


def get_funding_rate_rest(symbol: str) -> Dict[str, Any]:
    # Try futures premium index first
    try:
        prem = http_get(f"{BINANCE_FUT}/fapi/v1/premiumIndex", {"symbol": symbol})
        return {
            "fundingRate": float(prem.get("lastFundingRate", 0.0)),
            "nextFundingTime": prem.get("nextFundingTime", 0),
            "openInterest": float(prem.get("openInterest", 0) or 0),
            # iOS decoder expects a nested 'liquidations' object. Provide safe defaults.
            "liquidations": {
                "longLiquidations": 0.0,
                "shortLiquidations": 0.0,
                "liquidationPrice": 0.0,
            },
        }
    except Exception:
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


def get_orderbook(symbol: str, limit: int = 20, ttl: float = 2.0) -> Optional[Dict[str, Any]]:
    # Prefer WS depth cache
    ob_ws = WS_MANAGER.get_orderbook(symbol)
    if ob_ws.get('bids') or ob_ws.get('asks'):
        return ob_ws
    key = f"orderbook:{symbol}:{limit}"
    cached = CACHE.get(key)
    if isinstance(cached, dict):
        return cached
    data = get_orderbook_rest(symbol, limit, futures_first=True)
    if data:
        CACHE.set(key, data, ttl)
        return data
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
def last_rsi_value_for(symbol: str, interval: str, length: int = 14, limit: int = 120, ttl: float = 2.0):
    df = get_ohlcv(symbol, interval, limit, ttl)
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


@app.on_event("startup")
async def on_startup():
    # Start background refresher
    asyncio.create_task(background_refresher())
    # Start WS manager with a small preload; any new symbols will be added on demand
    try:
        asyncio.create_task(WS_MANAGER.run(["BTCUSDT", "ETHUSDT", "SOLUSDT"]))
    except Exception:
        pass

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"message": "API v2 çalışıyor!"}


@app.get("/scalping/{symbol}")
def scalping(symbol: str = "BTCUSDT"):
    try:
        df = get_ohlcv(symbol, "15m", 100, ttl=2.0)
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
                "RSI_1m": float(last_rsi_value_for(symbol, '1m', 14, 120, 2.0)),
                "RSI_5m": float(last_rsi_value_for(symbol, '5m', 14, 120, 2.0)),
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
        df = get_ohlcv(symbol, "5m", 100, ttl=2.0)
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
                "RSI_1m": float(last_rsi_value_for(symbol, '1m', 14, 120, 2.0)),
                "RSI_5m": float(last_rsi_value_for(symbol, '5m', 14, 120, 2.0)),
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/dailytrading/{symbol}")
def dailytrading(symbol: str = "BTCUSDT"):
    try:
        df = get_ohlcv(symbol, "1h", 100, ttl=2.0)
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
        df = get_ohlcv(symbol, interval, limit, ttl=2.0)
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
        # Use 1d data for swing context
        df = get_ohlcv(symbol, "1d", 100, ttl=2.0)
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
        # Fetch base data
        df = get_ohlcv(symbol, "5m", 120, ttl=2.0)
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

        # Orderbook spread
        ob = get_orderbook(symbol, limit=10, ttl=2.0) or {"bids": [], "asks": []}
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
