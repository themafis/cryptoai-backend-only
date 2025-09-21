import os
import math
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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
            "openInterest": prem.get("openInterest", 0)
        }
    except Exception:
        return {"fundingRate": 0.0001, "nextFundingTime": 0, "openInterest": 0}


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


def robust_bbands(close: pd.Series, win: int = 5) -> pd.DataFrame:
    try:
        bb = ta.bbands(close)
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
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

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
                "PP": safe_float(pp),
                "R1": safe_float(r1),
                "R2": safe_float(r2),
                "R3": safe_float(r3),
                "S1": safe_float(s1),
                "S2": safe_float(s2),
                "S3": safe_float(s3)
            },
            "fibonacciLevels": {k: safe_float(v) for k, v in fibonacci_levels.items()},
            "vwap": safe_list(vwap.dropna().tail(3).tolist()),
            "technicalIndicators": {
                "RSI": safe_list((ta.rsi(df['close'], length=14) or pd.Series(dtype='float64')).dropna().tail(3).tolist()),
                "MACD": safe_dict(ta.macd(df['close']).iloc[-1].to_dict() if not ta.macd(df['close']).empty else None),
                "WilliamsR": safe_list((ta.willr(df['high'], df['low'], df['close']) or pd.Series(dtype='float64')).dropna().tail(3).tolist()),
                "CCI": safe_list((ta.cci(df['high'], df['low'], df['close']) or pd.Series(dtype='float64')).dropna().tail(3).tolist()),
                "ATR": safe_list(atr_vals.tail(3).tolist() if not atr_vals.empty else []),
                "KeltnerChannels": safe_dict(keltner.iloc[-1].to_dict() if not keltner.empty else None),
                "BollingerBands": safe_dict(bb.iloc[-1].to_dict() if not bb.empty else None),
            },
            "microMetrics": {
                "RSI_1m": safe_float(((ta.rsi(get_ohlcv(symbol, '1m', 120, 2.0)['close'], length=14)) if (get_ohlcv(symbol, '1m', 120, 2.0) is not None) else pd.Series(dtype='float64')).dropna().iloc[-1]) if (get_ohlcv(symbol, '1m', 120, 2.0) is not None and not (ta.rsi(get_ohlcv(symbol, '1m', 120, 2.0)['close'], length=14) is None)) else MISSING,
                "RSI_5m": safe_float(((ta.rsi(get_ohlcv(symbol, '5m', 120, 2.0)['close'], length=14)) if (get_ohlcv(symbol, '5m', 120, 2.0) is not None) else pd.Series(dtype='float64')).dropna().iloc[-1]) if (get_ohlcv(symbol, '5m', 120, 2.0) is not None and not (ta.rsi(get_ohlcv(symbol, '5m', 120, 2.0)['close'], length=14) is None)) else MISSING,
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/miniscalping/{symbol}")
def miniscalping(symbol: str = "BTCUSDT"):
    try:
        df = get_ohlcv(symbol, "5m", 100, ttl=2.0)
        if df is None or df.empty:
            return {"error": "OHLCV verisi alınamadı"}
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors='coerce')

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
        funding = get_funding_rate_rest(symbol)

        # Response
        return {
            "priceData": {
                "currentPrice": safe_float(df['close'].iloc[-1]),
                "priceChange24h": MISSING,
                "priceChangePercent24h": MISSING,
                "volume24h": MISSING,
                "high24h": safe_float(df['high'].tail(24).max() if len(df) >= 24 else df['high'].max()),
                "low24h": safe_float(df['low'].tail(24).min() if len(df) >= 24 else df['low'].min()),
                "volatility": safe_float((df['close'].pct_change().dropna().std() or 0.0) * np.sqrt(24 * 365)),
                "atr": safe_float(atr.dropna().iloc[-1] if isinstance(atr, pd.Series) and not atr.dropna().empty else 0.0),
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
                "bids": ob.get('bids', [])[:10],
                "asks": ob.get('asks', [])[:10],
                "spread": safe_float(ob['asks'][0][0] - ob['bids'][0][0]) if ob.get('asks') and ob.get('bids') else MISSING,
                "bidVolume": safe_float(sum([float(b[1]) for b in ob.get('bids', [])[:10]])) if ob.get('bids') else MISSING,
                "askVolume": safe_float(sum([float(a[1]) for a in ob.get('asks', [])[:10]])) if ob.get('asks') else MISSING,
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
                "RSI": safe_list((rsi.dropna() if isinstance(rsi, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "MACD": safe_dict(macd_df.iloc[-1].to_dict() if isinstance(macd_df, pd.DataFrame) and not macd_df.empty else None),
                "Stochastic": safe_dict(stoch_df.iloc[-1].to_dict() if isinstance(stoch_df, pd.DataFrame) and not stoch_df.empty else None),
                "ATR": safe_list((atr.dropna() if isinstance(atr, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "Volume": safe_list(df['volume'].tail(3).tolist()),
                "EMA_12": safe_list((ema_12.dropna() if isinstance(ema_12, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "EMA_26": safe_list((ema_26.dropna() if isinstance(ema_26, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "SMA_20": safe_list((sma_20.dropna() if isinstance(sma_20, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "ADX": safe_dict(adx_df.iloc[-1].to_dict() if isinstance(adx_df, pd.DataFrame) and not adx_df.empty else None),
                "BollingerBands": safe_dict(bb_df.iloc[-1].to_dict() if isinstance(bb_df, pd.DataFrame) and not bb_df.empty else None),
                "StochRSI": safe_dict(stochrsi_df.iloc[-1].to_dict() if isinstance(stochrsi_df, pd.DataFrame) and not stochrsi_df.empty else None),
                "CCI": safe_list((cci.dropna() if isinstance(cci, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "OBV": safe_list((obv.dropna() if isinstance(obv, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "VWAP": safe_list(vwap.dropna().tail(3).tolist()),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
                },
            },
            "microMetrics": {
                "RSI_1m": safe_float((ta.rsi((get_ohlcv(symbol, '1m', 120, 2.0) or pd.DataFrame()).get('close', pd.Series(dtype='float64')), length=14)).dropna().iloc[-1]) if get_ohlcv(symbol, '1m', 120, 2.0) is not None else MISSING,
                "RSI_5m": safe_float((ta.rsi((get_ohlcv(symbol, '5m', 120, 2.0) or pd.DataFrame()).get('close', pd.Series(dtype='float64')), length=14)).dropna().iloc[-1]) if get_ohlcv(symbol, '5m', 120, 2.0) is not None else MISSING,
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
                "RSI": safe_list(rsi_values.tail(3).tolist() if not rsi_values.empty else []),
                "MACD": safe_dict(macd_values.iloc[-1].to_dict() if not macd_values.empty else None),
                "ADX": safe_dict(ta.adx(df['high'], df['low'], df['close']).iloc[-1].to_dict() if not ta.adx(df['high'], df['low'], df['close']).empty else None),
                "BollingerBands": safe_dict(bb.iloc[-1].to_dict() if not bb.empty else None),
                "ATR": safe_list(atr_values.tail(3).tolist() if not atr_values.empty else []),
                "Volume": safe_list(df['volume'].tail(10).tolist()),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0),
                },
            },
            "sentiment": {
                "fearGreedIndex": int(np.random.randint(20, 80)),
                "socialSentiment": safe_float(float(np.random.uniform(-1, 1))),
                "newsSentiment": safe_float(float(np.random.uniform(-1, 1))),
                "institutionalFlow": safe_float(float(np.random.uniform(1e7, 5e7)))
            },
            "correlation": {
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
            "bids": ob.get('bids', [])[:limit],
            "asks": ob.get('asks', [])[:limit],
            "spread": (float(ob['asks'][0][0]) - float(ob['bids'][0][0])) if ob.get('asks') and ob.get('bids') else None,
            "bidVolume": float(sum([float(b[1]) for b in ob.get('bids', [])[:limit]])) if ob.get('bids') else None,
            "askVolume": float(sum([float(a[1]) for a in ob.get('asks', [])[:limit]])) if ob.get('asks') else None,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
