import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import math
import ccxt
import requests
import os
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
from typing import Dict, Any, List
import json

try:
    import websockets
except Exception:
    websockets = None  # WebSocket bağımlılığı yoksa, REST fallback çalışır

app = FastAPI()

# Binance API bağlantısı
binance = ccxt.binance({
    'sandbox': False,
    'options': {
        'defaultType': 'future'
    }
})

# --- Binance Futures WebSocket Manager & Shared Cache ---
# Amaç: Çoklu kullanıcıda tek bağlantı ve paylaşılan cache ile rate-limit/ban riskini azaltmak

FSTREAM_WS = "wss://fstream.binance.com/stream"

class SymbolCache:
    def __init__(self):
        self.klines_1m: List[List[float]] = []  # [ts, o, h, l, c, v]
        self.klines_5m: List[List[float]] = []
        self.book_ticker: Dict[str, float] = {}
        self.depth_bids: List[List[float]] = []  # [[price, qty], ...]
        self.depth_asks: List[List[float]] = []
        self.last_bootstrap: float = 0.0
        self.subscribed: bool = False


class BinanceFuturesWSManager:
    def __init__(self):
        self.cache: Dict[str, SymbolCache] = {}
        self._task = None
        self._lock = asyncio.Lock()
        self._running = False
        # Funding rate cache (symbol -> {data, ts})
        self._funding_cache: Dict[str, Dict[str, Any]] = {}
        self._funding_ttl_sec: int = 600  # 10 dakika

    async def ensure_symbol(self, symbol: str):
        s = symbol.upper()
        if s not in self.cache:
            self.cache[s] = SymbolCache()
        sc = self.cache[s]

        # Bootstrap REST (throttled) only if needed
        now = time.time()
        if (now - sc.last_bootstrap) > 60 and (len(sc.klines_1m) < 50 or len(sc.klines_5m) < 50):
            try:
                ohlcv_1m = binance.fetch_ohlcv(s, timeframe='1m', limit=120)
                ohlcv_5m = binance.fetch_ohlcv(s, timeframe='5m', limit=120)
                sc.klines_1m = ohlcv_1m
                sc.klines_5m = ohlcv_5m
                # Ticker
                ticker = binance.fetch_ticker(s)
                sc.book_ticker = {
                    'b': float(ticker.get('bid', ticker.get('last', 0.0)) or 0.0),
                    'B': float(ticker.get('bidVolume', 0.0) or 0.0),
                    'a': float(ticker.get('ask', ticker.get('last', 0.0)) or 0.0),
                    'A': float(ticker.get('askVolume', 0.0) or 0.0),
                }
                # Order book (limited once)
                try:
                    ob = binance.fetch_order_book(s, limit=20)
                    sc.depth_bids = [[float(p), float(q)] for p, q in ob.get('bids', [])[:20]]
                    sc.depth_asks = [[float(p), float(q)] for p, q in ob.get('asks', [])[:20]]
                except Exception:
                    pass
                sc.last_bootstrap = now
            except Exception:
                # Sessizce geç; WS güncelleyecek
                sc.last_bootstrap = now

        # Start subscription once
        if not sc.subscribed:
            await self.subscribe_symbol(s)

    async def subscribe_symbol(self, symbol: str):
        async with self._lock:
            sc = self.cache[symbol]
            if sc.subscribed:
                return
            sc.subscribed = True

    async def run(self):
        if websockets is None:
            # WebSocket yoksa REST ile devam
            return
        self._running = True
        backoff = 1
        while self._running:
            try:
                # Build dynamic combined stream from currently tracked symbols
                streams = []
                for s in list(self.cache.keys()):
                    low = s.lower()
                    streams.extend([
                        f"{low}@kline_1m",
                        f"{low}@kline_5m",
                        f"{low}@bookTicker",
                        f"{low}@depth20@100ms",
                    ])
                if not streams:
                    await asyncio.sleep(0.5)
                    continue
                url = FSTREAM_WS + "?streams=" + "/".join(streams)
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
                    backoff = 1
                    async for raw in ws:
                        try:
                            msg = json.loads(raw)
                        except Exception:
                            continue
                        data = msg.get('data') or {}
                        stream = msg.get('stream', '')
                        if not data or not stream:
                            continue
                        # extract symbol
                        s = (data.get('s') or data.get('ps') or '').upper()
                        if not s:
                            # for depth updates, symbol might be in stream name
                            try:
                                s = stream.split('@')[0].upper()
                            except Exception:
                                continue
                        sc = self.cache.get(s)
                        if not sc:
                            continue
                        if '@kline_1m' in stream:
                            k = data.get('k') or {}
                            ts = k.get('t'); o=k.get('o'); h=k.get('h'); l=k.get('l'); c=k.get('c'); v=k.get('v')
                            try:
                                row = [int(ts), float(o), float(h), float(l), float(c), float(v)]
                                # upsert last
                                if sc.klines_1m and sc.klines_1m[-1][0] == row[0]:
                                    sc.klines_1m[-1] = row
                                else:
                                    sc.klines_1m.append(row)
                                    if len(sc.klines_1m) > 240:
                                        sc.klines_1m = sc.klines_1m[-240:]
                            except Exception:
                                pass
                        elif '@kline_5m' in stream:
                            k = data.get('k') or {}
                            ts = k.get('t'); o=k.get('o'); h=k.get('h'); l=k.get('l'); c=k.get('c'); v=k.get('v')
                            try:
                                row = [int(ts), float(o), float(h), float(l), float(c), float(v)]
                                if sc.klines_5m and sc.klines_5m[-1][0] == row[0]:
                                    sc.klines_5m[-1] = row
                                else:
                                    sc.klines_5m.append(row)
                                    if len(sc.klines_5m) > 240:
                                        sc.klines_5m = sc.klines_5m[-240:]
                            except Exception:
                                pass
                        elif '@bookTicker' in stream:
                            # best bid/ask
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
                            # partial depth update snapshot-like
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

    def get_funding_rate_cached(self, symbol: str) -> Dict[str, Any]:
        s = symbol.upper()
        now = time.time()
        cached = self._funding_cache.get(s)
        if cached and (now - cached.get('ts', 0)) < self._funding_ttl_sec:
            return cached['data']
        try:
            fr = binance.fetch_funding_rate(s)
            data = {
                'fundingRate': fr.get('fundingRate', 0.0001),
                'nextFundingTime': fr.get('nextFundingTime', 0),
                'openInterest': fr.get('openInterest', 0),
            }
            self._funding_cache[s] = {'data': data, 'ts': now}
            return data
        except Exception:
            # On failure, return last cached or default
            if cached:
                return cached['data']
            return {'fundingRate': 0.0001, 'nextFundingTime': 0, 'openInterest': 0}

ws_manager = BinanceFuturesWSManager()

@app.on_event("startup")
async def _startup_ws():
    # Başlangıçta BTCUSDT'yi izlemeye başla; diğer semboller ilk istek geldiğinde eklenecek
    preload = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    for s in preload:
        try:
            await ws_manager.ensure_symbol(s)
        except Exception:
            pass
    if websockets is not None:
        asyncio.create_task(ws_manager.run())

def get_cached_ohlcv(symbol: str, timeframe: str, limit: int = 120) -> pd.DataFrame:
    """Cache'den OHLCV al; yoksa REST ile bootstrap etmeye çalış."""
    s = symbol.upper()
    sc = ws_manager.cache.get(s)
    rows: List[List[float]] = []
    if sc:
        if timeframe == '1m':
            rows = sc.klines_1m[-limit:]
        elif timeframe == '5m':
            rows = sc.klines_5m[-limit:]
    if not rows:
        # Fallback REST (throttled by ensure_symbol)
        try:
            data = binance.fetch_ohlcv(s, timeframe=timeframe, limit=limit)
            rows = data
        except Exception:
            rows = []
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']) if rows else pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])
    return df

def get_cached_ticker(symbol: str) -> Dict[str, float]:
    s = symbol.upper()
    sc = ws_manager.cache.get(s)
    if sc and sc.book_ticker:
        return sc.book_ticker
    try:
        t = binance.fetch_ticker(s)
        return {
            'b': float(t.get('bid', t.get('last', 0.0)) or 0.0),
            'B': float(t.get('bidVolume', 0.0) or 0.0),
            'a': float(t.get('ask', t.get('last', 0.0)) or 0.0),
            'A': float(t.get('askVolume', 0.0) or 0.0),
        }
    except Exception:
        return {'b': 0.0, 'B': 0.0, 'a': 0.0, 'A': 0.0}

def get_cached_orderbook(symbol: str) -> Dict[str, List[List[float]]]:
    s = symbol.upper()
    sc = ws_manager.cache.get(s)
    if sc and (sc.depth_bids or sc.depth_asks):
        return {'bids': sc.depth_bids, 'asks': sc.depth_asks}
    try:
        ob = binance.fetch_order_book(s, limit=20)
        return {
            'bids': [[float(p), float(q)] for p, q in ob.get('bids', [])[:20]],
            'asks': [[float(p), float(q)] for p, q in ob.get('asks', [])[:20]],
        }
    except Exception:
        return {'bids': [], 'asks': []}

@app.get("/")
def read_root():
    return {"message": "API çalışıyor!"}

@app.get("/price/{coin_id}")
def get_coin_price(coin_id: str = "bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
    response = requests.get(url)
    data = response.json()
    return {coin_id: data.get(coin_id, {})}

@app.get("/ohlc/{coin_id}")
def get_coin_ohlc(coin_id: str = "bitcoin", days: int = 1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    # OHLCV: [timestamp, open, high, low, close]
    return {"ohlc": data}

@app.get("/rsi/{coin_id}")
def get_coin_rsi(coin_id: str = "bitcoin", days: int = 1):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    if not data:
        return {"error": "Veri yok"}
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    rsi = ta.rsi(df["close"], length=14)
    return {"rsi": rsi.dropna().tolist()}

@app.get("/indicators/{coin_id}")
def get_all_indicators(coin_id: str = "bitcoin", days: int = 30):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        response = requests.get(url)
        data = response.json()
        print(f"[DEBUG] API'den dönen ham veri: {data}")
        if not data or not isinstance(data, list) or len(data) == 0:
            print("[ERROR] Veri yok veya CoinGecko API boş döndü")
            return {"error": "Veri yok veya CoinGecko API boş döndü"}
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        print(f"[DEBUG] Oluşan DataFrame: {df.head()}")
        if df.empty or df.isnull().all().all():
            print("[ERROR] DataFrame tamamen boş veya tüm değerler NaN")
            return {"error": "Veri DataFrame'e aktarılamadı veya tamamen boş"}
        # Sütunlarda None veya tümü NaN ise
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns or df[col].isnull().all():
                print(f"[ERROR] {col} sütunu boş veya eksik!")
                return {"error": f"{col} sütunu boş veya eksik"}
        result = {}
        # Temel indikatörler
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist() if not df["close"].isnull().all() else []
        macd_df = ta.macd(df["close"]) if not df["close"].isnull().all() else pd.DataFrame()
        result["MACD"] = macd_df.iloc[-1].to_dict() if not macd_df.empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist() if not df["close"].isnull().all() else []
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist() if not df["close"].isnull().all() else []
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist() if not df["close"].isnull().all() else []
        adx_df = ta.adx(df["high"], df["low"], df["close"]) if not df["high"].isnull().all() and not df["low"].isnull().all() and not df["close"].isnull().all() else pd.DataFrame()
        result["ADX"] = adx_df.iloc[-1].to_dict() if not adx_df.empty else {}
        bb_df = ta.bbands(df["close"])
        result["BollingerBands"] = bb_df.iloc[-1].to_dict() if not bb_df.empty else {}
        stochrsi_df = ta.stochrsi(df["close"])
        result["StochRSI"] = stochrsi_df.iloc[-1].to_dict() if not stochrsi_df.empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        result["OBV"] = ta.obv(df["close"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        if "volume" in df.columns and not df["volume"].isnull().all():
            result["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        else:
            result["VWAP"] = None
        pivot = ((df["high"] + df["low"] + df["close"]) / 3)
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        psar_df = ta.psar(df["high"], df["low"])
        result["ParabolicSAR"] = psar_df.iloc[-1].to_dict() if not psar_df.empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        supertrend_df = ta.supertrend(df["high"], df["low"], df["close"])
        result["Supertrend"] = supertrend_df.iloc[-1].to_dict() if not supertrend_df.empty else {}
        result["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["close"]).dropna().tolist() if not df["close"].isnull().all() else []
        donchian_df = ta.donchian(df["high"], df["low"])
        result["Donchian"] = donchian_df.iloc[-1].to_dict() if not donchian_df.empty else {}
        keltner_df = ta.kc(df["high"], df["low"], df["close"])
        result["Keltner"] = keltner_df.iloc[-1].to_dict() if not keltner_df.empty else {}
        result["Volume"] = df["close"].tolist() if not df["close"].isnull().all() else []
        print(f"[DEBUG] Hesaplanan indikatörler: {result}")
        return clean_json(result)
    except Exception as e:
        print(f"[EXCEPTION] {str(e)}")
        return {"error": str(e)}

def get_binance_ohlcv(symbol="BTCUSDT", interval="1d", limit=30):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    # open_time'ı datetime'a çevir ve index yap
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df

def get_coingecko_ohlc(coin_id="bitcoin", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df

@app.get("/indicators_binance/{symbol}")
def get_all_indicators_binance(symbol: str = "BTCUSDT", interval: str = "1d", limit: int = 30):
    try:
        df = get_binance_ohlcv(symbol, interval, limit)
        result = {}
        # Hacim gerektirmeyenler
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist()
        result["MACD"] = ta.macd(df["close"]).iloc[-1].to_dict() if not ta.macd(df["close"]).empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist()
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist()
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist()
        result["ADX"] = ta.adx(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.adx(df["high"], df["low"], df["close"]).empty else {}
        result["BollingerBands"] = ta.bbands(df["close"]).iloc[-1].to_dict() if not ta.bbands(df["close"]).empty else {}
        result["StochRSI"] = ta.stochrsi(df["close"]).iloc[-1].to_dict() if not ta.stochrsi(df["close"]).empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist()
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist()
        # Klasik pivot noktası hesaplama (manuel)
        pivot = (df["high"] + df["low"] + df["close"]) / 3
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        result["ParabolicSAR"] = ta.psar(df["high"], df["low"]).iloc[-1].to_dict() if not ta.psar(df["high"], df["low"]).empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist()
        result["Donchian"] = ta.donchian(df["high"], df["low"]).iloc[-1].to_dict() if not ta.donchian(df["high"], df["low"]).empty else {}
        result["Keltner"] = ta.kc(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.kc(df["high"], df["low"], df["close"]).empty else {}
        # Hacim gerektirenler (sadece Binance ile)
        result["VWAP"] = ta.vwap(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        result["MFI"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"]).dropna().tolist()
        result["OBV"] = ta.obv(df["close"], df["volume"]).dropna().tolist()
        # Ekstra hacim
        result["Volume"] = df["volume"].tolist()
        return clean_json(result)
    except Exception as e:
        return {"error": str(e)}

# YENİ ENDPOINT'LER - SADECE AI İÇİN VERİLER

@app.get("/miniscalping/{symbol}")
def get_mini_scalping_data(symbol: str = "BTCUSDT"):
    """Mini Scalping için özel veriler - sadece AI'ya gönderilir"""
    
    # JSON serialization için güvenli değerler
    def safe_float(value):
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def safe_list(values):
        if values is None:
            return []
        return [safe_float(v) for v in values if not (np.isnan(v) or np.isinf(v))]
    
    def safe_dict(data_dict):
        if data_dict is None:
            return {}
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                result[key] = safe_float(value)
            elif isinstance(value, dict):
                result[key] = safe_dict(value)
            else:
                result[key] = value
        return result
    
    try:
        print(f"[DEBUG] Mini scalping endpoint çağrıldı: {symbol}")
        
        # Paylaşılan cache üzerinden Order book verisi (WS varsa WS'den)
        try:
            awaitable = ws_manager.ensure_symbol(symbol)
            if asyncio.iscoroutine(awaitable):
                # Fonksiyon sync, FastAPI onu sync çalıştıracak; event loop yoksa geç
                try:
                    asyncio.get_running_loop()
                    # içerisindeysek beklemeye gerek yok
                except RuntimeError:
                    asyncio.run(awaitable)
            orderbook = get_cached_orderbook(symbol)
            print(f"[DEBUG] Order book (cache) alındı: {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
        except Exception as e:
            print(f"[ERROR] Order book hatası: {e}")
            orderbook = {"bids": [], "asks": []}
        
        # Funding rate (futures için) - 10 dakika cache
        try:
            funding_rate = ws_manager.get_funding_rate_cached(symbol)
            print(f"[DEBUG] Funding rate (cache) alındı: {funding_rate}")
        except Exception as e:
            print(f"[ERROR] Funding rate (cache) hatası: {e}")
            funding_rate = {"fundingRate": 0.0001, "nextFundingTime": 0, "openInterest": 0}
        
        # Teknik indikatörler (5m) - cache'den
        try:
            awaitable = ws_manager.ensure_symbol(symbol)
            if asyncio.iscoroutine(awaitable):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(awaitable)
            df = get_cached_ohlcv(symbol, '5m', limit=120)
            print(f"[DEBUG] OHLCV (cache) alındı: {len(df)} veri noktası")
            if df.empty:
                raise Exception('Cache boş')
        except Exception as e:
            print(f"[ERROR] OHLCV hatası: {e}")
            return {"error": f"OHLCV verisi alınamadı: {e}"}
        
        # RSI hesaplama
        try:
            rsi = ta.rsi(df['close'], length=14)
        except Exception as e:
            print(f"RSI hesaplama hatası: {e}")
            rsi = pd.Series([0.0])
        
        # MACD hesaplama
        try:
            macd_df = ta.macd(df['close'])
        except Exception as e:
            print(f"MACD hesaplama hatası: {e}")
            macd_df = pd.DataFrame()
        
        # Stochastic hesaplama
        try:
            stoch_df = ta.stoch(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"Stochastic hesaplama hatası: {e}")
            stoch_df = pd.DataFrame()
        
        # ATR hesaplama
        try:
            atr = ta.atr(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"ATR hesaplama hatası: {e}")
            atr = pd.Series([0.0])
        
        # EMA hesaplama
        try:
            ema_12 = ta.ema(df['close'], length=12)
            ema_26 = ta.ema(df['close'], length=26)
        except Exception as e:
            print(f"EMA hesaplama hatası: {e}")
            ema_12 = pd.Series([0.0])
            ema_26 = pd.Series([0.0])
        
        # SMA hesaplama
        try:
            sma_20 = ta.sma(df['close'], length=20)
        except Exception as e:
            print(f"SMA hesaplama hatası: {e}")
            sma_20 = pd.Series([0.0])
        
        # ADX hesaplama
        try:
            adx_df = ta.adx(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"ADX hesaplama hatası: {e}")
            adx_df = pd.DataFrame()
        
        # Bollinger Bands hesaplama
        try:
            bb_df = ta.bbands(df['close'])
        except Exception as e:
            print(f"Bollinger Bands hesaplama hatası: {e}")
            bb_df = pd.DataFrame()
        
        # Stochastic RSI hesaplama
        try:
            stochrsi_df = ta.stochrsi(df['close'])
        except Exception as e:
            print(f"Stochastic RSI hesaplama hatası: {e}")
            stochrsi_df = pd.DataFrame()
        
        # CCI hesaplama
        try:
            cci = ta.cci(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"CCI hesaplama hatası: {e}")
            cci = pd.Series([0.0])
        
        # OBV hesaplama
        try:
            obv = ta.obv(df['close'], df['volume'])
        except Exception as e:
            print(f"OBV hesaplama hatası: {e}")
            obv = pd.Series([0.0])
        
        # VWAP hesaplama - Manuel hesaplama ile
        try:
            # Manuel VWAP hesaplama
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_values = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            vwap = vwap_values
        except Exception as e:
            print(f"VWAP hesaplama hatası: {e}")
            vwap = pd.Series([df['close'].mean() if not df.empty else 0.0])
        
        # Pivot Points hesaplama
        try:
            pivot = ((df['high'] + df['low'] + df['close']) / 3)
            r1 = (2 * pivot) - df['low']
            s1 = (2 * pivot) - df['high']
            r2 = pivot + (df['high'] - df['low'])
            s2 = pivot - (df['high'] - df['low'])
        except Exception as e:
            print(f"Pivot Points hesaplama hatası: {e}")
            pivot = pd.Series([0.0])
            r1 = pd.Series([0.0])
            s1 = pd.Series([0.0])
            r2 = pd.Series([0.0])
            s2 = pd.Series([0.0])
        
        # Ichimoku hesaplama - Manuel hesaplama ile
        try:
            # Manuel Ichimoku hesaplama
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            conversion = (high_9 + low_9) / 2
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            base = (high_26 + low_26) / 2
            
            span_a = ((conversion + base) / 2).shift(26)
            span_b = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
            
            # NaN değerleri kontrol et
            conversion_val = conversion.iloc[-1] if not conversion.empty and not pd.isna(conversion.iloc[-1]) else df['close'].iloc[-1]
            base_val = base.iloc[-1] if not base.empty and not pd.isna(base.iloc[-1]) else df['close'].iloc[-1]
            span_a_val = span_a.iloc[-1] if not span_a.empty and not pd.isna(span_a.iloc[-1]) else df['close'].iloc[-1]
            span_b_val = span_b.iloc[-1] if not span_b.empty and not pd.isna(span_b.iloc[-1]) else df['close'].iloc[-1]
            
            ichimoku_data = {
                "conversion": safe_float(conversion_val),
                "base": safe_float(base_val),
                "span_a": safe_float(span_a_val),
                "span_b": safe_float(span_b_val)
            }
        except Exception as e:
            print(f"Ichimoku hesaplama hatası: {e}")
            current_price = df['close'].iloc[-1] if not df.empty else 0.0
            ichimoku_data = {
                "conversion": safe_float(current_price),
                "base": safe_float(current_price),
                "span_a": safe_float(current_price),
                "span_b": safe_float(current_price)
            }
        
        # Parabolic SAR hesaplama
        try:
            psar_df = ta.psar(df['high'], df['low'])
        except Exception as e:
            print(f"Parabolic SAR hesaplama hatası: {e}")
            psar_df = pd.DataFrame()
        
        # Williams %R hesaplama
        try:
            williams_r = ta.willr(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"Williams %R hesaplama hatası: {e}")
            williams_r = pd.Series([0.0])
        
        # Supertrend hesaplama
        try:
            supertrend_df = ta.supertrend(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"Supertrend hesaplama hatası: {e}")
            supertrend_df = pd.DataFrame()
        
        # MFI hesaplama
        try:
            mfi = ta.mfi(df['high'], df['low'], df['close'], df['volume'])
        except Exception as e:
            print(f"MFI hesaplama hatası: {e}")
            mfi = pd.Series([0.0])
        
        # Donchian Channel hesaplama
        try:
            donchian_df = ta.donchian(df['high'], df['low'])
        except Exception as e:
            print(f"Donchian Channel hesaplama hatası: {e}")
            donchian_df = pd.DataFrame()
        
        # Keltner Channel hesaplama
        try:
            keltner_df = ta.kc(df['high'], df['low'], df['close'])
        except Exception as e:
            print(f"Keltner Channel hesaplama hatası: {e}")
            keltner_df = pd.DataFrame()
        
        # Real-time price data - cache'den bookTicker
        try:
            bt = get_cached_ticker(symbol)
            # last price approx mid
            mid = (bt.get('b', 0.0) + bt.get('a', 0.0)) / 2 if (bt.get('b', 0.0) and bt.get('a', 0.0)) else 0.0
            current_price = safe_float(mid or (df['close'].iloc[-1] if not df.empty else 0.0))
            # 24h alanları WS'de yok; gerekirse REST'e düşülebilir. Basitçe 0 bırakıyoruz.
            price_change_24h = 0.0
            price_change_percent_24h = 0.0
            volume_24h = float(df['volume'].tail(288).sum()) if not df.empty else 0.0
            high_24h = float(df['high'].tail(288).max()) if not df.empty else 0.0
            low_24h = float(df['low'].tail(288).min()) if not df.empty else 0.0
        except Exception as e:
            print(f"Ticker (cache) hatası: {e}")
            current_price = safe_float(df['close'].iloc[-1] if not df.empty else 0.0)
            price_change_24h = 0.0
            price_change_percent_24h = 0.0
            volume_24h = 0.0
            high_24h = 0.0
            low_24h = 0.0
        
        # Market sentiment data (simüle edilmiş)
        fear_greed_index = np.random.randint(20, 80)
        social_sentiment = np.random.uniform(-1, 1)
        news_sentiment = np.random.uniform(-1, 1)
        institutional_flow = np.random.uniform(10000000, 50000000)
        
        # Volatility metrics
        try:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365)  # Annualized volatility
            atr_current = atr.iloc[-1] if not atr.empty else 0.0
        except Exception as e:
            print(f"Volatility hesaplama hatası: {e}")
            volatility = 0.0
            atr_current = 0.0
        
        # Mikro zaman serisi metrikleri (1m/5m RSI, volatility burst, LH/LL, delta approx)
        def lower_high_lower_low_counts(series_high, series_low, lookback=10):
            lh = 0
            ll = 0
            for i in range(2, min(lookback+2, len(series_high))):
                if series_high.iloc[-i] < series_high.iloc[-i-1]:
                    lh += 1
                if series_low.iloc[-i] < series_low.iloc[-i-1]:
                    ll += 1
            return lh, ll

        def volatility_burst(close_series, window=20):
            rets = close_series.pct_change().dropna()
            if len(rets) < window + 1:
                return 0.0
            recent = rets.iloc[-1]
            std = rets.tail(window).std() or 0.0
            return float(recent / std) if std != 0 else 0.0

        def cvd_approx(close_series, volume_series):
            # Yaklaşık CVD: bar yönü * hacim'in kümülatifi
            if len(close_series) != len(volume_series):
                return 0.0
            direction = np.sign(close_series.diff().fillna(0))
            delta = direction * volume_series
            return float(delta.cumsum().iloc[-1])

        # 1m ve 5m OHLCV ile mikro metrikleri hesapla
        micro = {}
        try:
            d1 = get_cached_ohlcv(symbol, '1m', limit=120)
            micro['RSI_1m'] = float(ta.rsi(d1['close'], length=14).dropna().iloc[-1]) if not d1.empty else 0.0
            micro['volBurst_1m'] = volatility_burst(d1['close'], window=20)
            lh1, ll1 = lower_high_lower_low_counts(d1['high'], d1['low'], lookback=10)
            micro['lowerHighCount_1m'] = int(lh1)
            micro['lowerLowCount_1m'] = int(ll1)
            micro['cvd_approx_1m'] = cvd_approx(d1['close'], d1['volume'])
        except Exception:
            micro.update({
                'RSI_1m': 0.0,
                'volBurst_1m': 0.0,
                'lowerHighCount_1m': 0,
                'lowerLowCount_1m': 0,
                'cvd_approx_1m': 0.0
            })

        try:
            d5 = get_cached_ohlcv(symbol, '5m', limit=120)
            micro['RSI_5m'] = float(ta.rsi(d5['close'], length=14).dropna().iloc[-1]) if not d5.empty else 0.0
            micro['volBurst_5m'] = volatility_burst(d5['close'], window=20)
            lh5, ll5 = lower_high_lower_low_counts(d5['high'], d5['low'], lookback=10)
            micro['lowerHighCount_5m'] = int(lh5)
            micro['lowerLowCount_5m'] = int(ll5)
            micro['cvd_approx_5m'] = cvd_approx(d5['close'], d5['volume'])
        except Exception:
            micro.update({
                'RSI_5m': 0.0,
                'volBurst_5m': 0.0,
                'lowerHighCount_5m': 0,
                'lowerLowCount_5m': 0,
                'cvd_approx_5m': 0.0
            })

        # JSON response için güvenli veri hazırlama
        response_data = {
            "priceData": {
                "currentPrice": current_price,
                "priceChange24h": price_change_24h,
                "priceChangePercent24h": price_change_percent_24h,
                "volume24h": volume_24h,
                "high24h": high_24h,
                "low24h": low_24h,
                "volatility": safe_float(volatility),
                "atr": safe_float(atr_current)
            },
            "pivotPoints": {
                "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
            },
            "priceData": {
                "currentPrice": current_price,
                "priceChange24h": price_change_24h,
                "priceChangePercent24h": price_change_percent_24h,
                "volume24h": volume_24h,
                "high24h": high_24h,
                "low24h": low_24h,
                "volatility": safe_float(volatility),
                "atr": safe_float(atr_current)
            },
            "marketSentiment": {
                "fearGreedIndex": fear_greed_index,
                "socialSentiment": safe_float(social_sentiment),
                "newsSentiment": safe_float(news_sentiment),
                "institutionalFlow": safe_float(institutional_flow)
            },
            "orderBook": {
                "bids": orderbook['bids'][:10] if orderbook['bids'] else [],
                "asks": orderbook['asks'][:10] if orderbook['asks'] else [],
                "spread": safe_float(orderbook['asks'][0][0] - orderbook['bids'][0][0]) if orderbook['asks'] and orderbook['bids'] else 0.0,
                "bidVolume": safe_float(sum([bid[1] for bid in orderbook['bids'][:10]])) if orderbook['bids'] else 0.0,
                "askVolume": safe_float(sum([ask[1] for ask in orderbook['asks'][:10]])) if orderbook['asks'] else 0.0
            },
            "marketDepth": {
                "priceLevels": [
                    {"price": safe_float(orderbook['bids'][i][0]), "bidVolume": safe_float(orderbook['bids'][i][1]), "askVolume": 0.0} 
                    for i in range(min(5, len(orderbook['bids'])))
                ] + [
                    {"price": safe_float(orderbook['asks'][i][0]), "bidVolume": 0.0, "askVolume": safe_float(orderbook['asks'][i][1])} 
                    for i in range(min(5, len(orderbook['asks'])))
                ] if orderbook['bids'] and orderbook['asks'] else [],
                "totalBidVolume": safe_float(sum([bid[1] for bid in orderbook['bids'][:10]])) if orderbook['bids'] else 0.0,
                "totalAskVolume": safe_float(sum([ask[1] for ask in orderbook['asks'][:10]])) if orderbook['asks'] else 0.0,
                "imbalance": safe_float(sum([bid[1] for bid in orderbook['bids'][:10]]) / sum([ask[1] for ask in orderbook['asks'][:10]]) if sum([ask[1] for ask in orderbook['asks'][:10]]) > 0 else 1.0) if orderbook['bids'] and orderbook['asks'] else 1.0
            },
            "fundingRate": {
                "symbol": symbol,
                "fundingRate": safe_float(funding_rate.get('fundingRate', 0.0001)),
                "nextFundingTime": funding_rate.get('nextFundingTime', 0),
                "openInterest": funding_rate.get('openInterest', 0),
                "liquidations": {
                    "longLiquidations": 0.0,
                    "shortLiquidations": 0.0,
                    "liquidationPrice": safe_float(df['low'].iloc[-1] * 0.95) if not df.empty else 0.0
                }
            },
            "correlationData": {
                "btcCorrelation": safe_float(np.random.uniform(0.5, 0.95)),
                "ethCorrelation": safe_float(np.random.uniform(0.3, 0.8)),
                "marketCapRank": 1
            },
            "onChainMetrics": {
                "activeAddresses": np.random.randint(500000, 1000000),
                "transactionCount": np.random.randint(200000, 500000),
                "networkHashRate": safe_float(np.random.uniform(300000000, 600000000)),
                "stakingRatio": safe_float(np.random.uniform(0.1, 0.3)),
                "whaleTransactions": np.random.randint(20, 100)
            },
            "technicalIndicators": {
                "RSI": safe_list(rsi.dropna().tail(3).tolist()),
                "MACD": safe_dict(macd_df.iloc[-1].to_dict() if not macd_df.empty else {}),
                "Stochastic": safe_dict(stoch_df.iloc[-1].to_dict() if not stoch_df.empty else {}),
                "ATR": safe_list(atr.dropna().tail(3).tolist()),
                "Volume": safe_list(df['volume'].tail(3).tolist()),
                "EMA_12": safe_list(ema_12.dropna().tail(3).tolist()),
                "EMA_26": safe_list(ema_26.dropna().tail(3).tolist()),
                "SMA_20": safe_list(sma_20.dropna().tail(3).tolist()),
                "ADX": safe_dict(adx_df.iloc[-1].to_dict() if not adx_df.empty else {}),
                "BollingerBands": safe_dict(bb_df.iloc[-1].to_dict() if not bb_df.empty else {}),
                "StochRSI": safe_dict(stochrsi_df.iloc[-1].to_dict() if not stochrsi_df.empty else {}),
                "CCI": safe_list(cci.dropna().tail(3).tolist()),
                "OBV": safe_list(obv.dropna().tail(3).tolist()),
                "VWAP": safe_list(vwap.dropna().tail(3).tolist()),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
                },
                "Ichimoku": ichimoku_data,
                "ParabolicSAR": safe_dict(psar_df.iloc[-1].to_dict() if not psar_df.empty else {}),
                "WilliamsR": safe_list(williams_r.dropna().tail(3).tolist()),
                "Supertrend": safe_dict(supertrend_df.iloc[-1].to_dict() if not supertrend_df.empty else {}),
                "MFI": safe_list(mfi.dropna().tail(3).tolist()),
                "Donchian": safe_dict(donchian_df.iloc[-1].to_dict() if not donchian_df.empty else {}),
                "Keltner": safe_dict(keltner_df.iloc[-1].to_dict() if not keltner_df.empty else {})
            },
            "microMetrics": {k: safe_float(v) if isinstance(v, float) else v for k, v in micro.items()}
        }
        
        # JSON formatında döndür
        return response_data
    except Exception as e:
        return {"error": str(e)}

@app.get("/scalping/{symbol}")
def get_scalping_data(symbol: str = "BTCUSDT"):
    """Scalping için özel veriler - sadece AI'ya gönderilir"""
    try:
        # OHLCV verisi (15m) - 5m cache'den türet (3x5m = 15m)
        try:
            # ensure symbol tracked
            awaitable = ws_manager.ensure_symbol(symbol)
            if asyncio.iscoroutine(awaitable):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(awaitable)
            d5 = get_cached_ohlcv(symbol, '5m', limit=300)
            if d5.empty:
                raise Exception('5m cache boş')
            # 5m -> 15m aggregate
            d5['timestamp'] = pd.to_datetime(d5['timestamp'], unit='ms')
            d5.set_index('timestamp', inplace=True)
            agg = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            df = d5.resample('15T').agg(agg).dropna().tail(100).reset_index()
            df.rename(columns={'timestamp': 'ts'}, inplace=True)
        except Exception as e:
            # Fallback: tek seferlik REST çağrısı
            ohlcv = binance.fetch_ohlcv(symbol, '15m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        
        # Pivot points hesaplama
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
        
        # Fibonacci retracements
        swing_high = df['high'].max()
        swing_low = df['low'].min()
        diff = swing_high - swing_low
        
        fibonacci_levels = {
            "0.236": swing_high - 0.236 * diff,
            "0.382": swing_high - 0.382 * diff,
            "0.500": swing_high - 0.500 * diff,
            "0.618": swing_high - 0.618 * diff,
            "0.786": swing_high - 0.786 * diff
        }
        
        # VWAP hesaplama
        try:
            vwap = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        except:
            vwap = pd.Series([df['close'].mean()] * len(df))
        
        # Keltner Channels
        keltner = ta.kc(df['high'], df['low'], df['close'])
        
        # Current price and 24h high/low for Scalping
        current_price = float(df['close'].iloc[-1])
        high_24h = float(df['high'].tail(24).max() if len(df) >= 24 else df['high'].max())
        low_24h = float(df['low'].tail(24).min() if len(df) >= 24 else df['low'].min())
        
        # ATR hesaplama
        atr_values = ta.atr(df['high'], df['low'], df['close']).dropna()
        current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else 0.0
        
        # JSON serialization için güvenli değerler
        def safe_float(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        def safe_list(values):
            if values is None:
                return []
            return [safe_float(v) for v in values if not (np.isnan(v) or np.isinf(v))]
        
        def safe_dict(data_dict):
            if data_dict is None:
                return {}
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    result[key] = safe_float(value)
                elif isinstance(value, dict):
                    result[key] = safe_dict(value)
                else:
                    result[key] = value
            return result
        
        return {
            "priceData": {
                "currentPrice": safe_float(current_price),
                "high24h": safe_float(high_24h),
                "low24h": safe_float(low_24h),
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
                "RSI": safe_list(ta.rsi(df['close'], length=14).dropna().tail(3).tolist()),
                "MACD": safe_dict(ta.macd(df['close']).iloc[-1].to_dict() if not ta.macd(df['close']).empty else {}),
                "WilliamsR": safe_list(ta.willr(df['high'], df['low'], df['close']).dropna().tail(3).tolist()),
                "CCI": safe_list(ta.cci(df['high'], df['low'], df['close']).dropna().tail(3).tolist()),
                "ATR": safe_list(ta.atr(df['high'], df['low'], df['close']).dropna().tail(3).tolist()),
                "KeltnerChannels": safe_dict(keltner.iloc[-1].to_dict() if not keltner.empty else {})
            },
            "microMetrics": {
                # 1m/5m RSI ve approx CVD + volatility burst
                "RSI_1m": safe_float(ta.rsi(get_binance_ohlcv(symbol, '1m', 120)['close'], length=14).dropna().iloc[-1]) if not get_binance_ohlcv(symbol, '1m', 120).empty else 0.0,
                "RSI_5m": safe_float(ta.rsi(get_binance_ohlcv(symbol, '5m', 120)['close'], length=14).dropna().iloc[-1]) if not get_binance_ohlcv(symbol, '5m', 120).empty else 0.0
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/dailytrading/{symbol}")
def get_daily_trading_data(symbol: str = "BTCUSDT"):
    """Daily Trading için özel veriler - sadece AI'ya gönderilir"""
    try:
        # OHLCV verisi (1h)
        ohlcv = binance.fetch_ohlcv(symbol, '1h', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Pivot Points hesaplama (destek/direnç için gerekli)
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        
        # Teknik indikatörler
        rsi_values = ta.rsi(df['close'], length=14).dropna()
        macd_values = ta.macd(df['close'])
        
        # Market sentiment (simüle edilmiş)
        fear_greed_index = np.random.randint(20, 80)
        social_sentiment = np.random.uniform(-1, 1)
        news_sentiment = np.random.uniform(-1, 1)
        
        # Correlation data (simüle edilmiş)
        btc_correlation = np.random.uniform(0.5, 0.95)
        eth_correlation = np.random.uniform(0.3, 0.8)
        
        # On-chain metrics (simüle edilmiş)
        active_addresses = np.random.randint(500000, 1000000)
        transaction_count = np.random.randint(200000, 500000)
        whale_transactions = np.random.randint(20, 100)
        
        # Multi-timeframe analysis
        df_4h = get_binance_ohlcv(symbol, "4h", 50)
        df_1d = get_binance_ohlcv(symbol, "1d", 30)
        
        # Current price and 24h high/low
        current_price = float(df['close'].iloc[-1])
        high_24h = float(df['high'].tail(24).max())
        low_24h = float(df['low'].tail(24).min())
        
        # ATR hesaplama
        atr_values = ta.atr(df['high'], df['low'], df['close']).dropna()
        current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else 0.0
        
        # JSON serialization için güvenli değerler
        def safe_float(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        def safe_list(values):
            if values is None:
                return []
            return [safe_float(v) for v in values if not (np.isnan(v) or np.isinf(v))]
        
        def safe_dict(data_dict):
            if data_dict is None:
                return {}
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    result[key] = safe_float(value)
                elif isinstance(value, dict):
                    result[key] = safe_dict(value)
                else:
                    result[key] = value
            return result
        
        return {
            "priceData": {
                "currentPrice": safe_float(current_price),
                "high24h": safe_float(high_24h),
                "low24h": safe_float(low_24h),
                "atr": safe_float(current_atr)
            },
            "pivotPoints": {
                "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
            },
            "technicalIndicators": {
                "RSI": safe_list(rsi_values.tail(3).tolist() if not rsi_values.empty else []),
                "MACD": safe_dict(macd_values.iloc[-1].to_dict() if not macd_values.empty else {}),
                "ADX": safe_dict(ta.adx(df['high'], df['low'], df['close']).iloc[-1].to_dict() if not ta.adx(df['high'], df['low'], df['close']).empty else {}),
                "BollingerBands": safe_dict(ta.bbands(df['close']).iloc[-1].to_dict() if not ta.bbands(df['close']).empty else {}),
                "ATR": safe_list(atr_values.tail(3).tolist() if not atr_values.empty else []),
                "Volume": safe_list(df['volume'].tail(10).tolist()),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
                }
            },
            "sentiment": {
                "fearGreedIndex": fear_greed_index,
                "socialSentiment": safe_float(social_sentiment),
                "newsSentiment": safe_float(news_sentiment),
                "institutionalFlow": safe_float(np.random.uniform(10000000, 50000000))
            },
            "correlation": {
                "btcCorrelation": safe_float(btc_correlation),
                "ethCorrelation": safe_float(eth_correlation),
                "marketCapRank": 1
            },
            "onChain": {
                "activeAddresses": active_addresses,
                "transactionCount": transaction_count,
                "networkHashRate": safe_float(np.random.uniform(300000000, 600000000)),
                "stakingRatio": safe_float(np.random.uniform(0.1, 0.3)),
                "whaleTransactions": whale_transactions
            },
            "multiTimeframe": {
                "1h": {"trend": "bullish" if df['close'].iloc[-1] > df['close'].iloc[-2] else "bearish", "strength": safe_float(np.random.uniform(0.3, 0.9))},
                "4h": {"trend": "bullish" if df_4h['close'].iloc[-1] > df_4h['close'].iloc[-2] else "bearish", "strength": safe_float(np.random.uniform(0.3, 0.9))},
                "1d": {"trend": "bullish" if df_1d['close'].iloc[-1] > df_1d['close'].iloc[-2] else "bearish", "strength": safe_float(np.random.uniform(0.3, 0.9))}
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/swingtrading/{symbol}")
def get_swing_trading_data(symbol: str = "BTCUSDT"):
    """Swing Trading için özel veriler - sadece AI'ya gönderilir"""
    try:
        # OHLCV verisi (1d)
        ohlcv = binance.fetch_ohlcv(symbol, '1d', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Fundamental data (simüle edilmiş)
        market_cap = np.random.uniform(500000000000, 1000000000000)
        circulating_supply = np.random.uniform(18000000, 21000000)
        
        # Macro economic factors (simüle edilmiş)
        fed_rate = np.random.uniform(4.5, 6.0)
        inflation = np.random.uniform(2.5, 4.0)
        dollar_index = np.random.uniform(100, 105)
        
        # Regulatory data (simüle edilmiş)
        regulatory_risk = np.random.uniform(0.1, 0.5)
        
        # Institutional adoption (simüle edilmiş)
        institutional_adoption = np.random.uniform(0.4, 0.8)
        
        # Social sentiment (simüle edilmiş)
        twitter_sentiment = np.random.uniform(-1, 1)
        reddit_sentiment = np.random.uniform(-1, 1)
        
        # Pivot Points hesaplama (destek/direnç için gerekli)
        pivot = (df['high'] + df['low'] + df['close']) / 3
        r1 = (2 * pivot) - df['low']
        s1 = (2 * pivot) - df['high']
        r2 = pivot + (df['high'] - df['low'])
        s2 = pivot - (df['high'] - df['low'])
        
        # Current price and 24h high/low
        current_price = float(df['close'].iloc[-1])
        high_24h = float(df['high'].iloc[-1])
        low_24h = float(df['low'].iloc[-1])
        
        # ATR hesaplama
        atr_values = ta.atr(df['high'], df['low'], df['close']).dropna()
        current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else 0.0
        
        # JSON serialization için güvenli değerler
        def safe_float(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        def safe_list(values):
            if values is None:
                return []
            return [safe_float(v) for v in values if not (np.isnan(v) or np.isinf(v))]
        
        def safe_dict(data_dict):
            if data_dict is None:
                return {}
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    result[key] = safe_float(value)
                elif isinstance(value, dict):
                    result[key] = safe_dict(value)
                else:
                    result[key] = value
            return result
        
        return {
            "priceData": {
                "currentPrice": safe_float(current_price),
                "high24h": safe_float(high_24h),
                "low24h": safe_float(low_24h),
                "atr": safe_float(current_atr)
            },
            "pivotPoints": {
                "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
            },
            "fundamental": {
                "marketCap": safe_float(market_cap),
                "circulatingSupply": safe_float(circulating_supply),
                "maxSupply": 21000000,
                "developerActivity": safe_float(np.random.uniform(0.6, 0.9)),
                "githubCommits": np.random.randint(800, 2000),
                "roadmapProgress": safe_float(np.random.uniform(0.6, 0.9))
            },
            "macro": {
                "fedRate": safe_float(fed_rate),
                "inflation": safe_float(inflation),
                "dollarIndex": safe_float(dollar_index),
                "goldPrice": safe_float(np.random.uniform(1900, 2200)),
                "oilPrice": safe_float(np.random.uniform(70, 85))
            },
            "regulatory": {
                "secStatus": "pending",
                "euRegulation": "compliant",
                "asiaRegulation": "partial",
                "regulatoryRisk": safe_float(regulatory_risk)
            },
            "institutional": {
                "etfHoldings": safe_float(np.random.uniform(500000, 1000000)),
                "institutionalAdoption": safe_float(institutional_adoption),
                "corporateTreasury": safe_float(np.random.uniform(80000, 200000)),
                "adoptionScore": safe_float(np.random.uniform(0.6, 0.9))
            },
            "social": {
                "twitterSentiment": safe_float(twitter_sentiment),
                "redditSentiment": safe_float(reddit_sentiment),
                "telegramActivity": safe_float(np.random.uniform(0.6, 0.9)),
                "googleTrends": np.random.randint(70, 100)
            },
            "technicalIndicators": {
                "RSI": safe_list(ta.rsi(df['close'], length=14).dropna().tail(3).tolist()),
                "MACD": safe_dict(ta.macd(df['close']).iloc[-1].to_dict() if not ta.macd(df['close']).empty else {}),
                "ADX": safe_dict(ta.adx(df['high'], df['low'], df['close']).iloc[-1].to_dict() if not ta.adx(df['high'], df['low'], df['close']).empty else {}),
                "Ichimoku": {}
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/orderbook/{symbol}")
def get_order_book(symbol: str = "BTCUSDT", limit: int = 20):
    """Order book verisi - sadece AI'ya gönderilir"""
    try:
        orderbook = binance.fetch_order_book(symbol, limit=limit)
        return {
            "lastUpdateId": int(datetime.now().timestamp() * 1000),
            "bids": orderbook['bids'][:limit],
            "asks": orderbook['asks'][:limit],
            "spread": orderbook['asks'][0][0] - orderbook['bids'][0][0],
            "bidVolume": sum([bid[1] for bid in orderbook['bids'][:limit]]),
            "askVolume": sum([ask[1] for ask in orderbook['asks'][:limit]]),
            "imbalance": sum([bid[1] for bid in orderbook['bids'][:limit]]) / sum([ask[1] for ask in orderbook['asks'][:limit]]) if sum([ask[1] for ask in orderbook['asks'][:limit]]) > 0 else 1.0
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/sentiment/{symbol}")
def get_market_sentiment(symbol: str = "BTCUSDT"):
    """Market sentiment verisi - sadece AI'ya gönderilir"""
    try:
        # Simüle edilmiş sentiment verileri
        fear_greed_index = np.random.randint(20, 80)
        social_sentiment = np.random.uniform(-1, 1)
        news_sentiment = np.random.uniform(-1, 1)
        
        # JSON serialization için güvenli değerler
        def safe_float(value):
            if np.isnan(value) or np.isinf(value):
                return 0.0
            return float(value)
        
        return {
            "fearGreedIndex": fear_greed_index,
            "socialSentiment": safe_float(social_sentiment),
            "newsSentiment": safe_float(news_sentiment),
            "institutionalFlow": safe_float(np.random.uniform(10000000, 50000000)),
            "whaleActivity": {
                "largeTransactions": np.random.randint(15, 40),
                "totalVolume": safe_float(np.random.uniform(10000000, 20000000)),
                "direction": "buy" if np.random.random() > 0.5 else "sell"
            },
            "socialMetrics": {
                "twitterMentions": np.random.randint(8000, 15000),
                "redditPosts": np.random.randint(500, 1200),
                "telegramMembers": np.random.randint(200000, 300000),
                "googleSearches": np.random.randint(60000, 100000)
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/indicators_coingecko/{coin_id}")
def get_all_indicators_coingecko(coin_id: str = "bitcoin", days: int = 30):
    try:
        df = get_coingecko_ohlc(coin_id, days)
        result = {}
        result["RSI"] = ta.rsi(df["close"], length=14).dropna().tolist()
        result["MACD"] = ta.macd(df["close"]).iloc[-1].to_dict() if not ta.macd(df["close"]).empty else {}
        result["EMA_12"] = ta.ema(df["close"], length=12).dropna().tolist()
        result["EMA_26"] = ta.ema(df["close"], length=26).dropna().tolist()
        result["SMA_20"] = ta.sma(df["close"], length=20).dropna().tolist()
        result["ADX"] = ta.adx(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.adx(df["high"], df["low"], df["close"]).empty else {}
        result["BollingerBands"] = ta.bbands(df["close"]).iloc[-1].to_dict() if not ta.bbands(df["close"]).empty else {}
        result["StochRSI"] = ta.stochrsi(df["close"]).iloc[-1].to_dict() if not ta.stochrsi(df["close"]).empty else {}
        result["CCI"] = ta.cci(df["high"], df["low"], df["close"]).dropna().tolist()
        result["ATR"] = ta.atr(df["high"], df["low"], df["close"]).dropna().tolist()
        # Klasik pivot noktası hesaplama (manuel)
        pivot = (df["high"] + df["low"] + df["close"]) / 3
        r1 = (2 * pivot) - df["low"]
        s1 = (2 * pivot) - df["high"]
        r2 = pivot + (df["high"] - df["low"])
        s2 = pivot - (df["high"] - df["low"])
        result["PivotPoints"] = {
            "pivot": pivot.iloc[-1] if not pivot.empty else None,
            "resistance1": r1.iloc[-1] if not r1.empty else None,
            "support1": s1.iloc[-1] if not s1.empty else None,
            "resistance2": r2.iloc[-1] if not r2.empty else None,
            "support2": s2.iloc[-1] if not s2.empty else None
        }
        # Ichimoku hesaplama (tuple unpack ile, güvenli)
        try:
            ichimoku = ta.ichimoku(df["high"], df["low"], df["close"])
            if isinstance(ichimoku, tuple):
                conversion, base, span_a, span_b = ichimoku
                if all([hasattr(x, 'iloc') and not x.empty for x in [conversion, base, span_a, span_b]]):
                    result["Ichimoku"] = {
                        "conversion": conversion.iloc[-1],
                        "base": base.iloc[-1],
                        "span_a": span_a.iloc[-1],
                        "span_b": span_b.iloc[-1]
                    }
                else:
                    result["Ichimoku"] = {}
            else:
                result["Ichimoku"] = {}
        except Exception:
            result["Ichimoku"] = {}
        result["ParabolicSAR"] = ta.psar(df["high"], df["low"]).iloc[-1].to_dict() if not ta.psar(df["high"], df["low"]).empty else {}
        result["WilliamsR"] = ta.willr(df["high"], df["low"], df["close"]).dropna().tolist()
        result["Donchian"] = ta.donchian(df["high"], df["low"]).iloc[-1].to_dict() if not ta.donchian(df["high"], df["low"]).empty else {}
        result["Keltner"] = ta.kc(df["high"], df["low"], df["close"]).iloc[-1].to_dict() if not ta.kc(df["high"], df["low"], df["close"]).empty else {}
        return clean_json(result)
    except Exception as e:
        return {"error": str(e)}

@app.get("/coin_info/{symbol}")
def get_coin_info(symbol: str):
    # Tüm coin listesini çek
    coins = requests.get("https://api.coingecko.com/api/v3/coins/list").json()
    # Sembol eşleşmesi bul
    coin = next((c for c in coins if c["symbol"].lower() == symbol.lower()), None)
    if not coin:
        return {"error": "Coin bulunamadı"}
    # Detaylı bilgi çek
    details = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin['id']}").json()
    return {
        "id": coin["id"],
        "symbol": coin["symbol"],
        "name": coin["name"],
        "image": details.get("image", {}).get("large"),
        "thumb": details.get("image", {}).get("thumb"),
        "small": details.get("image", {}).get("small"),
    }

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

# =====================
# NEWS ENDPOINT (NewsAPI.org)
# =====================

@app.get("/news")
def get_news(symbol: str = "BTC", max_age_days: int = 7):
    print(f"🔍 [NEWS] Request: symbol={symbol}, max_age_days={max_age_days}")
    try:
        # NewsAPI.org
        newsapi_token = "209b2715e9544c65b2b9dc294fd225e0"
        base = "https://newsapi.org/v2/everything"
        
        # Coin-specific haber ara
        query = {
            "q": f"{symbol} cryptocurrency",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
            "apiKey": newsapi_token
        }
        
        print(f"🔍 [NEWS] Searching for: {symbol}")
        r = requests.get(base, params=query, timeout=10)
        print(f"🔍 [NEWS] Status: {r.status_code}")
        
        data = r.json()
        items = data.get("articles", [])
        print(f"🔍 [NEWS] Found {len(items)} items for {symbol}")

        # Eğer coin-specific haber yoksa, genel haber ara
        if len(items) == 0:
            print(f"🔍 [NEWS] No news for {symbol}, trying general crypto news...")
            general_query = {
                "q": "bitcoin ethereum cryptocurrency",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "apiKey": newsapi_token
            }
            
            r_general = requests.get(base, params=general_query, timeout=10)
            data_general = r_general.json()
            items = data_general.get("articles", [])
            print(f"🔍 [NEWS] Found {len(items)} general items")

        # Haberleri formatla
        simplified = []
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        
        for article in items:
            published_at = article.get("publishedAt")
            include = True
            
            # Tarih kontrolü
            if published_at:
                try:
                    ts = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                    ts_naive = ts.replace(tzinfo=None)
                    include = ts_naive >= cutoff
                except:
                    include = True
            
            if not include:
                continue
                
            simplified.append({
                "id": article.get("url", "").split("/")[-1] if article.get("url") else "",
                "title": article.get("title"),
                "url": article.get("url"),
                "source": article.get("source", {}).get("name") if isinstance(article.get("source"), dict) else None,
                "description": article.get("description"),
                "created_at": article.get("publishedAt"),
                "kind": "news",
                "votes": {},
                "currencies": []
            })
            
            if len(simplified) >= 10:
                break

        print(f"🔍 [NEWS] Returning {len(simplified)} items")
        return JSONResponse(content={"items": simplified})
        
    except Exception as e:
        print(f"❌ [NEWS] Error: {e}")
        return JSONResponse(content={"items": [], "error": str(e)}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
