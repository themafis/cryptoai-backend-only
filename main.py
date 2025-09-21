import sys
from pathlib import Path

# Ensure vendored third_party packages (e.g., pandas_ta) are importable BEFORE third-party imports
_THIRD_PARTY = Path(__file__).resolve().parent / "third_party"
if str(_THIRD_PARTY) not in sys.path:
    sys.path.append(str(_THIRD_PARTY))

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
import time
from threading import Lock

app = FastAPI()

# Binance API bağlantısı
binance = ccxt.binance({
    'sandbox': False
})

 

# --- Short TTL cache & rate limit / ban-backoff for Binance calls ---
_cache: dict = {}
_cache_lock = Lock()
_last_call_ts = 0.0
_min_interval_sec = 0.12  # ~8-9 calls/sec per IP
_ban_until_ms = 0  # if banned, store expiry (epoch ms)

def _get_cached(key: str, ttl: float):
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and (now - entry['t']) < ttl:
            return entry['v']
    return None

def _set_cached(key: str, value):
    with _cache_lock:
        _cache[key] = {'t': time.time(), 'v': value}

def binance_call(key: str, ttl: float, fn):
    """Unified wrapper: short TTL cache + simple rate limit + ban detection.
    Returns cached value if available during ban window.
    """
    global _last_call_ts, _ban_until_ms
    # If currently banned, return last cache (may be None)
    now_ms = int(time.time() * 1000)
    if now_ms < _ban_until_ms:
        cached = _get_cached(key, ttl)
        if cached is not None:
            return cached
        # no cache: raise a soft error handled by caller
        raise RuntimeError("Binance rate limit ban active; no cached data")

    # Serve fresh or cached
    cached = _get_cached(key, ttl)
    if cached is not None:
        return cached

    # Simple global pacing
    dt = time.time() - _last_call_ts
    if dt < _min_interval_sec:
        time.sleep(_min_interval_sec - dt)

    try:
        val = fn()
        _set_cached(key, val)
        return val
    except Exception as e:
        msg = str(e)
        if "Way too much request weight" in msg or "418" in msg or "-1003" in msg:
            # Try to parse banned-until epoch ms
            import re
            m = re.search(r"until\s+(\d+)", msg)
            if m:
                try:
                    _ban_until_ms = int(m.group(1))
                except Exception:
                    pass
            # Fall back to cache if any
            cached = _get_cached(key, ttl)
            if cached is not None:
                return cached
        # Re-raise for caller to handle gracefully
        raise
    finally:
        _last_call_ts = time.time()

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
    MISSING = "-"
    MISSING_OBJ = {"error": "veri çekilemedi"}

    def safe_float(value):
        if np.isnan(value) or np.isinf(value):
            return MISSING
        return float(value)
    
    def safe_list(values):
        if values is None:
            return MISSING
        cleaned = [safe_float(v) for v in values if not (np.isnan(v) or np.isinf(v))]
        return cleaned if cleaned else MISSING
    
    def safe_dict(data_dict):
        if data_dict is None:
            return MISSING_OBJ
        result = {}
        for key, value in data_dict.items():
            if isinstance(value, (int, float)):
                result[key] = safe_float(value)
            elif isinstance(value, dict):
                result[key] = safe_dict(value)
            else:
                result[key] = value
        return result if len(result) else MISSING_OBJ
    
    try:
        print(f"[DEBUG] Mini scalping endpoint çağrıldı: {symbol}")
        
        # Order book (cache 2s)
        try:
            orderbook = binance_call(f"ob:{symbol}:20", 2.0, lambda: binance.fetch_order_book(symbol, limit=20))
            if orderbook is None:
                orderbook = {"bids": [], "asks": []}
            print(f"[DEBUG] Order book alındı (cache): {len(orderbook.get('bids', []))} bids, {len(orderbook.get('asks', []))} asks")
        except Exception as e:
            print(f"[ERROR] Order book hatası: {e}")
            orderbook = {"bids": [], "asks": []}
        
        # Funding rate (cache 60s)
        try:
            funding_rate = binance_call(f"fr:{symbol}", 60.0, lambda: binance.fetch_funding_rate(symbol))
            if funding_rate is None:
                funding_rate = {"fundingRate": 0.0001, "nextFundingTime": 0, "openInterest": 0}
            print(f"[DEBUG] Funding rate alındı (cache): {funding_rate}")
        except Exception as e:
            print(f"[ERROR] Funding rate hatası: {e}")
            funding_rate = {"fundingRate": 0.0001, "nextFundingTime": 0, "openInterest": 0}
        
        # Teknik indikatörler (5m) - cache 8s
        try:
            ohlcv = binance_call(f"ohlcv:{symbol}:5m:100", 8.0, lambda: binance.fetch_ohlcv(symbol, '5m', limit=100))
            if ohlcv is None:
                raise RuntimeError("No cached 5m OHLCV available")
            print(f"[DEBUG] OHLCV(5m) alındı (cache): {len(ohlcv)} veri noktası")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"[DEBUG] DataFrame oluşturuldu: {df.shape}")
        except Exception as e:
            print(f"[ERROR] OHLCV(5m) hatası: {e}")
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
        
        # Real-time price data
        try:
            ticker = binance_call(f"ticker:{symbol}", 5.0, lambda: binance.fetch_ticker(symbol)) or {}
            current_price = safe_float(ticker['last'])
            price_change_24h = safe_float(ticker['change'])
            price_change_percent_24h = safe_float(ticker['percentage'])
            volume_24h = safe_float(ticker['quoteVolume'])
            high_24h = safe_float(ticker['high'])
            low_24h = safe_float(ticker['low'])
        except Exception as e:
            print(f"Ticker hatası: {e}")
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
            ohlcv_1m = binance_call(f"ohlcv:{symbol}:1m:120", 8.0, lambda: binance.fetch_ohlcv(symbol, '1m', limit=120)) or []
            d1 = pd.DataFrame(ohlcv_1m, columns=['timestamp','open','high','low','close','volume'])
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
            # Reuse 5m if possible; else fetch with cache 8s
            ohlcv_5m = ohlcv if len(ohlcv) >= 120 else binance_call(f"ohlcv:{symbol}:5m:120", 8.0, lambda: binance.fetch_ohlcv(symbol, '5m', limit=120)) or []
            d5 = pd.DataFrame(ohlcv_5m, columns=['timestamp','open','high','low','close','volume'])
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
                "VWAP": safe_list((vwap.dropna() if isinstance(vwap, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "PivotPoints": {
                    "pivot": safe_float(pivot.iloc[-1] if not pivot.empty else 0.0),
                    "resistance1": safe_float(r1.iloc[-1] if not r1.empty else 0.0),
                    "support1": safe_float(s1.iloc[-1] if not s1.empty else 0.0),
                    "resistance2": safe_float(r2.iloc[-1] if not r2.empty else 0.0),
                    "support2": safe_float(s2.iloc[-1] if not s2.empty else 0.0)
                },
                "Ichimoku": ichimoku_data,
                "ParabolicSAR": safe_dict(psar_df.iloc[-1].to_dict() if isinstance(psar_df, pd.DataFrame) and not psar_df.empty else None),
                "WilliamsR": safe_list((williams_r.dropna() if isinstance(williams_r, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "Supertrend": safe_dict(supertrend_df.iloc[-1].to_dict() if isinstance(supertrend_df, pd.DataFrame) and not supertrend_df.empty else None),
                "MFI": safe_list((mfi.dropna() if isinstance(mfi, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
                "Donchian": safe_dict(donchian_df.iloc[-1].to_dict() if isinstance(donchian_df, pd.DataFrame) and not donchian_df.empty else None),
                "Keltner": safe_dict(keltner_df.iloc[-1].to_dict() if isinstance(keltner_df, pd.DataFrame) and not keltner_df.empty else None)
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
        # OHLCV verisi (15m)
        ohlcv = binance.fetch_ohlcv(symbol, '15m', limit=100)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
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
        # Bollinger Bands
        try:
            bb_scalp = ta.bbands(df['close'])
        except Exception:
            bb_scalp = pd.DataFrame()
        
        # Current price and 24h high/low for Scalping
        current_price = float(df['close'].iloc[-1])
        high_24h = float(df['high'].tail(24).max() if len(df) >= 24 else df['high'].max())
        low_24h = float(df['low'].tail(24).min() if len(df) >= 24 else df['low'].min())
        
        # ATR hesaplama (güvenli)
        _atr_raw = None
        try:
            _atr_raw = ta.atr(df['high'], df['low'], df['close'])
        except Exception:
            _atr_raw = None
        atr_values = (_atr_raw.dropna() if isinstance(_atr_raw, pd.Series) else pd.Series(dtype='float64'))
        current_atr = float(atr_values.iloc[-1]) if not atr_values.empty else 0.0
        
        # JSON serialization için güvenli değerler (standartlaştırılmış)
        MISSING = "-"
        MISSING_OBJ = {"error": "veri çekilemedi"}

        def safe_float(value):
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    return MISSING
                return float(value)
            return MISSING
        
        def safe_list(values):
            if values is None:
                return MISSING
            cleaned = []
            for v in values:
                if isinstance(v, (int, float)) and not (np.isnan(v) or np.isinf(v)):
                    cleaned.append(float(v))
            return cleaned if cleaned else MISSING
        
        def safe_dict(data_dict):
            if data_dict is None:
                return MISSING_OBJ
            result = {}
            for key, value in data_dict.items():
                if isinstance(value, (int, float)):
                    result[key] = safe_float(value)
                elif isinstance(value, dict):
                    result[key] = safe_dict(value)
                else:
                    result[key] = value
            return result if len(result) else MISSING_OBJ
        
        # TA hesaplamalarını güvenli hazırla (None dönerse boş Seri kullan)
        _rsi_raw = ta.rsi(df['close'], length=14)
        rsi_safe = _rsi_raw if isinstance(_rsi_raw, pd.Series) else pd.Series(dtype='float64')

        _willr_raw = None
        try:
            _willr_raw = ta.willr(df['high'], df['low'], df['close'])
        except Exception:
            _willr_raw = None
        willr_safe = _willr_raw if isinstance(_willr_raw, pd.Series) else pd.Series(dtype='float64')

        _cci_raw = None
        try:
            _cci_raw = ta.cci(df['high'], df['low'], df['close'])
        except Exception:
            _cci_raw = None
        cci_safe = _cci_raw if isinstance(_cci_raw, pd.Series) else pd.Series(dtype='float64')

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
            "vwap": safe_list((vwap.dropna() if isinstance(vwap, pd.Series) else pd.Series(dtype='float64')).tail(3).tolist()),
            "technicalIndicators": {
                "RSI": safe_list(rsi_safe.dropna().tail(3).tolist()),
                "MACD": safe_dict(ta.macd(df['close']).iloc[-1].to_dict() if not ta.macd(df['close']).empty else {}),
                "WilliamsR": safe_list(willr_safe.dropna().tail(3).tolist()),
                "CCI": safe_list(cci_safe.dropna().tail(3).tolist()),
                "ATR": safe_list(atr_values.tail(3).tolist() if not atr_values.empty else []),
                "KeltnerChannels": safe_dict(keltner.iloc[-1].to_dict() if not keltner.empty else None),
                "BollingerBands": safe_dict(bb_scalp.iloc[-1].to_dict() if isinstance(bb_scalp, pd.DataFrame) and not bb_scalp.empty else None)
            },
            "microMetrics": {
                # 1m/5m RSI ve approx CVD + volatility burst
                "RSI_1m": (lambda _d: safe_float((ta.rsi(_d['close'], length=14) if not _d.empty else pd.Series(dtype='float64')).dropna().iloc[-1]) if not (ta.rsi(_d['close'], length=14) is None or _d.empty or ta.rsi(_d['close'], length=14).dropna().empty) else 0.0)(get_binance_ohlcv(symbol, '1m', 120)),
                "RSI_5m": (lambda _d: safe_float((ta.rsi(_d['close'], length=14) if not _d.empty else pd.Series(dtype='float64')).dropna().iloc[-1]) if not (ta.rsi(_d['close'], length=14) is None or _d.empty or ta.rsi(_d['close'], length=14).dropna().empty) else 0.0)(get_binance_ohlcv(symbol, '5m', 120))
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
