import requests
import pandas as pd
import pandas_ta as ta
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import os
import math
import ccxt
import numpy as np
from datetime import datetime, timedelta

app = FastAPI()

# Binance API bağlantısı
binance = ccxt.binance({
    'sandbox': False
})

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
        
        # Order book verisi
        try:
            orderbook = binance.fetch_order_book(symbol, limit=20)
            print(f"[DEBUG] Order book alındı: {len(orderbook['bids'])} bids, {len(orderbook['asks'])} asks")
        except Exception as e:
            print(f"[ERROR] Order book hatası: {e}")
            orderbook = {"bids": [], "asks": []}
        
        # Funding rate (futures için)
        try:
            funding_rate = binance.fetch_funding_rate(symbol)
            print(f"[DEBUG] Funding rate alındı: {funding_rate}")
        except Exception as e:
            print(f"[ERROR] Funding rate hatası: {e}")
            funding_rate = {"fundingRate": 0.0001, "nextFundingTime": 0, "openInterest": 0}
        
        # Teknik indikatörler (5m)
        try:
            ohlcv = binance.fetch_ohlcv(symbol, '5m', limit=100)
            print(f"[DEBUG] OHLCV alındı: {len(ohlcv)} veri noktası")
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(f"[DEBUG] DataFrame oluşturuldu: {df.shape}")
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
        
        # Real-time price data
        try:
            ticker = binance.fetch_ticker(symbol)
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
            }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# =====================
# NEWS ENDPOINT (CryptoPanic)
# =====================

NEWS_CACHE = {"data": None, "ts": None, "params": None}

@app.get("/news")
def get_news(currencies: str = "BTC,ETH", filter: str = "hot", limit: int = 10, region: str = "", max_age_days: int = 7):
    try:
        # Simple 120s cache per param-set
        now = int(datetime.utcnow().timestamp())
        params_signature = f"{currencies}|{filter}|{limit}|{region}"
        if (
            NEWS_CACHE["data"] is not None
            and NEWS_CACHE["params"] == params_signature
            and NEWS_CACHE["ts"] is not None
            and now - NEWS_CACHE["ts"] < 120
        ):
            return JSONResponse(content=NEWS_CACHE["data"]) 

        token = os.getenv("CRYPTOPANIC_TOKEN", "")
        base = "https://cryptopanic.com/api/developer/v2/posts/"
        query = {
            "filter": filter,
            "kind": "news",
            "currencies": currencies,
            "public": "true",
        }
        if region:
            query["regions"] = region
        if token:
            query["auth_token"] = token

        # Temporary mock data for testing
        items = [
            {
                "id": 1,
                "title": "Bitcoin ETF Onayı Piyasayı Hareketlendirdi",
                "url": "https://example.com/btc-news-1",
                "source": {"title": "CoinDesk"},
                "created_at": "2025-01-05T10:00:00Z",
                "kind": "news",
                "votes": {"positive": 25, "negative": 5},
                "currencies": [{"code": "BTC"}],
                "metadata": {"description": "Bitcoin ETF onayı sonrası kripto piyasalarında yükseliş trendi devam ediyor. Yatırımcılar pozitif tepki veriyor."}
            },
            {
                "id": 2,
                "title": "Cardano (ADA) Yeni Güncelleme Duyurusu",
                "url": "https://example.com/ada-news-1",
                "source": {"title": "CryptoNews"},
                "created_at": "2025-01-05T09:30:00Z",
                "kind": "news",
                "votes": {"positive": 15, "negative": 2},
                "currencies": [{"code": "ADA"}],
                "metadata": {"description": "Cardano ekibi yeni güncelleme hakkında önemli duyurular yaptı. Bu güncelleme ağ performansını artıracak."}
            }
        ]
        
        # Try real API
        try:
            r = requests.get(base, params=query, timeout=5)
            if r.status_code == 200:
                data = r.json()
                real_items = data.get("results", [])
                if real_items:
                    items = real_items
                    print(f"Got {len(real_items)} real news items")
        except Exception as e:
            print(f"CryptoPanic API error: {e}, using mock data")

        simplified = []
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        for it in items:
            created_at = it.get("created_at")
            include = True
            try:
                if created_at:
                    # Handle Z suffix
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00").replace("+00:00", "+00:00"))
                    # Convert to naive UTC for comparison
                    ts_naive = ts.replace(tzinfo=None)
                    include = ts_naive >= cutoff
            except Exception:
                include = True
            if not include:
                continue
            simplified.append({
                "id": it.get("id"),
                "title": it.get("title"),
                "url": it.get("url") or (it.get("source", {}) or {}).get("url"),
                "source": ((it.get("source", {}) or {}).get("title")) if isinstance(it.get("source"), dict) else None,
                "description": it.get("metadata", {}).get("description") if isinstance(it.get("metadata"), dict) else None,
                "created_at": it.get("created_at"),
                "kind": it.get("kind"),
                "votes": (it.get("votes") or {}),
                "currencies": [c.get("code") for c in (it.get("currencies") or []) if isinstance(c, dict)]
            })
            if len(simplified) >= limit:
                break

        response = {"items": simplified}
        NEWS_CACHE["data"] = response
        NEWS_CACHE["ts"] = now
        NEWS_CACHE["params"] = params_signature
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"items": [], "error": str(e)}, status_code=200)
