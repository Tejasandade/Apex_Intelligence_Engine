import pandas as pd
import numpy as np
import json
import aiohttp
from loguru import logger
from src.data.db import DatabaseManager

# --- Technical Indicator Functions ---
def compute_rsi(series: pd.Series, length: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0

def compute_ema(series: pd.Series, span: int) -> float:
    ema = series.ewm(span=span, adjust=False).mean()
    return float(ema.iloc[-1]) if not ema.empty else float(series.iloc[-1])

def compute_macd(series: pd.Series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return (
        float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0,
        float(macd_signal.iloc[-1]) if not pd.isna(macd_signal.iloc[-1]) else 0.0,
        float(macd_hist.iloc[-1]) if not pd.isna(macd_hist.iloc[-1]) else 0.0,
    )

class DataFetcher:
    """
    Fetches raw data from TimescaleDB, Redis, and Binance REST API,
    and builds normalized feature datasets for the ML models.
    """
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def fetch_recent_klines(self, symbol: str, interval: str = "1m", limit: int = 60) -> pd.DataFrame:
        """Fetches recent klines from Binance REST API for live technical indicators."""
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data, columns=[
                            "open_time", "open", "high", "low", "close", "volume",
                            "close_time", "quote_vol", "trades", "taker_buy_base",
                            "taker_buy_quote", "ignore"
                        ])
                        for col in ["open", "high", "low", "close", "volume"]:
                            df[col] = pd.to_numeric(df[col])
                        logger.debug(f"Fetched {len(df)} live klines for {symbol}")
                        return df
                    else:
                        logger.warning(f"Binance klines API returned status {response.status}")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch live klines: {e}")
            return pd.DataFrame()

    async def fetch_historical_liquidations(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Fetches recent liquidations from TimescaleDB."""
        if not self.db_manager.pg_pool:
            logger.error("Postgres pool not initialized.")
            return pd.DataFrame()
            
        query = """
            SELECT time, symbol, side, price, quantity
            FROM liquidations
            WHERE symbol = $1
            ORDER BY time DESC
            LIMIT $2
        """
        async with self.db_manager.pg_pool.acquire() as conn:
            rows = await conn.fetch(query, symbol, limit)
            
        if not rows:
            return pd.DataFrame(columns=['time', 'symbol', 'side', 'price', 'quantity'])
            
        df = pd.DataFrame(rows, columns=['time', 'symbol', 'side', 'price', 'quantity'])
        # Sort chronologically (oldest to newest)
        df = df.sort_values('time').reset_index(drop=True)
        return df

    async def fetch_current_lob(self, symbol: str) -> dict:
        """Fetches the latest order book state from Redis."""
        if not self.db_manager.redis_pool:
            logger.error("Redis pool not initialized.")
            return {}
            
        key = f"book:{symbol}"
        payload = await self.db_manager.redis_pool.get(key)
        if not payload:
            return {}
            
        return json.loads(payload)

    async def build_dataset(self, symbol: str) -> pd.DataFrame:
        """
        Builds a normalized feature dataset combining live klines (for indicators),
        LOB state, liquidations, and Macro Sentiment Score into a single 
        21-feature vector matching the trained XGBoost model.
        """
        # Fetch all data sources in parallel
        klines_df = await self.fetch_recent_klines(symbol)
        liq_df = await self.fetch_historical_liquidations(symbol)
        lob_data = await self.fetch_current_lob(symbol)
        macro_sentiment = await self.db_manager.fetch_latest_sentiment()

        # --- Compute LIVE Technical Indicators from Binance klines ---
        if not klines_df.empty and len(klines_df) >= 30:
            close = klines_df['close']
            rsi = compute_rsi(close, length=14)
            ema_14 = compute_ema(close, span=14)
            ema_50 = compute_ema(close, span=50)
            macd, macd_signal, macd_hist = compute_macd(close)
            
            # Use the latest candle for OHLCV
            latest = klines_df.iloc[-1]
            open_price = float(latest['open'])
            high_price = float(latest['high'])
            low_price = float(latest['low'])
            close_price = float(latest['close'])
            volume = float(latest['volume'])
            
            # CVD from recent candles
            direction = (klines_df['close'] - klines_df['open']).apply(lambda x: 1 if x >= 0 else -1)
            cvd = float((klines_df['volume'] * direction).sum())
            
            logger.debug(f"Live indicators: RSI={rsi:.1f} | EMA14={ema_14:.1f} | MACD={macd:.4f} | Close={close_price:.1f}")
        else:
            logger.warning("Insufficient kline data. Using neutral defaults.")
            rsi, ema_14, ema_50 = 50.0, 0.0, 0.0
            macd, macd_signal, macd_hist = 0.0, 0.0, 0.0
            open_price = high_price = low_price = close_price = 0.0
            volume, cvd = 0.0, 0.0

        # --- Extract LOB features ---
        best_bid = best_bid_qty = best_ask = best_ask_qty = 0.0
        if lob_data:
            bids = lob_data.get('bids', [])
            asks = lob_data.get('asks', [])
            if bids:
                best_bid = float(bids[0][0])
                best_bid_qty = float(bids[0][1])
            if asks:
                best_ask = float(asks[0][0])
                best_ask_qty = float(asks[0][1])
                
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else (high_price - low_price) * 0.1

        # --- Extract Liquidation features ---
        long_liq_vol = short_liq_vol = 0.0
        if not liq_df.empty:
            long_liq_vol = liq_df[liq_df['side'] == 'SELL']['quantity'].sum()
            short_liq_vol = liq_df[liq_df['side'] == 'BUY']['quantity'].sum()
        liq_imbalance = long_liq_vol - short_liq_vol

        # --- Build the EXACT 21-feature vector matching training order ---
        feature_vector = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.utcnow(),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'RSI': rsi,
            'EMA_14': ema_14,
            'EMA_50': ema_50,
            'MACD': macd,
            'MACD_signal': macd_signal,
            'MACD_hist': macd_hist,
            'spread': spread,
            'CVD': cvd,
            'best_bid': best_bid,
            'best_bid_qty': best_bid_qty,
            'best_ask': best_ask,
            'best_ask_qty': best_ask_qty,
            'recent_long_liq_vol': long_liq_vol,
            'recent_short_liq_vol': short_liq_vol,
            'liq_imbalance': liq_imbalance,
            'macro_sentiment_score': macro_sentiment,
        }
        
        return pd.DataFrame([feature_vector])
