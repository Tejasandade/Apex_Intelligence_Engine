import os
import sys
import asyncio
import pandas as pd
import numpy as np
import xgboost as xgb
from loguru import logger

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.db import db_manager

HISTORICAL_CSV = "D:/Apex_Intelligence_Engine/data/historical/cleaned/BTCUSDT_1m_cleaned.csv"
MODEL_PATH = "D:/Apex_Intelligence_Engine/src/models/final_model.json"


# --- Manual Technical Indicator Functions (no pandas-ta dependency) ---

def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Computes the Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Computes Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Computes MACD, MACD Signal, and MACD Histogram."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = compute_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


async def build_training_data():
    logger.info(f"Loading historical dataset: {HISTORICAL_CSV}")
    if not os.path.exists(HISTORICAL_CSV):
        logger.error("Historical CSV not found!")
        return None

    df = pd.read_csv(HISTORICAL_CSV)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of historical data.")

    # --- Feature Engineering: Technical Indicators ---
    logger.info("Calculating Technical Indicators (RSI, EMA, MACD, VWAP, ATR)...")
    df['RSI'] = compute_rsi(df['close'], length=14)
    df['EMA_14'] = compute_ema(df['close'], span=14)
    df['EMA_50'] = compute_ema(df['close'], span=50)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['close'])

    # Level 2 Institutional Features
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['VWAP'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14, min_periods=14).mean()

    # --- Synthetic LOB / CVD features (derived from OHLCV as proxies) ---
    df['spread'] = (df['high'] - df['low']) * 0.1
    direction = (df['close'] - df['open']).apply(lambda x: 1 if x >= 0 else -1)
    df['CVD'] = (df['volume'] * direction).cumsum()

    df['best_bid'] = df['close'] - df['spread'] / 2
    df['best_bid_qty'] = df['volume'] * 0.1
    df['best_ask'] = df['close'] + df['spread'] / 2
    df['best_ask_qty'] = df['volume'] * 0.1

    df['recent_long_liq_vol'] = 0.0
    df['recent_short_liq_vol'] = 0.0
    df['liq_imbalance'] = 0.0

    # --- Macro Sentiment (fetch from DB or default) ---
    logger.info("Connecting to DB to fetch latest sentiment score...")
    await db_manager.connect()
    sentiment = await db_manager.fetch_latest_sentiment()
    df['macro_sentiment_score'] = sentiment if sentiment else 0.5
    await db_manager.disconnect()

    # --- Target Variable: Will price be higher in 15 minutes? ---
    logger.info("Constructing target variable (next 15m price > current)...")
    df['future_close'] = df['close'].shift(-15)
    df['target'] = (df['future_close'] > df['close']).astype(int)

    # Drop rows with NaN (from indicator warm-up and target shift)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"Training-ready dataset: {len(df)} samples, {len(df.columns)} columns.")
    return df


def train_xgboost(df: pd.DataFrame):
    drop_cols = ['time', 'symbol', 'future_close', 'target']
    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df['target']

    logger.info(f"Features: {features}")
    logger.info(f"Training XGBoost on {len(df)} samples with {len(features)} features...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        n_jobs=-1,
    )

    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.success(f"Final XGBoost model saved to {MODEL_PATH}")
    logger.info(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")


async def main():
    df = await build_training_data()
    if df is not None:
        train_xgboost(df)


if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
