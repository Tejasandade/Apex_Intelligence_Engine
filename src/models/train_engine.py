import asyncio
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.db import db_manager
HISTORICAL_CSV = "D:/Apex_Intelligence_Engine/data/historical/cleaned/BTCUSDT_1m_cleaned.csv"
MODEL_PATH = "D:/Apex_Intelligence_Engine/src/models/final_model.json"


def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = compute_ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculating Institutional Structure features (FVG, Sweep, BOS, CHoCH)...")
    bullish_gap = df["low"] - df["high"].shift(2)
    bearish_gap = df["low"].shift(2) - df["high"]
    df["fvg_signal"] = np.where(bullish_gap > 0, 1.0, np.where(bearish_gap > 0, -1.0, 0.0))
    df["fvg_gap_pct"] = np.where(
        df["fvg_signal"] == 1.0,
        bullish_gap / df["close"].clip(lower=1.0),
        np.where(
            df["fvg_signal"] == -1.0,
            bearish_gap / df["close"].clip(lower=1.0),
            0.0,
        ),
    )

    prior_high = df["high"].shift(1).rolling(window=20, min_periods=20).max()
    prior_low = df["low"].shift(1).rolling(window=20, min_periods=20).min()
    atr_reference = df["ATR"].replace(0, np.nan).fillna(df["close"].abs() * 0.001).clip(lower=1.0)

    bullish_sweep = (df["low"] < prior_low) & (df["close"] > prior_low)
    bearish_sweep = (df["high"] > prior_high) & (df["close"] < prior_high)
    df["liquidity_sweep_signal"] = np.where(bullish_sweep, 1.0, np.where(bearish_sweep, -1.0, 0.0))
    df["liquidity_reclaim_strength"] = np.where(
        bullish_sweep,
        (df["close"] - prior_low).clip(lower=0.0) / atr_reference,
        np.where(
            bearish_sweep,
            (prior_high - df["close"]).clip(lower=0.0) / atr_reference,
            0.0,
        ),
    )

    structure_high = df["high"].shift(2).rolling(window=12, min_periods=12).max()
    structure_low = df["low"].shift(2).rolling(window=12, min_periods=12).min()
    trend_bias = df["close"].shift(1) - df["close"].shift(2).rolling(window=12, min_periods=12).mean()
    bullish_break = df["close"] > structure_high
    bearish_break = df["close"] < structure_low

    df["structure_break_signal"] = np.select(
        [
            bullish_break & (trend_bias >= 0),
            bullish_break & (trend_bias < 0),
            bearish_break & (trend_bias <= 0),
            bearish_break & (trend_bias > 0),
        ],
        [2.0, 1.0, -2.0, -1.0],
        default=0.0,
    )
    df["structure_break_strength"] = np.where(
        df["structure_break_signal"] > 0,
        (df["close"] - structure_high).clip(lower=0.0) / atr_reference,
        np.where(
            df["structure_break_signal"] < 0,
            (structure_low - df["close"]).clip(lower=0.0) / atr_reference,
            0.0,
        ),
    )
    df["structural_confluence"] = (
        (df["fvg_signal"] != 0.0) & (df["fvg_signal"] == df["liquidity_sweep_signal"])
    ).astype(float)

    df["fvg_gap_pct"] = df["fvg_gap_pct"].fillna(0.0)
    df["liquidity_reclaim_strength"] = df["liquidity_reclaim_strength"].fillna(0.0)
    df["structure_break_strength"] = df["structure_break_strength"].fillna(0.0)
    return df


async def build_training_data():
    logger.info(f"Loading historical dataset: {HISTORICAL_CSV}")
    if not os.path.exists(HISTORICAL_CSV):
        logger.error("Historical CSV not found!")
        return None

    df = pd.read_csv(HISTORICAL_CSV)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows of historical data.")

    logger.info("Calculating Technical Indicators (RSI, EMA, MACD, VWAP, ATR)...")
    df["RSI"] = compute_rsi(df["close"], length=14)
    df["EMA_14"] = compute_ema(df["close"], span=14)
    df["EMA_50"] = compute_ema(df["close"], span=50)
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = compute_macd(df["close"])

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    df["VWAP"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    df["ATR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14, min_periods=14).mean()

    df["spread"] = (df["high"] - df["low"]) * 0.1
    direction = (df["close"] - df["open"]).apply(lambda value: 1 if value >= 0 else -1)
    df["CVD"] = (df["volume"] * direction).cumsum()

    df["best_bid"] = df["close"] - df["spread"] / 2
    df["best_bid_qty"] = df["volume"] * 0.1
    df["best_ask"] = df["close"] + df["spread"] / 2
    df["best_ask_qty"] = df["volume"] * 0.1

    df["recent_long_liq_vol"] = 0.0
    df["recent_short_liq_vol"] = 0.0
    df["liq_imbalance"] = 0.0
    df = add_structure_features(df)

    logger.info("Connecting to DB to fetch latest sentiment score...")
    sentiment = 0.5
    try:
        await db_manager.connect()
        sentiment = await db_manager.fetch_latest_sentiment()
    except Exception as exc:
        logger.warning(f"Falling back to neutral sentiment for training: {exc}")
    finally:
        await db_manager.disconnect()

    df["macro_sentiment_score"] = sentiment if sentiment else 0.5

    logger.info("Constructing target variable (next 15m price > current)...")
    df["future_close"] = df["close"].shift(-15)
    df["target"] = (df["future_close"] > df["close"]).astype(int)

    df = df.dropna().reset_index(drop=True)
    feature_count = len([column for column in df.columns if column not in ["time", "symbol", "future_close", "target"]])
    logger.info(f"Training-ready dataset: {len(df)} samples with {feature_count} engineered features.")
    return df


def train_xgboost(df: pd.DataFrame):
    drop_cols = ["time", "symbol", "future_close", "target"]
    features = [column for column in df.columns if column not in drop_cols]

    X = df[features]
    y = df["target"]

    logger.info(f"Training XGBoost on {len(df)} samples with {len(features)} features...")
    logger.info(f"Feature set: {features}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.success(f"Final XGBoost model saved to {MODEL_PATH}")
    logger.info(f"Model file size: {os.path.getsize(MODEL_PATH)} bytes")


async def main():
    df = await build_training_data()
    if df is not None:
        train_xgboost(df)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
