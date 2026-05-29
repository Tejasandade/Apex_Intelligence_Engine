"""
Apex Intelligence Engine V5 — Volume & Order Flow Indicators
=============================================================
Volume-based signals that reveal institutional participation:
- CVD (Cumulative Volume Delta)
- OBV (On-Balance Volume)
- Order Flow Imbalance
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── CVD (Cumulative Volume Delta) ───────────────────────────────────────────
def compute_cvd(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute Cumulative Volume Delta — net buying vs selling pressure.
    Positive CVD = buyers dominating. Negative CVD = sellers dominating.
    CVD divergence from price = hidden accumulation/distribution.

    Uses close vs open to infer direction (tick rule approximation).
    Uses a rolling sum to maintain stationarity.

    Args:
        df: DataFrame with 'open', 'close', 'volume' columns.
        window: Rolling window period.

    Returns:
        Series of rolling cumulative volume delta values.
    """
    direction = np.where(df["close"] >= df["open"], 1, -1)
    delta = df["volume"] * direction
    return delta.rolling(window=window, min_periods=1).sum()


# ── OBV (On-Balance Volume) ────────────────────────────────────────────────
def compute_obv(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute On-Balance Volume — rolling sum of volume weighted by price direction.
    OBV trending up while price flat = accumulation.
    OBV trending down while price flat = distribution.

    Args:
        df: DataFrame with 'close', 'volume' columns.
        window: Rolling window period.

    Returns:
        Series of rolling OBV values.
    """
    direction = np.where(
        df["close"] > df["close"].shift(1),
        1,
        np.where(df["close"] < df["close"].shift(1), -1, 0),
    )
    delta = df["volume"] * direction
    return delta.rolling(window=window, min_periods=1).sum()


# ── Order Flow Imbalance ───────────────────────────────────────────────────
def compute_order_flow_imbalance(
    df: pd.DataFrame,
    window: int = 14,
) -> pd.Series:
    """
    Compute order flow imbalance from OHLCV data.
    Approximation: uses candle body position within range to estimate
    buy/sell pressure. Close near high = buying. Close near low = selling.

    This is a proxy when real order book data isn't available (training).
    Live inference should use actual bid/ask volume from LOB.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
        window: Rolling window for smoothing.

    Returns:
        Series of imbalance values (-1 to +1). Positive = buy pressure.
    """
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)

    # Position of close within the candle range (0 = at low, 1 = at high)
    close_position = (df["close"] - df["low"]) / candle_range

    # Volume-weighted position
    buy_volume = df["volume"] * close_position.fillna(0.5)
    sell_volume = df["volume"] * (1.0 - close_position.fillna(0.5))

    # Rolling imbalance ratio
    rolling_buy = buy_volume.rolling(window=window, min_periods=1).sum()
    rolling_sell = sell_volume.rolling(window=window, min_periods=1).sum()
    total = (rolling_buy + rolling_sell).replace(0, np.nan)

    imbalance = (rolling_buy - rolling_sell) / total
    return imbalance.fillna(0.0)


# ── Utility: Build All Volume Features ──────────────────────────────────────
def build_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all volume/order flow features and add them as columns.

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with volume feature columns added.
    """
    out = df.copy()
    out["CVD"] = compute_cvd(out)
    out["OBV"] = compute_obv(out)
    out["order_flow_imbalance"] = compute_order_flow_imbalance(out, window=14)
    return out
