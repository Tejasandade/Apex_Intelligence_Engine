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


# ── Anchored VWAP (AVWAP) ───────────────────────────────────────────────────
def compute_avwap(df: pd.DataFrame, anchor_signal: pd.Series) -> pd.Series:
    """
    Compute Anchored VWAP.
    Resets the VWAP calculation whenever anchor_signal > 0.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tv = tp * df["volume"]
    v = df["volume"]
    
    group_key = (anchor_signal > 0).cumsum()
    
    cum_tv = tv.groupby(group_key).cumsum()
    cum_v = v.groupby(group_key).cumsum()
    
    avwap = cum_tv / cum_v.replace(0, np.nan)
    return avwap.fillna(df['close'])


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

    # --- Phase 3 Advanced Features ---
    # 5. OBV Slope (Momentum of OBV over 10 periods)
    out["OBV_Slope"] = out["OBV"] - out["OBV"].shift(10).fillna(out["OBV"])
    
    # 6. Volume Profile POC Distance
    # Approximate POC as the close price of the candle with the highest volume in a 50-period window
    highest_vol_idx = out["volume"].rolling(window=50, min_periods=1).apply(np.argmax, raw=True)
    # Using numpy to quickly extract the prices
    close_prices = out["close"].values
    poc_prices = close_prices[highest_vol_idx.fillna(0).astype(int)]
    # This gives us index offset within the window, not absolute index, so we need a slightly different approach
    # Let's use pandas rolling with custom function or just simple logic:
    # Actually, the apply(np.argmax) returns the index relative to the window (0 to 49).
    # Absolute index = current_idx - window + 1 + argmax
    # A cleaner vectorised way:
    # Just compute price * volume, then rolling sum? No, POC is single price with max vol.
    # Let's just create a rolling max of volume, and take the close where volume == max_vol.
    roll_max_vol = out["volume"].rolling(window=50, min_periods=1).max()
    poc_series = out["close"].where(out["volume"] >= roll_max_vol).ffill().fillna(out["close"])
    out["Volume_Profile_POC_Dist"] = (out["close"] - poc_series) / poc_series.replace(0, np.nan)
    out["Volume_Profile_POC_Dist"] = out["Volume_Profile_POC_Dist"].fillna(0.0)

    # 7. Anchored VWAP Distance (Anchored to Major Structure Breaks)
    if "structure_break_signal" in out.columns:
        anchor = out["structure_break_signal"].abs()
    else:
        anchor = pd.Series(0, index=out.index)
        anchor.iloc[0] = 1
        
    out["AVWAP"] = compute_avwap(out, anchor)
    out["AVWAP_distance"] = ((out["close"] - out["AVWAP"]) / out["AVWAP"].replace(0, np.nan)).fillna(0.0)

    return out
