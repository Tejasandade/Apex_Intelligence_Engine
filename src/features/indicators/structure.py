"""
Apex Intelligence Engine V5 — Smart Money Concepts (SMC) Indicators
====================================================================
Institutional order flow structure detection:
- Fair Value Gaps (FVG)
- Break of Structure (BOS) / Change of Character (CHoCH)
- Liquidity Sweeps
- Structural Confluence

All functions are vectorised for batch computation in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Fair Value Gap (FVG) ────────────────────────────────────────────────────
def compute_fvg(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Detect Fair Value Gaps — institutional imbalances in price delivery.

    A bullish FVG occurs when candle[i].low > candle[i-2].high (gap up).
    A bearish FVG occurs when candle[i-2].low > candle[i].high (gap down).

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.

    Returns:
        Tuple of (fvg_signal, fvg_gap_pct):
        - fvg_signal: +1.0 bullish, -1.0 bearish, 0.0 none
        - fvg_gap_pct: gap size as percentage of close price
    """
    candle_1_high = df["high"].shift(2)
    candle_1_low = df["low"].shift(2)
    candle_3_low = df["low"]
    candle_3_high = df["high"]
    close = df["close"].replace(0, np.nan)

    bull_gap = candle_3_low - candle_1_high
    bear_gap = candle_1_low - candle_3_high

    signal = pd.Series(0.0, index=df.index)
    pct = pd.Series(0.0, index=df.index)

    bull_mask = bull_gap > 0
    bear_mask = (bear_gap > 0) & ~bull_mask

    signal[bull_mask] = 1.0
    signal[bear_mask] = -1.0
    pct[bull_mask] = (bull_gap[bull_mask] / close[bull_mask]).clip(lower=0)
    pct[bear_mask] = (bear_gap[bear_mask] / close[bear_mask]).clip(lower=0)

    return signal, pct


# ── Break of Structure (BOS) / Change of Character (CHoCH) ─────────────────
def compute_bos(
    df: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect Break of Structure — when price breaks a significant swing high/low.

    BOS in trend direction = trend continuation signal.
    BOS against trend (CHoCH) = potential reversal signal.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback window for swing high/low detection.

    Returns:
        Tuple of (bos_signal, bos_strength):
        - bos_signal: +1.0 bullish break, -1.0 bearish break, 0.0 none
        - bos_strength: magnitude of the break relative to the pivot level
    """
    pivot_high = df["high"].shift(1).rolling(window).max()
    pivot_low = df["low"].shift(1).rolling(window).min()
    close = df["close"]

    signal = pd.Series(0.0, index=df.index)
    strength = pd.Series(0.0, index=df.index)

    bull_mask = close > pivot_high
    bear_mask = close < pivot_low

    signal[bull_mask] = 1.0
    signal[bear_mask] = -1.0

    # Strength = how far price broke past the pivot (as ratio of pivot)
    strength[bull_mask] = (
        (close - pivot_high) / pivot_high.replace(0, np.nan)
    ).clip(lower=0)[bull_mask]
    strength[bear_mask] = (
        (pivot_low - close) / pivot_low.replace(0, np.nan)
    ).clip(lower=0)[bear_mask]

    return signal, strength


# ── Liquidity Sweep ─────────────────────────────────────────────────────────
def compute_liquidity_sweep(
    df: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Detect Liquidity Sweeps — when price wicks past a swing level
    but closes back inside (stop hunt / liquidity grab).

    A bullish sweep: price dips below swing low but closes above it.
    A bearish sweep: price spikes above swing high but closes below it.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        window: Lookback window for swing detection.

    Returns:
        Tuple of (sweep_signal, reclaim_strength):
        - sweep_signal: +1.0 bullish sweep, -1.0 bearish sweep, 0.0 none
        - reclaim_strength: how much of the candle range was reclaimed (0-1)
    """
    pivot_high = df["high"].shift(1).rolling(window).max()
    pivot_low = df["low"].shift(1).rolling(window).min()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    candle_range = (high - low).replace(0, np.nan)

    signal = pd.Series(0.0, index=df.index)
    reclaim = pd.Series(0.0, index=df.index)

    # Bullish sweep: wick below swing low, close back above
    bull_mask = (low < pivot_low) & (close > pivot_low)
    # Bearish sweep: wick above swing high, close back below
    bear_mask = (high > pivot_high) & (close < pivot_high)

    signal[bull_mask] = 1.0
    signal[bear_mask] = -1.0

    reclaim[bull_mask] = ((close - low) / candle_range).clip(0, 1)[bull_mask]
    reclaim[bear_mask] = ((high - close) / candle_range).clip(0, 1)[bear_mask]

    return signal, reclaim


# ── Structural Confluence ───────────────────────────────────────────────────
def compute_structural_confluence(
    fvg_signal: pd.Series,
    bos_signal: pd.Series,
) -> pd.Series:
    """
    Compute structural confluence — when FVG and BOS agree on direction.
    This is a high-conviction signal: institutional gaps + structural breaks.

    Args:
        fvg_signal: FVG signal series.
        bos_signal: BOS signal series.

    Returns:
        Series: +1.0 (bullish confluence), -1.0 (bearish confluence), 0.0 (none)
    """
    return pd.Series(
        np.where(
            (fvg_signal == 1.0) & (bos_signal == 1.0),
            1.0,
            np.where(
                (fvg_signal == -1.0) & (bos_signal == -1.0),
                -1.0,
                0.0,
            ),
        ),
        index=fvg_signal.index,
    )


# ── Utility: Build All Structure Features ───────────────────────────────────
def build_structure_features(
    df: pd.DataFrame,
    bos_window: int = 20,
    sweep_window: int = 20,
) -> pd.DataFrame:
    """
    Compute all SMC structure features and add them as columns.

    Args:
        df: DataFrame with OHLCV columns.
        bos_window: Lookback for BOS detection.
        sweep_window: Lookback for sweep detection.

    Returns:
        DataFrame with structure feature columns added.
    """
    out = df.copy()

    fvg_sig, fvg_pct = compute_fvg(out)
    bos_sig, bos_str = compute_bos(out, bos_window)
    sweep_sig, sweep_rcl = compute_liquidity_sweep(out, sweep_window)

    out["fvg_signal"] = fvg_sig
    out["fvg_gap_pct"] = fvg_pct
    out["structure_break_signal"] = bos_sig
    out["structure_break_strength"] = bos_str
    out["liquidity_sweep_signal"] = sweep_sig
    out["liquidity_reclaim_strength"] = sweep_rcl
    out["structural_confluence"] = compute_structural_confluence(fvg_sig, bos_sig)

    return out
