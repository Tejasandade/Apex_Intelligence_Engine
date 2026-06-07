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


# ── Pivot Distance (Order Block / Liquidity Pool proxy) ─────────────────────
def compute_pivot_distance(
    df: pd.DataFrame,
    window: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """
    Compute distance to the nearest swing high and swing low.
    This acts as a proxy for distance to nearest Order Block or Liquidity Pool.
    
    Returns:
        Tuple of (dist_to_high_pct, dist_to_low_pct).
        Values are positive percentages.
    """
    pivot_high = df["high"].shift(1).rolling(window).max()
    pivot_low = df["low"].shift(1).rolling(window).min()
    close = df["close"].replace(0, np.nan)
    
    dist_high = ((pivot_high - close) / close).clip(lower=0).fillna(0.0)
    dist_low = ((close - pivot_low) / close).clip(lower=0).fillna(0.0)
    
    return dist_high, dist_low


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


# ── True Liquidity Sweeps / Stop Runs (Advanced SMC) ────────────────────────
def detect_liquidity_pools(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 2
) -> tuple[pd.Series, pd.Series]:
    """
    Detects true structural swing highs and lows (fractals).
    A true swing high is higher than the `left_bars` before it and `right_bars` after it.
    
    Returns:
        Tuple of (swing_highs, swing_lows) where values are the price levels.
        Empty or non-swing bars are NaN.
    """
    high = df['high']
    low = df['low']
    
    # Use rolling window to check if current bar is local max/min
    window = left_bars + right_bars + 1
    
    # We shift the rolling max backwards so the current index aligns with the center
    local_max = high.rolling(window=window, center=False).max().shift(-right_bars)
    local_min = low.rolling(window=window, center=False).min().shift(-right_bars)
    
    is_swing_high = (high == local_max)
    is_swing_low = (low == local_min)
    
    swing_highs = pd.Series(np.nan, index=df.index)
    swing_lows = pd.Series(np.nan, index=df.index)
    
    swing_highs[is_swing_high] = high[is_swing_high]
    swing_lows[is_swing_low] = low[is_swing_low]
    
    # Forward fill to carry the liquidity pool level forward in time
    return swing_highs.ffill(), swing_lows.ffill()

def detect_stop_runs(df: pd.DataFrame, swing_highs: pd.Series, swing_lows: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Detects when price sweeps a mapped liquidity pool (stop run) and rejects.
    
    Bullish Stop Run: Wick goes below swing low, but candle closes above it.
    Bearish Stop Run: Wick goes above swing high, but candle closes below it.
    
    Returns:
        Tuple of (bullish_stop_runs, bearish_stop_runs) boolean series.
    """
    # We must shift the swing pools by 1 so we don't trigger a sweep on the bar that created the pool
    prev_swing_highs = swing_highs.shift(1)
    prev_swing_lows = swing_lows.shift(1)
    
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Bullish stop run: Low pierced previous swing low, but closed above it
    bullish_run = (low < prev_swing_lows) & (close > prev_swing_lows)
    
    # Bearish stop run: High pierced previous swing high, but closed below it
    bearish_run = (high > prev_swing_highs) & (close < prev_swing_highs)
    
    return bullish_run, bearish_run

# ── True Order Block (OB) Detection ───────────────────────────────────────────
def compute_order_blocks(df: pd.DataFrame, bos_signal: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Detects true Order Blocks based on BOS signals.
    Bullish OB: Last down candle before a Bullish BOS.
    Bearish OB: Last up candle before a Bearish BOS.
    
    Returns:
        Tuple of (dist_bull_ob, dist_bear_ob) as percentages.
    """
    is_down = df['close'] < df['open']
    is_up = df['close'] > df['open']
    
    last_down_high = df['high'].where(is_down, np.nan).ffill()
    last_up_low = df['low'].where(is_up, np.nan).ffill()
    
    bull_ob_high = last_down_high.shift(1).where(bos_signal == 1.0, np.nan)
    bear_ob_low = last_up_low.shift(1).where(bos_signal == -1.0, np.nan)
    
    active_bull_ob_high = bull_ob_high.ffill()
    active_bear_ob_low = bear_ob_low.ffill()
    
    dist_bull_ob = ((df['close'] - active_bull_ob_high) / df['close']).fillna(0.0)
    dist_bear_ob = ((active_bear_ob_low - df['close']) / df['close']).fillna(0.0)
    
    return dist_bull_ob, dist_bear_ob

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
    dist_high, dist_low = compute_pivot_distance(out, bos_window)

    out["fvg_signal"] = fvg_sig
    out["fvg_gap_pct"] = fvg_pct
    out["structure_break_signal"] = bos_sig
    out["structure_break_strength"] = bos_str
    out["liquidity_sweep_signal"] = sweep_sig
    out["liquidity_reclaim_strength"] = sweep_rcl
    out["dist_to_pivot_high"] = dist_high
    out["dist_to_pivot_low"] = dist_low
    out["structural_confluence"] = compute_structural_confluence(fvg_sig, bos_sig)

    # Advanced SMC True Liquidity Sweeps
    swing_highs, swing_lows = detect_liquidity_pools(out)
    bull_run, bear_run = detect_stop_runs(out, swing_highs, swing_lows)
    out["true_liquidity_pool_high"] = swing_highs
    out["true_liquidity_pool_low"] = swing_lows
    
    # Advanced SMC Order Blocks
    ob_bull_dist, ob_bear_dist = compute_order_blocks(out, bos_sig)
    out["OB_bull_dist"] = ob_bull_dist
    out["OB_bear_dist"] = ob_bear_dist
    
    out["bullish_stop_run"] = bull_run.astype(float)
    out["bearish_stop_run"] = bear_run.astype(float)

    # --- Phase 3 Advanced Features ---
    # 7. Liquidity Sweep 1H (Proxy: 60-period sweep signal)
    sweep_sig_1h, _ = compute_liquidity_sweep(out, 60)
    out["Liquidity_Sweep_1H"] = sweep_sig_1h


    return out
