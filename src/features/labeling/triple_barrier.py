"""
Apex Intelligence Engine V5 — Triple-Barrier Labeling
======================================================
The proper way to label trading targets. Instead of the naive
"will price be higher in N bars?" (coin flip), triple-barrier defines
THREE exit conditions:

1. UPPER BARRIER (Take Profit): Price hits profit target → Label = 1
2. LOWER BARRIER (Stop Loss): Price hits stop loss → Label = 0
3. TIME BARRIER (Expiry): Neither barrier hit within N bars → Label = 0

The label is 1 ONLY if the upper barrier is touched FIRST.
This directly models the actual trade outcome: "if I enter here with
this stop and target, will I win?"

Reference: Marcos López de Prado, "Advances in Financial Machine Learning"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger("apex.features.labeling")


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    profit_target_atr: float = 2.0,
    stop_loss_atr: float = 2.0,
    max_holding_bars: int = 30,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "ATR",
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a DataFrame of OHLCV data.

    For each bar, we look forward up to max_holding_bars and determine:
    - Did price reach close * (1 + profit_target_pct) via highs? (upper barrier)
    - Did price reach close * (1 - stop_loss_pct) via lows? (lower barrier)
    - Which barrier was hit FIRST?

    Args:
        df: DataFrame with OHLCV data.
        profit_target_atr: Take profit threshold as ATR multiple.
        stop_loss_atr: Stop loss threshold as ATR multiple.
        max_holding_bars: Maximum bars to hold before time expiry.
        close_col: Column name for close price.
        high_col: Column name for high price.
        low_col: Column name for low price.
        atr_col: Column name for ATR.

    Returns:
        DataFrame with added columns:
        - 'target_long': 1 if upper barrier hit first, 0 otherwise
        - 'target_short': 1 if lower barrier hit first, 0 otherwise
        - 'barrier_type': 'upper', 'lower', or 'time' indicating which barrier was hit
        - 'bars_to_barrier': number of bars until barrier was hit
    """
    close = df[close_col].values
    high = df[high_col].values
    low = df[low_col].values
    
    if atr_col not in df.columns:
        raise ValueError(f"ATR column '{atr_col}' not found in DataFrame.")
    atr = df[atr_col].values
    
    n = len(df)

    targets_long = np.full(n, np.nan)
    targets_short = np.full(n, np.nan)
    barrier_types = np.full(n, "", dtype=object)
    bars_to_barrier = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = close[i]
        entry_atr = atr[i]
        
        if entry_price <= 0 or entry_atr <= 0:
            targets_long[i] = 0
            targets_short[i] = 0
            barrier_types[i] = "invalid"
            bars_to_barrier[i] = 0
            continue

        # Evaluate Long and Short barriers independently so asymmetric TP:SL ratios are symmetric for both sides
        long_tp = entry_price + (entry_atr * profit_target_atr)
        long_sl = entry_price - (entry_atr * stop_loss_atr)
        short_tp = entry_price - (entry_atr * profit_target_atr)
        short_sl = entry_price + (entry_atr * stop_loss_atr)

        l_tp_bar, l_sl_bar = -1, -1
        s_tp_bar, s_sl_bar = -1, -1

        end_idx = min(i + max_holding_bars, n - 1)

        for j in range(i + 1, end_idx + 1):
            # Long barriers
            if l_tp_bar == -1 and high[j] >= long_tp:
                l_tp_bar = j - i
            if l_sl_bar == -1 and low[j] <= long_sl:
                l_sl_bar = j - i

            # Short barriers
            if s_tp_bar == -1 and low[j] <= short_tp:
                s_tp_bar = j - i
            if s_sl_bar == -1 and high[j] >= short_sl:
                s_sl_bar = j - i

            long_done = (l_tp_bar > 0 or l_sl_bar > 0)
            short_done = (s_tp_bar > 0 or s_sl_bar > 0)
            if long_done and short_done:
                break

        # Resolve Long outcome (conservative tie-break: same-bar SL assumed hit first)
        long_win = (l_tp_bar > 0) and (l_sl_bar == -1 or l_tp_bar < l_sl_bar)
        short_win = (s_tp_bar > 0) and (s_sl_bar == -1 or s_tp_bar < s_sl_bar)

        targets_long[i] = 1 if long_win else 0
        targets_short[i] = 1 if short_win else 0

        if long_win:
            barrier_types[i] = "upper"
            bars_to_barrier[i] = l_tp_bar
        elif short_win:
            barrier_types[i] = "lower"
            bars_to_barrier[i] = s_tp_bar
        else:
            barrier_types[i] = "time" if (l_sl_bar == -1 and s_sl_bar == -1) else "lower"
            bars_to_barrier[i] = l_sl_bar if l_sl_bar > 0 else (s_sl_bar if s_sl_bar > 0 else max_holding_bars)

    result = df.copy()
    result["target_long"] = targets_long
    result["target_short"] = targets_short
    result["barrier_type"] = barrier_types
    result["bars_to_barrier"] = bars_to_barrier

    # Drop rows where target is NaN (last max_holding_bars rows)
    valid_mask = result["target_long"].notna()
    result = result[valid_mask].copy()
    result["target_long"] = result["target_long"].astype(int)
    result["target_short"] = result["target_short"].astype(int)

    # Log statistics
    if len(result) > 0:
        long_pos_rate = result["target_long"].mean()
        short_pos_rate = result["target_short"].mean()
        upper_count = (result["barrier_type"] == "upper").sum()
        lower_count = (result["barrier_type"] == "lower").sum()
        time_count = (result["barrier_type"] == "time").sum()
        avg_bars = result["bars_to_barrier"].mean()

        logger.info(
            "triple_barrier_labels_applied",
            total_samples=len(result),
            long_positive_rate=f"{long_pos_rate:.2%}",
            short_positive_rate=f"{short_pos_rate:.2%}",
            upper_barrier_hits=int(upper_count),
            lower_barrier_hits=int(lower_count),
            time_barrier_hits=int(time_count),
            avg_bars_to_barrier=f"{avg_bars:.1f}",
            profit_target_atr=profit_target_atr,
            stop_loss_atr=stop_loss_atr,
            max_holding_bars=max_holding_bars,
        )

    return result
