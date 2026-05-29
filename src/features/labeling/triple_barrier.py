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
    profit_target_pct: float = 0.0020,
    stop_loss_pct: float = 0.0012,
    max_holding_bars: int = 20,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    Apply triple-barrier labeling to a DataFrame of OHLCV data.

    For each bar, we look forward up to max_holding_bars and determine:
    - Did price reach close * (1 + profit_target_pct) via highs? (upper barrier)
    - Did price reach close * (1 - stop_loss_pct) via lows? (lower barrier)
    - Which barrier was hit FIRST?

    Args:
        df: DataFrame with OHLCV data.
        profit_target_pct: Take profit threshold as percentage (e.g., 0.002 = 0.2%).
        stop_loss_pct: Stop loss threshold as percentage (e.g., 0.0012 = 0.12%).
        max_holding_bars: Maximum bars to hold before time expiry.
        close_col: Column name for close price.
        high_col: Column name for high price.
        low_col: Column name for low price.

    Returns:
        DataFrame with added columns:
        - 'target': 1 if profit target hit first, 0 otherwise
        - 'barrier_type': 'upper', 'lower', or 'time' indicating which barrier was hit
        - 'bars_to_barrier': number of bars until barrier was hit
    """
    close = df[close_col].values
    high = df[high_col].values
    low = df[low_col].values
    n = len(df)

    targets = np.full(n, np.nan)
    barrier_types = np.full(n, "", dtype=object)
    bars_to_barrier = np.full(n, np.nan)

    for i in range(n - 1):
        entry_price = close[i]
        if entry_price <= 0:
            targets[i] = 0
            barrier_types[i] = "invalid"
            bars_to_barrier[i] = 0
            continue

        upper_barrier = entry_price * (1.0 + profit_target_pct)
        lower_barrier = entry_price * (1.0 - stop_loss_pct)

        # Look forward bar by bar
        upper_hit_bar = -1
        lower_hit_bar = -1

        end_idx = min(i + max_holding_bars, n - 1)

        for j in range(i + 1, end_idx + 1):
            # Check if high touches upper barrier
            if upper_hit_bar == -1 and high[j] >= upper_barrier:
                upper_hit_bar = j - i

            # Check if low touches lower barrier
            if lower_hit_bar == -1 and low[j] <= lower_barrier:
                lower_hit_bar = j - i

            # If both hit on same bar, we need to determine which came first
            # Conservative assumption: if both barriers are hit on the same bar,
            # we assume the STOP was hit first (conservative bias)
            if upper_hit_bar > 0 and lower_hit_bar > 0:
                break

        # Determine the label
        if upper_hit_bar > 0 and lower_hit_bar > 0:
            if upper_hit_bar < lower_hit_bar:
                # Profit target hit first
                targets[i] = 1
                barrier_types[i] = "upper"
                bars_to_barrier[i] = upper_hit_bar
            elif lower_hit_bar < upper_hit_bar:
                # Stop loss hit first
                targets[i] = 0
                barrier_types[i] = "lower"
                bars_to_barrier[i] = lower_hit_bar
            else:
                # Same bar — conservative: assume stop hit first
                targets[i] = 0
                barrier_types[i] = "lower"
                bars_to_barrier[i] = lower_hit_bar
        elif upper_hit_bar > 0:
            # Only profit target hit
            targets[i] = 1
            barrier_types[i] = "upper"
            bars_to_barrier[i] = upper_hit_bar
        elif lower_hit_bar > 0:
            # Only stop loss hit
            targets[i] = 0
            barrier_types[i] = "lower"
            bars_to_barrier[i] = lower_hit_bar
        else:
            # Time barrier — neither barrier hit within max_holding_bars
            targets[i] = 0
            barrier_types[i] = "time"
            bars_to_barrier[i] = max_holding_bars

    result = df.copy()
    result["target"] = targets
    result["barrier_type"] = barrier_types
    result["bars_to_barrier"] = bars_to_barrier

    # Drop rows where target is NaN (last max_holding_bars rows)
    valid_mask = result["target"].notna()
    result = result[valid_mask].copy()
    result["target"] = result["target"].astype(int)

    # Log statistics
    if len(result) > 0:
        pos_rate = result["target"].mean()
        upper_count = (result["barrier_type"] == "upper").sum()
        lower_count = (result["barrier_type"] == "lower").sum()
        time_count = (result["barrier_type"] == "time").sum()
        avg_bars = result["bars_to_barrier"].mean()

        logger.info(
            "triple_barrier_labels_applied",
            total_samples=len(result),
            positive_rate=f"{pos_rate:.2%}",
            upper_barrier_hits=int(upper_count),
            lower_barrier_hits=int(lower_count),
            time_barrier_hits=int(time_count),
            avg_bars_to_barrier=f"{avg_bars:.1f}",
            profit_target_pct=f"{profit_target_pct:.4%}",
            stop_loss_pct=f"{stop_loss_pct:.4%}",
            max_holding_bars=max_holding_bars,
        )

    return result
