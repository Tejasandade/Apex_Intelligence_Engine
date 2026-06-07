"""
Apex Intelligence Engine V6 — Swing Detector
==============================================
Lightweight utility to detect structural swing points for the
STRUCTURAL and HYBRID trailing stop phases.

Designed to be O(n) fast, evaluating only the recent `lookback` window,
while applying SMC confirmation (a swing high is only confirmed if 
3 subsequent bars close below it).
"""

import pandas as pd

def find_trailing_swing(
    df: pd.DataFrame, 
    direction: str, 
    lookback: int = 20
) -> float | None:
    """
    Find the most recent confirmed swing point for trailing stop placement.
    
    A swing high is confirmed when:
    1. It is the highest high in the last `lookback` bars
    2. At least 3 subsequent bars have closed below it
    3. It has not been exceeded by any subsequent bar
    
    A swing low is confirmed when:
    1. It is the lowest low in the last `lookback` bars
    2. At least 3 subsequent bars have closed above it
    3. It has not been exceeded by any subsequent bar

    Args:
        df: The OHLCV dataframe (usually 5m or 15m timeframe).
        direction: "BUY" (find swing lows to trail long stops) or "SELL".
        lookback: The scan window length.
        
    Returns:
        The price level of the confirmed swing, or None if no valid swing exists.
    """
    if df is None or df.empty or len(df) < 5:
        return None
        
    # We only look at the most recent `lookback` bars
    scan_df = df.tail(lookback)
    
    if direction == "BUY":
        # Trailing a LONG position -> We need the most recent SWING LOW.
        # Find the absolute minimum in this window
        min_idx = scan_df["low"].idxmin()
        min_row_num = scan_df.index.get_loc(min_idx)
        
        # Check confirmation: Are there at least 3 bars after it?
        bars_after = len(scan_df) - 1 - min_row_num
        if bars_after < 3:
            # Too recent, not yet confirmed by 3 subsequent closes
            return None
            
        # Check condition 2: Did 3 subsequent bars close ABOVE it?
        # A swing low is confirmed if price moves away from it.
        subsequent = scan_df.iloc[min_row_num + 1:]
        closes_above = (subsequent["close"] > scan_df.loc[min_idx, "low"]).sum()
        
        if closes_above >= 3:
            return float(scan_df.loc[min_idx, "low"])
            
    else:
        # Trailing a SHORT position -> We need the most recent SWING HIGH.
        max_idx = scan_df["high"].idxmax()
        max_row_num = scan_df.index.get_loc(max_idx)
        
        bars_after = len(scan_df) - 1 - max_row_num
        if bars_after < 3:
            return None
            
        subsequent = scan_df.iloc[max_row_num + 1:]
        closes_below = (subsequent["close"] < scan_df.loc[max_idx, "high"]).sum()
        
        if closes_below >= 3:
            return float(scan_df.loc[max_idx, "high"])
            
    return None
