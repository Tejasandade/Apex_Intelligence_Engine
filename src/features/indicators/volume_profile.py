"""
Apex Intelligence Engine V5 — Volume Profile & Order Flow Depth
=============================================================
Calculates Volume Profile, Point of Control (POC), and Value Areas (VAH, VAL).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

def compute_volume_profile(
    df: pd.DataFrame, 
    lookback: int = 200, 
    bins: int = 50,
    value_area_pct: float = 0.70
) -> pd.DataFrame:
    """
    Computes rolling Volume Profile metrics (POC, VAH, VAL).
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume'.
        lookback: Number of candles for the profile.
        bins: Number of price bins for the distribution.
        value_area_pct: Percentage of volume contained in the Value Area.
        
    Returns:
        DataFrame with new columns: 'POC', 'VAH', 'VAL'
    """
    out = df.copy()
    out['POC'] = np.nan
    out['VAH'] = np.nan
    out['VAL'] = np.nan

    if len(df) < lookback:
        return out

    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        min_price = window['low'].min()
        max_price = window['high'].max()
        
        if min_price == max_price:
            out.iloc[i, out.columns.get_loc('POC')] = min_price
            out.iloc[i, out.columns.get_loc('VAH')] = max_price
            out.iloc[i, out.columns.get_loc('VAL')] = min_price
            continue
            
        bin_edges = np.linspace(min_price, max_price, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        vol_profile = np.zeros(bins)
        
        for _, row in window.iterrows():
            low, high, vol = row['low'], row['high'], row['volume']
            if high == low:
                # Assign all volume to the nearest bin
                idx = (np.abs(bin_centers - low)).argmin()
                vol_profile[idx] += vol
            else:
                # Distribute volume evenly across intersecting bins
                mask = (bin_centers >= low) & (bin_centers <= high)
                count = np.sum(mask)
                if count > 0:
                    vol_profile[mask] += vol / count
                else:
                    idx = (np.abs(bin_centers - ((low+high)/2))).argmin()
                    vol_profile[idx] += vol
                    
        # Find POC
        poc_idx = np.argmax(vol_profile)
        poc = bin_centers[poc_idx]
        
        # Calculate Value Area
        total_vol = np.sum(vol_profile)
        target_vol = total_vol * value_area_pct
        
        current_vol = vol_profile[poc_idx]
        up_idx = poc_idx + 1
        down_idx = poc_idx - 1
        
        while current_vol < target_vol and (up_idx < bins or down_idx >= 0):
            vol_up = vol_profile[up_idx] if up_idx < bins else 0
            vol_down = vol_profile[down_idx] if down_idx >= 0 else 0
            
            if vol_up >= vol_down and up_idx < bins:
                current_vol += vol_up
                up_idx += 1
            elif down_idx >= 0:
                current_vol += vol_down
                down_idx -= 1
            else:
                break
                
        vah = bin_centers[min(up_idx, bins-1)]
        val = bin_centers[max(down_idx, 0)]
        
        out.iloc[i, out.columns.get_loc('POC')] = poc
        out.iloc[i, out.columns.get_loc('VAH')] = vah
        out.iloc[i, out.columns.get_loc('VAL')] = val

    # Forward fill the remaining nans for first values if needed
    out['POC'] = out['POC'].ffill().fillna(out['close'])
    out['VAH'] = out['VAH'].ffill().fillna(out['high'])
    out['VAL'] = out['VAL'].ffill().fillna(out['low'])
    return out
