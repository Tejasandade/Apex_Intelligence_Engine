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


def compute_volume_profile_snapshot(
    df: pd.DataFrame,
    n_bars: int = 100,
    n_bins: int = 50,
    value_area_pct: float = 0.70,
) -> dict:
    """
    Compute Volume Profile snapshot for the most recent N bars.

    Returns a dict with:
        poc           : float  — Point of Control (price bin with most volume)
        vah           : float  — Value Area High
        val           : float  — Value Area Low
        hvn_prices    : list   — Top-3 High Volume Nodes besides POC
        lvn_prices    : list   — Low Volume Nodes (price bins with volume < 10th-percentile)
        current_in_value: bool — Is the last close inside the value area?
        profile_ready : bool   — False if not enough bars to compute
    """
    _empty = dict(poc=0.0, vah=0.0, val=0.0, hvn_prices=[], lvn_prices=[], current_in_value=False, profile_ready=False)
    if df is None or len(df) < 10:
        return _empty

    window = df.tail(n_bars).copy()
    required = {"high", "low", "close", "volume"}
    if not required.issubset(window.columns):
        return _empty

    for col in ["high", "low", "close", "volume"]:
        window[col] = pd.to_numeric(window[col], errors="coerce")
    window = window.dropna(subset=["high", "low", "close", "volume"])
    if len(window) < 5:
        return _empty

    min_price = window["low"].min()
    max_price = window["high"].max()
    if min_price >= max_price:
        return _empty

    bin_edges = np.linspace(min_price, max_price, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_profile = np.zeros(n_bins)

    for _, row in window.iterrows():
        low_p, high_p, vol = row["low"], row["high"], row["volume"]
        if vol <= 0:
            continue
        if high_p == low_p:
            idx = int(np.argmin(np.abs(bin_centers - low_p)))
            vol_profile[idx] += vol
        else:
            mask = (bin_centers >= low_p) & (bin_centers <= high_p)
            cnt = int(np.sum(mask))
            if cnt > 0:
                vol_profile[mask] += vol / cnt
            else:
                idx = int(np.argmin(np.abs(bin_centers - (low_p + high_p) / 2)))
                vol_profile[idx] += vol

    poc_idx = int(np.argmax(vol_profile))
    poc = float(bin_centers[poc_idx])

    # Value Area expansion
    total_vol = float(np.sum(vol_profile))
    target_vol = total_vol * value_area_pct
    current_vol = float(vol_profile[poc_idx])
    up_idx = poc_idx + 1
    dn_idx = poc_idx - 1

    while current_vol < target_vol and (up_idx < n_bins or dn_idx >= 0):
        vol_up = float(vol_profile[up_idx]) if up_idx < n_bins else 0.0
        vol_dn = float(vol_profile[dn_idx]) if dn_idx >= 0  else 0.0
        if vol_up >= vol_dn and up_idx < n_bins:
            current_vol += vol_up
            up_idx += 1
        elif dn_idx >= 0:
            current_vol += vol_dn
            dn_idx -= 1
        else:
            break

    vah = float(bin_centers[min(up_idx, n_bins - 1)])
    val = float(bin_centers[max(dn_idx, 0)])

    # HVN: top-3 bins by volume, excluding POC
    sorted_idxs = np.argsort(vol_profile)[::-1]
    hvn_prices = [float(bin_centers[i]) for i in sorted_idxs if i != poc_idx][:3]

    # LVN: bins below 10th-percentile of volume (fast-move zones)
    threshold = float(np.percentile(vol_profile[vol_profile > 0], 10)) if np.any(vol_profile > 0) else 0.0
    lvn_prices = [float(bin_centers[i]) for i in range(n_bins) if 0 < vol_profile[i] <= threshold]

    current_price = float(window["close"].iloc[-1])
    in_value = val <= current_price <= vah

    return dict(
        poc=round(poc, 1),
        vah=round(vah, 1),
        val=round(val, 1),
        hvn_prices=[round(p, 1) for p in hvn_prices],
        lvn_prices=[round(p, 1) for p in lvn_prices],
        current_in_value=in_value,
        profile_ready=True,
    )

