"""
Apex Intelligence Engine V5 — Technical Indicators
====================================================
Pure, vectorised indicator calculations using Pandas/NumPy.
No side effects. No state. Just math.

Every function operates on pd.Series/pd.DataFrame and returns pd.Series.
This allows both vectorised batch computation (training) and single-value
extraction (live inference) from the same code.

Extracted from legacy train_v3.py and dataset_builder.py, cleaned and unified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── RSI (Relative Strength Index) ────────────────────────────────────────────
def compute_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """
    Compute RSI using the standard Wilder smoothing (SMA of gains/losses).

    Args:
        close: Series of closing prices.
        length: Lookback period (default 14).

    Returns:
        Series of RSI values (0-100). NaN for insufficient data.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ── EMA (Exponential Moving Average) ────────────────────────────────────────
def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Compute EMA using the standard exponential weighting.

    Args:
        series: Input price series.
        span: EMA period.

    Returns:
        Series of EMA values.
    """
    return series.ewm(span=span, adjust=False).mean()


# ── MACD (Moving Average Convergence Divergence) ────────────────────────────
def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD line, signal line, and histogram.

    Args:
        close: Series of closing prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ── ATR (Average True Range) ────────────────────────────────────────────────
def compute_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute ATR — the average of the true range over N periods.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        length: Lookback period (default 14).

    Returns:
        Series of ATR values.
    """
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=length, min_periods=length).mean()


# ── ADX (Average Directional Index) ─────────────────────────────────────────
def compute_adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute ADX — measures trend strength regardless of direction.
    ADX > 25 = trending, ADX < 20 = ranging.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        length: Lookback period (default 14).

    Returns:
        Series of ADX values (0-100).
    """
    high, low, close = df["high"], df["low"], df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    atr = tr.ewm(span=length, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=length, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(span=length, adjust=False).mean() / atr)

    dx = (100.0 * (abs(plus_di - minus_di) / (plus_di + minus_di))).fillna(0.0)
    adx = dx.ewm(span=length, adjust=False).mean()

    return adx.fillna(0.0)


# ── CHOP (Choppiness Index) ─────────────────────────────────────────────────
def compute_chop(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Compute Choppiness Index — measures whether the market is trending or choppy.
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = strong trend.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.
        length: Lookback period (default 14).

    Returns:
        Series of CHOP values (0-100).
    """
    high, low, close = df["high"], df["low"], df["close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_sum = tr.rolling(window=length).sum()
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()

    range_hl = (highest_high - lowest_low).replace(0, np.nan)
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(length)

    return chop.fillna(0.0)


# ── VWAP (Volume Weighted Average Price) ────────────────────────────────────
def compute_vwap_continuous(df: pd.DataFrame) -> pd.Series:
    """
    Compute cumulative VWAP — for 24/7 markets (crypto).
    Resets not needed since crypto trades continuously.

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns.

    Returns:
        Series of VWAP values.
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum().replace(0, np.nan)
    return cum_tp_vol / cum_vol


def compute_vwap_session(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    timezone: str = "Asia/Kolkata",
) -> pd.Series:
    """
    Compute session-anchored VWAP — resets at each trading day boundary.
    For session-bound markets (NSE, Forex).

    If volume is all zeros (index data), falls back to cumulative mean
    of typical price per session (anchored TWAP).

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume', and timestamp columns.
        timestamp_col: Name of the timestamp column.
        timezone: Timezone for session grouping.

    Returns:
        Series of session-VWAP values.
    """
    df_v = df.copy()
    if timestamp_col in df_v.columns:
        ts = pd.to_datetime(df_v[timestamp_col], utc=True)
        df_v["_local_date"] = ts.dt.tz_convert(timezone).dt.date
    else:
        # Fallback: use index as date grouper
        df_v["_local_date"] = 0

    tp = (df_v["high"] + df_v["low"] + df_v["close"]) / 3.0

    if df_v["volume"].sum() > 0:
        # Real volume data
        tv = tp * df_v["volume"]
        df_v["_tv"] = tv
        cum_tv = df_v.groupby("_local_date")["_tv"].cumsum()
        cum_v = df_v.groupby("_local_date")["volume"].cumsum()
        vwap = cum_tv / cum_v.replace(0, np.nan)
    else:
        # No volume (index data) — use cumulative mean per session
        df_v["_tp"] = tp
        cum_tp = df_v.groupby("_local_date")["_tp"].cumsum()
        bar_num = df_v.groupby("_local_date").cumcount() + 1
        vwap = cum_tp / bar_num

    return vwap.fillna(tp).reindex(df.index)


def compute_vwap_zscore(
    close: pd.Series,
    vwap: pd.Series,
    window: int = 100,
) -> pd.Series:
    """
    Compute the Z-Score of the price's deviation from VWAP.
    High absolute values (>2.0) indicate mean reversion potential.
    
    Args:
        close: Series of closing prices.
        vwap: Series of VWAP values.
        window: Rolling window for mean/std of the deviation.
        
    Returns:
        Series of Z-Scores.
    """
    pct_dev = (close - vwap) / vwap.replace(0, np.nan)
    roll_mean = pct_dev.rolling(window=window, min_periods=window).mean()
    roll_std = pct_dev.rolling(window=window, min_periods=window).std().replace(0, np.nan)
    
    zscore = (pct_dev - roll_mean) / roll_std
    return zscore.fillna(0.0)


# ── Bollinger Bands ──────────────────────────────────────────────────────────
def compute_bollinger_bands(
    close: pd.Series,
    length: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands — middle band (SMA), upper and lower bands.

    Args:
        close: Series of closing prices.
        length: SMA period (default 20).
        std_dev: Standard deviation multiplier (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle = close.rolling(window=length, min_periods=length).mean()
    std = close.rolling(window=length, min_periods=length).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower


# ── Bollinger Band Width ────────────────────────────────────────────────────
def compute_bb_width(close: pd.Series, length: int = 20, std_dev: float = 2.0) -> pd.Series:
    """
    Bollinger Band Width — measures volatility squeeze/expansion.
    Low values = squeeze (upcoming breakout), high values = expansion.
    """
    upper, middle, lower = compute_bollinger_bands(close, length, std_dev)
    return ((upper - lower) / middle).fillna(0.0)


# ── Pseudo-Order Flow / Microstructure ───────────────────────────────────────
def compute_vsa_absorption(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Volume Spread Analysis: Absorption.
    High volume + small candle body = Absorption.
    Returns ratio of volume to body size, normalized by rolling mean.
    """
    body = (df['close'] - df['open']).abs()
    body = pd.Series(np.where(body == 0, df['close'] * 0.0001, body), index=df.index)
    
    vol_body_ratio = df['volume'] / body
    roll_mean = vol_body_ratio.rolling(window=window, min_periods=1).mean()
    absorption = vol_body_ratio / roll_mean.replace(0, np.nan)
    return absorption.fillna(1.0)

def compute_liquidity_sweep(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Detects Liquidity Sweeps (Stop Hunts).
    Bullish Sweep (+1): Low breaks 20-period low, but close is > open.
    Bearish Sweep (-1): High breaks 20-period high, but close is < open.
    """
    rolling_high = df['high'].shift(1).rolling(window=window).max()
    rolling_low = df['low'].shift(1).rolling(window=window).min()
    
    bearish_sweep = (df['high'] > rolling_high) & (df['close'] < df['open'])
    bullish_sweep = (df['low'] < rolling_low) & (df['close'] > df['open'])
    
    sweep = pd.Series(0.0, index=df.index)
    sweep[bullish_sweep] = 1.0
    sweep[bearish_sweep] = -1.0
    return sweep

def compute_trend_exhaustion(df: pd.DataFrame) -> pd.Series:
    """
    Detects Micro-Trend Exhaustion.
    3 consecutive green candles with decreasing volume = Bullish Exhaustion (-1).
    3 consecutive red candles with decreasing volume = Bearish Exhaustion (+1).
    """
    is_green = df['close'] > df['open']
    is_red = df['close'] < df['open']
    
    vol_down = df['volume'] < df['volume'].shift(1)
    vol_down_2 = vol_down & (df['volume'].shift(1) < df['volume'].shift(2))
    
    bull_exhaustion = is_green & is_green.shift(1) & is_green.shift(2) & vol_down_2
    bear_exhaustion = is_red & is_red.shift(1) & is_red.shift(2) & vol_down_2
    
    exhaustion = pd.Series(0.0, index=df.index)
    exhaustion[bull_exhaustion] = -1.0
    exhaustion[bear_exhaustion] = 1.0
    return exhaustion


# ── Utility: Build All Technical Features ───────────────────────────────────
def build_technical_features(
    df: pd.DataFrame,
    market_type: str = "crypto",
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Compute all technical indicators and add them as columns.

    This is the canonical feature builder — used in both training and inference
    to guarantee train/serve parity.

    Args:
        df: DataFrame with OHLCV columns.
        market_type: "crypto" or "india_equity" — determines VWAP mode.
        timestamp_col: Name of the timestamp column (for session VWAP).

    Returns:
        DataFrame with all technical indicator columns added.
    """
    out = df.copy()
    close = out["close"]

    # Core indicators
    out["RSI"] = compute_rsi(close, 14)
    ema_14 = compute_ema(close, 14)
    ema_50 = compute_ema(close, 50)
    out["EMA_14"] = ema_14  # Keep for council/other modules
    out["EMA_50"] = ema_50  # Keep for council/other modules
    macd, macd_sig, macd_hist = compute_macd(close)
    out["MACD"] = macd      # Keep raw for council
    out["MACD_signal"] = macd_sig  # Keep raw for council
    out["MACD_hist"] = macd_hist
    out["ATR"] = compute_atr(out, 14)
    out["ADX"] = compute_adx(out, 14)
    out["CHOP"] = compute_chop(out, 14)
    out["BB_width"] = compute_bb_width(close, 20, 2.0)

    # VWAP — mode depends on market type
    if market_type == "crypto":
        out["VWAP"] = compute_vwap_continuous(out)
    else:
        out["VWAP"] = compute_vwap_session(out, timestamp_col=timestamp_col)
        
    # VWAP Mean Reversion feature
    out["VWAP_zscore"] = compute_vwap_zscore(out["close"], out["VWAP"])

    # --- Normalized Price-Relative Features (no absolute price leakage) ---
    atr_safe = out["ATR"].replace(0, np.nan).fillna(1.0)
    
    # Distance from EMAs normalized by ATR
    out["price_vs_ema14"] = (close - ema_14) / atr_safe
    out["price_vs_ema50"] = (close - ema_50) / atr_safe
    out["ema_cross"] = (ema_14 - ema_50) / atr_safe
    
    # MACD normalized by ATR (scale-independent)
    out["MACD_norm"] = macd / atr_safe
    out["MACD_signal_norm"] = macd_sig / atr_safe

    # --- Phase 3 Advanced Features ---
    # 1. ATR Ratio (current ATR / 20-period moving average of ATR)
    out["ATR_Ratio"] = out["ATR"] / out["ATR"].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    out["ATR_Ratio"] = out["ATR_Ratio"].fillna(1.0)
    
    # 2. RSI Trend (Momentum of RSI over 10 periods)
    out["RSI_Trend"] = out["RSI"] - out["RSI"].shift(10).fillna(out["RSI"])
    
    # 3. VWAP Distance (Percentage distance from VWAP)
    out["VWAP_Distance"] = (out["close"] - out["VWAP"]) / out["VWAP"].replace(0, np.nan)
    out["VWAP_Distance"] = out["VWAP_Distance"].fillna(0.0)
    
    # 4. Volatility Regime (Binary: 1 if ATR_Ratio > 1.2 else 0)
    out["Volatility_Regime"] = (out["ATR_Ratio"] > 1.2).astype(float)

    # 5. Volume Spike (relative volume vs 20-bar average)
    vol_ma = out["volume"].rolling(window=20, min_periods=1).mean().replace(0, np.nan)
    out["volume_spike"] = out["volume"] / vol_ma
    out["volume_spike"] = out["volume_spike"].fillna(1.0)

    # 6. Time-of-day cyclic encoding (let model learn session patterns)
    if timestamp_col in out.columns:
        try:
            ts = out[timestamp_col]
            if ts.dtype in ('int64', 'float64'):
                hours = pd.to_datetime(ts, unit='ms', utc=True).dt.hour + \
                        pd.to_datetime(ts, unit='ms', utc=True).dt.minute / 60.0
            else:
                hours = pd.to_datetime(ts, utc=True).dt.hour + \
                        pd.to_datetime(ts, utc=True).dt.minute / 60.0
            out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
        except Exception:
            out["hour_sin"] = 0.0
            out["hour_cos"] = 0.0
    else:
        out["hour_sin"] = 0.0
        out["hour_cos"] = 0.0

    # 7. VSA Absorption
    out["vsa_absorption"] = compute_vsa_absorption(out)
    
    # 8. Liquidity Sweep
    out["liquidity_sweep_signal"] = compute_liquidity_sweep(out)
    
    # 9. Micro-Trend Exhaustion
    out["trend_exhaustion"] = compute_trend_exhaustion(out)

    return out

