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
        cum_tv = df_v.groupby("_local_date").apply(
            lambda g: tv.loc[g.index].cumsum()
        )
        cum_v = df_v.groupby("_local_date").apply(
            lambda g: df_v["volume"].loc[g.index].cumsum()
        )
        # Flatten multi-index from groupby
        if isinstance(cum_tv.index, pd.MultiIndex):
            cum_tv = cum_tv.droplevel(0)
            cum_v = cum_v.droplevel(0)
        vwap = cum_tv / cum_v.replace(0, np.nan)
    else:
        # No volume (index data) — use cumulative mean per session
        df_v["_tp"] = tp
        cum_tp = df_v.groupby("_local_date").apply(
            lambda g: df_v["_tp"].loc[g.index].cumsum()
        )
        bar_num = df_v.groupby("_local_date").cumcount() + 1
        if isinstance(cum_tp.index, pd.MultiIndex):
            cum_tp = cum_tp.droplevel(0)
        vwap = cum_tp / bar_num

    return vwap.fillna(tp).reindex(df.index)


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
    out["EMA_14"] = compute_ema(close, 14)
    out["EMA_50"] = compute_ema(close, 50)
    macd, macd_sig, macd_hist = compute_macd(close)
    out["MACD"] = macd
    out["MACD_signal"] = macd_sig
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

    return out
