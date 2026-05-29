"""
Apex Intelligence Engine V5 — India Equity Market Profile
===========================================================
NSE-specific feature engineering. Features unique to Indian equity markets:
- Session-relative time features (how far into the trading day)
- Gap-from-open analysis (opening gap is critical in Indian markets)
- Session filtering and opening bar noise removal
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

IST = ZoneInfo("Asia/Kolkata")
SESSION_OPEN_MINUTES = 9 * 60 + 15     # 09:15 IST in minutes
SESSION_CLOSE_MINUTES = 15 * 60 + 30   # 15:30 IST in minutes
SESSION_DURATION_MINUTES = SESSION_CLOSE_MINUTES - SESSION_OPEN_MINUTES  # 375 minutes


def filter_session_hours(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Keep only rows within NSE market hours (09:15 – 15:30 IST, Mon–Fri).

    Args:
        df: DataFrame with a UTC timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        Filtered DataFrame.
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    ist = ts.dt.tz_convert(IST)
    minutes = ist.dt.hour * 60 + ist.dt.minute
    weekday = ist.dt.dayofweek  # Mon=0, Fri=4

    mask = (
        (minutes >= SESSION_OPEN_MINUTES)
        & (minutes <= SESSION_CLOSE_MINUTES)
        & (weekday < 5)
    )
    return df[mask].reset_index(drop=True)


def drop_opening_bars(
    df: pd.DataFrame,
    n_bars: int = 5,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Remove the first N bars of each NSE trading session.
    Opening bars are noisy due to gap-up/gap-down fills.

    Args:
        df: DataFrame with a UTC timestamp column.
        n_bars: Number of opening bars to drop per session.
        timestamp_col: Name of the timestamp column.

    Returns:
        Filtered DataFrame.
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    ist = ts.dt.tz_convert(IST)
    minutes = ist.dt.hour * 60 + ist.dt.minute
    open_end = SESSION_OPEN_MINUTES + n_bars

    mask = minutes >= open_end
    return df[mask].reset_index(drop=True)


def compute_session_elapsed_pct(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Compute how far into the trading session each bar is (0.0 to 1.0).
    0.0 = market just opened, 1.0 = market closing.

    This captures time-of-day effects:
    - Early session: gap fill moves, high volatility
    - Mid session: institutional accumulation
    - Late session: squaring off, volatility spike

    Args:
        df: DataFrame with a UTC timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        Series of elapsed percentage values (0.0 to 1.0).
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    ist = ts.dt.tz_convert(IST)
    minutes = ist.dt.hour * 60 + ist.dt.minute

    elapsed = (minutes - SESSION_OPEN_MINUTES).clip(lower=0)
    pct = elapsed / SESSION_DURATION_MINUTES
    return pct.clip(0.0, 1.0)


def compute_gap_from_open_pct(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.Series:
    """
    Compute the current price's percentage distance from today's opening price.
    Captures gap-fill dynamics that dominate Indian intraday trading.

    Args:
        df: DataFrame with 'close', 'open' and a timestamp column.
        timestamp_col: Name of the timestamp column.

    Returns:
        Series of gap percentages from session open price.
    """
    ts = pd.to_datetime(df[timestamp_col], utc=True)
    ist = ts.dt.tz_convert(IST)
    df_tmp = df.copy()
    df_tmp["_local_date"] = ist.dt.date

    # Get the first open price of each session
    session_opens = df_tmp.groupby("_local_date")["open"].first()
    df_tmp["_session_open"] = df_tmp["_local_date"].map(session_opens)

    # Gap from session open as percentage
    gap_pct = (df_tmp["close"] - df_tmp["_session_open"]) / df_tmp[
        "_session_open"
    ].replace(0, np.nan)

    return gap_pct.fillna(0.0)


def build_india_equity_features(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Build India equity-specific features on top of the base OHLCV dataframe.

    Adds:
    - session_elapsed_pct: time-of-day position in session
    - gap_from_open_pct: distance from session open

    Args:
        df: DataFrame with OHLCV and timestamp columns.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with India equity-specific columns added.
    """
    out = df.copy()

    out["session_elapsed_pct"] = compute_session_elapsed_pct(out, timestamp_col)
    out["gap_from_open_pct"] = compute_gap_from_open_pct(out, timestamp_col)

    # Macro sentiment placeholder
    if "macro_sentiment_score" not in out.columns:
        out["macro_sentiment_score"] = 0.0

    return out
