"""
Apex Intelligence Engine V5 — Crypto Market Profile
=====================================================
Crypto-specific feature engineering. Features that only make sense
for 24/7 crypto markets: funding rate proxies, liquidation intensity, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_spread(df: pd.DataFrame) -> pd.Series:
    """
    Compute bid-ask spread estimate from OHLCV.
    In training (no LOB data), we approximate spread as 10% of candle range.
    Live inference should use actual best_bid/best_ask from order book.
    """
    if "best_bid" in df.columns and "best_ask" in df.columns:
        real_spread = df["best_ask"] - df["best_bid"]
        has_real = (real_spread > 0).any()
        if has_real:
            return real_spread
    return (df["high"] - df["low"]) * 0.1


def compute_book_imbalance(df: pd.DataFrame) -> pd.Series:
    """
    Compute order book imbalance from bid/ask quantities.
    Positive = more bids (buying pressure). Negative = more asks (selling).
    Returns 0 if no book data available.
    """
    if "best_bid_qty" in df.columns and "best_ask_qty" in df.columns:
        total = df["best_bid_qty"] + df["best_ask_qty"]
        total = total.replace(0, np.nan)
        return ((df["best_bid_qty"] - df["best_ask_qty"]) / total).fillna(0.0)
    return pd.Series(0.0, index=df.index)


def build_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build crypto-specific features on top of the base OHLCV dataframe.

    Adds:
    - spread: bid-ask spread estimate
    - book_imbalance: order book pressure ratio
    - best_bid/best_ask: pass-through or estimated from close

    Args:
        df: DataFrame with OHLCV and optionally LOB columns.

    Returns:
        DataFrame with crypto-specific columns added.
    """
    out = df.copy()

    out["spread"] = compute_spread(out)
    out["book_imbalance"] = compute_book_imbalance(out)

    # Ensure bid/ask columns exist (for model compatibility)
    if "best_bid" not in out.columns:
        out["best_bid"] = out["close"] - out["spread"] / 2
    if "best_ask" not in out.columns:
        out["best_ask"] = out["close"] + out["spread"] / 2

    # Macro sentiment placeholder (filled by oracle at inference time)
    if "macro_sentiment_score" not in out.columns:
        out["macro_sentiment_score"] = 0.0

    return out
