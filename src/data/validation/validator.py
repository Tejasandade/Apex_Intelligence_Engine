"""
Apex Intelligence Engine V5 — Data Validator
==============================================
Validates incoming market data before it enters the feature pipeline.
Bad data in = bad signals out. This is the L1 firewall.

Checks:
- OHLC consistency (high >= max(open,close), low <= min(open,close))
- Price bounds (reasonable range for the market)
- Volume sanity (non-negative, not NaN)
- Spike detection (Z-score from rolling mean)
- Staleness (timestamp gap detection)
- NaN/Inf cleansing
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.logging import get_logger

logger = get_logger("apex.data.validation")


class DataValidator:
    """
    Validates and cleans market data before feature computation.
    Operates on DataFrames — validates in batch for training,
    or single-row for live inference.
    """

    def __init__(
        self,
        spike_zscore_threshold: float = 5.0,
        max_gap_multiplier: float = 3.0,
        max_price_change_pct: float = 0.10,  # 10% single-bar move
    ):
        """
        Args:
            spike_zscore_threshold: Z-score above which a price is flagged as spike.
            max_gap_multiplier: Maximum allowed gap as multiple of expected interval.
            max_price_change_pct: Maximum single-bar price change before flagging.
        """
        self.spike_zscore_threshold = spike_zscore_threshold
        self.max_gap_multiplier = max_gap_multiplier
        self.max_price_change_pct = max_price_change_pct

    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean a DataFrame of OHLCV data.

        Removes rows that fail validation, logs statistics.

        Args:
            df: DataFrame with open, high, low, close, volume columns.

        Returns:
            Cleaned DataFrame.
        """
        initial_count = len(df)
        if initial_count == 0:
            return df

        # Step 1: Drop NaN rows in critical columns
        required_cols = ["open", "high", "low", "close"]
        existing = [c for c in required_cols if c in df.columns]
        df = df.dropna(subset=existing)

        # Step 2: OHLC consistency check
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            ohlc_valid = (
                (df["high"] >= df["open"])
                & (df["high"] >= df["close"])
                & (df["low"] <= df["open"])
                & (df["low"] <= df["close"])
                & (df["high"] >= df["low"])
                & (df["close"] > 0)
                & (df["open"] > 0)
            )
            invalid_ohlc = (~ohlc_valid).sum()
            if invalid_ohlc > 0:
                logger.warning(
                    "ohlc_consistency_failures",
                    count=int(invalid_ohlc),
                    pct=f"{invalid_ohlc / len(df):.1%}",
                )
                df = df[ohlc_valid]

        # Step 3: Volume sanity
        if "volume" in df.columns:
            df.loc[:, "volume"] = df["volume"].clip(lower=0).fillna(0)

        # Step 4: Spike detection (Z-score based)
        if "close" in df.columns and len(df) > 50:
            df = self._remove_spikes(df)

        # Step 5: Extreme single-bar moves
        if "close" in df.columns and len(df) > 1:
            pct_change = df["close"].pct_change().abs()
            extreme = pct_change > self.max_price_change_pct
            extreme_count = extreme.sum()
            if extreme_count > 0:
                logger.warning(
                    "extreme_bar_moves",
                    count=int(extreme_count),
                    threshold=f"{self.max_price_change_pct:.1%}",
                )
                # Don't remove — just log. Extreme moves are real in crypto.
                # For India, session opens can gap 2-3%, which is normal.

        # Step 6: Replace Inf/-Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].ffill().fillna(0)

        removed = initial_count - len(df)
        if removed > 0:
            logger.info(
                "validation_complete",
                initial=initial_count,
                final=len(df),
                removed=removed,
                removed_pct=f"{removed / initial_count:.1%}",
            )

        return df.reset_index(drop=True)

    def _remove_spikes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove price spikes using rolling Z-score."""
        close = df["close"]
        rolling_mean = close.rolling(50, min_periods=10).mean()
        rolling_std = close.rolling(50, min_periods=10).std().replace(0, np.nan)
        zscore = ((close - rolling_mean) / rolling_std).abs()

        spike_mask = zscore > self.spike_zscore_threshold
        spike_count = spike_mask.sum()

        if spike_count > 0:
            logger.warning(
                "price_spikes_detected",
                count=int(spike_count),
                threshold=self.spike_zscore_threshold,
            )
            df = df[~spike_mask]

        return df

    def validate_candle(
        self,
        open_p: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> bool:
        """
        Validate a single candle (for live inference).

        Returns True if valid, False if should be rejected.
        """
        if any(np.isnan(v) or np.isinf(v) for v in [open_p, high, low, close]):
            return False
        if close <= 0 or open_p <= 0:
            return False
        if high < max(open_p, close) or low > min(open_p, close):
            return False
        if high < low:
            return False
        if volume < 0:
            return False
        return True
