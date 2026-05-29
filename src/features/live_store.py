"""
Apex Intelligence Engine V5 — Live FeatureStore
==================================================
Rolling-window feature computation for live trading.

CRITICAL DESIGN: This produces the EXACT same features as the backtest
FeatureStore. Train/serve parity is the #1 requirement — any difference
means the model sees different inputs live vs training and will fail.

How it works:
1. Maintains a rolling buffer of the last N candles (default: 200)
2. On each new candle, appends to buffer and recomputes all features
3. Returns the feature vector for the LATEST bar only
4. Uses the same indicator functions as src.features.store.FeatureStore

Usage:
    live_store = LiveFeatureStore(market_type="crypto", buffer_size=200)
    live_store.warm_up(historical_candles_df)

    # On each new candle from WebSocket:
    features = live_store.update(candle_dict)
    # features is a 1-row DataFrame with 23 columns, ready for model.predict()
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from src.features.store import FeatureStore
from src.core.logging import get_logger

logger = get_logger("apex.features.live_store")


class LiveFeatureStore:
    """
    Rolling-window feature computation for live trading.

    Maintains a buffer of recent candles and recomputes features
    using the SAME FeatureStore pipeline as backtesting — ensuring
    train/serve parity.
    """

    def __init__(
        self,
        market_type: str = "crypto",
        buffer_size: int = 200,
    ):
        """
        Args:
            market_type: Market type for feature profiles ("crypto" or "india_equity").
            buffer_size: Number of candles to keep in the rolling buffer.
                         Must be >= max lookback of any indicator (default: 200).
        """
        self.market_type = market_type
        self.buffer_size = buffer_size
        self._feature_store = FeatureStore(market_type)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._warmed_up = False
        self._candles_processed = 0

    @property
    def is_warmed_up(self) -> bool:
        """True if enough candles have been buffered for reliable indicators."""
        return self._warmed_up

    @property
    def feature_columns(self) -> list[str]:
        """Feature column names (for model compatibility checks)."""
        return self._feature_store.feature_columns

    @property
    def num_features(self) -> int:
        return self._feature_store.num_features

    @property
    def buffer_length(self) -> int:
        return len(self._buffer)

    def warm_up(self, historical_data: pd.DataFrame) -> None:
        """
        Prime the buffer with historical candles.

        Call this ONCE at startup with the most recent N candles from
        the historical data provider.

        Args:
            historical_data: DataFrame with columns [timestamp, open, high, low, close, volume].
        """
        required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in historical_data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Take the last buffer_size rows
        recent = historical_data.tail(self.buffer_size)
        self._buffer.clear()

        for _, row in recent.iterrows():
            self._buffer.append({
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        self._warmed_up = len(self._buffer) >= 60  # Min warmup for indicators

        logger.info(
            "live_store_warmed_up",
            market_type=self.market_type,
            buffer_size=len(self._buffer),
            warmed_up=self._warmed_up,
        )

    def update(self, candle: dict[str, Any]) -> pd.DataFrame | None:
        """
        Process a new closed candle and return the latest feature vector.

        Args:
            candle: Dict with keys: timestamp, open, high, low, close, volume.

        Returns:
            A 1-row DataFrame with all features for the latest bar,
            or None if the buffer isn't warmed up yet.
        """
        # Append to buffer
        self._buffer.append({
            "timestamp": candle.get("timestamp", 0),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "close": float(candle.get("close", 0)),
            "volume": float(candle.get("volume", 0)),
        })
        self._candles_processed += 1

        if not self._warmed_up:
            if len(self._buffer) >= 60:
                self._warmed_up = True
                logger.info("live_store_warmup_complete", candles=len(self._buffer))
            else:
                return None

        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self._buffer))

        # Compute ALL features using the same pipeline as backtesting
        all_features = self._feature_store.build_features(df)

        if all_features.empty:
            return None

        # Return ONLY the last row (the new candle's features)
        latest = all_features.iloc[[-1]].reset_index(drop=True)

        # Sanity check: no NaN or Inf in features
        if latest.isna().any().any() or np.isinf(latest.values).any():
            nan_cols = latest.columns[latest.isna().any()].tolist()
            logger.warning(
                "live_features_nan",
                nan_columns=nan_cols,
                candles_processed=self._candles_processed,
            )
            # Replace NaN/Inf with 0 to prevent model crash
            latest = latest.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return latest

    def get_current_bar(self) -> dict[str, Any] | None:
        """Return the latest candle in the buffer."""
        if self._buffer:
            return self._buffer[-1]
        return None

    def get_buffer_as_df(self) -> pd.DataFrame:
        """Return the entire buffer as a DataFrame (for debugging)."""
        return pd.DataFrame(list(self._buffer))

    def get_stats(self) -> dict[str, Any]:
        """Return live store statistics."""
        return {
            "market_type": self.market_type,
            "buffer_size": self.buffer_size,
            "buffer_length": len(self._buffer),
            "warmed_up": self._warmed_up,
            "candles_processed": self._candles_processed,
            "num_features": self.num_features,
        }
