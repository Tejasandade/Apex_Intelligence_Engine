"""
Apex Intelligence Engine V5 — Multi-Timeframe Buffer
======================================================
Maintains rolling windows of synthetic 5m and 15m candles by
aggregating incoming 1m live candles.

Avoids the race conditions associated with subscribing to multiple
websocket streams by ensuring all higher timeframes are perfectly
derived from the base 1m stream.
"""

from __future__ import annotations

import pandas as pd
from src.features.indicators.technical import compute_adx

class MultiTimeframeBuffer:
    """
    Ingests 1m candles and maintains rolling 5m and 15m DataFrames.
    """

    def __init__(self, max_1m_candles: int = 1500):
        self._candles_1m: list[dict] = []
        self._max_1m_candles = max_1m_candles
        
        self._df_1m: pd.DataFrame = pd.DataFrame()
        self._df_5m: pd.DataFrame = pd.DataFrame()
        self._df_15m: pd.DataFrame = pd.DataFrame()
        self._df_1h: pd.DataFrame = pd.DataFrame()
        self._df_4h: pd.DataFrame = pd.DataFrame()
        self._ready = False

    def add_candle(self, candle: dict) -> None:
        """
        Add a closed 1m candle and immediately update the 5m and 15m buffers.
        Expects a dict with: timestamp, open, high, low, close, volume.
        """
        self._candles_1m.append(candle)
        if len(self._candles_1m) > self._max_1m_candles:
            self._candles_1m.pop(0)

        # Re-aggregate. For 1500 rows, this takes less than 2ms.
        df = pd.DataFrame(self._candles_1m)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        
        # Keep numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        self._df_1m = df

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "timestamp": "last"  # Keep the closing timestamp of the higher timeframe
        }
        
        # Generate synthetic higher timeframes
        self._df_5m = df.resample("5min").agg(agg_dict).dropna()
        self._df_15m = df.resample("15min").agg(agg_dict).dropna()

        # Ready when we have at least 15 candles on the 15m timeframe (for ADX/EMA)
        if len(self._df_15m) >= 15:
            self._ready = True

    def get_5m_df(self) -> pd.DataFrame:
        """Get the current 5m DataFrame."""
        return self._df_5m

    def get_15m_df(self) -> pd.DataFrame:
        """Get the current 15m DataFrame."""
        return self._df_15m

    def get_1h_df(self) -> pd.DataFrame:
        """Get the current 1h DataFrame."""
        return self._df_1h

    def get_4h_df(self) -> pd.DataFrame:
        """Get the current 4h DataFrame."""
        return self._df_4h

    def seed_direct(
        self,
        df_5m: pd.DataFrame | None = None,
        df_15m: pd.DataFrame | None = None,
        df_1h: pd.DataFrame | None = None,
        df_4h: pd.DataFrame | None = None,
    ) -> bool:
        """
        Directly populate 5m, 15m, 1h, and 4h buffers from pre-fetched historical klines.
        Eliminates the multi-candle warmup wait at startup.
        """
        if df_5m is not None and not df_5m.empty:
            d5 = df_5m.copy()
            if "datetime" not in d5.columns and "timestamp" in d5.columns:
                d5["datetime"] = pd.to_datetime(d5["timestamp"], unit="ms")
                d5.set_index("datetime", inplace=True)
            self._df_5m = d5

        if df_15m is not None and not df_15m.empty:
            d15 = df_15m.copy()
            if "datetime" not in d15.columns and "timestamp" in d15.columns:
                d15["datetime"] = pd.to_datetime(d15["timestamp"], unit="ms")
                d15.set_index("datetime", inplace=True)
            self._df_15m = d15
        elif self._df_5m is not None and len(self._df_5m) >= 3:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "timestamp": "last",
            }
            self._df_15m = self._df_5m.resample("15min").agg(agg_dict).dropna()

        if df_1h is not None and not df_1h.empty:
            d1h = df_1h.copy()
            if "datetime" not in d1h.columns and "timestamp" in d1h.columns:
                d1h["datetime"] = pd.to_datetime(d1h["timestamp"], unit="ms")
                d1h.set_index("datetime", inplace=True)
            self._df_1h = d1h
        elif self._df_15m is not None and len(self._df_15m) >= 4:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "timestamp": "last",
            }
            self._df_1h = self._df_15m.resample("1h").agg(agg_dict).dropna()

        if df_4h is not None and not df_4h.empty:
            d4h = df_4h.copy()
            if "datetime" not in d4h.columns and "timestamp" in d4h.columns:
                d4h["datetime"] = pd.to_datetime(d4h["timestamp"], unit="ms")
                d4h.set_index("datetime", inplace=True)
            self._df_4h = d4h
        elif self._df_1h is not None and len(self._df_1h) >= 4:
            agg_dict = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "timestamp": "last",
            }
            self._df_4h = self._df_1h.resample("4h").agg(agg_dict).dropna()

        self._ready = len(self._df_5m) >= 10 or len(self._df_15m) >= 5
        return self._ready

    def get_latest_15m_adx(self) -> float:
        """
        Compute and return the latest 14-period ADX on the 15m chart.
        Used by the engine to determine the macro trend (Swing vs Scalp mode).
        """
        if not self._ready or len(self._df_15m) < 15:
            return 0.0
            
        try:
            adx_series = compute_adx(self._df_15m, length=14)
            val = adx_series.iloc[-1]
            return float(val) if not pd.isna(val) else 0.0
        except Exception:
            return 0.0

    def is_ready(self) -> bool:
        """Returns True if the buffer has enough data for MTF indicators."""
        return self._ready

    def seed_from_historical(self, df_1m: pd.DataFrame) -> bool:
        """
        Bulk-load historical 1m candles to immediately populate all MTF buffers.
        Called at boot with data from the Oracle to avoid the 15+ minute warmup gap.

        Args:
            df_1m: DataFrame with 1m OHLCV data (up to 1500 rows).

        Returns:
            True if seeding met minimum candle requirements.
        """
        if df_1m.empty:
            return False

        # Ensure numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_1m.columns:
                df_1m[col] = pd.to_numeric(df_1m[col], errors="coerce").astype(float)

        # Convert each row to the candle dict format and add to buffer
        for _, row in df_1m.iterrows():
            candle = {
                "timestamp": row.get("timestamp", 0),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
            self._candles_1m.append(candle)

        # Trim to max size
        while len(self._candles_1m) > self._max_1m_candles:
            self._candles_1m.pop(0)

        # Rebuild all timeframes
        df = pd.DataFrame(self._candles_1m)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        self._df_1m = df

        agg_dict = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "timestamp": "last",
        }
        self._df_5m = df.resample("5min").agg(agg_dict).dropna()
        self._df_15m = df.resample("15min").agg(agg_dict).dropna()

        # Check minimum candle counts (from V6 response doc)
        min_ok = (
            len(self._df_1m) >= 200
            and len(self._df_5m) >= 50
            and len(self._df_15m) >= 15
        )
        self._ready = min_ok

        from src.core.logging import get_logger
        _logger = get_logger("apex.engine.mtf_buffer")
        _logger.info(
            "mtf_buffer_seeded",
            candles_1m=len(self._df_1m),
            candles_5m=len(self._df_5m),
            candles_15m=len(self._df_15m),
            ready=self._ready,
        )

        return self._ready

