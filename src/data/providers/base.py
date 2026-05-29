"""
Apex Intelligence Engine V5 — Abstract Market Data Provider
=============================================================
Every data source (Binance, Dhan, OANDA, CSV files) implements this interface.
The rest of the system NEVER talks to an exchange directly — only through providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MarketDataProvider(ABC):
    """
    Abstract interface for all market data sources.

    Implementations handle exchange-specific authentication, rate limits,
    data formats, and error handling. The consumer receives clean,
    normalized DataFrames regardless of the source.
    """

    @abstractmethod
    async def fetch_historical_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV candle data.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT", "BANKNIFTY").
            interval: Candle interval (e.g., "1m", "5m", "15m", "1h").
            limit: Maximum number of candles to fetch.
            start_time: Start time as Unix milliseconds (optional).
            end_time: End time as Unix milliseconds (optional).

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            - timestamp: Unix ms (int64) or datetime
            - open/high/low/close: float64
            - volume: float64
        """
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the data source."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close the connection."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g., 'Binance Futures')."""
        ...
