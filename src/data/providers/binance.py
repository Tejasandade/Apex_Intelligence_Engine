"""
Apex Intelligence Engine V5 — Binance Historical Data Provider
================================================================
Fetches historical candle data from Binance Futures REST API.
Direct httpx calls — no ccxt overhead or geo-restriction issues.

Used for:
- Training data download
- Backtesting data
- (Phase 1) Live WebSocket streaming will be added here
"""

from __future__ import annotations

import time

import httpx
import pandas as pd

from src.core.logging import get_logger
from src.data.providers.base import MarketDataProvider

logger = get_logger("apex.data.binance")

BINANCE_FUTURES_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


class BinanceHistoricalProvider(MarketDataProvider):
    """
    Binance Futures historical data provider.
    Downloads 1-minute candles via the REST klines endpoint.
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "Binance Futures"

    async def connect(self) -> None:
        """Create the HTTP client."""
        self._client = httpx.AsyncClient(timeout=30.0)
        logger.info("provider_connected", provider=self.name)

    async def disconnect(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("provider_disconnected", provider=self.name)

    async def fetch_historical_candles(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical candles from Binance Futures.

        Automatically paginates for large requests (limit > 1000).
        Returns a clean DataFrame with deduplicated, sorted data.
        """
        if self._client is None:
            await self.connect()

        all_rows: list = []
        remaining = limit
        current_end = end_time or int(time.time() * 1000)

        batch_size = min(1000, limit)  # Binance max is 1500, 1000 is safe

        logger.info(
            "fetch_candles_start",
            symbol=symbol,
            interval=interval,
            total_requested=limit,
        )

        while remaining > 0:
            params: dict = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(batch_size, remaining),
                "endTime": current_end,
            }
            if start_time is not None and not all_rows:
                params["startTime"] = start_time

            try:
                resp = await self._client.get(
                    BINANCE_FUTURES_KLINES_URL, params=params
                )
                resp.raise_for_status()
                candles = resp.json()
            except httpx.HTTPStatusError as e:
                logger.error(
                    "fetch_candles_http_error",
                    symbol=symbol,
                    status=e.response.status_code,
                    body=e.response.text[:200],
                )
                break
            except Exception as e:
                logger.error("fetch_candles_error", symbol=symbol, error=str(e))
                break

            if not candles:
                break

            all_rows = candles + all_rows  # Prepend older data
            remaining -= len(candles)
            current_end = int(candles[0][0]) - 1  # Step before earliest

            logger.debug(
                "fetch_candles_batch",
                batch_size=len(candles),
                total_collected=len(all_rows),
                remaining=remaining,
            )

            if len(candles) < batch_size:
                break  # No more data available

        if not all_rows:
            logger.warning("fetch_candles_empty", symbol=symbol)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Parse Binance format: [open_time, o, h, l, c, vol, close_time, ...]
        df = pd.DataFrame(
            all_rows[-limit:],
            columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "_ct", "_cq", "_nt", "_tbv", "_tqv", "_ignore",
            ],
        ).drop(columns=["_ct", "_cq", "_nt", "_tbv", "_tqv", "_ignore"])

        # Clean up
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        logger.info(
            "fetch_candles_complete",
            symbol=symbol,
            candles=len(df),
            start=pd.to_datetime(df["timestamp"].iloc[0], unit="ms").isoformat(),
            end=pd.to_datetime(df["timestamp"].iloc[-1], unit="ms").isoformat(),
        )

        return df
