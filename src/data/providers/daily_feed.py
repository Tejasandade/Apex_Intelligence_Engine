"""
Apex Intelligence Engine — Daily Feed Provider
================================================
Maintains rolling daily + 4H candle windows using Binance REST API.
Polls every 15 minutes for 4H data, daily at market open for daily data.

Usage:
    feed = DailyFeedProvider()
    await feed.refresh("BTCUSDT")

    daily_df = feed.get_daily_candles("BTCUSDT")
    daily_atr = feed.daily_atr("BTCUSDT")
    trend = feed.daily_trend("BTCUSDT")  # "BULL", "BEAR", "NEUTRAL"
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.data.daily_feed")

DAILY_REFRESH_SECS = 3600       # refresh daily candles every 1h
H4_REFRESH_SECS    = 900        # refresh 4H candles every 15min
BINANCE_FAPI       = "https://fapi.binance.com/fapi/v1/klines"


class DailyFeedProvider:
    """
    Provides daily + 4H candle data for swing trade analysis.

    All data fetched via public Binance fapi (no API key needed).
    Gracefully degrades if network is unavailable.
    """

    def __init__(self):
        self._daily: dict[str, pd.DataFrame] = {}
        self._h4:    dict[str, pd.DataFrame] = {}
        self._last_daily_update: dict[str, float] = {}
        self._last_h4_update:    dict[str, float] = {}
        self._lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_daily_candles(self, symbol: str) -> pd.DataFrame:
        return self._daily.get(symbol.upper(), pd.DataFrame())

    def get_4h_candles(self, symbol: str) -> pd.DataFrame:
        return self._h4.get(symbol.upper(), pd.DataFrame())

    def daily_atr(self, symbol: str, period: int = 14) -> float:
        """Average True Range of daily candles."""
        df = self.get_daily_candles(symbol)
        if df.empty or len(df) < period + 1:
            return 0.0
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def daily_trend(self, symbol: str) -> str:
        """
        Classify daily trend using EMA 20 vs EMA 50.
        Returns: "BULL", "BEAR", or "NEUTRAL"
        """
        df = self.get_daily_candles(symbol)
        if df.empty or len(df) < 51:
            return "NEUTRAL"
        close = df["close"]
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        price = float(close.iloc[-1])
        if ema20 > ema50 and price > ema20:
            return "BULL"
        elif ema20 < ema50 and price < ema20:
            return "BEAR"
        return "NEUTRAL"

    def h4_trend(self, symbol: str) -> str:
        """4H trend direction using EMA crossover."""
        df = self.get_4h_candles(symbol)
        if df.empty or len(df) < 21:
            return "NEUTRAL"
        close = df["close"]
        ema9  = float(close.ewm(span=9,  adjust=False).mean().iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        return "BULL" if ema9 > ema21 else ("BEAR" if ema9 < ema21 else "NEUTRAL")

    def daily_support_resistance(self, symbol: str, lookback: int = 20) -> dict:
        """Key daily swing highs/lows as S/R levels."""
        df = self.get_daily_candles(symbol)
        if df.empty or len(df) < lookback:
            return {"support": 0.0, "resistance": 0.0}
        window = df.tail(lookback)
        return {
            "support":    round(float(window["low"].min()),  1),
            "resistance": round(float(window["high"].max()), 1),
        }

    def is_ready(self, symbol: str) -> bool:
        return not self.get_daily_candles(symbol).empty

    # ── Refresh ────────────────────────────────────────────────────────────────

    async def refresh(self, symbol: str) -> None:
        """Refresh daily + 4H candles if cache is stale. Safe to call frequently."""
        sym = symbol.upper()
        now = time.time()
        tasks = []
        if now - self._last_daily_update.get(sym, 0) >= DAILY_REFRESH_SECS:
            tasks.append(self._fetch_daily(sym))
        if now - self._last_h4_update.get(sym, 0) >= H4_REFRESH_SECS:
            tasks.append(self._fetch_h4(sym))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_daily(self, symbol: str) -> None:
        try:
            df = await self._klines(symbol, "1d", 90)
            if df is not None and not df.empty:
                self._daily[symbol] = df
                self._last_daily_update[symbol] = time.time()
                logger.debug(f"[DailyFeed] {symbol} daily: {len(df)} candles")
        except Exception as exc:
            logger.debug(f"[DailyFeed] daily fetch error: {exc}")

    async def _fetch_h4(self, symbol: str) -> None:
        try:
            df = await self._klines(symbol, "4h", 48)
            if df is not None and not df.empty:
                self._h4[symbol] = df
                self._last_h4_update[symbol] = time.time()
                logger.debug(f"[DailyFeed] {symbol} 4H: {len(df)} candles")
        except Exception as exc:
            logger.debug(f"[DailyFeed] 4H fetch error: {exc}")

    async def _klines(self, symbol: str, interval: str, limit: int) -> pd.DataFrame | None:
        """Fetch klines from Binance fapi, return parsed DataFrame."""
        url = f"{BINANCE_FAPI}?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                    if resp.status != 200:
                        return None
                    raw = await resp.json()
        except ImportError:
            raw = await asyncio.to_thread(self._klines_sync, url)
            if raw is None:
                return None
        except Exception as exc:
            logger.debug(f"[DailyFeed] klines request error: {exc}")
            return None

        return self._parse_klines(raw)

    def _klines_sync(self, url: str):
        try:
            import requests
            r = requests.get(url, timeout=8)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    @staticmethod
    def _parse_klines(raw: list) -> pd.DataFrame:
        if not raw:
            return pd.DataFrame()
        cols = ["timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy", "tb_quote", "ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = df["timestamp"].astype("int64")
        return df[["timestamp", "open", "high", "low", "close", "volume"]].dropna()

