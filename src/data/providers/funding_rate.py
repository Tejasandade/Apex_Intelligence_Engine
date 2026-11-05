"""
Apex Intelligence Engine — FundingRateMonitor
===============================================
Monitors Binance perpetual funding rates and open interest.

Extreme positive funding = crowded longs  → lean SHORT / reduce LONG size
Extreme negative funding = crowded shorts → lean LONG  / reduce SHORT size

Endpoints used (all public, no API key needed):
  GET https://fapi.binance.com/fapi/v1/premiumIndex   — current funding rate
  GET https://fapi.binance.com/fapi/v1/openInterest   — open interest

Cache TTL: 5 minutes (non-blocking async).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

logger = logging.getLogger("apex.providers.funding_rate")

CACHE_TTL_SECONDS = 300          # 5 minutes
EXTREME_THRESHOLD = 0.001        # 0.1% — double the typical ~0.01% per 8h
HIGH_THRESHOLD    = 0.0005       # 0.05% — elevated, apply bias
NORMAL_THRESHOLD  = 0.0001       # 0.01% — typical neutral range


class FundingRateMonitor:
    """
    Monitors Binance perpetual funding rates.

    Usage:
        monitor = FundingRateMonitor()
        await monitor.update("BTCUSDT")

        bias  = monitor.get_signal_bias()   # -1.0 to +1.0
        rate  = monitor.get_funding_rate()  # e.g. 0.00012 = 0.012%
        label = monitor.get_label()         # "Bullish"/"Bearish"/"Neutral"
    """

    def __init__(self):
        self._rates: dict[str, float] = {}          # symbol → current funding rate
        self._oi:    dict[str, float] = {}           # symbol → open interest (USD)
        self._oi_prev: dict[str, float] = {}         # symbol → previous OI for delta
        self._last_update: dict[str, float] = {}
        self._update_lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """Current funding rate as a decimal (e.g. 0.00012 = 0.012%)."""
        return self._rates.get(symbol.upper(), 0.0)

    def get_funding_pct(self, symbol: str = "BTCUSDT") -> float:
        """Current funding rate as a percentage (e.g. 0.012)."""
        return self.get_funding_rate(symbol) * 100.0

    def get_signal_bias(self, symbol: str = "BTCUSDT") -> float:
        """
        Returns a directional bias float from -1.0 to +1.0.

          +0.3  → lean LONG  (crowded shorts → mean-revert up)
          -0.3  → lean SHORT (crowded longs  → mean-revert down)
           0.0  → neutral

        Use to scale position size, NOT as a primary signal.
        """
        rate = self.get_funding_rate(symbol)
        abs_rate = abs(rate)

        if abs_rate >= EXTREME_THRESHOLD:
            magnitude = -0.5 if rate > 0 else +0.5    # strong crowding → fade
        elif abs_rate >= HIGH_THRESHOLD:
            magnitude = -0.3 if rate > 0 else +0.3
        else:
            magnitude = 0.0

        return round(magnitude, 3)

    def is_extreme(self, symbol: str = "BTCUSDT") -> bool:
        """True if funding is at extreme levels (|rate| > 0.1%)."""
        return abs(self.get_funding_rate(symbol)) >= EXTREME_THRESHOLD

    def get_label(self, symbol: str = "BTCUSDT") -> str:
        """Human-readable funding rate label."""
        rate = self.get_funding_rate(symbol)
        abs_r = abs(rate)
        if abs_r >= EXTREME_THRESHOLD:
            return "Extreme Longs" if rate > 0 else "Extreme Shorts"
        elif abs_r >= HIGH_THRESHOLD:
            return "Elevated Longs" if rate > 0 else "Elevated Shorts"
        return "Neutral"

    def get_oi_change_pct(self, symbol: str = "BTCUSDT") -> float:
        """OI change % vs previous reading (+ve = OI increasing)."""
        cur  = self._oi.get(symbol.upper(), 0.0)
        prev = self._oi_prev.get(symbol.upper(), cur)
        if prev <= 0:
            return 0.0
        return round((cur - prev) / prev * 100.0, 3)

    def size_multiplier(self, symbol: str, direction: str) -> float:
        """
        Returns a position-size multiplier (0.5 – 1.0) based on funding.
        Applied BEFORE entering a trade to reduce size against the crowd.

        direction: "LONG" or "SHORT"
        """
        rate = self.get_funding_rate(symbol)
        # Trading WITH the funding crowd is risky (potential funding squeeze)
        if direction == "LONG" and rate >= HIGH_THRESHOLD:
            # Positive funding = crowd is long = we're going WITH crowd
            if abs(rate) >= EXTREME_THRESHOLD:
                return 0.5    # Half size — extreme crowding
            return 0.75
        if direction == "SHORT" and rate <= -HIGH_THRESHOLD:
            if abs(rate) >= EXTREME_THRESHOLD:
                return 0.5
            return 0.75
        return 1.0

    # ── Background Update ──────────────────────────────────────────────────────

    async def update(self, symbol: str = "BTCUSDT") -> None:
        """
        Fetch current funding rate and OI from Binance fapi.
        Respects cache TTL — safe to call on every candle.
        """
        sym = symbol.upper()
        last = self._last_update.get(sym, 0.0)
        if time.time() - last < CACHE_TTL_SECONDS:
            return

        async with self._update_lock:
            if time.time() - self._last_update.get(sym, 0.0) < CACHE_TTL_SECONDS:
                return
            try:
                await asyncio.gather(
                    self._fetch_funding(sym),
                    self._fetch_oi(sym),
                )
            except Exception as exc:
                logger.warning(f"[FundingRateMonitor] update error for {sym}: {exc}")
            finally:
                self._last_update[sym] = time.time()

    async def _fetch_funding(self, symbol: str) -> None:
        """Fetch funding rate from Binance premiumIndex endpoint."""
        try:
            import aiohttp
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        rate = float(data.get("lastFundingRate", 0.0))
                        self._rates[symbol] = rate
                        logger.debug(f"[FundingRateMonitor] {symbol} funding={rate:.6f}")
        except ImportError:
            # Fallback: use requests in a thread
            await asyncio.to_thread(self._fetch_funding_sync, symbol)
        except Exception as exc:
            logger.debug(f"[FundingRateMonitor] _fetch_funding error: {exc}")

    def _fetch_funding_sync(self, symbol: str) -> None:
        try:
            import requests
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                self._rates[symbol] = float(r.json().get("lastFundingRate", 0.0))
        except Exception as exc:
            logger.debug(f"[FundingRateMonitor] sync fetch error: {exc}")

    async def _fetch_oi(self, symbol: str) -> None:
        """Fetch open interest from Binance."""
        try:
            import aiohttp
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        new_oi = float(data.get("openInterest", 0.0))
                        self._oi_prev[symbol] = self._oi.get(symbol, new_oi)
                        self._oi[symbol] = new_oi
        except ImportError:
            await asyncio.to_thread(self._fetch_oi_sync, symbol)
        except Exception as exc:
            logger.debug(f"[FundingRateMonitor] _fetch_oi error: {exc}")

    def _fetch_oi_sync(self, symbol: str) -> None:
        try:
            import requests
            url = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                new_oi = float(r.json().get("openInterest", 0.0))
                self._oi_prev[symbol] = self._oi.get(symbol, new_oi)
                self._oi[symbol] = new_oi
        except Exception as exc:
            logger.debug(f"[FundingRateMonitor] sync OI error: {exc}")

