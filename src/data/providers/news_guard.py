"""
Apex Intelligence Engine — NewsGuard
======================================
Pre-trade blackout filter that prevents trading around high-impact economic events.

Data sources (priority order):
1. CryptoCompare News API  — uses CRYPTOCOMPARE_API_KEY from .env
2. Static economic calendar — FOMC/CPI/NFP hardcoded dates as fallback
3. Binance status API       — exchange maintenance check

Blackout window: 30 min before + 60 min after any high-impact event.
Cache TTL: 5 minutes (non-blocking).
"""

from __future__ import annotations

import asyncio
import os
import time
import datetime
import logging
from typing import Optional

logger = logging.getLogger("apex.providers.news_guard")

# ── Static FOMC schedule 2025-2026 (UTC dates) ──────────────────────────────
# Source: Federal Reserve announced dates
_FOMC_SCHEDULE: list[str] = [
    # 2025
    "2025-01-29 19:00",
    "2025-03-19 18:00",
    "2025-05-07 18:00",
    "2025-06-18 18:00",
    "2025-07-30 18:00",
    "2025-09-17 18:00",
    "2025-10-29 18:00",
    "2025-12-10 19:00",
    # 2026
    "2026-01-28 19:00",
    "2026-03-18 18:00",
    "2026-04-29 18:00",
    "2026-06-17 18:00",
    "2026-07-29 18:00",
    "2026-09-16 18:00",
    "2026-10-28 18:00",
    "2026-12-09 19:00",
]

# CPI releases (typically 8:30 AM ET = 13:30 UTC, second Tuesday of month)
_CPI_SCHEDULE: list[str] = [
    "2025-01-15 13:30",
    "2025-02-12 13:30",
    "2025-03-12 12:30",
    "2025-04-10 12:30",
    "2025-05-13 12:30",
    "2025-06-11 12:30",
    "2025-07-15 12:30",
    "2025-08-12 12:30",
    "2025-09-10 12:30",
    "2025-10-15 12:30",
    "2025-11-13 13:30",
    "2025-12-10 13:30",
    "2026-01-14 13:30",
    "2026-02-11 13:30",
    "2026-03-11 13:30",
    "2026-04-15 12:30",
    "2026-05-13 12:30",
    "2026-06-10 12:30",
    "2026-07-14 12:30",
    "2026-08-12 12:30",
    "2026-09-09 12:30",
    "2026-10-14 12:30",
    "2026-11-12 13:30",
    "2026-12-09 13:30",
]

# NFP releases (first Friday of month, 8:30 AM ET = 13:30 UTC)
_NFP_SCHEDULE: list[str] = [
    "2025-01-10 13:30",
    "2025-02-07 13:30",
    "2025-03-07 13:30",
    "2025-04-04 12:30",
    "2025-05-02 12:30",
    "2025-06-06 12:30",
    "2025-07-03 12:30",
    "2025-08-01 12:30",
    "2025-09-05 12:30",
    "2025-10-03 12:30",
    "2025-11-07 13:30",
    "2025-12-05 13:30",
    "2026-01-09 13:30",
    "2026-02-06 13:30",
    "2026-03-06 13:30",
    "2026-04-03 12:30",
    "2026-05-01 12:30",
    "2026-06-05 12:30",
    "2026-07-02 12:30",
    "2026-08-07 12:30",
    "2026-09-04 12:30",
    "2026-10-02 12:30",
    "2026-11-06 13:30",
    "2026-12-04 13:30",
]

# BTC/crypto keywords to filter CryptoCompare news for relevance
_BTC_KEYWORDS = {
    "bitcoin", "btc", "crypto", "cryptocurrency", "fed", "fomc",
    "interest rate", "inflation", "cpi", "nfp", "jobs", "etf", "sec",
    "regulation", "hack", "exploit", "crash", "dump", "pump", "liquidat",
    "binance", "exchange", "whale", "mt gox", "grayscale", "blackrock",
}

BLACKOUT_BEFORE_MINUTES = 30
BLACKOUT_AFTER_MINUTES = 60
CACHE_TTL_SECONDS = 300  # 5 minutes


def _parse_static_events() -> list[dict]:
    """Parse the hardcoded FOMC/CPI/NFP schedules into event dicts."""
    events = []
    for dt_str in _FOMC_SCHEDULE:
        events.append({
            "time": datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=datetime.timezone.utc
            ),
            "name": "FOMC Rate Decision",
            "impact": "HIGH",
            "source": "static",
        })
    for dt_str in _CPI_SCHEDULE:
        events.append({
            "time": datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=datetime.timezone.utc
            ),
            "name": "US CPI Release",
            "impact": "HIGH",
            "source": "static",
        })
    for dt_str in _NFP_SCHEDULE:
        events.append({
            "time": datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(
                tzinfo=datetime.timezone.utc
            ),
            "name": "Non-Farm Payrolls (NFP)",
            "impact": "HIGH",
            "source": "static",
        })
    return events


class NewsGuard:
    """
    Prevents trading around high-impact economic events.

    Data sources (in priority order):
    1. CryptoCompare News API (crypto-specific) — CRYPTOCOMPARE_API_KEY from .env
    2. Static economic calendar (FOMC, CPI, NFP — hardcoded monthly)
    3. Binance status API (exchange maintenance check)

    How it works:
    - is_blackout(symbol) → (True/False, reason_string)
    - Blackout window: 30 min before + 60 min after any high-impact event
    - Updates every 5 minutes (cached, non-blocking)
    """

    def __init__(self):
        self._api_key: str = os.getenv("CRYPTOCOMPARE_API_KEY", "")
        self._live_events: list[dict] = []          # Events from live API
        self._static_events: list[dict] = _parse_static_events()
        self._maintenance: bool = False             # Binance maintenance flag
        self._last_update: float = 0.0
        self._update_lock = asyncio.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_blackout(self, symbol: str = "BTCUSDT") -> tuple[bool, str]:
        """
        Check if we are inside a news blackout window.

        Returns:
            (True, reason) if trading is blocked; (False, "") otherwise.
        """
        if self._maintenance:
            return True, "Binance exchange maintenance in progress"

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        all_events = self._static_events + self._live_events

        for event in all_events:
            evt_time: datetime.datetime = event["time"]
            before_window = evt_time - datetime.timedelta(minutes=BLACKOUT_BEFORE_MINUTES)
            after_window  = evt_time + datetime.timedelta(minutes=BLACKOUT_AFTER_MINUTES)

            if before_window <= now_utc <= after_window:
                mins_to_event = (evt_time - now_utc).total_seconds() / 60.0
                if mins_to_event > 0:
                    reason = f"{event['name']} in {mins_to_event:.0f}m (pre-event blackout)"
                else:
                    reason = f"{event['name']} {abs(mins_to_event):.0f}m ago (post-event blackout)"
                return True, reason

        return False, ""

    def get_next_event(self) -> dict | None:
        """Return the next upcoming high-impact event (time, name, impact)."""
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        all_events = self._static_events + self._live_events
        future = [e for e in all_events if e["time"] > now_utc]
        if not future:
            return None
        return min(future, key=lambda e: e["time"])

    def seconds_since_update(self) -> float:
        return time.time() - self._last_update if self._last_update > 0 else float("inf")

    # ── Background Updates ────────────────────────────────────────────────────

    async def update_events(self) -> None:
        """
        Fetch upcoming events every 5 mins via aiohttp.
        Uses CryptoCompare /data/v2/news/ and Binance status API.
        Results are cached; safe to call on every candle.
        """
        # Respect cache TTL
        if time.time() - self._last_update < CACHE_TTL_SECONDS:
            return

        async with self._update_lock:
            # Double-check after acquiring lock
            if time.time() - self._last_update < CACHE_TTL_SECONDS:
                return

            try:
                await asyncio.gather(
                    self._fetch_cryptocompare_news(),
                    self._check_binance_status(),
                )
            except Exception as e:
                logger.warning(f"[NewsGuard] update_events error: {e}")
            finally:
                self._last_update = time.time()

    async def _fetch_cryptocompare_news(self) -> None:
        """Fetch and filter recent CryptoCompare news for high-impact items."""
        try:
            import aiohttp
        except ImportError:
            logger.debug("[NewsGuard] aiohttp not available, skipping CryptoCompare fetch")
            return

        try:
            url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=popular"
            headers = {}
            if self._api_key:
                headers["authorization"] = f"Apikey {self._api_key}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status != 200:
                        logger.debug(f"[NewsGuard] CryptoCompare status {resp.status}")
                        return
                    data = await resp.json()

            articles = data.get("Data", [])
            new_events: list[dict] = []
            now_utc = datetime.datetime.now(datetime.timezone.utc)

            for article in articles:
                title = (article.get("title", "") + " " + article.get("body", "")).lower()

                # Only consider articles with high-impact keywords
                is_high_impact = any(kw in title for kw in [
                    "fomc", "fed rate", "interest rate decision", "cpi release",
                    "inflation data", "nfp", "non-farm", "jobs report",
                    "emergency", "hack", "exploit", "banned", "shutdown",
                    "regulatory crackdown", "sec charges", "systemic risk",
                ])
                if not is_high_impact:
                    continue

                # Has crypto/BTC relevance?
                is_relevant = any(kw in title for kw in _BTC_KEYWORDS)
                if not is_relevant:
                    continue

                published_ts = article.get("published_on", 0)
                if published_ts:
                    pub_dt = datetime.datetime.fromtimestamp(
                        published_ts, tz=datetime.timezone.utc
                    )
                    # Treat article as a potential event: impose a 60-minute
                    # blackout window centred on publication time
                    effective_time = pub_dt + datetime.timedelta(minutes=5)  # slight delay
                    if abs((effective_time - now_utc).total_seconds()) < 7200:  # within 2h
                        new_events.append({
                            "time": effective_time,
                            "name": article.get("title", "High-Impact Crypto News")[:80],
                            "impact": "HIGH",
                            "source": "cryptocompare",
                        })

            self._live_events = new_events
            logger.debug(f"[NewsGuard] {len(new_events)} live events cached from CryptoCompare")

        except asyncio.TimeoutError:
            logger.debug("[NewsGuard] CryptoCompare request timed out")
        except Exception as e:
            logger.warning(f"[NewsGuard] CryptoCompare fetch error: {e}")

    async def _check_binance_status(self) -> None:
        """Check Binance status page for exchange maintenance."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://www.binancezh.io/api/v3/ping",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    self._maintenance = resp.status != 200
        except asyncio.TimeoutError:
            # Timeout = assume exchange is reachable (don't block trading on a timeout)
            self._maintenance = False
        except Exception as e:
            logger.debug(f"[NewsGuard] Binance status check failed: {e}")
            self._maintenance = False
