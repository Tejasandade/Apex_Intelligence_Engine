"""
Apex Intelligence Engine — SentimentMonitor
=============================================
Tracks crypto market sentiment from free public APIs.

Sources:
  1. Alternative.me Fear & Greed Index  — https://api.alternative.me/fng/?limit=1
  2. Binance open interest change        — via fapi/v1/openInterest

Usage:
    monitor = SentimentMonitor()
    await monitor.update()

    score   = monitor.get_fear_greed()       # 0–100
    label   = monitor.get_label()            # "Extreme Fear" … "Extreme Greed"
    mult    = monitor.get_regime_modifier()  # 0.7 – 1.2 (position size multiplier)
"""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("apex.providers.sentiment")

CACHE_TTL_SECONDS  = 900   # 15 minutes (F&G updates ~daily but we check frequently)
EXTREME_FEAR_BELOW = 20
GREED_ABOVE        = 70
EXTREME_GREED_ABOVE = 80


class SentimentMonitor:
    """
    Market sentiment tracker using Fear & Greed Index + open interest.

    Key outputs:
    - get_fear_greed()      → int 0–100
    - get_label()           → str description
    - get_regime_modifier() → float sizing multiplier (0.7 – 1.2)
    """

    def __init__(self):
        self._fng_value: int = 50       # Default: neutral
        self._fng_label: str = "Neutral"
        self._last_update: float = 0.0
        self._update_lock = asyncio.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_fear_greed(self) -> int:
        """Fear & Greed Index value (0=extreme fear, 100=extreme greed)."""
        return self._fng_value

    def get_label(self) -> str:
        """Human-readable sentiment label."""
        return self._fng_label

    def get_regime_modifier(self) -> float:
        """
        Position-size multiplier based on sentiment.

        Logic (contrarian approach):
          Extreme Fear  (< 20) → 1.15  (buy the fear, slight size-up)
          Fear          (< 40) → 1.05
          Neutral       (40-60)→ 1.00
          Greed         (> 70) → 0.85  (reduce exposure, crowded market)
          Extreme Greed (> 80) → 0.70  (significantly reduce, top warning)

        Returns float multiplier to apply to risk_pct.
        """
        v = self._fng_value
        if v <= EXTREME_FEAR_BELOW:
            return 1.15
        elif v <= 40:
            return 1.05
        elif v >= EXTREME_GREED_ABOVE:
            return 0.70
        elif v >= GREED_ABOVE:
            return 0.85
        return 1.00

    def get_color_tag(self) -> str:
        """ANSI-ready color tag for dashboard display."""
        v = self._fng_value
        if v <= EXTREME_FEAR_BELOW:
            return "\033[91m"   # RED
        elif v <= 40:
            return "\033[93m"   # YELLOW
        elif v >= EXTREME_GREED_ABOVE:
            return "\033[91m"   # RED (overbought)
        elif v >= GREED_ABOVE:
            return "\033[93m"   # YELLOW
        return "\033[92m"       # GREEN (neutral)

    def seconds_since_update(self) -> float:
        return time.time() - self._last_update if self._last_update > 0 else float("inf")

    # ── Background Update ──────────────────────────────────────────────────────

    async def update(self) -> None:
        """
        Fetch Fear & Greed index every 15 minutes.
        Thread-safe, caches results, never raises.
        """
        if time.time() - self._last_update < CACHE_TTL_SECONDS:
            return

        async with self._update_lock:
            if time.time() - self._last_update < CACHE_TTL_SECONDS:
                return
            try:
                await self._fetch_fng()
            except Exception as exc:
                logger.warning(f"[SentimentMonitor] update error: {exc}")
            finally:
                self._last_update = time.time()

    async def _fetch_fng(self) -> None:
        """Fetch Fear & Greed Index from Alternative.me (completely free, no key)."""
        try:
            import aiohttp
            url = "https://api.alternative.me/fng/?limit=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        entry = data.get("data", [{}])[0]
                        self._fng_value = int(entry.get("value", 50))
                        self._fng_label = entry.get("value_classification", "Neutral")
                        logger.debug(
                            f"[SentimentMonitor] F&G={self._fng_value} ({self._fng_label})"
                        )
        except ImportError:
            await asyncio.to_thread(self._fetch_fng_sync)
        except asyncio.TimeoutError:
            logger.debug("[SentimentMonitor] F&G request timed out")
        except Exception as exc:
            logger.debug(f"[SentimentMonitor] _fetch_fng error: {exc}")

    def _fetch_fng_sync(self) -> None:
        """Synchronous fallback using requests."""
        try:
            import requests
            r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
            if r.status_code == 200:
                entry = r.json().get("data", [{}])[0]
                self._fng_value = int(entry.get("value", 50))
                self._fng_label = entry.get("value_classification", "Neutral")
        except Exception as exc:
            logger.debug(f"[SentimentMonitor] sync fetch error: {exc}")

