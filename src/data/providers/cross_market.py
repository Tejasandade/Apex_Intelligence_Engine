"""
Apex Intelligence Engine — CrossMarketMonitor
==============================================
Monitors cross-market institutional signals:
1. US Dollar Index (DXY) momentum via inverse EUR/USD proxy.
2. ETH/BTC Relative Strength & Lead-Lag Divergence.

When USD is surging, crypto long breakouts frequently fail as liquidity is sucked into USD cash.
When ETH diverges negatively while BTC breaks out, it signals a fakeout bull trap.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp

logger = logging.getLogger("apex.providers.cross_market")


class CrossMarketMonitor:
    """
    Monitors Macro USD momentum and Crypto Market Breadth (ETH vs BTC).
    """

    def __init__(self):
        self._dxy_trend: str = "USD_NEUTRAL"
        self._dxy_momentum_pct: float = 0.0
        self._eth_trend: str = "ALIGNED"
        self._eth_btc_rel_pct: float = 0.0
        self._btc_15m_ret: float = 0.0
        self._eth_15m_ret: float = 0.0
        self._eur_15m_ret: float = 0.0
        self._last_update: float = 0.0
        self._update_lock = asyncio.Lock()

    @property
    def dxy_trend(self) -> str:
        return self._dxy_trend

    @property
    def dxy_momentum_pct(self) -> float:
        return self._dxy_momentum_pct

    @property
    def eth_trend(self) -> str:
        return self._eth_trend

    @property
    def eth_btc_rel_pct(self) -> float:
        return self._eth_btc_rel_pct

    def get_state(self) -> dict[str, Any]:
        """Return snapshot of current cross-market metrics."""
        return {
            "dxy_trend": self._dxy_trend,
            "dxy_momentum_pct": round(self._dxy_momentum_pct, 3),
            "eth_trend": self._eth_trend,
            "eth_btc_rel_pct": round(self._eth_btc_rel_pct, 3),
            "btc_15m_ret": round(self._btc_15m_ret, 3),
            "eth_15m_ret": round(self._eth_15m_ret, 3),
            "eur_15m_ret": round(self._eur_15m_ret, 3),
            "last_update": self._last_update,
        }

    async def update(self) -> None:
        """Fetch latest cross-asset price changes (MT5 primary with Binance fallback)."""
        async with self._update_lock:
            # 1. Try MT5 native queries in executor (zero network latency)
            loop = asyncio.get_event_loop()
            mt5_success = await loop.run_in_executor(None, self._update_from_mt5_sync)
            if mt5_success:
                self._last_update = time.time()
                return

            # 2. Fallback to Binance public REST
            await self._update_from_binance_async()
            self._last_update = time.time()

    def _update_from_mt5_sync(self) -> bool:
        """Query EURUSD, ETHUSD, and BTCUSD directly via MT5."""
        try:
            import MetaTrader5 as mt5
            if not mt5.initialize():
                return False

            # Helper to calculate 15m % change over last 3 candles
            def _get_return(symbol: str) -> Optional[float]:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 3)
                if rates is not None and len(rates) >= 2:
                    p_start = float(rates[0]["open"])
                    p_end = float(rates[-1]["close"])
                    if p_start > 0:
                        return ((p_end - p_start) / p_start) * 100.0
                return None

            eur_ret = _get_return("EURUSD")
            eth_ret = _get_return("ETHUSD#") or _get_return("ETHUSD") or _get_return("ETHEUR#")
            btc_ret = _get_return("BTCUSD#") or _get_return("BTCUSD")

            if btc_ret is not None and eth_ret is not None:
                self._btc_15m_ret = btc_ret
                self._eth_15m_ret = eth_ret
                self._eth_btc_rel_pct = eth_ret - btc_ret
                if self._eth_btc_rel_pct > 0.20:
                    self._eth_trend = "ETH_OUTPERFORMING"
                elif self._eth_btc_rel_pct < -0.20:
                    self._eth_trend = "ETH_UNDERPERFORMING"
                else:
                    self._eth_trend = "ALIGNED"

            if eur_ret is not None:
                self._eur_15m_ret = eur_ret
                # EUR/USD is the dominant inverse component of DXY (~57.6%)
                self._dxy_momentum_pct = -1.0 * eur_ret
                if self._dxy_momentum_pct > 0.08:
                    self._dxy_trend = "USD_SURGING"
                elif self._dxy_momentum_pct < -0.08:
                    self._dxy_trend = "USD_WEAKENING"
                else:
                    self._dxy_trend = "USD_NEUTRAL"

            return (btc_ret is not None and eth_ret is not None)
        except Exception as exc:
            logger.debug("mt5_cross_market_read_failed", error=str(exc))
            return False

    async def _update_from_binance_async(self) -> None:
        """Fallback: Query Binance Futures/Spot for ETH and BTC 15m return."""
        url = "https://fapi.binance.com/fapi/v1/klines"
        headers = {"User-Agent": "ApexEngine/7.2"}
        try:
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as session:
                # BTC klines
                async with session.get(url, params={"symbol": "BTCUSDT", "interval": "15m", "limit": "3"}) as r_btc:
                    if r_btc.status == 200:
                        data_btc = await r_btc.json()
                        if len(data_btc) >= 2:
                            o = float(data_btc[0][1])
                            c = float(data_btc[-1][4])
                            self._btc_15m_ret = ((c - o) / o) * 100.0

                # ETH klines
                async with session.get(url, params={"symbol": "ETHUSDT", "interval": "15m", "limit": "3"}) as r_eth:
                    if r_eth.status == 200:
                        data_eth = await r_eth.json()
                        if len(data_eth) >= 2:
                            o = float(data_eth[0][1])
                            c = float(data_eth[-1][4])
                            self._eth_15m_ret = ((c - o) / o) * 100.0

            self._eth_btc_rel_pct = self._eth_15m_ret - self._btc_15m_ret
            if self._eth_btc_rel_pct > 0.20:
                self._eth_trend = "ETH_OUTPERFORMING"
            elif self._eth_btc_rel_pct < -0.20:
                self._eth_trend = "ETH_UNDERPERFORMING"
            else:
                self._eth_trend = "ALIGNED"
        except Exception as exc:
            logger.debug("binance_cross_market_read_failed", error=str(exc))

