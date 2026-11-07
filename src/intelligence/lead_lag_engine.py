"""
Apex Intelligence Engine — Cross-Asset Lead-Lag & Spillover Engine
==================================================================
Exploits statistical lead-lag dynamics between crypto assets (BTC leads ETH)
and macroeconomic currency drivers (US Dollar DXY inversion with Gold).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("apex.intelligence.lead_lag")


class LeadLagEngine:
    """
    Computes cross-asset momentum spillovers and macro currency alignment.
    """

    def __init__(self):
        self._price_history: dict[str, list[tuple[float, float]]] = {}  # symbol -> [(timestamp, price)]
        self._diagnostics: dict[str, Any] = {
            "btc_15m_return_pct": 0.0,
            "eth_15m_return_pct": 0.0,
            "eth_spillover_bias": "NEUTRAL",
            "gold_dxy_bias": "NEUTRAL",
        }

    def update_price(self, symbol: str, price: float, timestamp: float) -> None:
        sym = symbol.upper()
        if sym not in self._price_history:
            self._price_history[sym] = []

        history = self._price_history[sym]
        history.append((timestamp, price))

        # Keep last 30 minutes of history
        cutoff = timestamp - 1800.0
        while history and history[0][0] < cutoff:
            history.pop(0)

    def compute_spillover(self, dxy_trend: str = "USD_NEUTRAL", btc_ofi: float = 0.0) -> dict[str, Any]:
        """
        Evaluate BTC velocity vs ETH velocity over the last 15 minutes.
        """
        now_btc = self._get_return_15m("BTCUSDT")
        now_eth = self._get_return_15m("ETHUSDT")

        self._diagnostics["btc_15m_return_pct"] = round(now_btc, 2)
        self._diagnostics["eth_15m_return_pct"] = round(now_eth, 2)

        # 1. BTC-to-ETH Spillover Alpha
        # If BTC surged > +0.70% with positive OFI (> +0.50), and ETH lagged (< +0.25%):
        eth_bias = "NEUTRAL"
        if now_btc >= 0.70 and btc_ofi >= 0.50 and now_eth < 0.25:
            eth_bias = "SPILLOVER_BULL"
        elif now_btc <= -0.70 and btc_ofi <= -0.50 and now_eth > -0.25:
            eth_bias = "SPILLOVER_BEAR"

        self._diagnostics["eth_spillover_bias"] = eth_bias

        # 2. Gold vs DXY Inversion Alpha
        gold_bias = "NEUTRAL"
        if dxy_trend in ("USD_WEAKENING", "DXY_BEARISH"):
            gold_bias = "DXY_TAILWIND_BULL"
        elif dxy_trend in ("USD_SURGING", "DXY_BULLISH"):
            gold_bias = "DXY_HEADWIND_BEAR"

        self._diagnostics["gold_dxy_bias"] = gold_bias
        return self._diagnostics

    def _get_return_15m(self, symbol: str) -> float:
        history = self._price_history.get(symbol, [])
        if len(history) < 2:
            return 0.0
        cur_p = history[-1][1]
        t_target = history[-1][0] - 900.0  # 15 minutes ago
        # Find closest price near 15m ago
        closest_p = history[0][1]
        min_diff = abs(history[0][0] - t_target)
        for t, p in history:
            diff = abs(t - t_target)
            if diff < min_diff:
                min_diff = diff
                closest_p = p

        if closest_p <= 0:
            return 0.0
        return ((cur_p - closest_p) / closest_p) * 100.0

    def evaluate_synergy(self, symbol: str, direction: str) -> tuple[float, str]:
        """
        Evaluate if cross-asset momentum or macro DXY supports the trade.
        Returns: (conviction_delta: float, reason: str)
        """
        sym = symbol.upper()
        dir_u = direction.upper()

        if "ETH" in sym:
            spill = self._diagnostics.get("eth_spillover_bias", "NEUTRAL")
            if dir_u in ("BUY", "LONG") and spill == "SPILLOVER_BULL":
                return +0.05, f"BTC Lead-Lag Spillover (+{self._diagnostics['btc_15m_return_pct']}% BTC burst)"
            if dir_u in ("SELL", "SHORT") and spill == "SPILLOVER_BEAR":
                return +0.05, f"BTC Lead-Lag Spillover ({self._diagnostics['btc_15m_return_pct']}% BTC drop)"

        elif "GOLD" in sym or "XAU" in sym or "PAXG" in sym:
            dxy_b = self._diagnostics.get("gold_dxy_bias", "NEUTRAL")
            if dir_u in ("BUY", "LONG") and dxy_b == "DXY_TAILWIND_BULL":
                return +0.05, "USD Weakening Tailwinds for Gold"
            if dir_u in ("SELL", "SHORT") and dxy_b == "DXY_HEADWIND_BEAR":
                return +0.05, "USD Surging Headwinds for Gold"

        return 0.0, "NEUTRAL"

