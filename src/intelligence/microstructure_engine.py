"""
Apex Intelligence Engine — Microstructure & CVD Absorption Engine
==================================================================
Analyzes Cumulative Volume Delta (CVD), aggressor tick imbalance,
and passive order absorption traps to filter false breakouts.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

logger = logging.getLogger("apex.intelligence.microstructure")


class MicrostructureEngine:
    """
    Computes real-time Cumulative Volume Delta (CVD) and institutional absorption traps.
    """

    def __init__(self, symbol: str, window_ticks: int = 1500):
        self.symbol = symbol.upper()
        self.window_ticks = window_ticks
        self._deltas: deque[tuple[float, float]] = deque(maxlen=window_ticks)  # (price, delta_volume)
        self._cvd_running: float = 0.0
        self._recent_high_p: float = 0.0
        self._recent_low_p: float = float("inf")
        self._recent_cvd_high: float = 0.0
        self._recent_cvd_low: float = float("inf")

    def add_tick(self, price: float, qty: float, is_buyer_maker: bool) -> None:
        """
        is_buyer_maker=True means market sell hit a passive bid (-delta).
        is_buyer_maker=False means market buy lifted a passive ask (+delta).
        """
        delta = -qty if is_buyer_maker else qty
        self._cvd_running += delta
        self._deltas.append((price, delta))

    def evaluate(self, direction: str) -> tuple[float, str]:
        """
        Evaluate if current order flow delta confirms direction or signals an absorption trap.
        Returns: (conviction_delta: float, summary: str)
        """
        if len(self._deltas) < 100:
            return 0.0, "CVD warming up"

        recent = list(self._deltas)[-300:]
        p_now = recent[-1][0]
        p_first = recent[0][0]
        p_change_pct = ((p_now - p_first) / max(p_first, 0.001)) * 100.0

        delta_sum = sum(d[1] for d in recent)
        total_vol = sum(abs(d[1]) for d in recent)
        delta_ratio = delta_sum / max(total_vol, 0.0001)

        dir_u = direction.upper()

        # 1. Delta Confirmation
        if dir_u in ("BUY", "LONG") and delta_ratio > 0.20 and p_change_pct > 0:
            return +0.04, f"CVD Confirmation (Delta +{delta_ratio:.0%})"
        if dir_u in ("SELL", "SHORT") and delta_ratio < -0.20 and p_change_pct < 0:
            return +0.04, f"CVD Confirmation (Delta {delta_ratio:.0%})"

        # 2. Absorption Trap Detection (Divergence)
        # Price dropped, but buyers heavily absorbed aggressive selling (delta positive)
        if dir_u in ("BUY", "LONG") and p_change_pct < -0.15 and delta_ratio > 0.25:
            return +0.06, f"Bullish Absorption Trap (Sellers absorbed at lows, Delta +{delta_ratio:.0%})"

        # Price rallied, but sellers heavily absorbed aggressive buying (delta negative)
        if dir_u in ("SELL", "SHORT") and p_change_pct > 0.15 and delta_ratio < -0.25:
            return +0.06, f"Bearish Absorption Trap (Buyers absorbed at highs, Delta {delta_ratio:.0%})"

        # 3. Exhaustion Warning: Buying into negative delta
        if dir_u in ("BUY", "LONG") and delta_ratio < -0.35:
            return -0.05, f"Warning: Heavy Aggressive Selling (Delta {delta_ratio:.0%})"
        if dir_u in ("SELL", "SHORT") and delta_ratio > 0.35:
            return -0.05, f"Warning: Heavy Aggressive Buying (Delta +{delta_ratio:.0%})"

        return 0.0, f"CVD Neutral (Delta {delta_ratio:+.0%})"
