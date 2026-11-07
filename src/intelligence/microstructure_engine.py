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

    def evaluate(self, direction: str, obi: float = 0.0) -> tuple[float, str]:
        """
        Evaluate if current order flow delta confirms direction or signals an absorption trap,
        and incorporate L2 Order Book Imbalance (OBI).
        Returns: (conviction_delta: float, summary: str)
        """
        dir_u = direction.upper()
        delta_score = 0.0
        reasons = []

        if len(self._deltas) >= 100:
            recent = list(self._deltas)[-300:]
            p_now = recent[-1][0]
            p_first = recent[0][0]
            p_change_pct = ((p_now - p_first) / max(p_first, 0.001)) * 100.0

            delta_sum = sum(d[1] for d in recent)
            total_vol = sum(abs(d[1]) for d in recent)
            delta_ratio = delta_sum / max(total_vol, 0.0001)

            # 1. Delta Confirmation
            if dir_u in ("BUY", "LONG") and delta_ratio > 0.20 and p_change_pct > 0:
                delta_score += 0.04
                reasons.append(f"CVD Confirmed (+{delta_ratio:.0%})")
            elif dir_u in ("SELL", "SHORT") and delta_ratio < -0.20 and p_change_pct < 0:
                delta_score += 0.04
                reasons.append(f"CVD Confirmed ({delta_ratio:.0%})")

            # 2. Absorption Trap Detection (Divergence)
            if dir_u in ("BUY", "LONG") and p_change_pct < -0.15 and delta_ratio > 0.25:
                delta_score += 0.06
                reasons.append(f"Bullish Absorption Trap (+{delta_ratio:.0%})")
            elif dir_u in ("SELL", "SHORT") and p_change_pct > 0.15 and delta_ratio < -0.25:
                delta_score += 0.06
                reasons.append(f"Bearish Absorption Trap ({delta_ratio:.0%})")

            # 3. Exhaustion Warning: Buying into heavy negative delta
            elif dir_u in ("BUY", "LONG") and delta_ratio < -0.35:
                delta_score -= 0.05
                reasons.append(f"Heavy Sell Aggression ({delta_ratio:.0%})")
            elif dir_u in ("SELL", "SHORT") and delta_ratio > 0.35:
                delta_score -= 0.05
                reasons.append(f"Heavy Buy Aggression (+{delta_ratio:.0%})")

        # 4. L2 Order Book Imbalance (OBI) Alignment
        if obi != 0.0:
            if dir_u in ("BUY", "LONG"):
                if obi >= 0.28:
                    delta_score += 0.03
                    reasons.append(f"L2 Bid Support (OBI {obi:+.2f})")
                elif obi <= -0.35:
                    delta_score -= 0.04
                    reasons.append(f"L2 Ask Wall Overhead (OBI {obi:+.2f})")
            elif dir_u in ("SELL", "SHORT"):
                if obi <= -0.28:
                    delta_score += 0.03
                    reasons.append(f"L2 Ask Pressure (OBI {obi:+.2f})")
                elif obi >= 0.35:
                    delta_score -= 0.04
                    reasons.append(f"L2 Bid Floor (OBI {obi:+.2f})")

        total_delta = round(max(-0.08, min(0.08, delta_score)), 3)
        summary = " | ".join(reasons) if reasons else "Neutral Microstructure"
        return total_delta, summary

