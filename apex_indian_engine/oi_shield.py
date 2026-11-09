"""
Apex Indian Engine — Open Interest (OI) & PCR Wall Shield
=========================================================
Prevents retail fakeout traps by tracking institutional Call/Put writing walls.
Blocks Call buying directly into massive Call resistance, and Put buying into Put support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("apex.indian.oi")


@dataclass
class OIMetrics:
    pcr: float                          # Put-Call Ratio = Total Put OI / Total Call OI
    total_call_oi: int
    total_put_oi: int
    max_call_wall_strike: float         # Strike with highest Call Open Interest (Resistance)
    max_put_wall_strike: float          # Strike with highest Put Open Interest (Support)
    sentiment: str                      # BULLISH | BEARISH | NEUTRAL | OVERBOUGHT | OVERSOLD


class OIShield:
    """
    Evaluates live Open Interest distributions across strike chains.
    """

    def __init__(self, wall_buffer_points: float = 60.0):
        self.wall_buffer = wall_buffer_points

    def analyze_chain(
        self,
        chain_data: list[dict[str, Any]],
        spot_price: float,
    ) -> OIMetrics:
        """
        Parses option chain list: [{strike, option_type, open_interest, ltp}, ...].
        """
        call_oi_by_strike: dict[float, int] = {}
        put_oi_by_strike: dict[float, int] = {}

        for c in chain_data:
            s = float(c.get("strike", 0.0))
            opt = str(c.get("option_type", "")).upper()
            oi = int(c.get("open_interest", 0))

            if opt == "CE":
                call_oi_by_strike[s] = call_oi_by_strike.get(s, 0) + oi
            elif opt == "PE":
                put_oi_by_strike[s] = put_oi_by_strike.get(s, 0) + oi

        tot_call = sum(call_oi_by_strike.values())
        tot_put = sum(put_oi_by_strike.values())
        pcr = round(tot_put / max(tot_call, 1), 2)

        # Find Overhead Call Resistance Wall (>= Spot) and Downside Put Support Wall (<= Spot)
        otm_calls = {s: oi for s, oi in call_oi_by_strike.items() if s >= spot_price}
        max_call_wall = max(otm_calls, key=otm_calls.get) if otm_calls else (
            max(call_oi_by_strike, key=call_oi_by_strike.get) if call_oi_by_strike else spot_price + 500
        )

        otm_puts = {s: oi for s, oi in put_oi_by_strike.items() if s <= spot_price}
        max_put_wall = max(otm_puts, key=otm_puts.get) if otm_puts else (
            max(put_oi_by_strike, key=put_oi_by_strike.get) if put_oi_by_strike else spot_price - 500
        )

        # Sentiment classification
        if pcr >= 1.45:
            sentiment = "OVERBOUGHT"
        elif pcr <= 0.60:
            sentiment = "OVERSOLD"
        elif pcr >= 1.10:
            sentiment = "BULLISH"
        elif pcr <= 0.90:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"

        return OIMetrics(
            pcr=pcr,
            total_call_oi=tot_call,
            total_put_oi=tot_put,
            max_call_wall_strike=max_call_wall,
            max_put_wall_strike=max_put_wall,
            sentiment=sentiment,
        )

    def validate_entry(
        self,
        direction: str,
        target_strike: float,
        spot_price: float,
        metrics: OIMetrics,
    ) -> tuple[bool, str]:
        """
        Validates if target strike is safe from institutional OI walls.
        """
        dir_up = direction.upper()

        if dir_up == "LONG":  # Buying Call (CE)
            # 1. PCR Exhaustion: Never buy calls into extreme overbought PCR
            if metrics.sentiment == "OVERBOUGHT":
                return False, f"OI SHIELD: Extreme Overbought PCR ({metrics.pcr:.2f} >= 1.45) — Exhaustion risk"

            # 2. Call Resistance Wall: Never buy calls within buffer of Max Call OI
            dist_to_call_wall = metrics.max_call_wall_strike - spot_price
            if 0 < dist_to_call_wall <= self.wall_buffer:
                return False, (
                    f"OI WALL SHIELD: Spot ({spot_price:.0f}) too close to institutional "
                    f"Call Writing Resistance Wall at {metrics.max_call_wall_strike:.0f} ({dist_to_call_wall:.0f} pts away)"
                )

        elif dir_up == "SHORT":  # Buying Put (PE)
            # 1. PCR Exhaustion: Never buy puts into extreme oversold PCR
            if metrics.sentiment == "OVERSOLD":
                return False, f"OI SHIELD: Extreme Oversold PCR ({metrics.pcr:.2f} <= 0.60) — Short squeeze risk"

            # 2. Put Support Wall: Never buy puts within buffer of Max Put OI
            dist_to_put_wall = spot_price - metrics.max_put_wall_strike
            if 0 < dist_to_put_wall <= self.wall_buffer:
                return False, (
                    f"OI WALL SHIELD: Spot ({spot_price:.0f}) too close to institutional "
                    f"Put Writing Support Floor at {metrics.max_put_wall_strike:.0f} ({dist_to_put_wall:.0f} pts away)"
                )

        return True, "CLEAR"

