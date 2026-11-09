"""
Apex Indian Engine — Institutional Key Levels & Liquidity Core
=============================================================
Calculates institutional reference levels for NIFTY & BANKNIFTY:
1. Previous Day High (PDH), Previous Day Low (PDL), Previous Day Close (PDC)
2. Central Pivot Range (CPR): Pivot (P), Top CPR (TC), Bottom CPR (BC)
   - Narrow CPR: High probability of trending expansion day
   - Wide CPR: High probability of rangebound mean-reverting day
3. Standard Floor Pivots: R1, R2, S1, S2
4. Opening Range Breakout (ORB: 09:15 - 09:30 AM IST): ORH & ORL
5. Real-Time Liquidity Sweep & Stop Hunt Detection:
   - Turtle Soup PDH Sweep Rejection (Institutional Short)
   - Turtle Soup PDL Sweep Reclaim (Institutional Long)
   - 15m Opening Range Breakout Momentum
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Optional

logger = logging.getLogger("apex.indian.key_levels")


@dataclass
class IndexKeyLevels:
    index_name: str
    pdh: float = 0.0                # Prev Day High
    pdl: float = 0.0                # Prev Day Low
    pdc: float = 0.0                # Prev Day Close
    pivot: float = 0.0              # Standard Pivot P
    tc: float = 0.0                 # Top CPR
    bc: float = 0.0                 # Bottom CPR
    cpr_width: float = 0.0          # abs(TC - BC)
    cpr_regime: str = "BALANCED"    # NARROW_TRENDING | WIDE_RANGING | BALANCED
    r1: float = 0.0
    s1: float = 0.0
    r2: float = 0.0
    s2: float = 0.0
    orh: float = 0.0                # 09:15 - 09:30 Opening Range High
    orl: float = 0.0                # 09:15 - 09:30 Opening Range Low
    orb_formed: bool = False
    active_sweep: str = "None"      # None | PDH_SWEEP_REJECT | PDL_SWEEP_RECLAIM | ORH_BREAKOUT | ORL_BREAKDOWN


class InstitutionalLevelsTracker:
    """
    Computes and updates institutional liquidity boundaries, CPR, and ORB for Indian Indices.
    """

    def __init__(self, index_name: str = "BANKNIFTY"):
        self.index_name = index_name.upper()
        self.levels = IndexKeyLevels(index_name=self.index_name)
        self._last_calc_date = ""

    def calculate_from_candles(
        self,
        candles_5m: list[dict[str, Any]],
        current_spot: float = 0.0,
    ) -> IndexKeyLevels:
        """
        Parses historical and intraday 5m candles to calculate PDH, PDL, PDC, CPR, and ORB.
        """
        if not candles_5m:
            return self.levels

        today_str = datetime.now().strftime("%Y-%m-%d")
        prev_day_candles = []
        today_candles = []

        for c in candles_5m:
            ts_str = str(c.get("timestamp", ""))
            if ts_str.startswith(today_str):
                today_candles.append(c)
            elif ts_str.startswith("20") and not ts_str.startswith("warmup"):
                prev_day_candles.append(c)

        # 1. Calculate Previous Day High, Low, Close
        if prev_day_candles:
            # Use last session's candles (up to 75 5m candles per NSE trading day)
            last_session = prev_day_candles[-75:]
            pdh = max(float(c["high"]) for c in last_session)
            pdl = min(float(c["low"]) for c in last_session)
            pdc = float(last_session[-1]["close"])
        else:
            # Fallback to earliest available candles or reasonable bounds
            base = float(candles_5m[0]["close"])
            step = 300.0 if "BANK" in self.index_name else 120.0
            pdh = base + step
            pdl = base - step
            pdc = base

        # 2. Central Pivot Range (CPR) & Standard Floor Pivots
        p = round((pdh + pdl + pdc) / 3.0, 2)
        bc = round((pdh + pdl) / 2.0, 2)
        tc = round((p - bc) + p, 2)
        cpr_w = round(abs(tc - bc), 2)
        
        # CPR width classification
        w_ratio = cpr_w / max(p, 1.0)
        if w_ratio <= 0.0020:
            cpr_regime = "NARROW (Trending Bias)"
        elif w_ratio >= 0.0050:
            cpr_regime = "WIDE (Sideways Bias)"
        else:
            cpr_regime = "MODERATE (Balanced)"

        r1 = round((2.0 * p) - pdl, 2)
        s1 = round((2.0 * p) - pdh, 2)
        r2 = round(p + (pdh - pdl), 2)
        s2 = round(p - (pdh - pdl), 2)

        # 3. Opening Range Breakout (09:15 - 09:30 AM IST = first 3 5m candles)
        orh = 0.0
        orl = 0.0
        orb_formed = False
        if len(today_candles) >= 3:
            first_three = today_candles[:3]
            orh = max(float(c["high"]) for c in first_three)
            orl = min(float(c["low"]) for c in first_three)
            orb_formed = True

        # 4. Detect Real-time Sweeps & Benchmarks
        active_sweep = "None"
        spot = current_spot or (float(today_candles[-1]["close"]) if today_candles else p)
        
        if orb_formed and orh > 0 and orl > 0:
            if spot > orh + (10.0 if "BANK" in self.index_name else 5.0):
                active_sweep = "ORH Bullish Breakout"
            elif spot < orl - (10.0 if "BANK" in self.index_name else 5.0):
                active_sweep = "ORL Bearish Breakdown"

        # Check PDH / PDL Liquidity Sweeps
        if pdh > 0 and pdl > 0:
            # Turtle Soup Sweep: High pierced PDH but current price returned below PDH
            if today_candles:
                day_high = max(float(c["high"]) for c in today_candles)
                day_low = min(float(c["low"]) for c in today_candles)
                if day_high > pdh and spot < pdh:
                    active_sweep = "PDH Sweep Rejection (Bearish Trap)"
                elif day_low < pdl and spot > pdl:
                    active_sweep = "PDL Sweep Reclaim (Bullish Trap)"

        self.levels = IndexKeyLevels(
            index_name=self.index_name,
            pdh=pdh,
            pdl=pdl,
            pdc=pdc,
            pivot=p,
            tc=tc,
            bc=bc,
            cpr_width=cpr_w,
            cpr_regime=cpr_regime,
            r1=r1,
            s1=s1,
            r2=r2,
            s2=s2,
            orh=orh,
            orl=orl,
            orb_formed=orb_formed,
            active_sweep=active_sweep,
        )
        return self.levels
