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
    fvg_supply: bool = False        # In Bearish FVG Supply Zone (Resist Calls)
    fvg_demand: bool = False        # In Bullish FVG Demand Zone (Resist Puts)
    fvg_zone_top: float = 0.0
    fvg_zone_bottom: float = 0.0


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
        official_pdc: float = 0.0,
        official_pdh: float = 0.0,
        official_pdl: float = 0.0,
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
        if official_pdh > 0 and official_pdl > 0:
            pdh = official_pdh
            pdl = official_pdl
            pdc = official_pdc if official_pdc > 0 else round((pdh + pdl) / 2.0, 2)
        elif prev_day_candles:
            # Use last session's candles (up to 75 5m candles per NSE trading day)
            last_session = prev_day_candles[-75:]
            pdh = max(float(c["high"]) for c in last_session)
            pdl = min(float(c["low"]) for c in last_session)
            pdc = float(last_session[-1]["close"])
        else:
            # Fallback to baseline price with realistic intraday variance
            base = float(candles_5m[0]["close"])
            step = 300.0 if "BANK" in self.index_name else 120.0
            pdh = round(base + step, 2)
            pdl = round(base - step, 2)
            pdc = round(base - (step * 0.15), 2)

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

        # 3. Opening Range Breakout (09:15 - 09:30 AM IST = 09:15, 09:20, 09:25 5m candles)
        orh = 0.0
        orl = 0.0
        orb_formed = False
        if today_candles:
            has_clock_ts = any("T" in str(c.get("timestamp", "")) or " " in str(c.get("timestamp", "")) for c in today_candles)
            if has_clock_ts:
                orb_candles = [
                    c for c in today_candles
                    if any(t in str(c.get("timestamp", "")) for t in ("09:15:", "09:20:", "09:25:"))
                ]
                if len(orb_candles) >= 3:
                    orh = max(float(c["high"]) for c in orb_candles)
                    orl = min(float(c["low"]) for c in orb_candles)
                    orb_formed = True
            elif len(today_candles) >= 3:
                first_three = today_candles[:3]
                orh = max(float(c["high"]) for c in first_three)
                orl = min(float(c["low"]) for c in first_three)
                orb_formed = True

        # 4. Detect Real-time Sweeps, Breakouts & Benchmarks
        active_sweep = "None"
        spot = current_spot or (float(today_candles[-1]["close"]) if today_candles else p)
        buf = 10.0 if "BANK" in self.index_name else 5.0
        today_low = min((float(c["low"]) for c in today_candles), default=spot)
        today_high = max((float(c["high"]) for c in today_candles), default=spot)
        # On large gap-up/gap-down days where spot never traded near PDH/PDL, anchor to intraday ORB instead
        pdh_in_play = (pdh > 0) and (today_low <= (pdh + (spot * 0.004)))
        pdl_in_play = (pdl > 0) and (today_high >= (pdl - (spot * 0.004)))

        # Prioritize intraday ORB when a session gaps far above PDH/below PDL or breaks intraday ORB opposite to prior PDH/PDL
        if orb_formed and orl > 0 and spot < (orl - buf):
            active_sweep = "PDL Bearish Breakdown" if (pdl_in_play and spot < (pdl - buf)) else "ORL Bearish Breakdown"
        elif orb_formed and orh > 0 and spot > (orh + buf) and not pdh_in_play:
            active_sweep = "ORH Bullish Breakout"
        elif pdh_in_play and spot > (pdh + buf):
            active_sweep = "PDH Bullish Breakout"
        elif pdl_in_play and spot < (pdl - buf):
            active_sweep = "PDL Bearish Breakdown"
        elif orb_formed and orh > 0 and spot > (orh + buf):
            active_sweep = "ORH Bullish Breakout"
        # Check Recent Turtle Soup Sweeps (Only active if occurred within recent 15 minutes / 3 candles)
        elif pdh > 0 and pdl > 0 and today_candles:
            recent_candles = today_candles[-3:]
            recent_high = max(float(c["high"]) for c in recent_candles)
            recent_low = min(float(c["low"]) for c in recent_candles)

            # Fresh PDH Sweep Rejection: Recent candle swept PDH, but spot rejected back below PDH
            if recent_high > pdh and spot < (pdh - buf):
                if not (orb_formed and spot > orh):
                    active_sweep = "PDH Sweep Rejection (Bearish Trap)"
            # Fresh PDL Sweep Reclaim: Recent candle swept PDL, but spot reclaimed back above PDL
            elif recent_low < pdl and spot > (pdl + buf):
                if not (orb_formed and spot < orl):
                    active_sweep = "PDL Sweep Reclaim (Bullish Trap)"

        # 5. Fair Value Gap (FVG) Imbalance Detection across recent candles (skip invalidated FVGs)
        fvg_supply = False
        fvg_demand = False
        fvg_top = 0.0
        fvg_bot = 0.0
        if len(candles_5m) >= 3:
            for i in range(len(candles_5m) - 1, max(len(candles_5m) - 15, 2), -1):
                c1 = candles_5m[i - 2]
                c3 = candles_5m[i]
                c1_h = float(c1.get("high", 0.0))
                c1_l = float(c1.get("low", 0.0))
                c3_h = float(c3.get("high", 0.0))
                c3_l = float(c3.get("low", 0.0))
                subsequent = candles_5m[i + 1 :] if i + 1 < len(candles_5m) else []

                # Bullish FVG Demand: c3_l > c1_h (invalidated once any later bar closes below c1_h)
                if c3_l > c1_h and c1_h > 0:
                    if subsequent and any(float(sc.get("close", c1_h)) < c1_h for sc in subsequent):
                        continue
                    if c1_h <= spot <= c3_l:
                        fvg_demand = True
                        fvg_top = c3_l
                        fvg_bot = c1_h
                        break

                # Bearish FVG Supply: c3_h < c1_l (invalidated once any later bar closes above c1_l)
                elif c3_h < c1_l and c3_h > 0:
                    if subsequent and any(float(sc.get("close", c1_l)) > c1_l for sc in subsequent):
                        continue
                    if c3_h <= spot <= c1_l:
                        fvg_supply = True
                        fvg_top = c1_l
                        fvg_bot = c3_h
                        break

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
            fvg_supply=fvg_supply,
            fvg_demand=fvg_demand,
            fvg_zone_top=fvg_top,
            fvg_zone_bottom=fvg_bot,
        )
        return self.levels

