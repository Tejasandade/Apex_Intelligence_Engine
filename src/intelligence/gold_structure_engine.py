"""
Apex Intelligence Engine — Gold Structural & MTL/ICT Intelligence Engine
========================================================================
Adapted from xau_c (Gold Observatory 0.21.0 / MTL V48 state machine).

Provides institutional-grade Gold structural intelligence:
1. Multi-timeframe Fair Value Gaps (FVG) across 15m, 1h, and 4h.
2. Origin candle Order Block (OB) proxy zones with lifecycle tracking (untouched, overlapped, invalidated).
3. Exact C1 and C2 equilibrium levels relative to dominant price legs.
4. Institutional Demand/Supply Zone mapping to prevent counter-trend shorts into demand or longs into supply.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MTLState:
    first: Optional[dict] = None
    second: Optional[dict] = None

    def process(self, bar: dict, bullish: bool) -> Optional[dict]:
        """
        Process a closed candle through the MTL state machine.
        Reflect prices for the bearish mirror of the bullish state machine.
        """
        def h(b: dict) -> float: return float(b["high"]) if bullish else -float(b["low"])
        def l(b: dict) -> float: return float(b["low"]) if bullish else -float(b["high"])
        def c(b: dict) -> float: return float(b["close"]) if bullish else -float(b["close"])

        if self.first is None:
            self.first = bar
            return None

        ref = self.second or self.first
        if h(bar) <= h(ref) and l(bar) >= l(ref):
            return None
        if l(bar) < l(ref):
            self.first, self.second = bar, None
            return None
        if c(bar) > h(ref):
            if self.second is None:
                self.second = bar
                return None
            found = None
            if h(self.first) < l(bar):
                bounds = (float(self.first["high"]), float(bar["low"])) if bullish else (float(bar["high"]), float(self.first["low"]))
                found = {
                    "lower": bounds[0],
                    "upper": bounds[1],
                    "midpoint": (bounds[0] + bounds[1]) / 2.0,
                    "c1_extreme": float(self.first["low"]) if bullish else float(self.first["high"]),
                    "origin_low": float(self.first["low"]),
                    "origin_high": float(self.first["high"]),
                    "origin_mid": (float(self.first["low"]) + float(self.first["high"])) / 2.0,
                    "bullish": bullish,
                    "status": "untouched",
                }
            self.first, self.second = self.second, bar
            return found
        if h(bar) > h(ref):
            self.first, self.second = bar, None
        return None


class GoldStructureEngine:
    """
    Dedicated structural analyst for Gold (XAUUSD / PAXGUSDT / GOLD.i#).
    Scans multi-timeframe candle streams to identify:
    - Active unmitigated Bullish FVGs (Demand) & Bearish FVGs (Supply)
    - Order Block origin zones
    - C1 and C2 equilibrium levels
    - Trade veto flags (prevents buying supply tops or selling demand bottoms)
    """

    def __init__(self):
        self._cache: dict[str, Any] = {}

    def scan_timeframe_fvgs(self, candles: list[dict], timeframe: str = "1h") -> list[dict]:
        """
        Scan a list of chronological closed candles for FVG zones and MTL structures.
        Supports both classic 3-bar ICT gaps and multi-bar MTL state formations.
        """
        if len(candles) < 3:
            return []

        bull_machine = MTLState()
        bear_machine = MTLState()
        detected_zones: list[dict] = []

        # Find rolling leg extremes
        leg_low = min(float(c["low"]) for c in candles)
        leg_high = max(float(c["high"]) for c in candles)

        # ── 1. Classic 3-Bar ICT Fair Value Gaps ──
        for i in range(2, len(candles)):
            c_prev2 = candles[i - 2]
            c_mid = candles[i - 1]
            c_curr = candles[i]

            p2_high = float(c_prev2["high"])
            p2_low = float(c_prev2["low"])
            curr_high = float(c_curr["high"])
            curr_low = float(c_curr["low"])

            # Bullish FVG: Low of candle i > High of candle i-2
            if curr_low > p2_high:
                lower = p2_high
                upper = curr_low
                c1 = (leg_low + upper) / 2.0
                c2 = (float(c_prev2["low"]) + upper) / 2.0
                detected_zones.append({
                    "lower": lower,
                    "upper": upper,
                    "midpoint": (lower + upper) / 2.0,
                    "c1_extreme": float(c_prev2["low"]),
                    "origin_low": float(c_prev2["low"]),
                    "origin_high": p2_high,
                    "origin_mid": (float(c_prev2["low"]) + p2_high) / 2.0,
                    "bullish": True,
                    "status": "untouched",
                    "timeframe": timeframe,
                    "direction": "bullish",
                    "c1": c1,
                    "c2": c2,
                    "bar_ts": c_curr.get("timestamp", 0),
                })

            # Bearish FVG: High of candle i < Low of candle i-2
            elif curr_high < p2_low:
                lower = curr_high
                upper = p2_low
                c1 = (leg_high + lower) / 2.0
                c2 = (float(c_prev2["high"]) + lower) / 2.0
                detected_zones.append({
                    "lower": lower,
                    "upper": upper,
                    "midpoint": (lower + upper) / 2.0,
                    "c1_extreme": float(c_prev2["high"]),
                    "origin_low": curr_high,
                    "origin_high": float(c_prev2["high"]),
                    "origin_mid": (curr_high + float(c_prev2["high"])) / 2.0,
                    "bullish": False,
                    "status": "untouched",
                    "timeframe": timeframe,
                    "direction": "bearish",
                    "c1": c1,
                    "c2": c2,
                    "bar_ts": c_curr.get("timestamp", 0),
                })

        # ── 2. MTL State Machine Expansion ──
        for bar in candles:
            b_found = bull_machine.process(bar, bullish=True)
            if b_found:
                edge = b_found["upper"]
                c1 = (leg_low + edge) / 2.0
                c2 = (b_found["c1_extreme"] + edge) / 2.0
                detected_zones.append({
                    **b_found,
                    "timeframe": timeframe,
                    "direction": "bullish",
                    "c1": c1,
                    "c2": c2,
                    "bar_ts": bar.get("timestamp", 0),
                })

            s_found = bear_machine.process(bar, bullish=False)
            if s_found:
                edge = s_found["lower"]
                c1 = (leg_high + edge) / 2.0
                c2 = (s_found["c1_extreme"] + edge) / 2.0
                detected_zones.append({
                    **s_found,
                    "timeframe": timeframe,
                    "direction": "bearish",
                    "c1": c1,
                    "c2": c2,
                    "bar_ts": bar.get("timestamp", 0),
                })

        # Track status against subsequent candles
        for z in detected_zones:
            status = "untouched"
            z_ts = z["bar_ts"]
            for c in candles:
                c_ts = c.get("timestamp", 0)
                if c_ts <= z_ts:
                    continue
                c_high = float(c["high"])
                c_low = float(c["low"])
                c_close = float(c["close"])

                # Check if price tapped the origin OB
                if status == "untouched" and c_low <= z["origin_high"] and c_high >= z["origin_low"]:
                    status = "overlapped"

                # Invalidation: candle closes through the origin candle extreme
                if z["direction"] == "bullish" and c_close < z["origin_low"]:
                    status = "invalidated"
                    break
                elif z["direction"] == "bearish" and c_close > z["origin_high"]:
                    status = "invalidated"
                    break
            z["status"] = status

        # Return only active (unmitigated or overlapped) zones
        return [z for z in detected_zones if z["status"] != "invalidated"]

    def evaluate_gold_structure(
        self,
        current_price: float,
        candles_15m: list[dict],
        candles_1h: list[dict],
        candles_4h: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """
        Evaluate current price relative to multi-timeframe Gold market structure.
        Returns safety vetoes, zone locations, and C1/C2 equilibrium metrics.
        """
        result = {
            "in_bullish_fvg": False,
            "in_bearish_fvg": False,
            "approaching_bullish_fvg": False,
            "approaching_bearish_fvg": False,
            "nearest_demand_fvg": None,
            "nearest_supply_fvg": None,
            "c1_level": 0.0,
            "c2_level": 0.0,
            "veto_short": False,
            "veto_long": False,
            "veto_reason": "",
            "structural_bias": "NEUTRAL",
            "active_zones_count": 0,
        }

        if current_price <= 0:
            return result

        # Scan 1h (primary institutional frame) and 15m (tactical execution frame)
        zones_1h = self.scan_timeframe_fvgs(candles_1h, "1h")
        zones_15m = self.scan_timeframe_fvgs(candles_15m, "15m")
        all_active = zones_1h + zones_15m
        result["active_zones_count"] = len(all_active)

        bullish_zones = [z for z in all_active if z["direction"] == "bullish"]
        bearish_zones = [z for z in all_active if z["direction"] == "bearish"]

        # ── 1. Check Bullish FVG / Demand Absorption ──
        for z in reversed(bullish_zones):
            lower, upper = z["lower"], z["upper"]
            # Inside the FVG zone: price is between lower and upper bounds
            if lower * 0.999 <= current_price <= upper * 1.001:
                result["in_bullish_fvg"] = True
                result["nearest_demand_fvg"] = z
                result["c1_level"] = z.get("c1", 0.0)
                result["c2_level"] = z.get("c2", 0.0)
                # FATAL SHORT VETO: Never short directly into an institutional demand FVG
                result["veto_short"] = True
                result["veto_reason"] = (
                    f"Gold Structure VETO: Price ${current_price:,.2f} is resting inside a {z['timeframe']} "
                    f"Bullish FVG Demand Zone [${lower:,.1f} - ${upper:,.1f}]. Short entries strictly locked."
                )
                result["structural_bias"] = "BULLISH_DEMAND"
                break
            # Approaching demand floor from above (within 0.20%)
            elif current_price > upper and (current_price - upper) / current_price < 0.002:
                result["approaching_bullish_fvg"] = True
                if not result["nearest_demand_fvg"]:
                    result["nearest_demand_fvg"] = z

        # ── 2. Check Bearish FVG / Supply Resistance ──
        for z in reversed(bearish_zones):
            lower, upper = z["lower"], z["upper"]
            # Inside the FVG zone: price is between lower and upper bounds
            if lower * 0.999 <= current_price <= upper * 1.001:
                result["in_bearish_fvg"] = True
                result["nearest_supply_fvg"] = z
                result["c1_level"] = z.get("c1", 0.0)
                result["c2_level"] = z.get("c2", 0.0)
                # FATAL LONG VETO: Never buy directly into an institutional supply FVG
                result["veto_long"] = True
                result["veto_reason"] = (
                    f"Gold Structure VETO: Price ${current_price:,.2f} is resting inside a {z['timeframe']} "
                    f"Bearish FVG Supply Zone [${lower:,.1f} - ${upper:,.1f}]. Long entries strictly locked."
                )
                result["structural_bias"] = "BEARISH_SUPPLY"
                break
            # Approaching supply ceiling from below (within 0.20%)
            elif current_price < lower and (lower - current_price) / current_price < 0.002:
                result["approaching_bearish_fvg"] = True
                if not result["nearest_supply_fvg"]:
                    result["nearest_supply_fvg"] = z

        # If not inside a zone, populate nearest C1/C2 from latest valid 1H zone
        if not result["c1_level"] and zones_1h:
            latest_1h = zones_1h[-1]
            result["c1_level"] = latest_1h.get("c1", 0.0)
            result["c2_level"] = latest_1h.get("c2", 0.0)

        return result


# Global singleton
gold_structure_engine = GoldStructureEngine()
