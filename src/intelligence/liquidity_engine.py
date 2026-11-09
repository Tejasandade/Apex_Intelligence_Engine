"""
Apex Intelligence Engine — Institutional Liquidity & Key Level Engine
====================================================================
Detects institutional liquidity pools, Fair Value Gaps (FVG),
Previous Day High/Low (PDH/PDL), Asian Session Initial Balance (ASH/ASL),
and Liquidity Sweep (Turtle Soup) patterns across all traded assets.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional
import pandas as pd


class LiquidityEngine:
    """
    Computes key liquidity levels and detects smart money liquidity sweeps.
    """

    def __init__(self):
        self._levels: dict[str, dict[str, Any]] = {}

    def update_levels(
        self,
        symbol: str,
        candles_1d: list[dict],
        candles_1h: list[dict],
        candles_5m: list[dict],
    ) -> dict[str, Any]:
        """
        Compute PDH, PDL, Asian Range, Fair Value Gaps, and Liquidity Sweeps.
        """
        sym = symbol.upper()
        res = {
            "symbol": sym,
            "pdh": 0.0,
            "pdl": 0.0,
            "pdc": 0.0,
            "ash": 0.0,  # Asian Session High (00:00 - 07:00 UTC)
            "asl": 0.0,  # Asian Session Low
            "fvg_bullish": False,
            "fvg_bearish": False,
            "sweep_bullish": False,  # Swept low and closed back inside (Turtle Soup Long)
            "sweep_bearish": False,  # Swept high and closed back inside (Turtle Soup Short)
            "sweep_level": "",
            "nearest_resistance": 0.0,
            "nearest_support": 0.0,
            "dist_to_resistance_pct": 99.0,
            "dist_to_support_pct": 99.0,
            "lbma_am_fix": 0.0,  # London Gold AM Fix (10:30 UTC benchmark)
            "lbma_pm_fix": 0.0,  # London Gold PM Fix (15:00 UTC benchmark)
        }

        # 1. Previous Day High & Low (PDH / PDL)
        if len(candles_1d) >= 2:
            prev_day = candles_1d[-2]
            res["pdh"] = float(prev_day["high"])
            res["pdl"] = float(prev_day["low"])
            res["pdc"] = float(prev_day["close"])

        # 2. Asian Session Initial Balance (00:00 to 07:00 UTC)
        if len(candles_1h) >= 12:
            now_dt = datetime.datetime.now(datetime.timezone.utc)
            today_start_ts = int(
                datetime.datetime(now_dt.year, now_dt.month, now_dt.day, tzinfo=datetime.timezone.utc).timestamp() * 1000
            )
            asian_candles = [
                c for c in candles_1h
                if c.get("timestamp", 0) >= today_start_ts and (c.get("timestamp", 0) - today_start_ts) <= (7 * 3600 * 1000)
            ]
            if asian_candles:
                res["ash"] = max(float(c["high"]) for c in asian_candles)
                res["asl"] = min(float(c["low"]) for c in asian_candles)

        # 3. Gold London Fix Benchmarks (LBMA AM 10:30 UTC & PM 15:00 UTC)
        if any(g in sym for g in ("PAXG", "XAU", "GOLD")) and len(candles_1h) >= 4:
            for c in reversed(candles_1h[-48:]):
                ts = c.get("timestamp", 0)
                if ts > 0:
                    dt = datetime.datetime.fromtimestamp(ts / 1000.0, tz=datetime.timezone.utc)
                    if dt.hour == 10 and res["lbma_am_fix"] == 0.0:
                        res["lbma_am_fix"] = float(c.get("close", 0.0))
                    elif dt.hour == 15 and res["lbma_pm_fix"] == 0.0:
                        res["lbma_pm_fix"] = float(c.get("close", 0.0))

        # 4. Fair Value Gap (FVG) on latest 5m candles
        if len(candles_5m) >= 3:
            c1 = candles_5m[-3]
            c2 = candles_5m[-2]
            c3 = candles_5m[-1]
            # Bullish FVG: Gap between candle 1 high and candle 3 low
            if float(c3["low"]) > float(c1["high"]):
                res["fvg_bullish"] = True
            # Bearish FVG: Gap between candle 1 low and candle 3 high
            elif float(c3["high"]) < float(c1["low"]):
                res["fvg_bearish"] = True

        # 4. Liquidity Sweep Detection (ICT Turtle Soup / AMD)
        if len(candles_5m) >= 2:
            cur_5m = candles_5m[-1]
            cur_high = float(cur_5m["high"])
            cur_low = float(cur_5m["low"])
            cur_close = float(cur_5m["close"])

            # Check PDL sweep
            if res["pdl"] > 0:
                if cur_low < res["pdl"] and cur_close > res["pdl"]:
                    res["sweep_bullish"] = True
                    res["sweep_level"] = "PDL"
            # Check ASL sweep
            if not res["sweep_bullish"] and res["asl"] > 0:
                if cur_low < res["asl"] and cur_close > res["asl"]:
                    res["sweep_bullish"] = True
                    res["sweep_level"] = "ASL"

            # Check PDH sweep
            if res["pdh"] > 0:
                if cur_high > res["pdh"] and cur_close < res["pdh"]:
                    res["sweep_bearish"] = True
                    res["sweep_level"] = "PDH"
            # Check ASH sweep
            if not res["sweep_bearish"] and res["ash"] > 0:
                if cur_high > res["ash"] and cur_close < res["ash"]:
                    res["sweep_bearish"] = True
                    res["sweep_level"] = "ASH"

            # Check Expansion Breakouts (Continuation moves above/below Session & Daily levels)
            if res["ash"] > 0 and cur_close > res["ash"]:
                res["breakout_bullish"] = True
                res["breakout_level"] = "ASH"
            elif res["pdh"] > 0 and cur_close > res["pdh"]:
                res["breakout_bullish"] = True
                res["breakout_level"] = "PDH"

            if res["asl"] > 0 and cur_close < res["asl"]:
                res["breakout_bearish"] = True
                res["breakout_level"] = "ASL"
            elif res["pdl"] > 0 and cur_close < res["pdl"]:
                res["breakout_bearish"] = True
                res["breakout_level"] = "PDL"

            # 5. Distance to nearest Key Levels
            cur_p = cur_close
            resists = [p for p in [res["pdh"], res["ash"]] if p > cur_p]
            supps = [p for p in [res["pdl"], res["asl"]] if 0 < p < cur_p]

            if resists:
                nr = min(resists)
                res["nearest_resistance"] = nr
                res["dist_to_resistance_pct"] = round(((nr - cur_p) / cur_p) * 100.0, 2)
            if supps:
                ns = max(supps)
                res["nearest_support"] = ns
                res["dist_to_support_pct"] = round(((cur_p - ns) / cur_p) * 100.0, 2)

        self._levels[sym] = res
        return res

    def get_levels(self, symbol: str) -> dict[str, Any]:
        return self._levels.get(symbol.upper(), {})

    def evaluate_signal_synergy(self, symbol: str, direction: str) -> tuple[float, str]:
        """
        Evaluate how current liquidity alignment affects signal conviction.
        Returns: (conviction_delta: float, reason: str)
        """
        sym = symbol.upper()
        lev = self.get_levels(sym)
        if not lev:
            return 0.0, ""

        dir_u = direction.upper()

        # 1. Direct Liquidity Grab Reversal (Highest Quality Setup)
        if dir_u in ("BUY", "LONG") and lev.get("sweep_bullish"):
            return +0.06, f"Bullish Liquidity Sweep of {lev.get('sweep_level')} confirmed"
        if dir_u in ("SELL", "SHORT") and lev.get("sweep_bearish"):
            return +0.06, f"Bearish Liquidity Sweep of {lev.get('sweep_level')} confirmed"

        # 1b. Institutional Expansion Breakout (Strong Momentum Acceleration)
        if dir_u in ("BUY", "LONG") and lev.get("breakout_bullish"):
            return +0.05, f"Bullish Breakout above {lev.get('breakout_level')} Expansion"
        if dir_u in ("SELL", "SHORT") and lev.get("breakout_bearish"):
            return +0.05, f"Bearish Breakdown below {lev.get('breakout_level')} Expansion"

        # 2. Bullish/Bearish Fair Value Gap
        if dir_u in ("BUY", "LONG") and lev.get("fvg_bullish"):
            return +0.03, "Bullish FVG Value Zone"
        if dir_u in ("SELL", "SHORT") and lev.get("fvg_bearish"):
            return +0.03, "Bearish FVG Value Zone"

        # 3. Proximity Warning: Buying directly into major resistance
        if dir_u in ("BUY", "LONG") and lev.get("dist_to_resistance_pct", 99.0) < 0.20:
            return -0.05, f"Warning: Too close to Resistance ({lev.get('dist_to_resistance_pct')}%)"
        if dir_u in ("SELL", "SHORT") and lev.get("dist_to_support_pct", 99.0) < 0.20:
            return -0.05, f"Warning: Too close to Support ({lev.get('dist_to_support_pct')}%)"

        # 4. Gold London Fix Institutional Anchor Synergy
        if any(g in sym for g in ("PAXG", "XAU", "GOLD")):
            cur_p = lev.get("nearest_support", 0.0) or lev.get("nearest_resistance", 0.0)
            if cur_p == 0.0 and lev.get("pdc", 0.0) > 0.0:
                cur_p = lev.get("pdc", 0.0)
            for fix_name, fix_val in [("LBMA PM Fix", lev.get("lbma_pm_fix", 0.0)), ("LBMA AM Fix", lev.get("lbma_am_fix", 0.0))]:
                if fix_val > 0 and cur_p > 0:
                    dist = abs(cur_p - fix_val) / fix_val
                    if dist <= 0.0025:  # Within 0.25% of London Fix
                        if dir_u in ("BUY", "LONG") and cur_p >= fix_val:
                            return +0.05, f"Gold Reclaiming Institutional {fix_name} (${fix_val:,.1f})"
                        elif dir_u in ("SELL", "SHORT") and cur_p <= fix_val:
                            return +0.05, f"Gold Rejecting Institutional {fix_name} (${fix_val:,.1f})"

        return 0.0, "NEUTRAL"

