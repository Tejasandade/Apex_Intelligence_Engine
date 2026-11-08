"""
Apex Indian Engine — Signal Engine & Multi-Advisor Council
==========================================================
Generates directional option signals on NIFTY & BANKNIFTY using
multi-timeframe EMA trend alignment, VWAP breakouts, and Volume Surges.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger("apex.indian.signals")


@dataclass
class OptionSignal:
    signal_id: str
    index_name: str                 # "NIFTY" or "BANKNIFTY"
    direction: str                  # "LONG" (Buy CE) or "SHORT" (Buy PE)
    option_type: str                # "CE" or "PE"
    spot_price: float
    conviction: float               # 0.0 to 1.0
    council_score: float
    strike_token: str
    strike_symbol: str
    strike_price: float
    lot_size: int
    expiry: str
    sl_points: float                # Stop loss in option premium points
    tp1_points: float               # TP1 (+1.5R)
    tp2_points: float               # TP2 (+2.5R to +3.5R)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    advisors_log: list[str] = field(default_factory=list)


class IndianSignalEngine:
    """
    Analyzes live candle data for Nifty and BankNifty, votes via a 4-Advisor Council,
    and maps the consensus into the optimal ATM/ITM Option Contract.
    """

    def __init__(self, index_name: str = "BANKNIFTY"):
        self.index_name = index_name.upper()
        self.candles_5m: list[dict[str, float]] = []
        self.min_consensus_threshold = 0.58

    def add_candle(self, c: dict[str, float]) -> None:
        """Add a completed 5-minute candle: {open, high, low, close, volume, vwap}."""
        self.candles_5m.append(c)
        if len(self.candles_5m) > 100:
            self.candles_5m.pop(0)

    def evaluate_signal(self, current_spot: float, current_vwap: float = 0.0) -> Optional[tuple[str, str, float, float, list[str]]]:
        """
        Evaluates current technical posture across 4 Council Advisors.
        Returns: (direction, option_type, conviction, council_score, advisor_notes) or None
        """
        if len(self.candles_5m) < 15:
            return None

        closes = [c["close"] for c in self.candles_5m]
        volumes = [c.get("volume", 1.0) for c in self.candles_5m]
        cur_close = closes[-1]
        vwap = current_vwap or self.candles_5m[-1].get("vwap", cur_close)

        # ── Technical Calculations ──
        # EMAs
        ema9 = self._calc_ema(closes, 9)
        ema21 = self._calc_ema(closes, 21)
        
        # ATR (14 period)
        atr = self._calc_atr(14)
        
        # Volume Surge: current 5m vol vs 10-period avg
        avg_vol = sum(volumes[-11:-1]) / max(len(volumes[-11:-1]), 1)
        vol_ratio = volumes[-1] / max(avg_vol, 1.0)

        # ── 4 Council Advisors ──
        # Advisor 1: EMA Fast Momentum (Weight: 0.30)
        adv1_score = 0.0
        if ema9 > ema21:
            adv1_score = 0.30 * min(1.0, (ema9 - ema21) / max(atr * 0.5, 1.0))
        elif ema9 < ema21:
            adv1_score = -0.30 * min(1.0, (ema21 - ema9) / max(atr * 0.5, 1.0))

        # Advisor 2: VWAP Positioning (Weight: 0.25)
        adv2_score = 0.0
        if cur_close > vwap:
            adv2_score = 0.25
        elif cur_close < vwap:
            adv2_score = -0.25

        # Advisor 3: Volume Expansion (Weight: 0.20)
        adv3_score = 0.0
        if vol_ratio >= 1.3:
            adv3_score = 0.20 if cur_close > self.candles_5m[-2]["close"] else -0.20

        # Advisor 4: Candle Convexity / Range Breakout (Weight: 0.25)
        adv4_score = 0.0
        c_open = self.candles_5m[-1]["open"]
        body = abs(cur_close - c_open)
        if body >= atr * 0.7:
            adv4_score = 0.25 if cur_close > c_open else -0.25

        # Combined Weighted Score (-1.0 to +1.0)
        net_score = adv1_score + adv2_score + adv3_score + adv4_score
        abs_score = abs(net_score)

        notes = [
            f"EMA9: {ema9:.1f} vs EMA21: {ema21:.1f} (Score: {adv1_score:+.2f})",
            f"Spot: {cur_close:.1f} vs VWAP: {vwap:.1f} (Score: {adv2_score:+.2f})",
            f"Vol Surge: {vol_ratio:.1f}x (Score: {adv3_score:+.2f})",
            f"Range Body: {body:.1f} pts (Score: {adv4_score:+.2f})",
        ]

        # Consensus Threshold
        if net_score >= self.min_consensus_threshold:
            # Bullish: Buy CALL (CE)
            conviction = min(0.95, 0.50 + abs_score * 0.5)
            return "LONG", "CE", conviction, round(net_score, 2), notes
        elif net_score <= -self.min_consensus_threshold:
            # Bearish: Buy PUT (PE)
            conviction = min(0.95, 0.50 + abs_score * 0.5)
            return "SHORT", "PE", conviction, round(abs(net_score), 2), notes

        return None

    def _calc_ema(self, series: list[float], period: int) -> float:
        if len(series) < period:
            return series[-1]
        k = 2.0 / (period + 1.0)
        ema = series[0]
        for val in series[1:]:
            ema = (val * k) + (ema * (1.0 - k))
        return ema

    def _calc_atr(self, period: int = 14) -> float:
        if len(self.candles_5m) < 2:
            return 25.0
        trs = []
        for i in range(1, len(self.candles_5m)):
            h = self.candles_5m[i]["high"]
            l = self.candles_5m[i]["low"]
            prev_c = self.candles_5m[i-1]["close"]
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return sum(trs[-period:]) / max(len(trs[-period:]), 1)
