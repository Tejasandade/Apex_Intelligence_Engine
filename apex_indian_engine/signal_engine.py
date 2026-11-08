"""
Apex Indian Engine — Signal Engine & Multi-Advisor Council
==========================================================
Generates directional option signals on NIFTY & BANKNIFTY using:
1. Regime-Conditioned XGBoost ML Inference (23 Institutional Features)
2. Multi-Timeframe Trend Alignment (EMA 9 / 21 / 50)
3. Volume-Weighted Average Price (VWAP) Positioning & Breakout
4. Volume Surge Expansion & Candle Convexity
5. Open Interest (OI) & Put-Call Ratio (PCR) Wall Shield
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from apex_indian_engine.ml_predictor import IndianMLPredictor
from apex_indian_engine.oi_shield import OIShield, OIMetrics

logger = logging.getLogger("apex.indian.signals")


@dataclass
class OptionSignal:
    signal_id: str
    index_name: str                 # "NIFTY" or "BANKNIFTY"
    direction: str                  # "LONG" (Buy CE) or "SHORT" (Buy PE)
    option_type: str                # "CE" or "PE"
    spot_price: float
    conviction: float               # 0.0 to 1.0
    council_score: float            # -1.0 to +1.0
    strike_token: str = ""
    strike_symbol: str = ""
    strike_price: float = 0.0
    lot_size: int = 30
    expiry: str = ""
    sl_points: float = 25.0         # Stop loss in option premium points
    tp1_points: float = 37.5        # TP1 (+1.5R)
    tp2_points: float = 62.5        # TP2 (+2.5R to +3.5R)
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    advisors_log: list[str] = field(default_factory=list)
    oi_metrics: Optional[OIMetrics] = None


class IndianSignalEngine:
    """
    Analyzes live candle data for Nifty and BankNifty, votes via a 4-Advisor Council
    (including 23-feature XGBoost ML models), and gates entries via the OI Wall Shield.
    """

    def __init__(self, index_name: str = "BANKNIFTY"):
        self.index_name = index_name.upper()
        self.candles_5m: list[dict[str, Any]] = []
        self.min_consensus_threshold = 0.55

        # Initialize ML Predictor (XGBoost models for BankNifty)
        self.ml_predictor = IndianMLPredictor()

        # Initialize OI Wall Shield (buffer: 60 pts for BankNifty, 30 pts for Nifty)
        wall_buf = 60.0 if "BANK" in self.index_name else 30.0
        self.oi_shield = OIShield(wall_buffer_points=wall_buf)

    def add_candle(self, c: dict[str, Any]) -> None:
        """
        Add a completed 5-minute candle:
        {timestamp, open, high, low, close, volume, [vwap]}
        """
        self.candles_5m.append(c)
        if len(self.candles_5m) > 120:
            self.candles_5m.pop(0)

    def evaluate_signal(
        self,
        current_spot: float,
        current_vwap: float = 0.0,
        oi_chain: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[tuple[str, str, float, float, list[str], Optional[OIMetrics]]]:
        """
        Evaluates current technical posture across 4 Council Advisors + OI Wall Shield.
        Returns: (direction, option_type, conviction, council_score, advisor_notes, oi_metrics) or None
        """
        if len(self.candles_5m) < 20:
            return None

        closes = [float(c["close"]) for c in self.candles_5m]
        volumes = [float(c.get("volume", 1.0)) for c in self.candles_5m]
        cur_close = closes[-1]
        vwap = current_vwap or float(self.candles_5m[-1].get("vwap", cur_close))

        # ── 1. Technical Calculations ──
        ema9 = self._calc_ema(closes, 9)
        ema21 = self._calc_ema(closes, 21)
        ema50 = self._calc_ema(closes, 50)
        atr = self._calc_atr(14)

        avg_vol = sum(volumes[-11:-1]) / max(len(volumes[-11:-1]), 1)
        vol_ratio = volumes[-1] / max(avg_vol, 1.0)
        c_open = float(self.candles_5m[-1]["open"])
        body = abs(cur_close - c_open)

        # ── 2. Advisor 1: Institutional ML Model (Weight: 0.35) ──
        adv1_score = 0.0
        ml_log = "ML: Inactive / Insufficient Data"
        try:
            df = pd.DataFrame(self.candles_5m)
            # Ensure proper column names
            req_cols = ["timestamp", "open", "high", "low", "close", "volume"]
            for col in req_cols:
                if col not in df.columns:
                    if col == "timestamp":
                        df["timestamp"] = range(len(df))
                    else:
                        df[col] = cur_close

            p_bull, regime, ml_conv = self.ml_predictor.predict(df)
            if p_bull >= 0.52:
                # Bullish Call Bias
                adv1_score = 0.35 * min(1.0, (p_bull - 0.50) / 0.30)
                ml_log = f"ML (Weight: 35%): Bull P(CE)={p_bull*100:.1f}% | Regime={regime} (Score: {adv1_score:+.2f})"
            elif p_bull <= 0.48:
                # Bearish Put Bias
                adv1_score = -0.35 * min(1.0, (0.50 - p_bull) / 0.30)
                ml_log = f"ML (Weight: 35%): Bear P(PE)={(1-p_bull)*100:.1f}% | Regime={regime} (Score: {adv1_score:+.2f})"
            else:
                ml_log = f"ML (Weight: 35%): Neutral P(CE)={p_bull*100:.1f}% (Score: +0.00)"
        except Exception as e:
            logger.debug(f"ML advisor error: {e}")

        # ── 3. Advisor 2: Multi-Timeframe Trend & EMA Alignment (Weight: 0.25) ──
        adv2_score = 0.0
        if ema9 > ema21 > ema50:
            # Full Bullish Alignment
            adv2_score = 0.25
        elif ema9 < ema21 < ema50:
            # Full Bearish Alignment
            adv2_score = -0.25
        elif ema9 > ema21:
            adv2_score = 0.12
        elif ema9 < ema21:
            adv2_score = -0.12
        adv2_log = f"Trend EMA (Weight: 25%): 9={ema9:.0f}, 21={ema21:.0f}, 50={ema50:.0f} (Score: {adv2_score:+.2f})"

        # ── 4. Advisor 3: VWAP Positioning & Breakout (Weight: 0.20) ──
        adv3_score = 0.0
        vwap_diff = cur_close - vwap
        if vwap_diff >= atr * 0.1:
            adv3_score = 0.20
        elif vwap_diff <= -atr * 0.1:
            adv3_score = -0.20
        adv3_log = f"VWAP (Weight: 20%): Spot={cur_close:.0f} vs VWAP={vwap:.0f} [Diff: {vwap_diff:+.1f}] (Score: {adv3_score:+.2f})"

        # ── 5. Advisor 4: Volume Expansion & Candle Convexity (Weight: 0.20) ──
        adv4_score = 0.0
        is_bull_candle = cur_close > c_open
        if vol_ratio >= 1.25 and body >= atr * 0.5:
            adv4_score = 0.20 if is_bull_candle else -0.20
        elif vol_ratio >= 1.10:
            adv4_score = 0.10 if is_bull_candle else -0.10
        adv4_log = f"Vol Surge (Weight: 20%): {vol_ratio:.1f}x vol, Body={body:.1f} pts (Score: {adv4_score:+.2f})"

        # ── 6. Council Aggregation ──
        net_score = round(adv1_score + adv2_score + adv3_score + adv4_score, 2)
        notes = [ml_log, adv2_log, adv3_log, adv4_log]

        # ── 7. Open Interest & PCR Wall Shield Evaluation ──
        oi_metrics: Optional[OIMetrics] = None
        if oi_chain:
            try:
                oi_metrics = self.oi_shield.analyze_chain(oi_chain, spot_price=cur_close)
                notes.append(
                    f"OI Shield: PCR={oi_metrics.pcr:.2f} ({oi_metrics.sentiment}) | "
                    f"Call Wall={oi_metrics.max_call_wall_strike:.0f} | Put Wall={oi_metrics.max_put_wall_strike:.0f}"
                )
            except Exception as e:
                logger.debug(f"OI shield calculation error: {e}")

        # Check Threshold
        if net_score >= self.min_consensus_threshold:
            # Bullish: Buy CALL (CE)
            direction, opt_type = "LONG", "CE"
            conviction = min(0.95, round(0.50 + (net_score * 0.45), 2))

            # Validate against OI Wall Shield
            if oi_metrics:
                safe, reason = self.oi_shield.validate_entry(direction, cur_close, cur_close, oi_metrics)
                if not safe:
                    logger.warn(f"Signal Vetoed: {reason}")
                    notes.append(f"VETO: {reason}")
                    return None

            return direction, opt_type, conviction, net_score, notes, oi_metrics

        elif net_score <= -self.min_consensus_threshold:
            # Bearish: Buy PUT (PE)
            direction, opt_type = "SHORT", "PE"
            conviction = min(0.95, round(0.50 + (abs(net_score) * 0.45), 2))

            # Validate against OI Wall Shield
            if oi_metrics:
                safe, reason = self.oi_shield.validate_entry(direction, cur_close, cur_close, oi_metrics)
                if not safe:
                    logger.warn(f"Signal Vetoed: {reason}")
                    notes.append(f"VETO: {reason}")
                    return None

            return direction, opt_type, conviction, net_score, notes, oi_metrics

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
            h = float(self.candles_5m[i]["high"])
            l = float(self.candles_5m[i]["low"])
            prev_c = float(self.candles_5m[i-1]["close"])
            tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
            trs.append(tr)
        return sum(trs[-period:]) / max(len(trs[-period:]), 1)
