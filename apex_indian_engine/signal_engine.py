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
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from apex_indian_engine.ml_predictor import IndianMLPredictor
from apex_indian_engine.oi_shield import OIShield, OIMetrics
from apex_indian_engine.key_levels import IndexKeyLevels

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

        # Initialize ML Predictor (XGBoost models for BankNifty / Nifty)
        self.ml_predictor = IndianMLPredictor(index_name=self.index_name)

        # Initialize OI Wall Shield (buffer: 60 pts for BankNifty, 30 pts for Nifty)
        wall_buf = 60.0 if "BANK" in self.index_name else 30.0
        self.oi_shield = OIShield(wall_buffer_points=wall_buf)
        self._last_veto_log: dict[str, float] = {}
        self.last_diagnostics: dict[str, Any] = {
            "net_score": 0.0,
            "threshold": self.min_consensus_threshold,
            "veto": "",
            "candidate": "NEUTRAL",
            "ml_reliable": self.ml_predictor.is_reliable,
        }

    def add_candle(self, c: dict[str, Any]) -> None:
        """
        Add a completed 5-minute candle (deduplicating matching boot/close timestamps):
        {timestamp, open, high, low, close, volume, [vwap]}
        """
        ts = str(c.get("timestamp", ""))
        if self.candles_5m and ts and str(self.candles_5m[-1].get("timestamp", "")) == ts:
            self.candles_5m[-1] = c
            return
        self.candles_5m.append(c)
        if len(self.candles_5m) > 120:
            self.candles_5m.pop(0)

    def _compute_htf_bias(self, closes: list[float]) -> str:
        """
        Evaluates 15-minute intraday trend structure from 5m close series.
        Uses 15m EMA9 and EMA21 momentum to assess whether the session
        is trending bullish, bearish, or consolidating.
        Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
        """
        if len(closes) < 15:
            return "NEUTRAL"

        # Resample last 45 5m closes (up to 3.75 hours) into 15m close bars
        recent_closes = closes[-45:] if len(closes) >= 45 else closes
        c15 = recent_closes[::3]
        if len(c15) < 3:
            return "NEUTRAL"

        ema9_15 = self._calc_ema(c15, min(9, len(c15)))
        ema21_15 = self._calc_ema(c15, min(21, len(c15)))
        cur = closes[-1]

        if cur > ema9_15 and ema9_15 >= ema21_15:
            return "BULLISH"
        elif cur < ema9_15 and ema9_15 <= ema21_15:
            return "BEARISH"
        elif cur > ema9_15:
            return "BULLISH"
        elif cur < ema9_15:
            return "BEARISH"
        return "NEUTRAL"

    def evaluate_signal(
        self,
        current_spot: float,
        current_vwap: float = 0.0,
        oi_chain: Optional[list[dict[str, Any]]] = None,
        key_levels: Optional[IndexKeyLevels] = None,
        is_preview: bool = False,
        heavyweights: Optional[dict[str, Any]] = None,
        oi_flow: str = "",
    ) -> Optional[tuple[str, str, float, float, list[str], Optional[OIMetrics]]]:
        """
        Evaluates current technical posture across 5 Council Advisors + OI Wall Shield + Liquidity Traps.
        Returns: (direction, option_type, conviction, council_score, advisor_notes, oi_metrics) or None
        """
        if len(self.candles_5m) < 20:
            return None

        closes = [float(c["close"]) for c in self.candles_5m]
        volumes = [float(c.get("volume", 1.0)) for c in self.candles_5m]
        cur_close = float(current_spot) if (is_preview and current_spot > 0) else closes[-1]
        eval_closes = closes[:-1] + [cur_close] if (is_preview and current_spot > 0) else closes
        vwap = current_vwap or float(self.candles_5m[-1].get("vwap", cur_close))

        # ── 1. Technical Calculations ──
        ema9 = self._calc_ema(eval_closes, 9)
        ema21 = self._calc_ema(eval_closes, 21)
        ema50 = self._calc_ema(eval_closes, 50)
        atr = self._calc_atr(14)

        avg_vol = sum(volumes[-11:-1]) / max(len(volumes[-11:-1]), 1)
        vol_ratio = volumes[-1] / max(avg_vol, 1.0)
        c_open = float(self.candles_5m[-1]["open"])
        c_high = max(float(self.candles_5m[-1]["high"]), cur_close)
        c_low = min(float(self.candles_5m[-1]["low"]), cur_close)
        body = abs(cur_close - c_open)

        # Dynamically rebalance council weights when ML weights are uncalibrated/out-of-scale
        ml_active = self.ml_predictor.is_reliable
        if ml_active:
            w_ml, w_ema, w_vwap, w_disp, w_lvl = 0.30, 0.20, 0.15, 0.15, 0.20
        else:
            w_ml, w_ema, w_vwap, w_disp, w_lvl = 0.00, 0.30, 0.25, 0.20, 0.25

        # ── 2. Advisor 1: Institutional ML Model ──
        adv1_score = 0.0
        ml_log = "ML: Structural Bypass (Uncalibrated/Out-of-Scale weights)"
        if ml_active:
            ml_log = "ML: Inactive / Insufficient Data"
            try:
                df = pd.DataFrame(self.candles_5m)
                req_cols = ["timestamp", "open", "high", "low", "close", "volume"]
                for col in req_cols:
                    if col not in df.columns:
                        if col == "timestamp":
                            df["timestamp"] = range(len(df))
                        else:
                            df[col] = cur_close

                p_bull, regime, ml_conv = self.ml_predictor.predict(df)
                if p_bull >= 0.52:
                    adv1_score = w_ml * min(1.0, (p_bull - 0.50) / 0.30)
                    ml_log = f"ML (Weight: {int(w_ml*100)}%): Bull P(CE)={p_bull*100:.1f}% | Regime={regime} (Score: {adv1_score:+.2f})"
                elif p_bull <= 0.48:
                    adv1_score = -w_ml * min(1.0, (0.50 - p_bull) / 0.30)
                    ml_log = f"ML (Weight: {int(w_ml*100)}%): Bear P(PE)={(1-p_bull)*100:.1f}% | Regime={regime} (Score: {adv1_score:+.2f})"
                else:
                    ml_log = f"ML (Weight: {int(w_ml*100)}%): Neutral P(CE)={p_bull*100:.1f}% (Score: +0.00)"
            except Exception as e:
                logger.debug(f"ML advisor error: {e}")

        # ── 3. Advisor 2: Multi-Timeframe Trend & EMA Alignment ──
        adv2_score = 0.0
        if ema9 > ema21 > ema50:
            adv2_score = w_ema
        elif ema9 < ema21 < ema50:
            adv2_score = -w_ema
        elif ema9 > ema21:
            adv2_score = round(w_ema * 0.5, 2)
        elif ema9 < ema21:
            adv2_score = -round(w_ema * 0.5, 2)
        adv2_log = f"Trend EMA (Weight: {int(w_ema*100)}%): 9={ema9:.0f}, 21={ema21:.0f}, 50={ema50:.0f} (Score: {adv2_score:+.2f})"

        # ── 4. Advisor 3: Session VWAP Positioning & Breakout ──
        adv3_score = 0.0
        vwap_diff = cur_close - vwap
        if abs(vwap_diff) > atr * 3.5:
            # Overextended trend: keep same-side bias at reduced weight instead of flipping counter-trend
            adv3_score = round(w_vwap * 0.35, 2) * (1.0 if vwap_diff > 0 else -1.0)
            adv3_log = f"VWAP (Weight: {int(w_vwap*100)}%): Spot={cur_close:.0f} vs VWAP={vwap:.0f} [EXTENDED: {vwap_diff:+.1f} pts] (Score: {adv3_score:+.2f})"
        elif vwap_diff >= atr * 0.08:
            adv3_score = w_vwap
            adv3_log = f"VWAP (Weight: {int(w_vwap*100)}%): Spot={cur_close:.0f} vs VWAP={vwap:.0f} [Diff: {vwap_diff:+.1f}] (Score: {adv3_score:+.2f})"
        elif vwap_diff <= -atr * 0.08:
            adv3_score = -w_vwap
            adv3_log = f"VWAP (Weight: {int(w_vwap*100)}%): Spot={cur_close:.0f} vs VWAP={vwap:.0f} [Diff: {vwap_diff:+.1f}] (Score: {adv3_score:+.2f})"
        else:
            adv3_log = f"VWAP (Weight: {int(w_vwap*100)}%): Spot={cur_close:.0f} vs VWAP={vwap:.0f} [At Equilibrium] (Score: +0.00)"

        # ── 5. Advisor 4: Volume Surge / Spot Index Displacement Expansion ──
        adv4_score = 0.0
        is_bull_candle = cur_close > c_open
        if avg_vol > 10.0:
            if vol_ratio >= 1.25 and body >= atr * 0.45:
                adv4_score = w_disp if is_bull_candle else -w_disp
            elif vol_ratio >= 1.10:
                half_d = round(w_disp * 0.5, 2)
                adv4_score = half_d if is_bull_candle else -half_d
            adv4_log = f"Vol Surge (Weight: {int(w_disp*100)}%): {vol_ratio:.1f}x vol, Body={body:.1f} pts (Score: {adv4_score:+.2f})"
        else:
            # NSE cash index tokens report volume=0.0; measure institutional candle displacement & follow-through
            c_range = max(c_high - c_low, 0.01)
            close_pos = (cur_close - c_low) / c_range
            prev_c = self.candles_5m[-2] if len(self.candles_5m) >= 2 else self.candles_5m[-1]
            prev_bull = float(prev_c["close"]) >= float(prev_c["open"])
            prev_bear = float(prev_c["close"]) <= float(prev_c["open"])
            if is_bull_candle and close_pos >= 0.65 and ((body >= atr * 0.50 and prev_bull) or body >= atr * 0.75):
                adv4_score = w_disp
            elif (not is_bull_candle) and close_pos <= 0.35 and ((body >= atr * 0.50 and prev_bear) or body >= atr * 0.75):
                adv4_score = -w_disp
            elif is_bull_candle and close_pos >= 0.60 and body >= atr * 0.30:
                adv4_score = round(w_disp * 0.5, 2)
            elif (not is_bull_candle) and close_pos <= 0.40 and body >= atr * 0.30:
                adv4_score = -round(w_disp * 0.5, 2)
            adv4_log = f"Displacement (Weight: {int(w_disp*100)}%): Body={body:.1f} ({body/max(atr,1):.2f} ATR), Pos={close_pos:.2f} (Score: {adv4_score:+.2f})"

        # ── 6. Advisor 5: Institutional CPR, ORB & PDH/PDL Confluence ──
        adv5_score = 0.0
        adv5_log = "Levels: Inactive"
        if key_levels:
            cpr_top = max(key_levels.tc, key_levels.bc)
            cpr_bot = min(key_levels.tc, key_levels.bc)
            half_l = w_lvl * 0.5
            cpr_bias = 0.0
            if cur_close > cpr_top:
                cpr_bias = half_l
            elif cur_close < cpr_bot:
                cpr_bias = -half_l

            orb_bias = 0.0
            if key_levels.orb_formed and key_levels.orh > 0 and key_levels.orl > 0:
                if cur_close > key_levels.orh:
                    orb_bias = half_l
                elif cur_close < key_levels.orl:
                    orb_bias = -half_l
            if orb_bias == 0.0 and key_levels.pdh > 0 and key_levels.pdl > 0:
                if cur_close > key_levels.pdh:
                    orb_bias = half_l
                elif cur_close < key_levels.pdl:
                    orb_bias = -half_l

            adv5_score = round(max(-w_lvl, min(w_lvl, cpr_bias + orb_bias)), 2)
            adv5_log = f"CPR/ORB (Weight: {int(w_lvl*100)}%): Spot={cur_close:.0f} vs CPR=[{cpr_bot:.0f}-{cpr_top:.0f}] | ORB=[{key_levels.orl:.0f}-{key_levels.orh:.0f}] (Score: {adv5_score:+.2f})"

        # ── 7. Council Aggregation ──
        net_score = round(adv1_score + adv2_score + adv3_score + adv4_score + adv5_score, 2)
        notes = [ml_log, adv2_log, adv3_log, adv4_log, adv5_log]

        # ── 8. Open Interest & PCR Wall Shield Evaluation ──
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

        # ── 9. Higher-Timeframe Trend Structure ──
        htf_bias = self._compute_htf_bias(eval_closes)
        self.last_diagnostics = {
            "net_score": net_score,
            "threshold": self.min_consensus_threshold,
            "veto": "",
            "candidate": "LONG (CE)" if net_score >= self.min_consensus_threshold else ("SHORT (PE)" if net_score <= -self.min_consensus_threshold else "NEUTRAL"),
            "ml_reliable": ml_active,
            "htf_bias": htf_bias,
        }

        def _veto(msg: str) -> None:
            self.last_diagnostics["veto"] = msg
            notes.append(f"VETO: {msg}")
            return None

        # ── 10. Institutional Council Gate & Veto Protocol ──
        if net_score >= self.min_consensus_threshold:
            # Bullish Candidate: Buy CALL (CE)
            direction, opt_type = "LONG", "CE"
            conviction = min(0.95, round(0.50 + (net_score * 0.45), 2))

            # VETO 1: VWAP Absolute Rule (Never buy CE below VWAP)
            if cur_close < (vwap - 5.0):
                if not is_preview:
                    now_t = time.time()
                    if (now_t - self._last_veto_log.get("VWAP_CE_VETO", 0.0)) >= 60.0:
                        logger.warning(f"CE Entry VETOED on {self.index_name}: Spot (₹{cur_close:,.1f}) below VWAP (₹{vwap:,.1f})")
                        self._last_veto_log["VWAP_CE_VETO"] = now_t
                return _veto("Bearish VWAP alignment (Spot < VWAP)")

            # VETO 2: Central Pivot Range (CPR) Chop Zone
            if key_levels:
                cpr_top = max(key_levels.tc, key_levels.bc)
                cpr_bot = min(key_levels.tc, key_levels.bc)
                if cpr_bot <= cur_close <= cpr_top:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("CPR_CHOP_VETO", 0.0)) >= 60.0:
                            logger.warning(f"CE Entry VETOED on {self.index_name}: Trapped inside CPR chop zone ({cpr_bot:.0f}-{cpr_top:.0f})")
                            self._last_veto_log["CPR_CHOP_VETO"] = now_t
                    return _veto("Trapped inside CPR (Theta chop zone)")

            # VETO 3: Higher-Timeframe Trend (15m) Bearish Opposing (permits breakouts & PDL Sweep Reclaim turns)
            if htf_bias == "BEARISH":
                is_breakout = key_levels and (
                    "Breakout" in key_levels.active_sweep
                    or "ORH Bullish" in key_levels.active_sweep
                    or "Reclaim" in key_levels.active_sweep
                )
                if not (is_breakout and cur_close > vwap):
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("HTF_CE_VETO", 0.0)) >= 60.0:
                            logger.warning(f"CE Entry VETOED on {self.index_name}: 15m Trend is BEARISH")
                            self._last_veto_log["HTF_CE_VETO"] = now_t
                    return _veto("Macro 15m Trend is Bearish")

            # VETO 4: Banking Heavyweight Opposing (BankNifty)
            if heavyweights and "BANK" in self.index_name:
                if heavyweights.get("BANKING_FLOW") == "BEARISH":
                    if not is_preview:
                        logger.warning(f"CE Entry VETOED on {self.index_name}: Heavyweights HDFC/ICICI are BEARISH")
                    return _veto("Heavyweights HDFC/ICICI opposing")

            # VETO 5: Institutional PDH Sweep Trap
            if key_levels and "PDH Sweep Rejection" in key_levels.active_sweep:
                is_defused = cur_close > vwap and (key_levels.orh > 0 and cur_close >= key_levels.orh)
                if not is_defused:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("PDH_TRAP", 0.0)) >= 60.0:
                            logger.warning(f"CE Entry VETOED on {self.index_name}: Institutional PDH Sweep Trap detected!")
                            self._last_veto_log["PDH_TRAP"] = now_t
                    return _veto("PDH Sweep Rejection Trap (Liquidity swept, sellers active)")

            # VETO 6: OI Wall Shield (Call Wall / PCR ceiling)
            if oi_metrics:
                safe, reason = self.oi_shield.validate_entry(direction, cur_close, cur_close, oi_metrics)
                if not safe:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get(f"OI_{reason}", 0.0)) >= 60.0:
                            logger.warning(f"Signal Vetoed by OI Shield: {reason}")
                            self._last_veto_log[f"OI_{reason}"] = now_t
                    return _veto(reason)

            # VETO 6b: Counter-Macro + Bearish OI Flow Requires Strong Confirmation
            if key_levels and cur_close < min(key_levels.tc, key_levels.bc) and oi_flow in ("CALL_WRITING_BEARISH", "LONG_UNWINDING_CRASH"):
                req_hi = max(self.min_consensus_threshold + 0.12, 0.70)
                if net_score < req_hi:
                    return _veto(f"Counter-macro CE below CPR vs {oi_flow} (need {req_hi:.2f}, got {net_score:.2f})")

            # VETO 7: Fair Value Gap (FVG) Bearish Supply Veto
            if key_levels and key_levels.fvg_supply:
                if not is_preview:
                    now_t = time.time()
                    if (now_t - self._last_veto_log.get("FVG_SUPPLY_VETO", 0.0)) >= 60.0:
                        logger.warning(f"CE Entry VETOED on {self.index_name}: Inside Bearish FVG Supply Zone ({key_levels.fvg_zone_bottom:.0f}-{key_levels.fvg_zone_top:.0f})")
                        self._last_veto_log["FVG_SUPPLY_VETO"] = now_t
                return _veto(f"Inside Bearish FVG Supply Zone ({key_levels.fvg_zone_bottom:.0f}-{key_levels.fvg_zone_top:.0f})")

            return direction, opt_type, conviction, net_score, notes, oi_metrics

        elif net_score <= -self.min_consensus_threshold:
            # Bearish Candidate: Buy PUT (PE)
            direction, opt_type = "SHORT", "PE"
            conviction = min(0.95, round(0.50 + (abs(net_score) * 0.45), 2))

            # VETO 1: VWAP Absolute Rule (Never buy PE above VWAP)
            if cur_close > (vwap + 5.0):
                if not is_preview:
                    now_t = time.time()
                    if (now_t - self._last_veto_log.get("VWAP_PE_VETO", 0.0)) >= 60.0:
                        logger.warning(f"PE Entry VETOED on {self.index_name}: Spot (₹{cur_close:,.1f}) above VWAP (₹{vwap:,.1f})")
                        self._last_veto_log["VWAP_PE_VETO"] = now_t
                return _veto("Bullish VWAP alignment (Spot > VWAP)")

            # VETO 2: Central Pivot Range (CPR) Chop Zone
            if key_levels:
                cpr_top = max(key_levels.tc, key_levels.bc)
                cpr_bot = min(key_levels.tc, key_levels.bc)
                if cpr_bot <= cur_close <= cpr_top:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("CPR_CHOP_VETO", 0.0)) >= 60.0:
                            logger.warning(f"PE Entry VETOED on {self.index_name}: Trapped inside CPR chop zone ({cpr_bot:.0f}-{cpr_top:.0f})")
                            self._last_veto_log["CPR_CHOP_VETO"] = now_t
                    return _veto("Trapped inside CPR (Theta chop zone)")

            # VETO 3: Higher-Timeframe Trend (15m) Bullish Opposing (permits breakdowns & PDH Sweep Rejection turns)
            if htf_bias == "BULLISH":
                is_breakdown = key_levels and (
                    "Breakdown" in key_levels.active_sweep
                    or "ORL Bearish" in key_levels.active_sweep
                    or "Rejection" in key_levels.active_sweep
                )
                if not (is_breakdown and cur_close < vwap):
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("HTF_PE_VETO", 0.0)) >= 60.0:
                            logger.warning(f"PE Entry VETOED on {self.index_name}: 15m Trend is BULLISH")
                            self._last_veto_log["HTF_PE_VETO"] = now_t
                    return _veto("Macro 15m Trend is Bullish")

            # VETO 4: Banking Heavyweight Opposing (BankNifty)
            if heavyweights and "BANK" in self.index_name:
                if heavyweights.get("BANKING_FLOW") == "BULLISH":
                    if not is_preview:
                        logger.warning(f"PE Entry VETOED on {self.index_name}: Heavyweights HDFC/ICICI are BULLISH")
                    return _veto("Heavyweights HDFC/ICICI opposing")

            # VETO 5: Institutional PDL Sweep Trap
            if key_levels and "PDL Sweep Reclaim" in key_levels.active_sweep:
                is_defused = cur_close < vwap and (key_levels.orl > 0 and cur_close <= key_levels.orl)
                if not is_defused:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get("PDL_TRAP", 0.0)) >= 60.0:
                            logger.warning(f"PE Entry VETOED on {self.index_name}: Institutional PDL Sweep Trap detected!")
                            self._last_veto_log["PDL_TRAP"] = now_t
                    return _veto("PDL Sweep Reclaim Trap (Liquidity swept, buyers active)")

            # VETO 6: OI Wall Shield (Put Wall / PCR floor)
            if oi_metrics:
                safe, reason = self.oi_shield.validate_entry(direction, cur_close, cur_close, oi_metrics)
                if not safe:
                    if not is_preview:
                        now_t = time.time()
                        if (now_t - self._last_veto_log.get(f"OI_{reason}", 0.0)) >= 60.0:
                            logger.warning(f"Signal Vetoed by OI Shield: {reason}")
                            self._last_veto_log[f"OI_{reason}"] = now_t
                    return _veto(reason)

            # VETO 6b: Counter-Macro + Bullish OI Flow Requires Strong Confirmation (Stops weak PE vs Bull Day dips)
            if key_levels and cur_close > max(key_levels.tc, key_levels.bc) and oi_flow in ("PUT_WRITING_BULLISH", "SHORT_COVERING_RALLY"):
                req_hi = max(self.min_consensus_threshold + 0.12, 0.70)
                if abs(net_score) < req_hi:
                    return _veto(f"Counter-macro PE above CPR vs {oi_flow} (need -{req_hi:.2f}, got {net_score:+.2f})")

            # VETO 7: Fair Value Gap (FVG) Bullish Demand Veto
            if key_levels and key_levels.fvg_demand:
                if not is_preview:
                    now_t = time.time()
                    if (now_t - self._last_veto_log.get("FVG_DEMAND_VETO", 0.0)) >= 60.0:
                        logger.warning(f"PE Entry VETOED on {self.index_name}: Inside Bullish FVG Demand Zone ({key_levels.fvg_zone_bottom:.0f}-{key_levels.fvg_zone_top:.0f})")
                        self._last_veto_log["FVG_DEMAND_VETO"] = now_t
                return _veto(f"Inside Bullish FVG Demand Zone ({key_levels.fvg_zone_bottom:.0f}-{key_levels.fvg_zone_top:.0f})")

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
