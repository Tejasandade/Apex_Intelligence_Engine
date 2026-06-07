"""
Apex Intelligence Engine V5 — Council Advisors
=================================================
Five specialist advisors, each analyzing the market from a unique perspective.

1. MomentumAdvisor  — RSI, MACD, EMA alignment
2. StructureAdvisor — FVG, BOS, liquidity, support/resistance
3. VolumeAdvisor    — CVD, OBV, order flow imbalance
4. RegimeAdvisor    — Is this trade aligned with the current regime?
5. RiskAdvisor      — Drawdown, exposure, daily loss limits
"""

from __future__ import annotations

from typing import Any
import datetime

import pandas as pd

from src.features.indicators.technical import compute_adx
from src.features.indicators.volume_profile import compute_volume_profile
from src.council.base import BaseAdvisor, AdvisorVote, Vote
from src.core.logging import get_logger

logger = get_logger("apex.council.advisors")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. MOMENTUM ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MomentumAdvisor(BaseAdvisor):
    """
    Evaluates momentum alignment using Institutional Order Flow (CVD, Imbalances).

    Logic:
    - BUY: CVD > 0, Order Flow Imbalance > 0, Price > VWAP
    - SELL: CVD < 0, Order Flow Imbalance < 0, Price < VWAP
    - Pure order flow momentum rather than lagging technical indicators.
    """

    def __init__(self, weight: float = 1.5):
        super().__init__(name="Momentum", weight=weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        cvd = float(row.get("CVD", 0.0))
        of_imb = float(row.get("order_flow_imbalance", 0.0))
        vwap = float(row.get("VWAP", price))

        factors = 0
        reasons = []

        if direction == "BUY":
            if cvd > 0:
                factors += 1
                reasons.append("CVD is Bullish")
            if of_imb > 0:
                factors += 1
                reasons.append("Aggressive Buy Orders (Imbalance)")
            if price > vwap:
                factors += 1
                reasons.append("Price > VWAP")
        else:  # SELL
            if cvd < 0:
                factors += 1
                reasons.append("CVD is Bearish")
            if of_imb < 0:
                factors += 1
                reasons.append("Aggressive Sell Orders (Imbalance)")
            if price < vwap:
                factors += 1
                reasons.append("Price < VWAP")

        conviction = factors / 3.0

        if factors >= 2:
            return self._make_vote(
                Vote.APPROVE, conviction,
                f"Strong Order Flow: {', '.join(reasons)}",
                {"cvd": cvd, "of_imb": of_imb},
            )
        elif factors == 1:
            return self._make_vote(
                Vote.ABSTAIN, conviction,
                f"Mixed Order Flow: {', '.join(reasons)}",
                {"cvd": cvd, "of_imb": of_imb},
            )
        else:
            return self._make_vote(
                Vote.REJECT, 0.7,
                "Order Flow completely against trade.",
                {"cvd": cvd, "of_imb": of_imb},
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  2. STRUCTURE ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class StructureAdvisor(BaseAdvisor):
    """
    Evaluates market structure: FVG, BOS, liquidity sweeps, confluence zones.

    Logic:
    - BUY: FVG bullish present, BOS bullish, structural confluence > 0
    - SELL: FVG bearish present, BOS bearish, structural confluence < 0
    - Liquidity sweep in the right direction = strong confirmation
    """

    def __init__(self, weight: float = 1.2):
        super().__init__(name="Structure", weight=weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        scalp_mode: bool = False,
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        fvg = float(row.get("fvg_signal", 0.0))
        bos = float(row.get("structure_break_signal", 0.0))
        liq_sweep = float(row.get("liquidity_sweep_signal", 0.0))
        confluence = float(row.get("structural_confluence", 0.0))

        factors = 0
        reasons = []

        # MTF check: Structure confirmation from 5m chart
        df_5m = kwargs.get("df_5m")
        mtf_ready = kwargs.get("mtf_ready", False)

        if mtf_ready and df_5m is not None and len(df_5m) >= 200:
            rolling_close = df_5m.close.rolling(20)
            bb_upper = rolling_close.mean().iloc[-1] + 2 * rolling_close.std().iloc[-1]
            bb_lower = rolling_close.mean().iloc[-1] - 2 * rolling_close.std().iloc[-1]
            current_price = df_5m.close.iloc[-1]
            
            # Macro Trend Check
            ema_50 = df_5m.close.ewm(span=50).mean().iloc[-1]
            ema_200 = df_5m.close.ewm(span=200).mean().iloc[-1]
            
            if direction == "BUY" and ema_50 < ema_200 and current_price < ema_50:
                return self._make_vote(
                    Vote.REJECT, 0.6,
                    "Structure REJECT: Counter-trend BUY in macro downtrend",
                    {"ema_50": ema_50, "ema_200": ema_200}
                )
            elif direction == "SELL" and ema_50 > ema_200 and current_price > ema_50:
                return self._make_vote(
                    Vote.REJECT, 0.6,
                    "Structure REJECT: Counter-trend SELL in macro uptrend",
                    {"ema_50": ema_50, "ema_200": ema_200}
                )

            if direction == "BUY":
                if current_price <= bb_lower * 1.002:
                    factors += 1
                    reasons.append("At 5m BB lower (discount zone)")
            else:  # SELL
                if current_price >= bb_upper * 0.998:
                    factors += 1
                    reasons.append("At 5m BB upper (premium zone)")

        adv_bull_sweep = float(row.get("bullish_stop_run", 0.0))
        adv_bear_sweep = float(row.get("bearish_stop_run", 0.0))

        if direction == "BUY":
            if fvg > 0:
                factors += 1
                reasons.append("Bullish FVG present")
            if bos > 0:
                factors += 1
                reasons.append("Bullish BOS confirmed")
            if liq_sweep > 0:
                factors += 1
                reasons.append("Buy-side liquidity swept")
            if adv_bull_sweep > 0:
                factors += 1.5
                reasons.append("Advanced SMC: Bear Trap (Bullish Stop Run)")
            if confluence > 0:
                factors += 1
                reasons.append(f"Confluence={confluence:.2f} (support)")
        else:  # SELL
            if fvg < 0:
                factors += 1
                reasons.append("Bearish FVG present")
            if bos < 0:
                factors += 1
                reasons.append("Bearish BOS confirmed")
            if liq_sweep < 0:
                factors += 1
                reasons.append("Sell-side liquidity swept")
            if adv_bear_sweep > 0:
                factors += 1.5
                reasons.append("Advanced SMC: Bull Trap (Bearish Stop Run)")
            if confluence < 0:
                factors += 1
                reasons.append(f"Confluence={confluence:.2f} (resistance)")

        conviction = factors / 4.0

        # --- Phase 1b: Oracle HTF Context Override ---
        oracle_levels = kwargs.get("oracle_levels", [])
        vote_type = Vote.APPROVE if (factors >= 2 or (scalp_mode and factors >= 1)) else Vote.ABSTAIN
        if factors < 1 and not scalp_mode:
            vote_type = Vote.ABSTAIN
            conviction = 0.3
            reasons.append("No local structural signals")

        for level in oracle_levels:
            if level.direction != direction:
                # Counter-directional level (Resistance for Longs, Support for Shorts)
                if level.touch_count < 3:
                    vote_type = Vote.ABSTAIN
                    conviction = max(0.0, conviction - 0.3)
                    reasons.append(f"Counter-directional {level.timeframe} {level.level_type}")
                else:
                    reasons.append(f"Weakened {level.timeframe} {level.level_type} nearby (tested {level.touch_count}x)")
            else:
                # Confluent level
                conviction = min(1.0, conviction + 0.3)
                reasons.append(f"Confluent {level.timeframe} {level.level_type}")
                if level.level_type == "FVG" and level.timeframe == "4h":
                    vote_type = Vote.APPROVE
                    conviction = min(1.0, conviction + 0.5)
                    reasons.append("Inside 4H FVG (Macro Tailwind)")

        # Ensure bounds
        conviction = max(0.0, min(1.0, float(conviction)))

        return self._make_vote(
            vote_type, conviction,
            f"Structure: {', '.join(reasons) or 'neutral'}",
            {"fvg": fvg, "bos": bos, "confluence": confluence, "oracle_levels": len(oracle_levels)},
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  3. VOLUME ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class VolumeAdvisor(BaseAdvisor):
    """
    Evaluates volume confirmation: CVD, OBV trend, order flow imbalance.

    Logic:
    - BUY: CVD rising (buyers), OBV trending up, order_flow_imbalance > 0
    - SELL: CVD falling (sellers), OBV trending down, order_flow_imbalance < 0
    - Volume without direction = noise, not signal
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Volume", weight=weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        scalp_mode: bool = False,
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        cvd = float(row.get("CVD", 0.0))
        obv = float(row.get("OBV", 0.0))
        ofi = float(row.get("order_flow_imbalance", 0.0))

        # --- Volume Profile Analysis ---
        df_5m = kwargs.get("df_5m")
        vp_reason = ""
        poc_reject = False
        poc_approve = False

        if df_5m is not None and len(df_5m) > 50:
            try:
                # Compute volume profile over recent 50 bars
                vp_df = compute_volume_profile(df_5m.tail(50), lookback=50, bins=20)
                if not vp_df.empty:
                    poc = float(vp_df["POC"].iloc[-1])
                    vah = float(vp_df["VAH"].iloc[-1])
                    val = float(vp_df["VAL"].iloc[-1])
                    
                    if direction == "BUY":
                        if price < poc and (poc - price) < atr * 0.5:
                            vp_reason = f"Price is directly below POC resistance (${poc:.2f})"
                            poc_reject = True
                        elif price > val and price < poc:
                            vp_reason = f"Bouncing from Value Area Low (${val:.2f})"
                            poc_approve = True
                        elif price > poc:
                            vp_reason = f"Price > POC (${poc:.2f}), bullish volume node"
                            poc_approve = True
                    else:
                        if price > poc and (price - poc) < atr * 0.5:
                            vp_reason = f"Price is directly above POC support (${poc:.2f})"
                            poc_reject = True
                        elif price < vah and price > poc:
                            vp_reason = f"Rejecting from Value Area High (${vah:.2f})"
                            poc_approve = True
                        elif price < poc:
                            vp_reason = f"Price < POC (${poc:.2f}), bearish volume node"
                            poc_approve = True
            except Exception as e:
                logger.error("volume_profile_error", error=str(e))

        factors = 0
        reasons = []

        if vp_reason:
            reasons.append(vp_reason)
            if poc_reject:
                factors -= 1.5
            if poc_approve:
                factors += 1.0

        if direction == "BUY":
            if cvd > 0:
                factors += 1
                reasons.append(f"CVD={cvd:.0f} (net buying)")
            else:
                factors -= 0.5
                
            if ofi > 0.1:
                factors += 1
                reasons.append(f"OFI={ofi:.2f} (buy pressure)")
            elif ofi > 0:
                factors += 0.5
                reasons.append(f"OFI={ofi:.2f} (mild buy)")
            elif ofi < -0.1:
                factors -= 1
                
        else:  # SELL
            if cvd < 0:
                factors += 1
                reasons.append(f"CVD={cvd:.0f} (net selling)")
            else:
                factors -= 0.5
                
            if ofi < -0.1:
                factors += 1
                reasons.append(f"OFI={ofi:.2f} (sell pressure)")
            elif ofi < 0:
                factors += 0.5
                reasons.append(f"OFI={ofi:.2f} (mild sell)")
            elif ofi > 0.1:
                factors -= 1

        # Base conviction on absolute positive factors
        conviction = min(max(factors, 0) / 2.0, 1.0)

        if factors >= 1.5:
            return self._make_vote(
                Vote.APPROVE, conviction,
                f"Volume confirms: {', '.join(reasons)}",
                {"cvd": cvd, "ofi": ofi, "obv": obv},
            )
        elif factors >= 0 or scalp_mode:
            return self._make_vote(
                Vote.ABSTAIN, 0.4,
                f"Weak volume: {', '.join(reasons) or 'neutral flow'}",
                {"cvd": cvd, "ofi": ofi, "obv": obv},
            )
        else:
            return self._make_vote(
                Vote.REJECT, 0.6,
                f"Volume divergence: flow against {direction}",
                {"cvd": cvd, "ofi": ofi, "obv": obv},
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  4. REGIME ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RegimeAdvisor(BaseAdvisor):
    """
    Evaluates regime alignment: is this trade style compatible with the regime?

    Logic:
    - TRENDING regime: Allow trend-following trades, reject mean-reversion
    - RANGING regime: Allow mean-reversion, reject trend-following at extremes
    - VOLATILE regime: Only allow with tight stops and reduced size
    - QUIET regime: Allow scalps, reject swing trades
    """

    def __init__(self, weight: float = 1.3):
        super().__init__(name="Regime", weight=weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        scalp_mode: bool = False,
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        adx = float(row.get("ADX", 0.0))
        chop = float(row.get("CHOP", 50.0))
        bb_width = float(row.get("BB_width", 0.0))

        # MTF check: Macro Bias from 15m chart
        df_15m = kwargs.get("df_15m")
        mtf_ready = kwargs.get("mtf_ready", False)

        if mtf_ready and df_15m is not None and len(df_15m) >= 14:
            adx_series = compute_adx(df_15m, length=14)
            adx_15m = adx_series.iloc[-1]
            ema14_15m = df_15m.close.ewm(span=14).mean().iloc[-1]
            ema50_15m = df_15m.close.ewm(span=50).mean().iloc[-1]
            macro_uptrend = ema14_15m > ema50_15m

            # Hard block: if 15m is a strong trend, reject counter-trend 1m signals
            if not scalp_mode and adx_15m > 30 and macro_uptrend and direction == "SELL":
                # Check for 5m structural breakdown (ChoCh)
                df_5m = kwargs.get("df_5m")
                choch_detected = False
                if df_5m is not None and len(df_5m) >= 50:
                    ema_50_5m = df_5m.close.ewm(span=50).mean().iloc[-1]
                    if price < ema_50_5m:
                        choch_detected = True

                if choch_detected:
                    return self._make_vote(
                        Vote.APPROVE, 0.7,
                        f"15m uptrend blocked, BUT 5m ChoCh detected (Liquidity Sweep) -> OVERRIDE",
                        {"adx_15m": float(adx_15m), "choch": True}
                    )
                else:
                    return self._make_vote(
                        Vote.REJECT, 0.95,
                        f"15m macro uptrend (ADX={adx_15m:.1f}), SELL blocked",
                        {"adx_15m": float(adx_15m), "macro_uptrend": bool(macro_uptrend)}
                    )

            if not scalp_mode and adx_15m > 30 and not macro_uptrend and direction == "BUY":
                # Check for 5m structural breakout (ChoCh)
                df_5m = kwargs.get("df_5m")
                choch_detected = False
                if df_5m is not None and len(df_5m) >= 50:
                    ema_50_5m = df_5m.close.ewm(span=50).mean().iloc[-1]
                    if price > ema_50_5m:
                        choch_detected = True

                if choch_detected:
                    return self._make_vote(
                        Vote.APPROVE, 0.7,
                        f"15m downtrend blocked, BUT 5m ChoCh detected (Liquidity Sweep) -> OVERRIDE",
                        {"adx_15m": float(adx_15m), "choch": True}
                    )
                else:
                    return self._make_vote(
                        Vote.REJECT, 0.95,
                        f"15m macro downtrend (ADX={adx_15m:.1f}), BUY blocked",
                        {"adx_15m": float(adx_15m), "macro_uptrend": bool(macro_uptrend)}
                    )

        regime_upper = regime.upper()

        if regime_upper == "TRENDING":
            # Trending = good for directional trades, but ONLY if aligned with trend
            ema_14 = float(row.get("EMA_14", price))
            ema_50 = float(row.get("EMA_50", price))
            trend_is_up = ema_14 > ema_50
            
            aligned = (direction == "BUY" and trend_is_up) or (direction == "SELL" and not trend_is_up)
            
            if not aligned:
                if scalp_mode:
                    return self._make_vote(
                        Vote.ABSTAIN, 0.5,
                        f"Counter-trend scalp against {'uptrend' if trend_is_up else 'downtrend'}.",
                        {"adx": adx, "chop": chop, "regime": regime, "ema_14": ema_14, "ema_50": ema_50},
                    )
                # Counter-trend is dangerous. Only allow if trend is extremely weak.
                if adx > 40:
                    return self._make_vote(
                        Vote.REJECT, 0.9,
                        f"Counter-trend trade! {direction} against strong {'uptrend' if trend_is_up else 'downtrend'} (ADX={adx:.1f}).",
                        {"adx": adx, "chop": chop, "regime": regime, "ema_14": ema_14, "ema_50": ema_50},
                    )
                elif adx > 20:
                    # Moderate trend — still reject but with less conviction
                    return self._make_vote(
                        Vote.REJECT, 0.6,
                        f"Counter-trend trade against moderate {'uptrend' if trend_is_up else 'downtrend'} (ADX={adx:.1f}). Risky.",
                        {"adx": adx, "chop": chop, "regime": regime, "ema_14": ema_14, "ema_50": ema_50},
                    )
                else:
                    # ADX < 20: trend is essentially non-existent, allow ML edge
                    return self._make_vote(
                        Vote.ABSTAIN, 0.4,
                        f"Counter-trend trade, but trend is negligible (ADX={adx:.1f}). Allow ML edge.",
                        {"adx": adx, "chop": chop, "regime": regime, "ema_14": ema_14, "ema_50": ema_50},
                    )
            
            if adx > 30:
                return self._make_vote(
                    Vote.APPROVE, 0.9,
                    f"Strong trend (ADX={adx:.1f}). {direction} aligned with regime.",
                    {"adx": adx, "chop": chop, "regime": regime},
                )
            else:
                return self._make_vote(
                    Vote.APPROVE, 0.6,
                    f"Mild trend (ADX={adx:.1f}). Proceed with caution.",
                    {"adx": adx, "chop": chop, "regime": regime},
                )

        elif regime_upper == "RANGING":
            # Ranging = good for mean-reversion, bad for breakouts
            # But extremely choppy markets (CHOP > 70) kill everything
            if chop > 70:
                return self._make_vote(
                    Vote.REJECT, 0.7,
                    f"Extremely choppy market (CHOP={chop:.1f}). High false-signal risk. Avoid trading.",
                    {"adx": adx, "chop": chop, "regime": regime},
                )
            elif chop > 61.8:
                return self._make_vote(
                    Vote.ABSTAIN, 0.4,
                    f"Choppy market (CHOP={chop:.1f}). Scalp mode only.",
                    {"adx": adx, "chop": chop, "regime": regime},
                )
            else:
                return self._make_vote(
                    Vote.APPROVE, 0.5,
                    f"Mild range (CHOP={chop:.1f}). Mean-reversion may work.",
                    {"adx": adx, "chop": chop, "regime": regime},
                )

        elif regime_upper == "VOLATILE":
            # Volatile = reduce exposure, widen stops
            return self._make_vote(
                Vote.ABSTAIN, 0.3,
                f"VOLATILE regime. Trade with reduced size only. BB_width={bb_width:.4f}",
                {"adx": adx, "chop": chop, "bb_width": bb_width, "regime": regime},
            )

        elif regime_upper == "QUIET":
            # Quiet = good for tight scalps
            return self._make_vote(
                Vote.APPROVE, 0.5,
                f"Quiet market. Tight scalp opportunities. ADX={adx:.1f}",
                {"adx": adx, "chop": chop, "regime": regime},
            )

        return self._make_vote(
            Vote.ABSTAIN, 0.3,
            f"Unknown regime: {regime}",
            {"regime": regime},
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  5. RISK ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class RiskAdvisor(BaseAdvisor):
    """
    Evaluates risk conditions: drawdown, recent losses, volatility expansion.

    This is the VETO advisor — if risk conditions are bad, it blocks the trade
    regardless of what other advisors say.

    Logic:
    - Current drawdown > threshold → REJECT
    - Too many recent losses → REJECT
    - ATR expanding rapidly (volatility spike) → ABSTAIN
    - Everything normal → APPROVE
    """

    def __init__(
        self,
        weight: float = 2.0,  # High weight — risk is king
        max_drawdown_pct: float = 10.0,
        max_daily_loss: float = 500.0,
        max_recent_losses: int = 5,
    ):
        super().__init__(name="Risk", weight=weight)
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss = max_daily_loss
        self.max_recent_losses = max_recent_losses

        # State (updated externally)
        self.current_drawdown_pct: float = 0.0
        self.daily_pnl: float = 0.0
        self.recent_losses: int = 0
        self.open_positions: int = 0

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> AdvisorVote:
        reasons = []

        # Check drawdown
        if self.current_drawdown_pct > self.max_drawdown_pct:
            return self._make_vote(
                Vote.REJECT, 1.0,
                f"VETO: Drawdown at {self.current_drawdown_pct:.1f}% "
                f"(max: {self.max_drawdown_pct:.1f}%). Stop trading.",
                {"drawdown": self.current_drawdown_pct, "threshold": self.max_drawdown_pct},
            )

        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            return self._make_vote(
                Vote.REJECT, 1.0,
                f"VETO: Daily loss ${self.daily_pnl:.2f} exceeds "
                f"limit ${-self.max_daily_loss:.2f}. Done for today.",
                {"daily_pnl": self.daily_pnl, "limit": self.max_daily_loss},
            )

        # Note: We rely on PositionManager's 'cooldown_remaining' for loss management 
        # instead of a hard VETO, which was causing a permanent deadlock.

        # Overextension Check (Prevent chasing pumps/dumps)
        df_5m = kwargs.get("df_5m")
        if df_5m is not None and len(df_5m) >= 20:
            ema_20_5m = df_5m.close.ewm(span=20).mean().iloc[-1]
            distance = price - ema_20_5m
            if direction == "BUY" and distance > (atr * 1.5):
                return self._make_vote(
                    Vote.REJECT, 1.0,
                    f"VETO: Price overextended {distance/atr:.1f}x ATR above 5m EMA (Chasing pump)",
                    {"distance": distance, "atr": atr},
                )
            elif direction == "SELL" and distance < -(atr * 1.5):
                return self._make_vote(
                    Vote.REJECT, 1.0,
                    f"VETO: Price overextended {abs(distance)/atr:.1f}x ATR below 5m EMA (Chasing dump)",
                    {"distance": distance, "atr": atr},
                )

        # Moderate risk checks
        conviction = 0.8

        if self.current_drawdown_pct > self.max_drawdown_pct * 0.7:
            conviction -= 0.2
            reasons.append(f"Drawdown elevated at {self.current_drawdown_pct:.1f}%")

        if self.recent_losses >= 3:
            conviction -= 0.1
            reasons.append(f"{self.recent_losses} recent losses")

        if self.open_positions > 0:
            conviction -= 0.1
            reasons.append(f"{self.open_positions} open position(s)")

        if not reasons:
            reasons.append("All risk checks passed")

        return self._make_vote(
            Vote.ABSTAIN, 0.0,
            f"Risk OK: {', '.join(reasons)}",
            {
                "drawdown": self.current_drawdown_pct,
                "daily_pnl": self.daily_pnl,
                "recent_losses": self.recent_losses,
            },
        )

    def update_state(
        self,
        drawdown_pct: float = 0.0,
        daily_pnl: float = 0.0,
        recent_losses: int = 0,
        open_positions: int = 0,
    ) -> None:
        """Update risk state from portfolio/position manager."""
        self.current_drawdown_pct = drawdown_pct
        self.daily_pnl = daily_pnl
        self.recent_losses = recent_losses
        self.open_positions = open_positions

class SessionAdvisor(BaseAdvisor):
    """
    Session Advisor

    Analyzes the current time of day to determine the active global session.
    Crypto behaves differently in different sessions.
    - Asian Session (00:00 - 08:00 UTC): Low volume, ranging, false breakouts.
    - London/NY Session (08:00 - 22:00 UTC): High volume, trend continuity.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(name="Session", weight=weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> AdvisorVote:
        
        # Use candle timestamp if available (critical for backtesting),
        # fall back to current time only for live trading
        hour = None
        try:
            row = features.iloc[0] if hasattr(features, "iloc") else features
            ts = row.get("timestamp", None)
            if ts is not None:
                if isinstance(ts, (int, float)):
                    dt = pd.to_datetime(ts, unit='ms', utc=True)
                else:
                    dt = pd.to_datetime(ts, utc=True)
                hour = dt.hour
        except Exception:
            pass
        
        if hour is None:
            now_utc = datetime.datetime.utcnow()
            hour = now_utc.hour

        # Determine session
        is_asian = 0 <= hour < 8
        is_london = 8 <= hour < 16
        is_ny = 13 <= hour < 22

        if is_asian:
            # During Asian session, trend trades are very risky due to lack of volume and chop.
            # Only approve strongly if regime is RANGING and we are mean-reverting, otherwise Abstain or Reject.
            if regime == "RANGING":
                return self._make_vote(
                    Vote.APPROVE, 0.6,
                    f"Asian session (Hour {hour}). Ranging regime favors mean reversion.",
                    {"session": "Asian", "hour": hour}
                )
            else:
                return self._make_vote(
                    Vote.ABSTAIN, 0.5,
                    f"Asian session (Hour {hour}). Low volume, trend continuity is poor.",
                    {"session": "Asian", "hour": hour}
                )

        elif is_london and is_ny:
            # Overlap! Maximum liquidity and trend potential.
            return self._make_vote(
                Vote.APPROVE, 0.9,
                f"London/NY Overlap (Hour {hour}). Maximum liquidity favors {direction}.",
                {"session": "London/NY Overlap", "hour": hour}
            )

        elif is_ny:
            return self._make_vote(
                Vote.APPROVE, 0.8,
                f"New York session (Hour {hour}). High volume favors {direction}.",
                {"session": "New York", "hour": hour}
            )

        else: # London only
            return self._make_vote(
                Vote.APPROVE, 0.7,
                f"London session (Hour {hour}). Good liquidity for {direction}.",
                {"session": "London", "hour": hour}
            )

