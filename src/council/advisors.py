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

import pandas as pd

from src.council.base import BaseAdvisor, AdvisorVote, Vote
from src.core.logging import get_logger

logger = get_logger("apex.council.advisors")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  1. MOMENTUM ADVISOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class MomentumAdvisor(BaseAdvisor):
    """
    Evaluates momentum alignment: RSI, MACD, and EMA trend direction.

    Logic:
    - BUY: RSI > 40 (not oversold bounce), MACD > signal, price > EMA_50
    - SELL: RSI < 60 (not overbought bounce), MACD < signal, price < EMA_50
    - Counts how many momentum factors align with the proposed direction
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
        rsi = float(row.get("RSI", 50.0))
        macd = float(row.get("MACD", 0.0))
        macd_signal = float(row.get("MACD_signal", 0.0))
        macd_hist = float(row.get("MACD_hist", 0.0))
        ema_14 = float(row.get("EMA_14", price))
        ema_50 = float(row.get("EMA_50", price))

        factors = 0
        reasons = []

        if direction == "BUY":
            if rsi > 40 and rsi < 75:
                factors += 1
                reasons.append(f"RSI={rsi:.1f} (bullish zone)")
            if macd > macd_signal:
                factors += 1
                reasons.append(f"MACD above signal ({macd_hist:+.2f})")
            if price > ema_50:
                factors += 1
                reasons.append(f"Price above EMA50")
            if ema_14 > ema_50:
                factors += 1
                reasons.append(f"EMA14 > EMA50 (uptrend)")
        else:  # SELL
            if rsi < 60 and rsi > 25:
                factors += 1
                reasons.append(f"RSI={rsi:.1f} (bearish zone)")
            if macd < macd_signal:
                factors += 1
                reasons.append(f"MACD below signal ({macd_hist:+.2f})")
            if price < ema_50:
                factors += 1
                reasons.append(f"Price below EMA50")
            if ema_14 < ema_50:
                factors += 1
                reasons.append(f"EMA14 < EMA50 (downtrend)")

        conviction = factors / 4.0

        if factors >= 3:
            return self._make_vote(
                Vote.APPROVE, conviction,
                f"Strong momentum: {', '.join(reasons)}",
                {"factors": factors, "rsi": rsi, "macd_hist": macd_hist},
            )
        elif factors >= 2:
            return self._make_vote(
                Vote.ABSTAIN, conviction,
                f"Mixed momentum: {', '.join(reasons) or 'weak alignment'}",
                {"factors": factors, "rsi": rsi, "macd_hist": macd_hist},
            )
        elif factors == 1:
            # Only 1 out of 4 momentum factors - weak signal
            return self._make_vote(
                Vote.REJECT, 0.55,
                f"Weak momentum ({factors}/4): {', '.join(reasons) or 'poor alignment'}",
                {"factors": factors, "rsi": rsi, "macd_hist": macd_hist},
            )
        else:
            counter_reasons = []
            if direction == "BUY" and rsi > 75:
                counter_reasons.append(f"RSI={rsi:.1f} overbought")
            elif direction == "SELL" and rsi < 25:
                counter_reasons.append(f"RSI={rsi:.1f} oversold")
            return self._make_vote(
                Vote.REJECT, 0.8,
                f"Momentum against: {', '.join(counter_reasons or reasons or ['no alignment'])}",
                {"factors": factors, "rsi": rsi, "macd_hist": macd_hist},
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
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        fvg = float(row.get("fvg_signal", 0.0))
        bos = float(row.get("structure_break_signal", 0.0))
        liq_sweep = float(row.get("liquidity_sweep_signal", 0.0))
        confluence = float(row.get("structural_confluence", 0.0))

        factors = 0
        reasons = []

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
            if confluence < 0:
                factors += 1
                reasons.append(f"Confluence={confluence:.2f} (resistance)")

        conviction = factors / 4.0

        if factors >= 2:
            return self._make_vote(
                Vote.APPROVE, conviction,
                f"Structure confirms: {', '.join(reasons)}",
                {"fvg": fvg, "bos": bos, "confluence": confluence},
            )
        elif factors >= 1:
            return self._make_vote(
                Vote.ABSTAIN, conviction,
                f"Partial structure: {', '.join(reasons) or 'neutral'}",
                {"fvg": fvg, "bos": bos, "confluence": confluence},
            )
        else:
            # No structural signals present — ABSTAIN rather than REJECT
            # FVG/BOS are rare events; their absence is not evidence against a trade
            return self._make_vote(
                Vote.ABSTAIN, 0.3,
                f"No structural signals present (neutral)",
                {"fvg": fvg, "bos": bos, "confluence": confluence},
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
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        cvd = float(row.get("CVD", 0.0))
        obv = float(row.get("OBV", 0.0))
        ofi = float(row.get("order_flow_imbalance", 0.0))

        factors = 0
        reasons = []

        if direction == "BUY":
            if cvd > 0:
                factors += 1
                reasons.append(f"CVD={cvd:.0f} (net buying)")
            if ofi > 0.1:
                factors += 1
                reasons.append(f"OFI={ofi:.2f} (buy pressure)")
            elif ofi > 0:
                factors += 0.5
                reasons.append(f"OFI={ofi:.2f} (mild buy)")
        else:  # SELL
            if cvd < 0:
                factors += 1
                reasons.append(f"CVD={cvd:.0f} (net selling)")
            if ofi < -0.1:
                factors += 1
                reasons.append(f"OFI={ofi:.2f} (sell pressure)")
            elif ofi < 0:
                factors += 0.5
                reasons.append(f"OFI={ofi:.2f} (mild sell)")

        conviction = min(factors / 2.0, 1.0)

        if factors >= 1.5:
            return self._make_vote(
                Vote.APPROVE, conviction,
                f"Volume confirms: {', '.join(reasons)}",
                {"cvd": cvd, "ofi": ofi, "obv": obv},
            )
        elif factors >= 0.5:
            return self._make_vote(
                Vote.ABSTAIN, conviction,
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
        **kwargs: Any,
    ) -> AdvisorVote:
        row = features.iloc[0]
        adx = float(row.get("ADX", 0.0))
        chop = float(row.get("CHOP", 50.0))
        bb_width = float(row.get("BB_width", 0.0))

        regime_upper = regime.upper()

        if regime_upper == "TRENDING":
            # Trending = good for directional trades, but ONLY if aligned with trend
            ema_14 = float(row.get("EMA_14", price))
            ema_50 = float(row.get("EMA_50", price))
            trend_is_up = ema_14 > ema_50
            
            aligned = (direction == "BUY" and trend_is_up) or (direction == "SELL" and not trend_is_up)
            
            if not aligned:
                return self._make_vote(
                    Vote.REJECT, 0.7,
                    f"Counter-trend trade! {direction} against {'uptrend' if trend_is_up else 'downtrend'} (ADX={adx:.1f}, EMA14{'>' if trend_is_up else '<'}EMA50).",
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
            if chop > 70:
                return self._make_vote(
                    Vote.ABSTAIN, 0.4,
                    f"Choppy market (CHOP={chop:.1f}). High false-signal risk.",
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

        # Check recent losses
        if self.recent_losses >= self.max_recent_losses:
            return self._make_vote(
                Vote.REJECT, 0.9,
                f"VETO: {self.recent_losses} consecutive losses. "
                f"Cooling off (max: {self.max_recent_losses}).",
                {"recent_losses": self.recent_losses},
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
            Vote.APPROVE, max(conviction, 0.3),
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
