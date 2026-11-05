"""
Apex Intelligence Engine — Swing Trade Engine
===============================================
Generates multi-day swing trade signals using Daily + 4H timeframes.

Philosophy:
  - Daily trend filter first (EMA 20 > 50 = only longs; EMA 20 < 50 = only shorts)
  - 4H pullback into value zone as entry trigger
  - ATR-based SL (2× daily ATR — wider than intraday)
  - TP at next swing high/low (daily S/R)
  - Max hold: 7 days with daily close confirmation stop

Capital is SEPARATE from scalp pool — swing pool never mixes with intraday.

Usage:
    engine = SwingTradeEngine("BTCUSDT", capital=500.0)
    engine.update_feeds(daily_df, h4_df)
    signal = engine.evaluate()   # None or dict signal
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.engine.swing")

MAX_HOLD_DAYS    = 7
SWING_RISK_PCT   = 0.015    # 1.5% capital risk per swing trade
SL_ATR_MULTIPLE  = 2.0      # SL = 2 × daily ATR below entry
TP1_RR           = 2.0      # TP1 at 2R
TP2_RR           = 4.0      # TP2 at 4R
RSI_OVERSOLD     = 40       # Daily RSI pullback zone for longs
RSI_OVERBOUGHT   = 60       # Daily RSI pullback zone for shorts


@dataclass
class SwingPosition:
    symbol: str
    direction: str         # "LONG" or "SHORT"
    entry_price: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    notional: float
    risk_usd: float
    daily_atr: float
    entry_time: float = field(default_factory=time.time)
    phase: int = 1         # 1=full, 2=runner after TP1
    banked_pnl: float = 0.0
    highest_fav: float = 0.0
    days_held: int = 0

    @property
    def age_days(self) -> float:
        return (time.time() - self.entry_time) / 86400.0


@dataclass
class SwingSignal:
    symbol: str
    direction: str         # "LONG" or "SHORT"
    entry_zone_low: float  # suggested entry range
    entry_zone_high: float
    sl_price: float
    tp1_price: float
    tp2_price: float
    daily_atr: float
    conviction: float      # 0.0–1.0
    reason: str            # human-readable rationale


class SwingTradeEngine:
    """
    Multi-day swing trade signal generator.

    Key differences from scalp engine:
    - Operates on Daily + 4H timeframes
    - SL = 2× daily ATR (much wider)
    - Hold duration: 2–7 days
    - Separate capital pool
    - Lower trade frequency (1–3 per week)
    """

    def __init__(self, symbol: str, capital: float = 500.0, max_hold_days: int = MAX_HOLD_DAYS):
        self.symbol = symbol.upper()
        self.capital = capital
        self.max_hold_days = max_hold_days

        self._daily_df: Optional[pd.DataFrame] = None
        self._h4_df:    Optional[pd.DataFrame] = None
        self._position: Optional[SwingPosition] = None

        # EMA state (recursive, O(1) update)
        self._daily_ema20: Optional[float] = None
        self._daily_ema50: Optional[float] = None
        self._h4_ema9:     Optional[float] = None
        self._h4_ema21:    Optional[float] = None

    # ── Feed Updates ──────────────────────────────────────────────────────────

    def update_feeds(self, daily_df: pd.DataFrame, h4_df: pd.DataFrame) -> None:
        """Update internal candle data from DailyFeedProvider."""
        if daily_df is not None and not daily_df.empty:
            self._daily_df = daily_df.copy()
            self._recompute_daily_emas()
        if h4_df is not None and not h4_df.empty:
            self._h4_df = h4_df.copy()
            self._recompute_h4_emas()

    def _recompute_daily_emas(self) -> None:
        if self._daily_df is None or len(self._daily_df) < 2:
            return
        closes = self._daily_df["close"].values
        ema20 = closes[0]; ema50 = closes[0]
        a20 = 2 / (20 + 1); a50 = 2 / (50 + 1)
        for c in closes[1:]:
            ema20 = c * a20 + ema20 * (1 - a20)
            ema50 = c * a50 + ema50 * (1 - a50)
        self._daily_ema20 = ema20
        self._daily_ema50 = ema50

    def _recompute_h4_emas(self) -> None:
        if self._h4_df is None or len(self._h4_df) < 2:
            return
        closes = self._h4_df["close"].values
        ema9 = closes[0]; ema21 = closes[0]
        a9 = 2 / (9 + 1); a21 = 2 / (21 + 1)
        for c in closes[1:]:
            ema9  = c * a9  + ema9  * (1 - a9)
            ema21 = c * a21 + ema21 * (1 - a21)
        self._h4_ema9  = ema9
        self._h4_ema21 = ema21

    # ── Signal Generation ─────────────────────────────────────────────────────

    def evaluate(self) -> Optional[SwingSignal]:
        """
        Evaluate current market conditions for a swing trade setup.
        Returns SwingSignal if conditions are met, None otherwise.

        Conditions for LONG:
        1. Daily EMA 20 > EMA 50 (uptrend)
        2. Daily RSI in pullback zone (35–55) — not overbought
        3. 4H EMA 9 < 21 (pullback in progress) or just crossed back up
        4. 4H price near daily EMA 20 (mean reversion entry)

        Conditions for SHORT: mirror of above.
        """
        if self._position is not None:
            return None   # Already in a trade
        if self._daily_df is None or len(self._daily_df) < 51:
            return None
        if self._h4_df is None or len(self._h4_df) < 22:
            return None
        if self._daily_ema20 is None or self._daily_ema50 is None:
            return None

        daily_atr  = self._compute_daily_atr()
        daily_rsi  = self._compute_daily_rsi()
        price      = float(self._daily_df["close"].iloc[-1])
        h4_price   = float(self._h4_df["close"].iloc[-1]) if self._h4_df is not None else price

        ema20 = self._daily_ema20
        ema50 = self._daily_ema50

        # ── LONG setup ──────────────────────────────────────────────────────
        if ema20 > ema50 * 1.001:   # Daily uptrend with 0.1% margin
            # RSI pullback zone
            if RSI_OVERSOLD <= daily_rsi <= 60:
                # 4H pulling back toward daily EMA 20
                if h4_price <= ema20 * 1.005:   # price within 0.5% of EMA 20
                    sl   = price - daily_atr * SL_ATR_MULTIPLE
                    tp1  = price + daily_atr * SL_ATR_MULTIPLE * TP1_RR
                    tp2  = price + daily_atr * SL_ATR_MULTIPLE * TP2_RR
                    conv = self._score_long_setup(daily_rsi, price, ema20, ema50)
                    if conv >= 0.55:
                        return SwingSignal(
                            symbol=self.symbol,
                            direction="LONG",
                            entry_zone_low=round(ema20 * 0.998, 1),
                            entry_zone_high=round(ema20 * 1.002, 1),
                            sl_price=round(sl, 1),
                            tp1_price=round(tp1, 1),
                            tp2_price=round(tp2, 1),
                            daily_atr=round(daily_atr, 1),
                            conviction=round(conv, 3),
                            reason=f"Daily uptrend pullback | EMA20>{ema20:,.0f} EMA50>{ema50:,.0f} | RSI={daily_rsi:.0f}",
                        )

        # ── SHORT setup ─────────────────────────────────────────────────────
        elif ema20 < ema50 * 0.999:   # Daily downtrend
            if 40 <= daily_rsi <= RSI_OVERBOUGHT:
                if h4_price >= ema20 * 0.995:
                    sl   = price + daily_atr * SL_ATR_MULTIPLE
                    tp1  = price - daily_atr * SL_ATR_MULTIPLE * TP1_RR
                    tp2  = price - daily_atr * SL_ATR_MULTIPLE * TP2_RR
                    conv = self._score_short_setup(daily_rsi, price, ema20, ema50)
                    if conv >= 0.55:
                        return SwingSignal(
                            symbol=self.symbol,
                            direction="SHORT",
                            entry_zone_low=round(ema20 * 0.998, 1),
                            entry_zone_high=round(ema20 * 1.002, 1),
                            sl_price=round(sl, 1),
                            tp1_price=round(tp1, 1),
                            tp2_price=round(tp2, 1),
                            daily_atr=round(daily_atr, 1),
                            conviction=round(conv, 3),
                            reason=f"Daily downtrend pullback | EMA20<{ema20:,.0f} EMA50<{ema50:,.0f} | RSI={daily_rsi:.0f}",
                        )

        return None

    # ── Position Management ───────────────────────────────────────────────────

    def open_position(self, signal: SwingSignal, fill_price: float) -> SwingPosition:
        """Record a swing position opening."""
        sl_dist  = abs(fill_price - signal.sl_price)
        risk_usd = self.capital * SWING_RISK_PCT
        notional = risk_usd / (sl_dist / fill_price) if sl_dist > 0 else 0.0

        self._position = SwingPosition(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=fill_price,
            sl_price=signal.sl_price,
            tp1_price=signal.tp1_price,
            tp2_price=signal.tp2_price,
            notional=notional,
            risk_usd=risk_usd,
            daily_atr=signal.daily_atr,
            highest_fav=fill_price,
        )
        return self._position

    def should_exit(self, current_price: float) -> tuple[bool, str]:
        """
        Check exit conditions for the active swing position.
        Returns (should_exit, reason).
        """
        pos = self._position
        if pos is None:
            return False, ""

        # Time stop
        if pos.age_days >= self.max_hold_days:
            return True, f"TIME STOP ({self.max_hold_days}d)"

        if pos.direction == "LONG":
            pos.highest_fav = max(pos.highest_fav, current_price)
            # Hard SL
            if current_price <= pos.sl_price:
                return True, "SWING SL"
            # TP1 — take 50% off, move SL to breakeven
            if pos.phase == 1 and current_price >= pos.tp1_price:
                pos.phase = 2
                pos.sl_price = pos.entry_price * 1.001   # slight BE buffer
                return False, ""   # partial — don't exit fully yet
            # TP2 — full exit
            if pos.phase == 2 and current_price >= pos.tp2_price:
                return True, "SWING TP2"
            # Trailing: once at phase 2, trail SL at 1.5× daily ATR below high
            if pos.phase == 2:
                trail_sl = pos.highest_fav - pos.daily_atr * 1.5
                pos.sl_price = max(pos.sl_price, trail_sl)
        else:  # SHORT
            pos.highest_fav = min(pos.highest_fav if pos.highest_fav > 0 else current_price, current_price)
            if current_price >= pos.sl_price:
                return True, "SWING SL"
            if pos.phase == 1 and current_price <= pos.tp1_price:
                pos.phase = 2
                pos.sl_price = pos.entry_price * 0.999
                return False, ""
            if pos.phase == 2 and current_price <= pos.tp2_price:
                return True, "SWING TP2"
            if pos.phase == 2:
                trail_sl = pos.highest_fav + pos.daily_atr * 1.5
                pos.sl_price = min(pos.sl_price, trail_sl)

        return False, ""

    def close_position(self) -> Optional[SwingPosition]:
        pos = self._position
        self._position = None
        return pos

    @property
    def position(self) -> Optional[SwingPosition]:
        return self._position

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_daily_atr(self, period: int = 14) -> float:
        df = self._daily_df
        if df is None or len(df) < period + 1:
            return 0.0
        h, l, c = df["high"], df["low"], df["close"]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        return float(tr.rolling(period).mean().iloc[-1])

    def _compute_daily_rsi(self, period: int = 14) -> float:
        df = self._daily_df
        if df is None or len(df) < period + 1:
            return 50.0
        delta = df["close"].diff()
        gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss  = (-delta).where(delta < 0, 0.0).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi   = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def _score_long_setup(self, rsi: float, price: float, ema20: float, ema50: float) -> float:
        score = 0.5
        # RSI in sweet spot (40–55 = best pullback zone)
        if 40 <= rsi <= 55:    score += 0.15
        elif 55 < rsi <= 60:   score += 0.07
        # Price near EMA 20 (tighter = better)
        dist_pct = abs(price - ema20) / ema20
        if dist_pct < 0.005:   score += 0.15
        elif dist_pct < 0.01:  score += 0.08
        # EMA separation (wider = stronger trend)
        sep = (ema20 - ema50) / ema50
        if sep > 0.02:  score += 0.10
        elif sep > 0.01: score += 0.05
        # 4H aligned
        if self._h4_ema9 and self._h4_ema21 and self._h4_ema9 > self._h4_ema21:
            score += 0.10
        return min(score, 1.0)

    def _score_short_setup(self, rsi: float, price: float, ema20: float, ema50: float) -> float:
        score = 0.5
        if 45 <= rsi <= 60:    score += 0.15
        elif 40 <= rsi < 45:   score += 0.07
        dist_pct = abs(price - ema20) / ema20
        if dist_pct < 0.005:   score += 0.15
        elif dist_pct < 0.01:  score += 0.08
        sep = (ema50 - ema20) / ema50
        if sep > 0.02:  score += 0.10
        elif sep > 0.01: score += 0.05
        if self._h4_ema9 and self._h4_ema21 and self._h4_ema9 < self._h4_ema21:
            score += 0.10
        return min(score, 1.0)

