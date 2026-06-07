"""
Apex Intelligence Engine V5 — Smart Trailing Stop (Regime-Adaptive)
=====================================================================
3-phase exit management with DUAL-MODE operation:

  TREND MODE  (TRENDING / VOLATILE)
    → Wide stops, let winners run, scale-out milestones
    → Breakeven at 1.5 ATR, Trailing at 2.0 ATR

  SCALP MODE  (RANGING / QUIET)
    → Tight stops, quick exits, no scale-out
    → Breakeven at 0.8 ATR, Trailing at 1.0 ATR

Phases:
    INITIAL    → Fixed stop loss. Standard protection.
    BREAKEVEN  → Stop moves to entry + fees. Risk-free trade.
    TRAILING   → Stop follows price by N×ATR. Locks in profit.

Key design decisions:
    1. Trail only ratchets on CANDLE CLOSE (anti-wick protection)
       - Stop-hit checks still happen on every tick (instant exit)
       - But the trail level only moves on confirmed candle closes
       - This prevents liquidity sweep wicks from ruining the trail
    
    2. Regime-aware trail distance AND phase triggers
       - TRENDING: 2.0× ATR trail, breakeven at 1.5 ATR profit
       - RANGING:  0.8× ATR trail, breakeven at 0.8 ATR profit
       - VOLATILE: 2.5× ATR trail, breakeven at 1.5 ATR profit
       - QUIET:    0.6× ATR trail, breakeven at 0.8 ATR profit
    
    3. Partial scale-out milestones (TREND MODE ONLY)
       - At 1× R:R target: close 33% of position
       - At 2× R:R target: close another 33%
       - Remaining 34% rides the trailing stop to maximise winners
       - Disabled in SCALP MODE (take full profit at target)

Architecture:
    LiveRunner → on_tick → PositionManager.update_price() → TrailingStop.check_stop_hit()
    LiveRunner → on_candle_close → PositionManager.update_trail() → TrailingStop.update_trail()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd
from src.core.logging import get_logger
from src.execution.swing_detector import find_trailing_swing

logger = get_logger("apex.execution.trailing_stop")


class TrailPhase(str, Enum):
    """Trailing stop state machine phases."""
    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    HYBRID = "HYBRID"
    STRUCTURAL = "STRUCTURAL"


# Regime → trail multiplier (× ATR)
REGIME_TRAIL_MULTIPLIERS: dict[str, float] = {
    "TRENDING": 3.0,   # Tighter than 3.5, but looser than 2.5
    "RANGING": 2.0,    # Room for chops
    "VOLATILE": 4.0,   # Wide — avoid noise
    "QUIET": 2.0,      # Tight trail
}

# Default trail multiplier if regime is unknown
DEFAULT_TRAIL_MULTIPLIER = 2.0

# Regime-aware breakeven and trailing trigger thresholds (in ATR multiples)
REGIME_BREAKEVEN_TRIGGERS: dict[str, float] = {
    "TRENDING": 1.5,   
    "RANGING": 1.5,    
    "VOLATILE": 1.5,   
    "QUIET": 1.5,      
}

REGIME_TRAILING_TRIGGERS: dict[str, float] = {
    "TRENDING": 2.0,  
    "RANGING": 2.0,   
    "VOLATILE": 2.0,   
    "QUIET": 2.0,      
}

# Regimes where scale-out is disabled (scalp mode takes full profit)
SCALP_REGIMES = set() # Always scale out!

# 100% Runner Strategy: No scale-outs. Hold 100% of the position to capture massive runners.
SCALE_OUT_MILESTONES: list[tuple[float, float]] = []


@dataclass
class TrailingStopState:
    """Per-position trailing stop state."""

    # Identity
    symbol: str = ""
    side: str = ""  # "LONG" or "SHORT"
    regime: str = "TRENDING"  # Market regime at entry — determines TREND vs SCALP mode

    # Entry data
    entry_price: float = 0.0
    initial_stop: float = 0.0
    initial_atr: float = 0.0
    stop_distance: float = 0.0  # Initial distance: entry ↔ stop
    commission_paid: float = 0.0

    # Current state
    phase: TrailPhase = TrailPhase.INITIAL
    current_stop: float = 0.0
    trail_distance: float = 0.0  # Current trail distance (regime-aware)

    # Regime-adaptive thresholds (set at creation based on regime)
    breakeven_trigger: float = 1.5  # ATR multiples to trigger breakeven
    trailing_trigger: float = 2.0   # ATR multiples to trigger trailing
    is_scalp_mode: bool = False     # True in RANGING/QUIET — disables scale-out

    # Peak tracking (for trailing)
    highest_price: float = 0.0  # Best price since entry (for LONG)
    lowest_price: float = float("inf")  # Best price since entry (for SHORT)
    peak_profit_r: float = 0.0  # Peak profit in R-multiples
    drawdown_guard_hit: bool = False # Peak drawdown guard triggered

    # Scale-out tracking
    initial_quantity: float = 0.0
    remaining_quantity: float = 0.0
    milestones_hit: int = 0  # How many scale-out milestones triggered
    partial_profit_taken: float = 0.0  # Total $ from partial closes

    # Metadata
    phase_changes: list[str] = field(default_factory=list)


class SmartTrailingStop:
    """
    Smart trailing stop manager.

    Creates and manages TrailingStopState for each open position.
    Called by PositionManager on ticks (stop-hit checks) and
    candle closes (trail ratcheting).
    """

    def __init__(
        self,
        breakeven_trigger_atr: float = 0.5,
        trailing_trigger_atr: float = 1.0,
        default_trail_multiplier: float = DEFAULT_TRAIL_MULTIPLIER,
        enable_scale_out: bool = True,
    ):
        """
        Args:
            breakeven_trigger_atr: Profit in ATR multiples to trigger breakeven.
            trailing_trigger_atr: Profit in ATR multiples to trigger trailing.
            default_trail_multiplier: Default trail distance in ATR multiples.
            enable_scale_out: Whether to enable partial position scaling.
        """
        self.breakeven_trigger_atr = breakeven_trigger_atr
        self.trailing_trigger_atr = trailing_trigger_atr
        self.default_trail_multiplier = default_trail_multiplier
        self.enable_scale_out = enable_scale_out

        self._states: dict[str, TrailingStopState] = {}

    def create_state(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        stop_loss: float,
        atr: float,
        quantity: float,
        commission: float = 0.0,
        regime: str = "TRENDING",
    ) -> TrailingStopState:
        """
        Create a new trailing stop state for a position.

        Args:
            symbol: Trading symbol.
            side: "LONG" or "SHORT".
            entry_price: Position entry price.
            stop_loss: Initial stop loss level.
            atr: Current ATR at time of entry.
            quantity: Position quantity.
            commission: Commission paid to open the position.
            regime: Market regime at entry (determines TREND vs SCALP mode).

        Returns:
            New TrailingStopState for the position.
        """
        stop_distance = abs(entry_price - stop_loss)
        regime_upper = regime.upper()
        is_scalp = regime_upper in SCALP_REGIMES

        # Set regime-adaptive thresholds
        be_trigger = REGIME_BREAKEVEN_TRIGGERS.get(regime_upper, self.breakeven_trigger_atr)
        trail_trigger = REGIME_TRAILING_TRIGGERS.get(regime_upper, self.trailing_trigger_atr)
        trail_mult = REGIME_TRAIL_MULTIPLIERS.get(regime_upper, self.default_trail_multiplier)

        # --- FEE PROTECTION ---
        # Ensure we don't trigger breakeven before we've even covered commissions
        per_unit_commission = commission / max(quantity, 1e-8)
        commission_in_atr = per_unit_commission / max(atr, 1e-8)
        
        # Cover entry + exit (2x) plus a small 0.5 ATR buffer
        min_be_trigger = round((commission_in_atr * 2.0) + 0.5, 1)
        if be_trigger < min_be_trigger:
            logger.info("breakeven_adjusted_for_fees", old=be_trigger, new=min_be_trigger, atr=atr, comm_atr=commission_in_atr)
            be_trigger = min_be_trigger
            
        if trail_trigger <= be_trigger:
            trail_trigger = be_trigger + 0.5

        mode_str = "SCALP" if is_scalp else "TREND"

        state = TrailingStopState(
            symbol=symbol,
            side=side,
            regime=regime_upper,
            entry_price=entry_price,
            initial_stop=stop_loss,
            initial_atr=atr,
            stop_distance=stop_distance,
            commission_paid=commission,
            phase=TrailPhase.INITIAL,
            current_stop=stop_loss,
            trail_distance=atr * trail_mult,
            breakeven_trigger=be_trigger,
            trailing_trigger=trail_trigger,
            is_scalp_mode=is_scalp,
            highest_price=entry_price,
            lowest_price=entry_price,
            initial_quantity=quantity,
            remaining_quantity=quantity,
            milestones_hit=0,
            partial_profit_taken=0.0,
            phase_changes=[f"INITIAL @ ${entry_price:.2f} [{mode_str} mode]"],
        )

        self._states[symbol] = state

        logger.info(
            "trailing_stop_created",
            symbol=symbol,
            side=side,
            regime=regime_upper,
            mode=mode_str,
            entry=f"${entry_price:.2f}",
            stop=f"${stop_loss:.2f}",
            atr=f"${atr:.2f}",
            stop_distance=f"${stop_distance:.2f}",
            breakeven_at=f"{be_trigger:.1f} ATR",
            trailing_at=f"{trail_trigger:.1f} ATR",
            trail_distance=f"{trail_mult:.1f}x ATR",
        )

        return state

    def sync_entry_price(self, symbol: str, new_entry_price: float) -> None:
        """
        Syncs the trailing stop state when a position is pyramided (scaled-in),
        which changes the average entry price.
        """
        state = self._states.get(symbol)
        if not state:
            return

        old_entry = state.entry_price
        state.entry_price = new_entry_price
        
        # Recalculate stop distance based on new entry price
        if state.side == "LONG":
            state.stop_distance = state.entry_price - state.initial_stop
        else:
            state.stop_distance = state.initial_stop - state.entry_price

        logger.debug(
            "trailing_stop_entry_synced",
            symbol=symbol,
            old_entry=old_entry,
            new_entry=new_entry_price,
            new_stop_distance=state.stop_distance
        )

    def get_state(self, symbol: str) -> TrailingStopState | None:
        """Get trailing stop state for a symbol."""
        return self._states.get(symbol)

    def remove_state(self, symbol: str) -> None:
        """Remove trailing stop state when position is fully closed."""
        self._states.pop(symbol, None)

    def check_stop_hit(self, symbol: str, price: float) -> bool:
        """
        Check if the current price has hit the stop loss.
        Called on EVERY TICK for real-time exit.

        Args:
            symbol: Trading symbol.
            price: Current market price.

        Returns:
            True if stop was hit, False otherwise.
        """
        state = self._states.get(symbol)
        if state is None:
            return False

        if state.side == "LONG":
            return price <= state.current_stop
        else:  # SHORT
            return price >= state.current_stop

    def check_scale_out(
        self, symbol: str, price: float
    ) -> tuple[bool, float, float] | None:
        """
        Check if a scale-out milestone has been reached.
        Called on EVERY TICK.

        Args:
            symbol: Trading symbol.
            price: Current market price.

        Returns:
            Tuple of (should_scale, fraction_to_close, milestone_price) or None.
        """
        if not self.enable_scale_out:
            return None

        state = self._states.get(symbol)
        if state is None:
            return None

        # Scalp mode: no scale-out, take full profit at target
        if state.is_scalp_mode:
            return None

        # Check if there are remaining milestones
        if state.milestones_hit >= len(SCALE_OUT_MILESTONES):
            return None

        milestone_atr, fraction = SCALE_OUT_MILESTONES[state.milestones_hit]
        target_profit = state.initial_atr * milestone_atr

        # Calculate current profit
        if state.side == "LONG":
            current_profit = price - state.entry_price
        else:
            current_profit = state.entry_price - price

        if current_profit >= target_profit:
            return (True, fraction, price)

        return None

    def record_scale_out(
        self, symbol: str, quantity_closed: float, pnl: float
    ) -> None:
        """Record that a partial close happened."""
        state = self._states.get(symbol)
        if state is None:
            return

        state.milestones_hit += 1
        state.remaining_quantity -= quantity_closed
        state.partial_profit_taken += pnl

        logger.info(
            "scale_out_recorded",
            symbol=symbol,
            milestone=state.milestones_hit,
            qty_closed=f"{quantity_closed:.6f}",
            remaining=f"{state.remaining_quantity:.6f}",
            partial_pnl=f"${pnl:+.2f}",
            total_partial=f"${state.partial_profit_taken:+.2f}",
        )

    def update_trail_on_candle_close(
        self,
        symbol: str,
        close_price: float,
        atr: float,
        regime: str = "TRENDING",
        df_5m: pd.DataFrame | None = None,
        df_15m: pd.DataFrame | None = None,
    ) -> dict[str, Any] | None:
        """
        Update the trailing stop on a CANDLE CLOSE.
        """
        state = self._states.get(symbol)
        if state is None or state.drawdown_guard_hit:
            return None

        updates: dict[str, Any] = {
            "symbol": symbol,
            "phase_changed": False,
            "stop_moved": False,
            "old_phase": state.phase.value,
            "old_stop": state.current_stop,
        }

        # Update peak price tracking
        if state.side == "LONG":
            if close_price > state.highest_price:
                state.highest_price = close_price
            current_profit = close_price - state.entry_price
        else:  # SHORT
            if close_price < state.lowest_price:
                state.lowest_price = close_price
            current_profit = state.entry_price - close_price

        profit_in_atr = current_profit / state.initial_atr if state.initial_atr > 0 else 0
        profit_in_r = current_profit / state.stop_distance if state.stop_distance > 0 else 0

        if profit_in_r > state.peak_profit_r:
            state.peak_profit_r = profit_in_r

        # ── Peak Drawdown Guard (Phase 2 spec) ──────────────────────────────
        if state.phase in (TrailPhase.HYBRID, TrailPhase.STRUCTURAL) and state.peak_profit_r > 0:
            drawdown_pct = ((state.peak_profit_r - profit_in_r) / state.peak_profit_r) * 100.0
            
            guard_pct = 100.0
            if state.phase == TrailPhase.HYBRID:
                guard_pct = 35.0
            elif state.phase == TrailPhase.STRUCTURAL:
                guard_pct = 30.0 if regime.upper() in ("RANGING", "QUIET") else 50.0
                
            if drawdown_pct >= guard_pct:
                logger.info("peak_drawdown_guard_hit", symbol=symbol, phase=state.phase.value, peak_r=state.peak_profit_r, current_r=profit_in_r, drawdown=drawdown_pct)
                # Force close by setting current stop to current close_price (or just slightly beyond)
                if state.side == "LONG":
                    state.current_stop = close_price + (atr * 0.01)
                else:
                    state.current_stop = close_price - (atr * 0.01)
                
                state.drawdown_guard_hit = True
                updates["stop_moved"] = True
                updates["new_stop"] = state.current_stop
                updates["new_phase"] = state.phase.value
                return updates

        # ── Phase transitions (REGIME-AWARE — uses state.breakeven_trigger / trailing_trigger) ──

        mode_str = "SCALP" if state.is_scalp_mode else "TREND"

        # INITIAL → BREAKEVEN (uses regime-aware breakeven_trigger in ATR multiples)
        if state.phase == TrailPhase.INITIAL and profit_in_atr >= state.breakeven_trigger:
            state.phase = TrailPhase.BREAKEVEN

            total_cost_per_unit = (state.commission_paid * 2.0) / max(state.initial_quantity, 1e-8)
            if state.side == "LONG":
                new_stop = state.entry_price + total_cost_per_unit + (state.initial_atr * 0.2)
            else:
                new_stop = state.entry_price - total_cost_per_unit - (state.initial_atr * 0.2)
                
            if self._is_better_stop(state, new_stop):
                state.current_stop = new_stop
                updates["stop_moved"] = True

            state.phase_changes.append(f"BREAKEVEN @ ${close_price:.2f} (profit={profit_in_atr:.1f}ATR, trigger={state.breakeven_trigger:.1f}ATR) [{mode_str}]")
            updates["phase_changed"] = True

        # BREAKEVEN → HYBRID (uses regime-aware trailing_trigger in ATR multiples)
        if state.phase == TrailPhase.BREAKEVEN and profit_in_atr >= state.trailing_trigger:
            state.phase = TrailPhase.HYBRID
            state.phase_changes.append(f"HYBRID @ ${close_price:.2f} (profit={profit_in_atr:.1f}ATR, trigger={state.trailing_trigger:.1f}ATR) [{mode_str}]")
            updates["phase_changed"] = True

        # HYBRID → STRUCTURAL (At +2.5 ATR profit)
        if state.phase == TrailPhase.HYBRID and profit_in_atr >= 2.5:
            state.phase = TrailPhase.STRUCTURAL
            state.phase_changes.append(f"STRUCTURAL @ ${close_price:.2f} (profit={profit_in_atr:.1f}ATR) [{mode_str}]")
            updates["phase_changed"] = True

        # ── Trail Ratchet ──────────────────────────────
        if state.phase in (TrailPhase.HYBRID, TrailPhase.STRUCTURAL):
            atr_multiplier = REGIME_TRAIL_MULTIPLIERS.get(regime.upper(), self.default_trail_multiplier)
            atr_trail_distance = atr * atr_multiplier
            
            if state.side == "LONG":
                atr_stop = state.highest_price - atr_trail_distance
            else:
                atr_stop = state.lowest_price + atr_trail_distance

            new_stop = state.current_stop

            if state.phase == TrailPhase.HYBRID:
                # max(atr_trail, structural_swing) for longs
                swing_stop = find_trailing_swing(df_5m, state.side, lookback=20) if df_5m is not None else None
                
                if swing_stop is not None:
                    new_stop = max(atr_stop, swing_stop) if state.side == "LONG" else min(atr_stop, swing_stop)
                else:
                    new_stop = atr_stop
                    
            elif state.phase == TrailPhase.STRUCTURAL:
                # Pure structural swing (5m/15m)
                swing_stop = find_trailing_swing(df_15m, state.side, lookback=40) if df_15m is not None else None
                if swing_stop is None and df_5m is not None:
                    swing_stop = find_trailing_swing(df_5m, state.side, lookback=40)
                    
                if swing_stop is not None:
                    new_stop = swing_stop
                else:
                    new_stop = atr_stop  # Fallback

            if self._is_better_stop(state, new_stop):
                state.current_stop = new_stop
                updates["stop_moved"] = True
                state.trail_distance = atr_trail_distance # For logging/dashboard

        updates["new_phase"] = state.phase.value
        updates["new_stop"] = state.current_stop
        updates["profit_atr"] = round(profit_in_atr, 2)
        updates["peak_price"] = state.highest_price if state.side == "LONG" else state.lowest_price

        if updates["phase_changed"] or updates["stop_moved"]:
            return updates

        return None

    def _is_better_stop(self, state: TrailingStopState, new_stop: float) -> bool:
        """Check if a new stop level is better (more protective) than current."""
        if state.side == "LONG":
            return new_stop > state.current_stop  # Higher stop = better for long
        else:
            return new_stop < state.current_stop  # Lower stop = better for short

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return all trailing stop states (for dashboard/debugging)."""
        result = {}
        for symbol, state in self._states.items():
            result[symbol] = {
                "phase": state.phase.value,
                "current_stop": state.current_stop,
                "initial_stop": state.initial_stop,
                "entry_price": state.entry_price,
                "highest_price": state.highest_price,
                "lowest_price": state.lowest_price,
                "trail_distance": state.trail_distance,
                "milestones_hit": state.milestones_hit,
                "remaining_quantity": state.remaining_quantity,
                "partial_profit_taken": state.partial_profit_taken,
                "phase_changes": state.phase_changes,
            }
        return result
"""

    Apex Intelligence Engine V5 — Smart Trailing Stop Module
    
"""
