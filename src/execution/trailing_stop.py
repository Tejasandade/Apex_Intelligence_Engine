"""
Apex Intelligence Engine V5 — Smart Trailing Stop
====================================================
3-phase exit management system designed for crypto volatility.

Phases:
    INITIAL    → Fixed stop loss (3x ATR from entry). Standard protection.
    BREAKEVEN  → Stop moves to entry + fees. Triggered at 1 ATR profit. Risk-free trade.
    TRAILING   → Stop follows price by N×ATR. Triggered at 2 ATR profit. Locks in profit.

Key design decisions:
    1. Trail only ratchets on CANDLE CLOSE (anti-wick protection)
       - Stop-hit checks still happen on every tick (instant exit)
       - But the trail level only moves on confirmed candle closes
       - This prevents liquidity sweep wicks from ruining the trail
    
    2. Regime-aware trail distance
       - TRENDING: 2.5× ATR (wide, let trends run)
       - RANGING:  1.5× ATR (tight, take quick profits)
       - VOLATILE: 3.0× ATR (very wide, avoid noise)
       - QUIET:    1.0× ATR (very tight, small moves)
    
    3. Partial scale-out milestones
       - At 1× R:R target: close 33% of position
       - At 2× R:R target: close another 33%
       - Remaining 34% rides the trailing stop to maximise winners

Architecture:
    LiveRunner → on_tick → PositionManager.update_price() → TrailingStop.check_stop_hit()
    LiveRunner → on_candle_close → PositionManager.update_trail() → TrailingStop.update_trail()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.core.logging import get_logger

logger = get_logger("apex.execution.trailing_stop")


class TrailPhase(str, Enum):
    """Trailing stop state machine phases."""
    INITIAL = "INITIAL"
    BREAKEVEN = "BREAKEVEN"
    TRAILING = "TRAILING"


# Regime → trail multiplier (× ATR)
REGIME_TRAIL_MULTIPLIERS: dict[str, float] = {
    "TRENDING": 2.5,
    "RANGING": 1.5,
    "VOLATILE": 3.0,
    "QUIET": 1.0,
}

# Default trail multiplier if regime is unknown
DEFAULT_TRAIL_MULTIPLIER = 2.0

# Scale-out milestones: (profit_in_R_multiples, fraction_to_close)
# R = risk amount = stop_distance
SCALE_OUT_MILESTONES: list[tuple[float, float]] = [
    (1.0, 0.33),  # At 1× R:R → close 33%
    (2.0, 0.33),  # At 2× R:R → close another 33%
    # Remaining 34% rides the trailing stop
]


@dataclass
class TrailingStopState:
    """Per-position trailing stop state."""

    # Identity
    symbol: str = ""
    side: str = ""  # "LONG" or "SHORT"

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

    # Peak tracking (for trailing)
    highest_price: float = 0.0  # Best price since entry (for LONG)
    lowest_price: float = float("inf")  # Best price since entry (for SHORT)

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
        breakeven_trigger_atr: float = 1.0,
        trailing_trigger_atr: float = 2.0,
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

        Returns:
            New TrailingStopState for the position.
        """
        stop_distance = abs(entry_price - stop_loss)

        state = TrailingStopState(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            initial_stop=stop_loss,
            initial_atr=atr,
            stop_distance=stop_distance,
            commission_paid=commission,
            phase=TrailPhase.INITIAL,
            current_stop=stop_loss,
            trail_distance=atr * self.default_trail_multiplier,
            highest_price=entry_price,
            lowest_price=entry_price,
            initial_quantity=quantity,
            remaining_quantity=quantity,
            milestones_hit=0,
            partial_profit_taken=0.0,
            phase_changes=[f"INITIAL @ ${entry_price:.2f}"],
        )

        self._states[symbol] = state

        logger.info(
            "trailing_stop_created",
            symbol=symbol,
            side=side,
            entry=f"${entry_price:.2f}",
            stop=f"${stop_loss:.2f}",
            atr=f"${atr:.2f}",
            stop_distance=f"${stop_distance:.2f}",
        )

        return state

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

        # Check if there are remaining milestones
        if state.milestones_hit >= len(SCALE_OUT_MILESTONES):
            return None

        milestone_r, fraction = SCALE_OUT_MILESTONES[state.milestones_hit]
        target_profit = state.stop_distance * milestone_r

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
    ) -> dict[str, Any] | None:
        """
        Update the trailing stop on a CANDLE CLOSE.
        This is where the trail ratchets — NOT on ticks.

        Anti-wick protection: by only moving the stop on confirmed
        candle closes, we avoid wicks pushing the trail to bad levels.

        Args:
            symbol: Trading symbol.
            close_price: Candle close price.
            atr: Current ATR value.
            regime: Current market regime.

        Returns:
            Dict with update details if the stop or phase changed, None otherwise.
        """
        state = self._states.get(symbol)
        if state is None:
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

        # ── Phase transitions ──────────────────────────────────────────────

        # INITIAL → BREAKEVEN
        if state.phase == TrailPhase.INITIAL and profit_in_atr >= self.breakeven_trigger_atr:
            state.phase = TrailPhase.BREAKEVEN

            # Move stop to entry + a small buffer to cover commissions
            # Buffer = commission / quantity (cost per unit) + tiny margin
            per_unit_cost = state.commission_paid / max(state.initial_quantity, 1e-8)

            if state.side == "LONG":
                state.current_stop = state.entry_price + per_unit_cost + (state.initial_atr * 0.1)
            else:
                state.current_stop = state.entry_price - per_unit_cost - (state.initial_atr * 0.1)

            state.phase_changes.append(
                f"BREAKEVEN @ ${close_price:.2f} (profit={profit_in_atr:.1f} ATR)"
            )
            updates["phase_changed"] = True
            updates["stop_moved"] = True

            logger.info(
                "trail_phase_change",
                symbol=symbol,
                phase="BREAKEVEN",
                stop=f"${state.current_stop:.2f}",
                profit_atr=f"{profit_in_atr:.1f}",
                price=f"${close_price:.2f}",
            )

        # BREAKEVEN → TRAILING
        if state.phase == TrailPhase.BREAKEVEN and profit_in_atr >= self.trailing_trigger_atr:
            state.phase = TrailPhase.TRAILING

            # Set regime-aware trail distance
            multiplier = REGIME_TRAIL_MULTIPLIERS.get(
                regime.upper(), self.default_trail_multiplier
            )
            state.trail_distance = atr * multiplier

            # Set initial trailing stop
            if state.side == "LONG":
                new_stop = state.highest_price - state.trail_distance
            else:
                new_stop = state.lowest_price + state.trail_distance

            # Only move stop if it's better than current
            if self._is_better_stop(state, new_stop):
                state.current_stop = new_stop
                updates["stop_moved"] = True

            state.phase_changes.append(
                f"TRAILING @ ${close_price:.2f} (trail={multiplier:.1f}×ATR, "
                f"regime={regime})"
            )
            updates["phase_changed"] = True

            logger.info(
                "trail_phase_change",
                symbol=symbol,
                phase="TRAILING",
                stop=f"${state.current_stop:.2f}",
                trail_distance=f"${state.trail_distance:.2f}",
                regime=regime,
                multiplier=f"{multiplier:.1f}",
                profit_atr=f"{profit_in_atr:.1f}",
            )

        # ── Trail ratchet (in TRAILING phase) ──────────────────────────────

        if state.phase == TrailPhase.TRAILING:
            # Update trail distance for current regime (it can change mid-trade)
            multiplier = REGIME_TRAIL_MULTIPLIERS.get(
                regime.upper(), self.default_trail_multiplier
            )
            state.trail_distance = atr * multiplier

            if state.side == "LONG":
                new_stop = state.highest_price - state.trail_distance
            else:
                new_stop = state.lowest_price + state.trail_distance

            # Trail only ratchets in the favorable direction (never moves back)
            if self._is_better_stop(state, new_stop):
                state.current_stop = new_stop
                updates["stop_moved"] = True

                logger.debug(
                    "trailing_stop_updated",
                    symbol=symbol,
                    stop=f"${state.current_stop:.2f}",
                    peak=f"${state.highest_price if state.side == 'LONG' else state.lowest_price:.2f}",
                    trail=f"${state.trail_distance:.2f}",
                    regime=regime,
                )

        updates["new_phase"] = state.phase.value
        updates["new_stop"] = state.current_stop
        updates["profit_atr"] = round(profit_in_atr, 2)
        updates["peak_price"] = (
            state.highest_price if state.side == "LONG" else state.lowest_price
        )

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
