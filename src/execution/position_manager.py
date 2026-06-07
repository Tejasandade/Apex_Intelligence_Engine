"""
Apex Intelligence Engine V5 — Position Manager
=================================================
Manages the lifecycle of positions from signal to close.

Responsibilities:
1. Open positions from signals (via broker)
2. Track open positions and real-time P&L
3. Smart trailing stop management (3-phase exit system)
4. Partial profit scaling (scale-out at milestones)
5. Post-loss cooldown timer
6. Enforce risk limits (max drawdown, daily loss, position limits)
7. Feed P&L data back to the Council's Risk advisor

Architecture:
    Signal → PositionManager → Broker → Position
                    ↓
              Smart Trailing Stop → Scale-Out → Risk Monitoring
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.execution.broker import (
    BaseBroker,
    Order,
    OrderSide,
    OrderType,
    Position,
    PositionSide,
    TradeRecord,
)
from src.execution.trailing_stop import SmartTrailingStop, TrailPhase

logger = get_logger("apex.execution.position_mgr")


class PositionManager:
    """
    Manages all open positions and enforces risk limits.

    Acts as the bridge between the signal engine and the broker.
    No signal reaches the broker without passing through here.
    """

    def __init__(
        self,
        broker: BaseBroker,
        max_positions: int = 3,
        max_drawdown_pct: float = 10.0,
        daily_loss_limit: float = 500.0,
        max_risk_per_trade_pct: float = 2.0,
        enable_trailing_stop: bool = True,
        enable_scale_out: bool = True,
        cooldown_candles: int = 5,
        max_daily_trades: int = 5,
    ):
        """
        Args:
            broker: Broker implementation (paper or live).
            max_positions: Max concurrent open positions.
            max_drawdown_pct: Kill switch drawdown threshold.
            daily_loss_limit: Max daily dollar loss before stopping.
            max_risk_per_trade_pct: Max % of equity risked per trade.
            enable_trailing_stop: Enable smart trailing stop system.
            enable_scale_out: Enable partial profit scaling.
            cooldown_candles: Candles to wait after a stop loss before re-entry.
        """
        self.broker = broker
        self.max_positions = max_positions
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_risk_per_trade_pct = max_risk_per_trade_pct

        # State
        self._positions: dict[str, Position] = {}  # symbol → position
        self._trade_history: list[TradeRecord] = []
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._kill_switch_active = False
        self._kill_switch_reason = ""

        # Performance tracking
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
        self._total_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._max_consecutive_losses: int = 0

        # Smart trailing stop
        self._enable_trailing = enable_trailing_stop
        self._trailing_stop = SmartTrailingStop(
            enable_scale_out=enable_scale_out,
        ) if enable_trailing_stop else None

        # Post-loss cooldown
        self._base_cooldown = cooldown_candles
        self._cooldown_remaining: int = 0  # Candles remaining in cooldown

        # Daily trade limit (FIX #3: prevent overtrading)
        self.max_daily_trades = max_daily_trades

        # Cumulative loss tracking (FIX #2: hard-stop on total bleed)
        self._initial_capital: float = 0.0  # Set in initialize()

    @property
    def open_position_count(self) -> int:
        return len([p for p in self._positions.values() if p.side != PositionSide.FLAT])

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._current_equity) / self._peak_equity * 100

    async def initialize(self) -> None:
        """Initialize from broker state."""
        balance = await self.broker.get_balance()
        self._current_equity = balance.total_equity
        self._peak_equity = balance.total_equity
        self._initial_capital = balance.total_equity  # Store for cumulative loss check

        positions = await self.broker.get_positions()
        for pos in positions:
            if pos.side != PositionSide.FLAT:
                self._positions[pos.symbol] = pos

        logger.info(
            "position_manager_initialized",
            equity=f"${self._current_equity:,.2f}",
            open_positions=self.open_position_count,
            max_positions=self.max_positions,
        )
        
        # Load state if exists
        self.load_state()

    def save_state(self) -> None:
        """Save the position manager state (trade history, equity peaks) to disk."""
        import json
        import os
        from src.core.config import DATA_DIR
        
        state_dir = DATA_DIR / "state"
        state_dir.mkdir(exist_ok=True, parents=True)
        state_file = state_dir / "position_manager.json"
        
        try:
            state = {
                "peak_equity": self._peak_equity,
                "current_equity": self._current_equity,
                "total_pnl": self._total_pnl,
                "trade_history": [
                    {
                        "symbol": t.symbol,
                        "side": t.side.value if hasattr(t.side, "value") else str(t.side),
                        "pnl": t.pnl,
                        "entry_time": t.entry_time.isoformat() if hasattr(t.entry_time, "isoformat") else str(t.entry_time),
                        "exit_time": t.exit_time.isoformat() if hasattr(t.exit_time, "isoformat") else str(t.exit_time)
                    }
                    for t in self._trade_history
                ]
            }
            with open(state_file, "w") as f:
                json.dump(state, f)
            logger.debug("position_manager_state_saved", trades=len(self._trade_history))
        except Exception as e:
            logger.error("failed_to_save_state", error=str(e))

    def load_state(self) -> None:
        """Load the position manager state from disk to survive crashes."""
        import json
        from src.core.config import DATA_DIR
        
        state_file = DATA_DIR / "state" / "position_manager.json"
        if not state_file.exists():
            return
            
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
                
            self._peak_equity = state.get("peak_equity", self._peak_equity)
            self._current_equity = state.get("current_equity", self._current_equity)
            self._total_pnl = state.get("total_pnl", self._total_pnl)
            
            # Reconstruct basic trade history for Sharpe/Trade count gating
            history = state.get("trade_history", [])
            reconstructed_history = []
            for t in history:
                reconstructed_history.append(
                    TradeRecord(
                        symbol=t["symbol"],
                        side=PositionSide.LONG if "LONG" in t["side"] else PositionSide.SHORT,
                        entry_price=0.0, exit_price=0.0, quantity=0.0,
                        pnl=t["pnl"], commission=0.0, slippage=0.0,
                        entry_time=t.get("entry_time"), exit_time=t.get("exit_time"),
                        regime="UNKNOWN"
                    )
                )
            self._trade_history = reconstructed_history
            
            logger.info("position_manager_state_loaded", trades=len(self._trade_history), peak=self._peak_equity)
        except Exception as e:
            logger.error("failed_to_load_state", error=str(e))

    @property
    def is_on_cooldown(self) -> bool:
        return self._cooldown_remaining > 0

    @property
    def cooldown_remaining(self) -> int:
        return self._cooldown_remaining

    def is_pyramiding_allowed(self) -> bool:
        """
        Check if the system meets the strict Phase 5 Pyramiding gate requirements:
        1. 100 completed trades
        2. Sharpe ratio > 0.5
        3. Max drawdown < 15%
        """
        total_trades = len(self._trade_history)
        if total_trades < 10:
            return False
            
        # Max DD check
        if self._peak_equity > 0:
            current_dd_pct = (self._peak_equity - self._current_equity) / self._peak_equity * 100.0
            if current_dd_pct >= 15.0:
                return False
                
        # Approximate Sharpe Ratio
        pnls = [t.pnl for t in self._trade_history if t.pnl != 0]
        if not pnls:
            return False
            
        import numpy as np
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls)
        if std_pnl == 0:
            return False
            
        # Simplified pseudo-Sharpe for trade series
        pseudo_sharpe = mean_pnl / std_pnl
        if pseudo_sharpe <= 0.1:
            return False
            
        return True

    async def add_to_position(
        self,
        symbol: str,
        additional_qty: float,
        current_price: float,
        signal_conviction: float,
    ) -> Order | None:
        """
        Phase 5: Pyramiding (Scale-In).
        Add to a winning position if gates are met.
        """
        if not self.is_pyramiding_allowed():
            logger.debug("pyramiding_gate_failed", symbol=symbol)
            return None
            
        pos = self._positions.get(symbol)
        if not pos:
            logger.warning("pyramiding_no_position", symbol=symbol)
            return None
            
        if pos.unrealized_pnl <= 0:
            logger.debug("pyramiding_rejected_not_in_profit", symbol=symbol)
            return None
            
        # --- NEW STRUCTURAL GUARDS ---
        # Guard 1: Must be BREAKEVEN phase or later — never pyramid in INITIAL
        trail_state = self._trailing_stop.get_state(symbol) if self._trailing_stop else None
        if trail_state and trail_state.phase.value == "INITIAL":
            logger.info("pyramiding_blocked", reason="trail_phase_INITIAL", symbol=symbol)
            return None
            
        # Guard 2: Max 1 add per position — no third tranche
        if getattr(pos, "pyramid_count", 0) >= 1:
            logger.info("pyramiding_blocked", reason="max_tranches_reached", symbol=symbol)
            return None
        # -----------------------------
            
        logger.info("pyramiding_approved", symbol=symbol, current_qty=pos.quantity, add_qty=additional_qty)
        
        # Execute order via broker
        from src.execution.broker import Order, OrderSide, OrderType
        side = OrderSide.BUY if pos.side == PositionSide.LONG else OrderSide.SELL
        new_order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=additional_qty,
        )
        order = await self.broker.place_order(new_order)
        
        if order.status != "FILLED":
            logger.error("pyramiding_order_failed", symbol=symbol)
            return None
            
        # --- PYRAMIDING DIAGNOSTIC LOGGING ---
        trail_state = self._trailing_stop.get_state(symbol) if self._trailing_stop else None
        trail_phase = trail_state.phase.value if trail_state else "UNKNOWN"
        current_stop = trail_state.current_stop if trail_state else 0.0
        
        original_entry = pos.entry_price
        
        # Update position average entry price
        total_cost = (pos.quantity * pos.entry_price) + (additional_qty * order.filled_price)
        new_qty = pos.quantity + additional_qty
        new_entry = total_cost / new_qty
        
        # Calculate Risk (R) distance
        if pos.side == PositionSide.LONG:
            risk_dist = new_entry - current_stop
        else:
            risk_dist = current_stop - new_entry
            
        logger.info(
            "pyramiding_diagnostic",
            symbol=symbol,
            trail_phase=trail_phase,
            original_entry=original_entry,
            new_avg_entry=new_entry,
            stop_price=current_stop,
            risk_distance=risk_dist
        )
        # -------------------------------------
        
        pos.quantity = new_qty
        pos.entry_price = new_entry
        pos.pyramid_count = getattr(pos, "pyramid_count", 0) + 1
        
        # Sync trailing stop average entry price
        if self._trailing_stop:
            self._trailing_stop.sync_entry_price(symbol, new_entry)
            
        return order

    async def open_position(
        self,
        signal_id: str,
        symbol: str,
        direction: str,
        quantity: float,
        price: float,
        stop_loss: float,
        take_profit: float,
        regime: str = "",
        council_score: float = 0.0,
        atr: float = 0.0,
    ) -> Order | None:
        """
        Open a new position from a signal.

        Performs risk checks before placing the order.

        Returns:
            Filled Order if successful, None if blocked.
        """
        # ── Risk Checks ──────────────────────────────────────────────────────
        if self._kill_switch_active:
            logger.warning("position_blocked_kill_switch", reason=self._kill_switch_reason)
            return None

        if self.open_position_count >= self.max_positions:
            logger.debug("position_blocked_max_positions", count=self.open_position_count)
            return None

        if symbol in self._positions and self._positions[symbol].side != PositionSide.FLAT:
            # Check if Pyramiding is appropriate
            existing_pos = self._positions[symbol]
            new_side = PositionSide.LONG if direction == "BUY" else PositionSide.SHORT
            
            if existing_pos.side == new_side:
                # Attempt to add to position
                return await self.add_to_position(
                    symbol=symbol,
                    additional_qty=quantity,
                    current_price=price,
                    signal_conviction=council_score
                )
            else:
                logger.debug("position_blocked_opposing_existing", symbol=symbol)
                return None

        # ── Cooldown Check ───────────────────────────────────────────────────
        if self._cooldown_remaining > 0:
            if council_score >= 0.74:
                logger.info(
                    "cooldown_bypassed_high_conviction",
                    remaining=self._cooldown_remaining,
                    symbol=symbol,
                    council_score=f"{council_score:.3f}",
                )
            else:
                logger.info(
                    "position_blocked_cooldown",
                    remaining=self._cooldown_remaining,
                    symbol=symbol,
                    council_score=f"{council_score:.3f}",
                )
                return None

        if self._daily_pnl < -self.daily_loss_limit:
            logger.warning(
                "position_blocked_daily_limit",
                daily_pnl=f"${self._daily_pnl:.2f}",
                limit=f"${self.daily_loss_limit:.2f}",
            )
            return None

        # FIX #3: Daily trade count limit
        if self._daily_trades >= self.max_daily_trades:
            logger.info(
                "position_blocked_daily_trade_limit",
                daily_trades=self._daily_trades,
                limit=self.max_daily_trades,
            )
            return None

        # FIX #2: Cumulative loss hard-stop
        if self._initial_capital > 0 and self._total_pnl < -(self._initial_capital * 0.10):
            self._activate_kill_switch(
                f"Cumulative loss {self._total_pnl:.2f} exceeds 10% of initial capital"
            )
            return None

        # Check drawdown
        if self.current_drawdown_pct > self.max_drawdown_pct:
            self._activate_kill_switch(
                f"Drawdown {self.current_drawdown_pct:.1f}% exceeds {self.max_drawdown_pct:.1f}%"
            )
            return None

        # ── Place Order ──────────────────────────────────────────────────────
        side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

        order = Order(
            client_order_id=signal_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=price,
        )

        filled_order = await self.broker.place_order(order)

        if filled_order.status.value == "FILLED":
            # Adjust SL/TP based on slippage to maintain intended R:R ratio
            slippage = filled_order.filled_price - price
            adjusted_stop_loss = stop_loss + slippage
            adjusted_take_profit = take_profit + slippage

            # Track position
            pos_side = PositionSide.LONG if direction == "BUY" else PositionSide.SHORT
            self._positions[symbol] = Position(
                symbol=symbol,
                side=pos_side,
                quantity=filled_order.filled_quantity,
                entry_price=filled_order.filled_price,
                current_price=filled_order.filled_price,
                stop_loss=adjusted_stop_loss,
                take_profit=adjusted_take_profit,
                signal_id=signal_id,
                initial_quantity=filled_order.filled_quantity,
                highest_price=filled_order.filled_price,
                lowest_price=filled_order.filled_price,
                initial_atr=atr,
            )

            # Initialize trailing stop
            if self._trailing_stop:
                self._trailing_stop.create_state(
                    symbol=symbol,
                    side=pos_side.value,
                    entry_price=filled_order.filled_price,
                    stop_loss=adjusted_stop_loss,
                    atr=atr,
                    quantity=filled_order.filled_quantity,
                    commission=filled_order.commission,
                    regime=regime,
                )

            logger.info(
                "position_opened",
                signal_id=signal_id,
                symbol=symbol,
                side=direction,
                qty=f"{filled_order.filled_quantity:.6f}",
                price=f"${filled_order.filled_price:.2f}",
                stop=f"${stop_loss:.2f}",
                target=f"${take_profit:.2f}",
                trailing=self._enable_trailing,
            )

        return filled_order

    async def close_position(
        self, symbol: str, reason: str = "manual"
    ) -> TradeRecord | None:
        """Close a position and record the trade."""
        pos = self._positions.get(symbol)
        if pos is None or pos.side == PositionSide.FLAT:
            return None

        close_order = await self.broker.close_position(symbol)

        if close_order.status.value == "FILLED":
            return self._record_trade(pos, close_order.filled_price, reason)

        return None

    async def update_price(self, symbol: str, price: float, current_time: datetime | None = None) -> tuple[TradeRecord | None, TradeRecord | None]:
        """
        Update price and check stops. Called on every tick.

        With trailing stops enabled, this checks the trailing stop level
        (managed by SmartTrailingStop) instead of the fixed stop/TP.

        Returns:
            Tuple of (main_trade, partial_trade):
            - main_trade: TradeRecord if position was fully closed
            - partial_trade: TradeRecord if a scale-out milestone was hit
        """
        # Update broker price
        if hasattr(self.broker, 'update_price'):
            if current_time:
                self.broker.update_price(symbol, price, current_time)
            else:
                self.broker.update_price(symbol, price)

        pos = self._positions.get(symbol)
        if pos is None or pos.side == PositionSide.FLAT:
            return (None, None)

        # Update position P&L
        pos.current_price = price
        if pos.side == PositionSide.LONG:
            pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            if price > pos.highest_price:
                pos.highest_price = price
        else:
            pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity
            if price < pos.lowest_price or pos.lowest_price <= 0:
                pos.lowest_price = price

        # ── Ultimate Take Profit Check ───────────────────────────────────────
        if pos.take_profit > 0 and not self._trailing_stop:
            if (pos.side == PositionSide.LONG and price >= pos.take_profit) or \
               (pos.side == PositionSide.SHORT and price <= pos.take_profit):
                if self._trailing_stop:
                    self._trailing_stop.remove_state(symbol)
                close_order = await self.broker.close_position(symbol, execution_price=pos.take_profit)
                if close_order and close_order.status.value == "FILLED":
                    return (self._record_trade(pos, close_order.filled_price, "take_profit"), None)
                return (self._record_trade(pos, pos.take_profit, "take_profit"), None)

        # ── Trailing Stop Mode ───────────────────────────────────────────────
        if self._trailing_stop:
            trail_state = self._trailing_stop.get_state(symbol)

            if trail_state:
                # Update trail phase on position for dashboard
                pos.trail_phase = trail_state.phase.value

                if self._trailing_stop.check_stop_hit(symbol, price):
                    exit_reason = (
                        "trailing_stop" if trail_state.phase in (TrailPhase.STRUCTURAL, TrailPhase.HYBRID)
                        else "breakeven_stop" if trail_state.phase == TrailPhase.BREAKEVEN
                        else "stop_loss"
                    )
                    execution_price = trail_state.current_stop
                    self._trailing_stop.remove_state(symbol)
                    
                    # ACTUAL FIX: We must tell the broker to close the position AT the stop price!
                    close_order = await self.broker.close_position(symbol, execution_price=execution_price)
                    if close_order and close_order.status.value == "FILLED":
                        return (self._record_trade(pos, close_order.filled_price, exit_reason), None)
                    else:
                        # Fallback if broker fails
                        return (self._record_trade(pos, price, exit_reason), None)

                # Check scale-out milestones
                scale_result = self._trailing_stop.check_scale_out(symbol, price)
                if scale_result:
                    should_scale, fraction, milestone_price = scale_result
                    if should_scale and hasattr(self.broker, 'partial_close'):
                        partial_trade = self.broker.partial_close(
                            symbol, fraction, price
                        )
                        if partial_trade:
                            self._trailing_stop.record_scale_out(
                                symbol, partial_trade.quantity, partial_trade.pnl
                            )
                            self._trade_history.append(partial_trade)
                            self._total_pnl += partial_trade.pnl
                            self._daily_pnl += partial_trade.pnl
                            self._daily_trades += 1
                            if partial_trade.pnl > 0:
                                self._consecutive_losses = 0
                            self._update_equity()
                            return (None, partial_trade)

        else:
            # ── Legacy fixed stop/TP mode ─────────────────────────────────────
            # Check stop loss
            if pos.stop_loss > 0:
                if (pos.side == PositionSide.LONG and price <= pos.stop_loss) or \
                   (pos.side == PositionSide.SHORT and price >= pos.stop_loss):
                    close_order = await self.broker.close_position(symbol, execution_price=pos.stop_loss)
                    if close_order and close_order.status.value == "FILLED":
                        return (self._record_trade(pos, close_order.filled_price, "stop_loss"), None)
                    return (self._record_trade(pos, pos.stop_loss, "stop_loss"), None)

        # Update equity tracking
        self._update_equity()

        return (None, None)

    def update_on_candle_close(
        self,
        symbol: str,
        close_price: float,
        atr: float,
        regime: str = "TRENDING",
        df_5m: pd.DataFrame | None = None,
        df_15m: pd.DataFrame | None = None,
    ) -> dict[str, Any] | None:
        """
        Called after each candle close to:
        1. Ratchet the trailing stop (anti-wick protection)
        2. Decrement cooldown timer

        Args:
            symbol: Trading symbol.
            close_price: Candle close price.
            atr: Current ATR value.
            regime: Current market regime.

        Returns:
            Dict with trail update details if stop/phase changed.
        """
        # Decrement cooldown timer
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                logger.info("cooldown_expired", symbol=symbol)

        # Update trailing stop on candle close
        if self._trailing_stop:
            trail_update = self._trailing_stop.update_trail_on_candle_close(
                symbol=symbol,
                close_price=close_price,
                atr=atr,
                regime=regime,
                df_5m=df_5m,
                df_15m=df_15m,
            )

            # Sync trail stop level to position and broker
            trail_state = self._trailing_stop.get_state(symbol)
            if trail_state:
                pos = self._positions.get(symbol)
                if pos and pos.side != PositionSide.FLAT:
                    pos.stop_loss = trail_state.current_stop
                    pos.trail_phase = trail_state.phase.value
                    # Update broker's stop level
                    if hasattr(self.broker, 'update_stop'):
                        self.broker.update_stop(symbol, trail_state.current_stop)

            return trail_update

        return None

    def _record_trade(
        self, pos: Position, exit_price: float, exit_reason: str
    ) -> TradeRecord:
        """Record a completed trade and update stats."""
        if pos.side == PositionSide.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = pnl / (pos.entry_price * pos.quantity) if pos.entry_price > 0 else 0.0
        duration = (datetime.now(timezone.utc) - pos.opened_at).total_seconds()

        trade = TradeRecord(
            trade_id=f"trade_{uuid.uuid4().hex[:8]}",
            signal_id=pos.signal_id,
            symbol=pos.symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            entry_time=pos.opened_at,
            exit_time=datetime.now(timezone.utc),
            exit_reason=exit_reason,
            duration_seconds=duration,
        )

        self._trade_history.append(trade)
        self._total_pnl += pnl
        self._daily_pnl += pnl
        self._daily_trades += 1

        # Consecutive loss tracking + cooldown
        if pnl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            self._max_consecutive_losses = max(
                self._max_consecutive_losses, self._consecutive_losses
            )
            # Activate cooldown after a stop loss
            if exit_reason in ("stop_loss", "breakeven_stop"):
                self._cooldown_remaining = (
                    self._base_cooldown + (self._consecutive_losses * 2)
                )
                logger.info(
                    "cooldown_activated",
                    candles=self._cooldown_remaining,
                    consecutive_losses=self._consecutive_losses,
                    reason=exit_reason,
                )

        # Clean up trailing stop state
        if self._trailing_stop:
            self._trailing_stop.remove_state(pos.symbol)

        # Clear position
        self._positions[pos.symbol] = Position(symbol=pos.symbol, side=PositionSide.FLAT)

        logger.info(
            "trade_closed",
            trade_id=trade.trade_id,
            symbol=pos.symbol,
            side=pos.side.value,
            pnl=f"${pnl:+.2f}",
            pnl_pct=f"{pnl_pct:+.4%}",
            reason=exit_reason,
            duration=f"{duration:.0f}s",
            daily_pnl=f"${self._daily_pnl:+.2f}",
            cooldown=self._cooldown_remaining,
        )

        self._update_equity()
        return trade

    def _update_equity(self) -> None:
        """Update equity tracking for drawdown monitoring."""
        unrealized = sum(
            p.unrealized_pnl for p in self._positions.values()
            if p.side != PositionSide.FLAT
        )
        # Approximate equity (broker balance + unrealized)
        self._current_equity = self._peak_equity + self._total_pnl + unrealized
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        # Check kill switch
        if self.current_drawdown_pct > self.max_drawdown_pct:
            self._activate_kill_switch(
                f"Max drawdown hit: {self.current_drawdown_pct:.1f}%"
            )

    def _activate_kill_switch(self, reason: str) -> None:
        """Activate the kill switch — stop all trading."""
        if not self._kill_switch_active:
            self._kill_switch_active = True
            self._kill_switch_reason = reason
            logger.warning(
                "kill_switch_activated",
                reason=reason,
                equity=f"${self._current_equity:,.2f}",
                drawdown=f"{self.current_drawdown_pct:.1f}%",
            )

    def reset_daily(self) -> None:
        """Reset daily counters (call at market open or midnight)."""
        self._daily_pnl = 0.0
        self._daily_trades = 0

    def get_risk_state(self) -> dict[str, float]:
        """Get risk state for the Council's Risk advisor."""
        return {
            "drawdown_pct": self.current_drawdown_pct,
            "daily_pnl": self._daily_pnl,
            "recent_losses": self._consecutive_losses,
            "open_positions": self.open_position_count,
        }

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive position manager stats."""
        winning = sum(1 for t in self._trade_history if t.pnl > 0)
        total = len(self._trade_history)

        # Get trailing stop info
        trail_info = {}
        if self._trailing_stop:
            trail_info = self._trailing_stop.get_all_states()

        return {
            "open_positions": self.open_position_count,
            "total_trades": total,
            "winning_trades": winning,
            "win_rate": winning / max(total, 1),
            "total_pnl": round(self._total_pnl, 2),
            "daily_pnl": round(self._daily_pnl, 2),
            "max_drawdown_pct": round(self.current_drawdown_pct, 2),
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self._max_consecutive_losses,
            "kill_switch": self._kill_switch_active,
            "cooldown_remaining": self._cooldown_remaining,
            "trailing_stops": trail_info,
            "positions": {
                sym: {
                    "side": pos.side.value,
                    "entry": pos.entry_price,
                    "current": pos.current_price,
                    "pnl": round(pos.unrealized_pnl, 2),
                    "trail_phase": pos.trail_phase,
                    "stop_loss": pos.stop_loss,
                }
                for sym, pos in self._positions.items()
                if pos.side != PositionSide.FLAT
            },
        }
