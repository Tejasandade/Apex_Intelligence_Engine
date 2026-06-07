"""
Apex Intelligence Engine V5 — Paper Broker
=============================================
Full trading simulation without real money. Implements the BaseBroker
interface so the execution engine treats it identically to a real broker.

Features:
- Realistic fill simulation (market orders fill at current price + slippage)
- Commission modeling (configurable bps)
- Position tracking with real-time P&L
- Trade history with full metadata
- Kill switch support

This is used for:
1. System validation before going live
2. Strategy testing on real-time data
3. Performance benchmarking
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.logging import get_logger
from src.execution.broker import (
    AccountBalance,
    BaseBroker,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    TradeRecord,
)

logger = get_logger("apex.execution.paper")


class PaperBroker(BaseBroker):
    """
    Simulated broker for paper trading.

    Executes orders instantly at current price with configurable
    slippage and commission. Tracks positions and P&L in memory.
    """

    def __init__(
        self,
        initial_balance: float = 10_000.0,
        commission_bps: float = 10.0,  # 0.10% per trade
        slippage_bps: float = 5.0,  # 0.05% slippage
        currency: str = "USD",
    ):
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._commission_bps = commission_bps
        self._slippage_bps = slippage_bps
        self._currency = currency

        # State
        self._connected = False
        self._positions: dict[str, Position] = {}  # symbol → position
        self._orders: dict[str, Order] = {}  # order_id → order
        self._trades: list[TradeRecord] = []
        self._current_prices: dict[str, float] = {}
        self._current_time: datetime | None = None

        # Stats
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._peak_equity = initial_balance
        self._max_drawdown = 0.0

    @property
    def broker_name(self) -> str:
        return "paper"

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> bool:
        self._connected = True
        logger.info(
            "paper_broker_connected",
            balance=f"{self._currency}{self._balance:,.2f}",
            commission=f"{self._commission_bps}bps",
            slippage=f"{self._slippage_bps}bps",
        )
        return True

    async def disconnect(self) -> None:
        self._connected = False
        logger.info(
            "paper_broker_disconnected",
            final_balance=f"{self._currency}{self._balance:,.2f}",
            total_pnl=f"{self._currency}{self._total_pnl:,.2f}",
            trades=self._total_trades,
            peak_equity=f"{self._currency}{self._peak_equity:,.2f}",
        )

    # ── Orders ───────────────────────────────────────────────────────────────

    async def place_order(self, order: Order) -> Order:
        """Simulate order placement with instant fill."""
        order.order_id = f"paper_{uuid.uuid4().hex[:12]}"
        order.broker = "paper"

        symbol = order.symbol
        current_price = self._current_prices.get(symbol, order.price)

        if current_price <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning("paper_order_rejected", reason="no price available", symbol=symbol)
            return order

        # Apply slippage
        slippage_mult = self._slippage_bps / 10000.0
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + slippage_mult)
        else:
            fill_price = current_price * (1 - slippage_mult)

        # Calculate commission
        notional = order.quantity * fill_price
        commission = notional * (self._commission_bps / 10000.0)

        # Fill order
        order.filled_price = round(fill_price, 8)
        order.filled_quantity = order.quantity
        order.commission = round(commission, 4)
        order.slippage = round(abs(fill_price - current_price), 8)
        order.status = OrderStatus.FILLED
        order.filled_at = self._current_time or datetime.now(timezone.utc)

        self._orders[order.order_id] = order

        # Update position
        self._update_position(order)

        # Deduct commission
        self._balance -= commission

        logger.info(
            "paper_order_filled",
            order_id=order.order_id,
            side=order.side.value,
            symbol=symbol,
            qty=f"{order.quantity:.6f}",
            price=f"{fill_price:.2f}",
            commission=f"${commission:.4f}",
            slippage=f"${order.slippage:.4f}",
        )

        return order

    async def cancel_order(self, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order and order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            return True
        return False

    async def get_order_status(self, order_id: str) -> Order:
        return self._orders.get(order_id, Order(order_id=order_id, status=OrderStatus.REJECTED))

    # ── Positions ────────────────────────────────────────────────────────────

    async def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    async def close_position(self, symbol: str, execution_price: float = 0.0) -> Order:
        """Close a position by placing an opposing market order."""
        pos = self._positions.get(symbol)
        if pos is None or pos.side == PositionSide.FLAT:
            return Order(status=OrderStatus.REJECTED)

        close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        
        # Calculate slippage adjusted execution price if we're simulating a limit/stop
        if execution_price > 0:
            slippage_mult = self._slippage_bps / 10000.0
            if close_side == OrderSide.SELL:
                execution_price *= (1 - slippage_mult)
            else:
                execution_price *= (1 + slippage_mult)
        else:
            execution_price = self._current_prices.get(symbol, 0.0)
            
        close_order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            price=execution_price,
        )
        
        # Override the current price temporarily to ensure place_order fills at the correct price
        old_price = self._current_prices.get(symbol, 0.0)
        self._current_prices[symbol] = execution_price
        try:
            return await self.place_order(close_order)
        finally:
            self._current_prices[symbol] = old_price

    # ── Account ──────────────────────────────────────────────────────────────

    async def get_balance(self) -> AccountBalance:
        unrealized = sum(p.unrealized_pnl for p in self._positions.values())
        return AccountBalance(
            total_equity=self._balance + unrealized,
            available_balance=self._balance,
            used_margin=sum(
                p.quantity * p.entry_price for p in self._positions.values()
            ),
            unrealized_pnl=unrealized,
            currency=self._currency,
            broker="paper",
        )

    async def get_current_price(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0.0)

    # ── Paper-specific methods ───────────────────────────────────────────────

    def update_price(self, symbol: str, price: float, current_time: datetime | None = None) -> None:
        """Update the current price for a symbol (called by tick handler)."""
        self._current_prices[symbol] = price
        if current_time:
            self._current_time = current_time

        # Update position P&L
        pos = self._positions.get(symbol)
        if pos and pos.side != PositionSide.FLAT:
            pos.current_price = price
            if pos.side == PositionSide.LONG:
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

    def check_stops(self, symbol: str) -> TradeRecord | None:
        """Check if any position hit stop loss or take profit."""
        pos = self._positions.get(symbol)
        if pos is None or pos.side == PositionSide.FLAT:
            return None

        price = self._current_prices.get(symbol, 0.0)
        if price <= 0:
            return None

        exit_reason = ""

        if pos.side == PositionSide.LONG:
            if pos.stop_loss > 0 and price <= pos.stop_loss:
                exit_reason = "stop_loss"
            elif pos.take_profit > 0 and price >= pos.take_profit:
                exit_reason = "take_profit"
        else:  # SHORT
            if pos.stop_loss > 0 and price >= pos.stop_loss:
                exit_reason = "stop_loss"
            elif pos.take_profit > 0 and price <= pos.take_profit:
                exit_reason = "take_profit"

        if exit_reason:
            # Execute at the actual stop price to prevent massive inaccurate slippage
            # from the candle's absolute high/low
            if exit_reason == "stop_loss":
                execution_price = pos.stop_loss
            elif exit_reason == "take_profit":
                execution_price = pos.take_profit
            else:
                execution_price = price
                
            return self._close_position_internal(symbol, execution_price, exit_reason)


        return None

    def _update_position(self, order: Order) -> None:
        """Update position tracking after an order fill."""
        symbol = order.symbol
        pos = self._positions.get(symbol)

        if pos is None or pos.side == PositionSide.FLAT:
            # Opening new position
            side = PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT
            self._positions[symbol] = Position(
                symbol=symbol,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price,
                signal_id=order.client_order_id,
            )
            # Override the default datetime.now() with the simulated filled time
            self._positions[symbol].opened_at = order.filled_at
        else:
            # Closing existing position
            if (pos.side == PositionSide.LONG and order.side == OrderSide.SELL) or \
               (pos.side == PositionSide.SHORT and order.side == OrderSide.BUY):
                self._close_position_internal(symbol, order.filled_price, "signal")

    def _close_position_internal(
        self, symbol: str, exit_price: float, exit_reason: str
    ) -> TradeRecord:
        """Close a position and record the trade."""
        pos = self._positions[symbol]

        # Calculate Gross P&L
        if pos.side == PositionSide.LONG:
            gross_pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.quantity

        # Calculate Commissions
        entry_commission = (pos.entry_price * pos.quantity) * (self._commission_bps / 10000.0)
        exit_commission = (exit_price * pos.quantity) * (self._commission_bps / 10000.0)
        total_commission = entry_commission + exit_commission

        # Calculate Net P&L
        net_pnl = gross_pnl - total_commission

        pnl_pct = net_pnl / (pos.entry_price * pos.quantity) if pos.entry_price > 0 else 0.0
        
        now = self._current_time or datetime.now(timezone.utc)
        duration = (now - pos.opened_at).total_seconds() if pos.opened_at else 0.0

        # Record trade
        trade = TradeRecord(
            trade_id=f"trade_{uuid.uuid4().hex[:8]}",
            signal_id=pos.signal_id,
            symbol=symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=round(net_pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            entry_time=pos.opened_at,
            exit_time=now,
            exit_reason=exit_reason,
            duration_seconds=duration,
        )
        self._trades.append(trade)

        # Update stats
        self._total_trades += 1
        self._total_pnl += net_pnl
        
        # entry_commission was deducted when opening the position via place_order.
        # exit_commission is deducted by place_order right after this returns.
        # So we only add gross_pnl here.
        self._balance += gross_pnl

        if net_pnl > 0:
            self._winning_trades += 1

        # Track drawdown
        equity = self._balance + sum(
            p.unrealized_pnl for p in self._positions.values() if p.symbol != symbol
        )
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity * 100
        self._max_drawdown = max(self._max_drawdown, drawdown)

        # Clear position
        self._positions[symbol] = Position(symbol=symbol, side=PositionSide.FLAT)

        logger.info(
            "paper_trade_closed",
            trade_id=trade.trade_id,
            symbol=symbol,
            side=pos.side.value,
            entry=f"${pos.entry_price:.2f}",
            exit=f"${exit_price:.2f}",
            pnl=f"${net_pnl:+.2f}",
            pnl_pct=f"{pnl_pct:+.4%}",
            reason=exit_reason,
            balance=f"${self._balance:,.2f}",
        )

        return trade

    def partial_close(
        self, symbol: str, fraction: float, price: float
    ) -> TradeRecord | None:
        """
        Close a fraction of a position (for scale-out).

        Args:
            symbol: Trading symbol.
            fraction: Fraction to close (0.0 to 1.0, e.g., 0.33 for 33%).
            price: Current market price for the close.

        Returns:
            TradeRecord for the partial close, or None if no position exists.
        """
        pos = self._positions.get(symbol)
        if pos is None or pos.side == PositionSide.FLAT:
            return None

        close_qty = pos.quantity * fraction

        # Calculate P&L for the partial close
        if pos.side == PositionSide.LONG:
            pnl = (price - pos.entry_price) * close_qty
        else:
            pnl = (pos.entry_price - price) * close_qty

        pnl_pct = pnl / (pos.entry_price * close_qty) if pos.entry_price > 0 else 0.0
        
        now = self._current_time or datetime.now(timezone.utc)
        duration = (now - pos.opened_at).total_seconds() if pos.opened_at else 0.0

        # Apply slippage and commission
        slippage_mult = self._slippage_bps / 10000.0
        slippage_cost = price * close_qty * slippage_mult
        commission = price * close_qty * (self._commission_bps / 10000.0)

        pnl -= commission  # Net P&L after commission
        self._balance += pnl  # Add net P&L to balance (commission already included)

        # Record partial trade
        trade = TradeRecord(
            trade_id=f"trade_partial_{uuid.uuid4().hex[:8]}",
            signal_id=pos.signal_id,
            symbol=symbol,
            side=pos.side.value,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=close_qty,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            entry_time=pos.opened_at,
            exit_time=now,
            exit_reason="scale_out",
            duration_seconds=duration,
        )
        self._trades.append(trade)
        self._total_pnl += pnl

        if pnl > 0:
            self._winning_trades += 1
        self._total_trades += 1

        # Reduce position quantity (keep position open with remaining)
        pos.quantity -= close_qty

        logger.info(
            "paper_partial_close",
            symbol=symbol,
            fraction=f"{fraction:.0%}",
            qty_closed=f"{close_qty:.6f}",
            qty_remaining=f"{pos.quantity:.6f}",
            pnl=f"${pnl:+.2f}",
            price=f"${price:.2f}",
            commission=f"${commission:.4f}",
        )

        return trade

    def update_stop(self, symbol: str, new_stop: float) -> None:
        """Update the stop loss level for a position (for trailing stop)."""
        pos = self._positions.get(symbol)
        if pos and pos.side != PositionSide.FLAT:
            pos.stop_loss = new_stop

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive paper trading stats."""
        win_rate = self._winning_trades / max(self._total_trades, 1)
        return {
            "broker": "paper",
            "initial_balance": self._initial_balance,
            "current_balance": round(self._balance, 2),
            "total_pnl": round(self._total_pnl, 2),
            "total_return_pct": round(
                (self._balance - self._initial_balance) / self._initial_balance * 100, 2
            ),
            "total_trades": self._total_trades,
            "winning_trades": self._winning_trades,
            "win_rate": round(win_rate, 4),
            "max_drawdown_pct": round(self._max_drawdown, 2),
            "open_positions": len([p for p in self._positions.values() if p.side != PositionSide.FLAT]),
        }

    def get_trade_history(self) -> list[TradeRecord]:
        """Return all completed trades."""
        return self._trades.copy()

