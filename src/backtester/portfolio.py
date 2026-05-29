"""
Apex Intelligence Engine V5 — Portfolio Tracker
=================================================
Tracks portfolio state during backtesting:
- Cash balance, equity curve
- Open positions with entry/stop/target
- Trade lifecycle management
- PnL accounting with cost model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from src.backtester.cost_model import CostModel
from src.core.logging import get_logger

logger = get_logger("apex.backtester.portfolio")


@dataclass
class Position:
    """An open position being tracked by the portfolio."""

    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_bar: int
    max_holding_bars: int = 20
    # Tracking
    peak_price: float = 0.0
    trough_price: float = float("inf")
    scaled_out: bool = False

    def __post_init__(self):
        self.peak_price = self.entry_price
        self.trough_price = self.entry_price


@dataclass
class ClosedTrade:
    """A completed trade with full accounting."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_bar: int
    exit_bar: int
    bars_held: int
    gross_pnl: float
    net_pnl: float
    pnl_pct: float
    exit_reason: str  # "take_profit", "stop_loss", "time_expiry", "signal_exit"


class PortfolioTracker:
    """
    Manages portfolio state during a backtest simulation.

    Handles:
    - Opening/closing positions with cost modeling
    - Stop loss and take profit monitoring
    - Scale-out at 1:1 R:R (configurable)
    - Time-based position expiry
    - Equity curve construction
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        cost_model: CostModel | None = None,
        max_concurrent_positions: int = 3,
        scale_out_at_rr: float = 1.0,
        scale_out_pct: float = 0.5,
        trail_to_breakeven: bool = True,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.cost_model = cost_model or CostModel()
        self.max_concurrent_positions = max_concurrent_positions
        self.scale_out_at_rr = scale_out_at_rr
        self.scale_out_pct = scale_out_pct
        self.trail_to_breakeven = trail_to_breakeven

        self.open_positions: list[Position] = []
        self.closed_trades: list[ClosedTrade] = []
        self.equity_history: list[float] = []
        self._current_bar: int = 0

    @property
    def equity(self) -> float:
        """Current portfolio equity (cash + unrealized positions)."""
        return self.cash + sum(
            self._unrealized_pnl(pos) for pos in self.open_positions
        )

    @property
    def num_open_positions(self) -> int:
        return len(self.open_positions)

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.open_positions) < self.max_concurrent_positions

    def _unrealized_pnl(self, pos: Position) -> float:
        """Compute unrealized PnL for an open position (without exit costs)."""
        if pos.side == "BUY":
            return (pos.peak_price - pos.entry_price) * pos.quantity
        else:
            return (pos.entry_price - pos.trough_price) * pos.quantity

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        bar_index: int,
        max_holding_bars: int = 20,
    ) -> Position | None:
        """
        Open a new position.

        Uses RISK-BASED margin: the capital reserved is the maximum potential
        loss (quantity × distance to stop), NOT the full notional.
        This models leveraged/futures trading correctly.

        Returns the Position if opened, None if rejected.
        """
        if not self.can_open_position():
            return None

        # Apply entry cost
        effective_entry = self.cost_model.apply_entry_cost(price)

        # Risk-based margin: capital at risk = quantity × stop distance
        # This is what we'd actually LOSE if stopped out (plus a safety buffer)
        stop_distance = abs(effective_entry - stop_loss)
        risk_amount = quantity * stop_distance * 1.5  # 1.5x buffer for slippage
        risk_amount = max(risk_amount, 1.0)  # Minimum $1 to avoid division issues

        if risk_amount > self.cash:
            logger.debug(
                "position_rejected_insufficient_cash",
                symbol=symbol,
                risk_required=f"{risk_amount:.2f}",
                available=f"{self.cash:.2f}",
            )
            return None

        self.cash -= risk_amount

        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=effective_entry,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar=bar_index,
            max_holding_bars=max_holding_bars,
        )
        # Store the reserved margin for returning on close
        pos._reserved_margin = risk_amount
        self.open_positions.append(pos)

        logger.debug(
            "position_opened",
            symbol=symbol,
            side=side,
            entry=f"{effective_entry:.2f}",
            qty=f"{quantity:.6f}",
            stop=f"{stop_loss:.2f}",
            target=f"{take_profit:.2f}",
            risk=f"{risk_amount:.2f}",
            bar=bar_index,
        )

        return pos

    def update_bar(
        self,
        bar_index: int,
        high: float,
        low: float,
        close: float,
    ) -> list[ClosedTrade]:
        """
        Process a new bar: check stop losses, take profits, time expiry.

        Args:
            bar_index: Current bar index.
            high: Bar high price.
            low: Bar low price.
            close: Bar close price.

        Returns:
            List of trades closed during this bar.
        """
        self._current_bar = bar_index
        closed_this_bar: list[ClosedTrade] = []
        remaining_positions: list[Position] = []

        for pos in self.open_positions:
            # Update peak/trough
            if pos.side == "BUY":
                pos.peak_price = max(pos.peak_price, high)
            else:
                pos.trough_price = min(pos.trough_price, low)

            exit_reason: str | None = None
            exit_price: float = 0.0

            # Check stop loss
            if pos.side == "BUY" and low <= pos.stop_loss:
                exit_reason = "stop_loss"
                exit_price = pos.stop_loss
            elif pos.side == "SELL" and high >= pos.stop_loss:
                exit_reason = "stop_loss"
                exit_price = pos.stop_loss

            # Check take profit (only if stop wasn't hit — conservative)
            if exit_reason is None:
                if pos.side == "BUY" and high >= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit
                elif pos.side == "SELL" and low <= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit

            # Check time expiry
            if exit_reason is None:
                bars_held = bar_index - pos.entry_bar
                if bars_held >= pos.max_holding_bars:
                    exit_reason = "time_expiry"
                    exit_price = close

            # Scale-out at 1:1 R:R (if not already done)
            if exit_reason is None and not pos.scaled_out:
                risk = abs(pos.entry_price - pos.stop_loss)
                if risk > 0:
                    if pos.side == "BUY":
                        current_profit = high - pos.entry_price
                    else:
                        current_profit = pos.entry_price - low

                    rr_multiple = current_profit / risk
                    if rr_multiple >= self.scale_out_at_rr:
                        # Scale out: close partial position
                        scale_qty = pos.quantity * self.scale_out_pct
                        pos.quantity -= scale_qty
                        pos.scaled_out = True

                        # Move stop to breakeven
                        if self.trail_to_breakeven:
                            pos.stop_loss = pos.entry_price

                        # Record partial close
                        partial_exit_price = self.cost_model.apply_exit_cost(
                            pos.entry_price + current_profit
                            if pos.side == "BUY"
                            else pos.entry_price - current_profit
                        )
                        partial_pnl = self.cost_model.compute_net_pnl(
                            pos.entry_price, partial_exit_price, scale_qty, pos.side
                        )
                        # Return proportional margin + partial PnL
                        reserved = getattr(pos, "_reserved_margin", 0.0)
                        partial_margin = reserved * self.scale_out_pct
                        pos._reserved_margin = reserved - partial_margin
                        self.cash += partial_margin + partial_pnl

                        logger.debug(
                            "scale_out",
                            symbol=pos.symbol,
                            rr=f"{rr_multiple:.1f}",
                            scaled_qty=scale_qty,
                            remaining_qty=pos.quantity,
                        )

            # Close position if exit triggered
            if exit_reason is not None:
                effective_exit = self.cost_model.apply_exit_cost(exit_price)
                net_pnl = self.cost_model.compute_net_pnl(
                    pos.entry_price, exit_price, pos.quantity, pos.side
                )
                bars_held = bar_index - pos.entry_bar

                if pos.entry_price > 0:
                    if pos.side == "BUY":
                        pnl_pct = (effective_exit - pos.entry_price) / pos.entry_price
                    else:
                        pnl_pct = (pos.entry_price - effective_exit) / pos.entry_price
                else:
                    pnl_pct = 0.0

                # Return reserved margin + net PnL to cash
                reserved = getattr(pos, "_reserved_margin", 0.0)
                self.cash += reserved + net_pnl

                trade = ClosedTrade(
                    symbol=pos.symbol,
                    side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=effective_exit,
                    quantity=pos.quantity,
                    entry_bar=pos.entry_bar,
                    exit_bar=bar_index,
                    bars_held=bars_held,
                    gross_pnl=(exit_price - pos.entry_price) * pos.quantity
                    if pos.side == "BUY"
                    else (pos.entry_price - exit_price) * pos.quantity,
                    net_pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                )
                self.closed_trades.append(trade)
                closed_this_bar.append(trade)

                logger.debug(
                    "position_closed",
                    symbol=pos.symbol,
                    side=pos.side,
                    reason=exit_reason,
                    pnl=f"{net_pnl:.2f}",
                    pnl_pct=f"{pnl_pct:.4%}",
                    bars_held=bars_held,
                )
            else:
                remaining_positions.append(pos)

        self.open_positions = remaining_positions

        # Record equity
        self.equity_history.append(self.equity)

        return closed_this_bar

    def get_equity_curve(self) -> pd.Series:
        """Return the equity curve as a pandas Series."""
        return pd.Series(self.equity_history, name="equity")

    def get_trade_list(self) -> list[dict]:
        """Return closed trades as list of dicts (for metrics computation)."""
        return [
            {
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "bars_held": t.bars_held,
                "pnl": t.net_pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
            }
            for t in self.closed_trades
        ]

    def get_summary(self) -> dict:
        """Quick summary of portfolio state."""
        return {
            "initial_capital": self.initial_capital,
            "current_equity": self.equity,
            "cash": self.cash,
            "open_positions": len(self.open_positions),
            "closed_trades": len(self.closed_trades),
            "total_return_pct": (
                (self.equity - self.initial_capital) / self.initial_capital * 100
                if self.initial_capital > 0
                else 0.0
            ),
        }
