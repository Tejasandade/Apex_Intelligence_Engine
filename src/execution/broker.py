"""
Apex Intelligence Engine V5 — Broker Abstraction
===================================================
Unified interface for all brokers. Every broker (Binance, Dhan, OANDA)
implements this interface, so the execution engine doesn't care which
broker is behind it.

This is the key to multi-market support:
    Signal Engine → Execution Engine → BrokerInterface → [Binance | Dhan | OANDA]

The execution engine speaks ONE language. Each broker adapter translates
to the specific broker's API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Order Types ──────────────────────────────────────────────────────────────

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    STOP_MARKET = "STOP_MARKET"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Order:
    """A trading order."""

    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: float = 0.0  # For LIMIT orders
    stop_price: float = 0.0  # For STOP orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    filled_at: datetime | None = None
    broker: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass
class Position:
    """An open trading position."""

    symbol: str = ""
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signal_id: str = ""  # Link back to the signal that opened this

    # Trailing stop fields (managed by SmartTrailingStop)
    trail_phase: str = "INITIAL"  # INITIAL, BREAKEVEN, TRAILING
    initial_quantity: float = 0.0  # Original qty before partial closes
    highest_price: float = 0.0  # Peak price since entry (for LONG trailing)
    lowest_price: float = 0.0  # Lowest price since entry (for SHORT trailing)
    initial_atr: float = 0.0  # ATR at time of entry

    @property
    def pnl_pct(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) / self.entry_price
        else:
            return (self.entry_price - self.current_price) / self.entry_price

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0


@dataclass
class AccountBalance:
    """Broker account balance snapshot."""

    total_equity: float = 0.0
    available_balance: float = 0.0
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    currency: str = "USD"
    broker: str = ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TradeRecord:
    """A completed trade (entry + exit)."""

    trade_id: str = ""
    signal_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    entry_time: datetime | None = None
    exit_time: datetime | None = None
    exit_reason: str = ""  # "stop_loss", "take_profit", "signal", "manual"
    duration_seconds: float = 0.0
    regime: str = ""
    council_score: float = 0.0


# ── Broker Interface ─────────────────────────────────────────────────────────

class BaseBroker(ABC):
    """
    Abstract broker interface. All broker adapters must implement this.

    The execution engine calls these methods without knowing which
    broker is behind them.
    """

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Unique broker identifier (e.g., 'binance', 'dhan')."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Whether the broker connection is active."""
        ...

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker API. Returns True on success."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the broker API."""
        ...

    # ── Orders ───────────────────────────────────────────────────────────────

    @abstractmethod
    async def place_order(self, order: Order) -> Order:
        """
        Place an order with the broker.

        Args:
            order: Order to place.

        Returns:
            Updated order with broker-assigned ID and status.
        """
        ...

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True on success."""
        ...

    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Get the current status of an order."""
        ...

    # ── Positions ────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    @abstractmethod
    async def close_position(self, symbol: str) -> Order:
        """Close a position for a symbol via market order."""
        ...

    # ── Account ──────────────────────────────────────────────────────────────

    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """Get account balance."""
        ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get the current market price for a symbol."""
        ...
