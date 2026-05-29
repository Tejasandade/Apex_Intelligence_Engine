"""
Apex Intelligence Engine V5 — Event Bus
=========================================
Typed events for all inter-module communication.
Modules publish events to the bus; other modules subscribe to events they care about.
This decouples all components — no direct imports between pillars.

Usage:
    from src.core.events import event_bus, MarketTick, SignalGenerated

    # Subscribe
    @event_bus.on(MarketTick)
    async def handle_tick(event: MarketTick):
        ...

    # Publish
    await event_bus.emit(MarketTick(symbol="BTCUSDT", price=68000.0, ...))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Type


# ── Event Base ───────────────────────────────────────────────────────────────
@dataclass
class Event:
    """Base event class. All events carry a timestamp."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Market Data Events ──────────────────────────────────────────────────────
@dataclass
class MarketTick(Event):
    """A single price update from a data provider."""

    symbol: str = ""
    price: float = 0.0
    volume: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    source: str = ""  # "binance", "dhan", etc.


@dataclass
class CandleClose(Event):
    """A finalized 1-minute candle."""

    symbol: str = ""
    interval: str = "1m"
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    open_time: int = 0  # Unix ms
    close_time: int = 0


@dataclass
class OrderBookSnapshot(Event):
    """Top-of-book or full order book update."""

    symbol: str = ""
    bids: list[list[float]] = field(default_factory=list)
    asks: list[list[float]] = field(default_factory=list)
    is_top_of_book: bool = True


# ── Signal Events ───────────────────────────────────────────────────────────
class Direction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalGenerated(Event):
    """A trade signal produced by the Council."""

    symbol: str = ""
    direction: Direction = Direction.HOLD
    conviction: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size_pct: float = 0.0
    quantity: float = 0.0
    regime: str = ""
    reasoning: str = ""
    advisor_votes: dict[str, Any] = field(default_factory=dict)


# ── Execution Events ───────────────────────────────────────────────────────
@dataclass
class OrderPlaced(Event):
    """An order was sent to the broker."""

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    order_type: str = "MARKET"
    broker: str = ""


@dataclass
class OrderFilled(Event):
    """An order was filled by the broker."""

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class TradeResult(Event):
    """A trade was closed — the final outcome."""

    order_id: str = ""
    symbol: str = ""
    side: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    duration_seconds: float = 0.0


# ── System Events ──────────────────────────────────────────────────────────
@dataclass
class KillSwitchActivated(Event):
    """The kill switch was triggered."""

    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelDecayAlert(Event):
    """A model's performance has decayed beyond acceptable thresholds."""

    model_name: str = ""
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0


# ── Event Bus ───────────────────────────────────────────────────────────────
EventHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """
    Async publish/subscribe event bus.

    Subscribers register handlers for specific event types.
    When an event is emitted, all registered handlers for that type are called.
    Handlers that raise exceptions are logged but don't crash the bus.
    """

    def __init__(self):
        self._handlers: dict[Type[Event], list[EventHandler]] = {}
        self._lock = asyncio.Lock()

    def on(self, event_type: Type[Event]):
        """Decorator to register a handler for an event type."""

        def decorator(func: EventHandler) -> EventHandler:
            self.subscribe(event_type, func)
            return func

        return decorator

    def subscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[Event], handler: EventHandler) -> None:
        """Remove a handler for an event type."""
        if event_type in self._handlers:
            self._handlers[event_type] = [
                h for h in self._handlers[event_type] if h is not handler
            ]

    async def emit(self, event: Event) -> None:
        """Publish an event to all subscribed handlers."""
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            return

        # Fire all handlers concurrently; isolate failures
        results = await asyncio.gather(
            *[self._safe_call(h, event) for h in handlers],
            return_exceptions=True,
        )

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Import here to avoid circular dependency
                import structlog

                logger = structlog.get_logger("apex.events")
                logger.error(
                    "event_handler_error",
                    event_type=event_type.__name__,
                    handler=handlers[i].__name__,
                    error=str(result),
                )

    @staticmethod
    async def _safe_call(handler: EventHandler, event: Event) -> None:
        """Call a handler, catching and returning any exception."""
        await handler(event)

    def clear(self) -> None:
        """Remove all handlers. Useful for testing."""
        self._handlers.clear()


# ── Global Singleton ────────────────────────────────────────────────────────
event_bus = EventBus()
