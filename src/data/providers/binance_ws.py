"""
Apex Intelligence Engine V5 — Binance WebSocket Client
========================================================
Real-time 1-minute candle streaming from Binance Futures via WebSocket.

Features:
- Auto-reconnection with exponential backoff
- Heartbeat monitoring (detects stale connections)
- Clean candle event emission via the event bus
- Multi-symbol streaming support

Usage:
    ws = BinanceWebSocket(symbols=["btcusdt", "ethusdt"])
    ws.on_candle = my_callback  # Called with (symbol, candle_dict)
    await ws.connect()
    # ... runs forever, reconnecting on failures
    await ws.disconnect()
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Coroutine

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from src.core.events import event_bus, MarketTick
from src.core.logging import get_logger

logger = get_logger("apex.data.binance_ws")

# Binance Spot WebSocket (Futures is geo-blocked in India)
WS_BASE_URL = "wss://stream.binance.com:9443/ws"

# Reconnection settings
INITIAL_BACKOFF_S = 1.0
MAX_BACKOFF_S = 60.0
BACKOFF_MULTIPLIER = 2.0
HEARTBEAT_TIMEOUT_S = 30.0  # If no message in 30s, reconnect


CandleCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]


class BinanceWebSocket:
    """
    Binance Futures WebSocket client for real-time kline (candle) streaming.

    Streams 1-minute candles and emits events on each closed candle.
    Handles reconnection automatically with exponential backoff.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        interval: str = "1m",
    ):
        """
        Args:
            symbols: List of symbols to stream (e.g., ["btcusdt", "ethusdt"]).
            interval: Candle interval (default: "1m").
        """
        self.symbols = [s.lower() for s in (symbols or ["btcusdt"])]
        self.interval = interval

        # Connection state
        self._ws: Any = None
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._last_message_time: float = 0.0

        # Callbacks
        self.on_candle: CandleCallback | None = None
        self.on_tick: CandleCallback | None = None  # Every tick, not just closed

        # Stats
        self._ticks_received = 0
        self._candles_received = 0
        self._connect_time: float = 0.0

        # Orderbook State (Phase 4)
        self._orderbook_state: dict[str, dict[str, float]] = {
            sym: {"best_bid": 0.0, "best_bid_qty": 0.0, "best_ask": 0.0, "best_ask_qty": 0.0}
            for sym in self.symbols
        }

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    @property
    def uptime_seconds(self) -> float:
        if self._connect_time > 0:
            return time.time() - self._connect_time
        return 0.0

    def _build_stream_url(self) -> str:
        """Build the combined stream URL for all symbols."""
        streams = []
        for sym in self.symbols:
            streams.append(f"{sym}@kline_{self.interval}")
            streams.append(f"{sym}@bookTicker")
        stream_path = "/".join(streams)
        return f"wss://stream.binance.com:9443/stream?streams={stream_path}"

    async def connect(self) -> None:
        """
        Connect to Binance WebSocket and start streaming.
        Runs forever, auto-reconnecting on failures.
        """
        self._running = True
        backoff = INITIAL_BACKOFF_S

        logger.info(
            "ws_connecting",
            symbols=self.symbols,
            interval=self.interval,
        )

        while self._running:
            try:
                url = self._build_stream_url()
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    self._connect_time = time.time()
                    self._last_message_time = time.time()
                    backoff = INITIAL_BACKOFF_S  # Reset backoff on success

                    logger.info(
                        "ws_connected",
                        url=url[:80],
                        reconnect_count=self._reconnect_count,
                    )

                    # Run the message loop + heartbeat monitor concurrently
                    await asyncio.gather(
                        self._message_loop(ws),
                        self._heartbeat_monitor(),
                    )

            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                self._connected = False
                logger.warning(
                    "ws_disconnected",
                    reason=str(e),
                    reconnect_in=f"{backoff:.1f}s",
                )

            except Exception as e:
                self._connected = False
                logger.error(
                    "ws_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    reconnect_in=f"{backoff:.1f}s",
                )

            if not self._running:
                break

            # Exponential backoff
            self._reconnect_count += 1
            logger.info("ws_reconnecting", attempt=self._reconnect_count, backoff=f"{backoff:.1f}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF_S)

    async def _message_loop(self, ws: Any) -> None:
        """Process incoming WebSocket messages."""
        async for raw_msg in ws:
            if not self._running:
                break

            self._last_message_time = time.time()

            try:
                msg = json.loads(raw_msg)

                # Combined stream format: {"stream": "...", "data": {...}}
                if "data" in msg:
                    data = msg["data"]
                else:
                    data = msg

                if "e" not in data:
                    # bookTicker doesn't have an "e" field, but we can detect it by its structure
                    if "u" in data and "b" in data and "B" in data and "a" in data and "A" in data:
                        symbol = data["s"].lower()
                        if symbol in self._orderbook_state:
                            self._orderbook_state[symbol] = {
                                "best_bid": float(data["b"]),
                                "best_bid_qty": float(data["B"]),
                                "best_ask": float(data["a"]),
                                "best_ask_qty": float(data["A"]),
                            }
                    continue

                if data["e"] != "kline":
                    continue

                kline = data["k"]
                symbol = kline["s"].lower()  # e.g., "btcusdt"
                is_closed = kline["x"]  # True when candle is complete
                
                # Fetch latest orderbook state
                lob = self._orderbook_state.get(symbol, {})

                candle = {
                    "symbol": symbol,
                    "timestamp": kline["t"],  # Kline start time (ms)
                    "open": float(kline["o"]),
                    "high": float(kline["h"]),
                    "low": float(kline["l"]),
                    "close": float(kline["c"]),
                    "volume": float(kline["v"]),
                    "is_closed": is_closed,
                    "trades": int(kline["n"]),
                    "quote_volume": float(kline["q"]),
                    "taker_buy_volume": float(kline["V"]),
                    # Orderbook fields
                    "best_bid": lob.get("best_bid", 0.0),
                    "best_bid_qty": lob.get("best_bid_qty", 0.0),
                    "best_ask": lob.get("best_ask", 0.0),
                    "best_ask_qty": lob.get("best_ask_qty", 0.0),
                }

                self._ticks_received += 1

                # Emit tick event (every update)
                if self.on_tick:
                    await self.on_tick(symbol, candle)

                # Emit candle event (only on close)
                if is_closed:
                    self._candles_received += 1

                    if self.on_candle:
                        await self.on_candle(symbol, candle)

                    # Also emit via event bus
                    await event_bus.emit(MarketTick(
                        symbol=symbol,
                        price=candle["close"],
                        volume=candle["volume"],
                        source="binance",
                    ))

                    if self._candles_received % 10 == 0:
                        logger.debug(
                            "ws_candle_stats",
                            candles=self._candles_received,
                            ticks=self._ticks_received,
                            uptime=f"{self.uptime_seconds:.0f}s",
                        )

            except json.JSONDecodeError:
                logger.warning("ws_invalid_json", raw=str(raw_msg)[:100])
            except Exception as e:
                logger.error("ws_message_error", error=str(e), error_type=type(e).__name__)

    async def _heartbeat_monitor(self) -> None:
        """Monitor for stale connections and force reconnect if no messages received."""
        while self._running and self._connected:
            await asyncio.sleep(HEARTBEAT_TIMEOUT_S)

            if not self._running:
                break

            elapsed = time.time() - self._last_message_time
            if elapsed > HEARTBEAT_TIMEOUT_S:
                logger.warning(
                    "ws_heartbeat_timeout",
                    elapsed=f"{elapsed:.1f}s",
                    threshold=f"{HEARTBEAT_TIMEOUT_S:.1f}s",
                )
                # Force close — the connect() loop will reconnect
                if self._ws:
                    await self._ws.close()
                break

    async def disconnect(self) -> None:
        """Gracefully disconnect from WebSocket."""
        self._running = False
        self._connected = False

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        logger.info(
            "ws_disconnected_graceful",
            candles_received=self._candles_received,
            ticks_received=self._ticks_received,
            uptime=f"{self.uptime_seconds:.0f}s",
            reconnects=self._reconnect_count,
        )

    def get_stats(self) -> dict[str, Any]:
        """Return connection statistics."""
        return {
            "connected": self._connected,
            "uptime_s": self.uptime_seconds,
            "candles_received": self._candles_received,
            "ticks_received": self._ticks_received,
            "reconnect_count": self._reconnect_count,
            "symbols": self.symbols,
        }
