"""
Apex Intelligence Engine V5 — Simulation WebSocket Client
==========================================================
Simulates a real-time data feed for markets without an active websocket (e.g. BankNifty in Phase 4).
Generates random-walk ticks and 1-minute candles.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Coroutine

from src.core.events import event_bus, MarketTick
from src.core.logging import get_logger

logger = get_logger("apex.data.simulation_ws")

CandleCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]

class SimulationWebSocket:
    def __init__(self, symbols: list[str] | None = None, interval: str = "1m"):
        self.symbols = [s.lower() for s in (symbols or ["banknifty"])]
        self.interval = interval

        self._running = False
        self._connected = False
        
        self.on_candle: CandleCallback | None = None
        self.on_tick: CandleCallback | None = None

        self._ticks_received = 0
        self._candles_received = 0
        self._connect_time = 0.0

        # State for random walk
        self._current_price = 54200.0  # Approx BankNifty starting price
        self._candle_open = self._current_price
        self._candle_high = self._current_price
        self._candle_low = self._current_price
        self._candle_volume = 0.0
        self._candle_start_time = time.time() * 1000

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def uptime_seconds(self) -> float:
        if self._connect_time > 0:
            return time.time() - self._connect_time
        return 0.0

    async def connect(self) -> None:
        self._running = True
        self._connected = True
        self._connect_time = time.time()
        self._candle_start_time = time.time() * 1000

        logger.info("sim_ws_connected", symbols=self.symbols)

        while self._running:
            try:
                # Random walk: -10 to +10
                move = (random.random() - 0.5) * 20.0
                self._current_price += move
                
                self._candle_high = max(self._candle_high, self._current_price)
                self._candle_low = min(self._candle_low, self._current_price)
                
                vol = random.random() * 100.0
                self._candle_volume += vol

                now_ms = time.time() * 1000
                is_closed = (now_ms - self._candle_start_time) >= 60000

                candle = {
                    "symbol": self.symbols[0],
                    "timestamp": self._candle_start_time,
                    "open": self._candle_open,
                    "high": self._candle_high,
                    "low": self._candle_low,
                    "close": self._current_price,
                    "volume": self._candle_volume,
                    "is_closed": is_closed,
                    "trades": 10,
                    "quote_volume": self._candle_volume * self._current_price,
                    "taker_buy_volume": self._candle_volume * 0.5,
                }

                self._ticks_received += 1

                if self.on_tick:
                    await self.on_tick(self.symbols[0], candle)

                if is_closed:
                    self._candles_received += 1
                    if self.on_candle:
                        await self.on_candle(self.symbols[0], candle)

                    await event_bus.emit(MarketTick(
                        symbol=self.symbols[0],
                        price=candle["close"],
                        volume=candle["volume"],
                        source="simulation",
                    ))

                    # Reset for next candle
                    self._candle_open = self._current_price
                    self._candle_high = self._current_price
                    self._candle_low = self._current_price
                    self._candle_volume = 0.0
                    self._candle_start_time = now_ms

                # Sleep 2 seconds per tick
                await asyncio.sleep(2.0)

            except Exception as e:
                logger.error("sim_ws_error", error=str(e))
                await asyncio.sleep(5.0)

    async def disconnect(self) -> None:
        self._running = False
        self._connected = False
        logger.info("sim_ws_disconnected", candles=self._candles_received)

    def get_stats(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "uptime_s": self.uptime_seconds,
            "candles_received": self._candles_received,
            "ticks_received": self._ticks_received,
            "symbols": self.symbols,
        }
