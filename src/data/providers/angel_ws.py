"""
Apex Intelligence Engine V5 — Angel One WebSocket Client
==========================================================
Real-time tick streaming from Angel One SmartAPI, aggregated into 1-minute candles.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine
import threading

from SmartApi.smartWebSocketV2 import SmartWebSocketV2

from src.core.events import event_bus, MarketTick
from src.core.logging import get_logger
from src.data.providers.angel_api import AngelApiProvider

logger = get_logger("apex.data.angel_ws")

CandleCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]

class AngelWebSocket:
    """
    Angel One WebSocket client for real-time tick streaming.
    Aggregates incoming ticks into 1-minute candles.
    """

    def __init__(self, symbols: list[str] | None = None, interval: str = "1m"):
        self.symbols = [s.lower() for s in (symbols or ["banknifty"])]
        self.interval = interval

        self._connected = False
        self._running = False
        
        self.on_candle: CandleCallback | None = None
        self.on_tick: CandleCallback | None = None

        self._ticks_received = 0
        self._candles_received = 0
        self._connect_time = 0.0

        # SmartApi WebSocket instance
        self._sws: SmartWebSocketV2 | None = None
        self._api_provider = AngelApiProvider()
        
        # Thread sync
        self._loop = asyncio.get_event_loop()
        self._stop_event = threading.Event()

        # Candle Aggregation State
        self._current_candle: dict[str, Any] | None = None
        self._candle_start_time_ms = 0
        
        # Token maps (hardcoded for index spot for Phase 4)
        self._token_map = {
            "banknifty": "99926009",
            "nifty": "99926000"
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def uptime_seconds(self) -> float:
        if self._connect_time > 0:
            return time.time() - self._connect_time
        return 0.0

    async def connect(self) -> None:
        """Connect to Angel One WebSocket and start streaming."""
        self._running = True
        
        # Authenticate
        if not self._api_provider.auth_token:
            logger.info("angel_ws_authenticating")
            if not self._api_provider.connect():
                logger.error("angel_ws_auth_failed")
                self._running = False
                return

        self._sws = SmartWebSocketV2(
            self._api_provider.auth_token,
            self._api_provider.api_key,
            self._api_provider.client_id,
            self._api_provider.feed_token
        )
        
        self._sws.on_open = self._on_open
        self._sws.on_data = self._on_data
        self._sws.on_error = self._on_error
        self._sws.on_close = self._on_close

        logger.info("angel_ws_connecting", symbols=self.symbols)
        
        # Start WebSocket connection in a background thread to not block asyncio
        def run_ws():
            self._sws.connect()
            
        self._ws_thread = threading.Thread(target=run_ws, daemon=True)
        self._ws_thread.start()
        
        # Block until cancelled
        while self._running:
            await asyncio.sleep(1.0)
            
            # Check for closed candle in case no ticks arrive on the boundary
            if self._current_candle:
                now_ms = time.time() * 1000
                if now_ms - self._candle_start_time_ms >= 60000:
                    self._close_candle(now_ms)

    def _on_open(self, wsapp):
        self._connected = True
        self._connect_time = time.time()
        logger.info("angel_ws_connected")
        
        tokens = []
        for sym in self.symbols:
            if sym in self._token_map:
                tokens.append(self._token_map[sym])
                
        if tokens:
            token_list = [{"exchangeType": 1, "tokens": tokens}]
            # Mode 3 = SNAP_QUOTE (provides last_traded_price, vol, etc.)
            self._sws.subscribe("apex_engine", 3, token_list)
            logger.info("angel_ws_subscribed", tokens=tokens)

    def _on_data(self, wsapp, msg):
        if not self._running:
            return
            
        if "last_traded_price" in msg:
            # Price comes multiplied by 100
            price = float(msg["last_traded_price"]) / 100.0
            
            # Map token back to symbol
            token = str(msg.get("token"))
            symbol = self.symbols[0]  # Fallback
            for sym, tkn in self._token_map.items():
                if tkn == token:
                    symbol = sym
                    break
                    
            # Use asyncio.run_coroutine_threadsafe to dispatch to event loop
            asyncio.run_coroutine_threadsafe(
                self._process_tick(symbol, price),
                self._loop
            )

    def _on_error(self, wsapp, error):
        logger.error("angel_ws_error", error=str(error))

    def _on_close(self, wsapp):
        self._connected = False
        logger.warning("angel_ws_closed")

    async def _process_tick(self, symbol: str, price: float) -> None:
        """Process incoming tick and manage candle aggregation."""
        now_ms = time.time() * 1000
        
        self._ticks_received += 1
        
        if self._current_candle is None:
            # Start new candle
            self._candle_start_time_ms = now_ms
            self._current_candle = {
                "symbol": symbol,
                "timestamp": self._candle_start_time_ms,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 0.0,
                "is_closed": False,
                "trades": 0,
                "quote_volume": 0.0,
                "taker_buy_volume": 0.0,
            }
        else:
            # Update existing candle
            c = self._current_candle
            c["high"] = max(c["high"], price)
            c["low"] = min(c["low"], price)
            c["close"] = price
            c["trades"] += 1
            
            # Check if 1 minute has elapsed
            if now_ms - self._candle_start_time_ms >= 60000:
                self._close_candle(now_ms)
                
        # Emit tick event (for dashboard live price)
        if self._current_candle:
            tick_data = self._current_candle.copy()
            tick_data["close"] = price
            if self.on_tick:
                await self.on_tick(symbol, tick_data)

    def _close_candle(self, now_ms: float) -> None:
        """Close the current candle and dispatch."""
        if not self._current_candle:
            return
            
        c = self._current_candle.copy()
        c["is_closed"] = True
        
        self._candles_received += 1
        
        if self.on_candle:
            # Since this is called from the main thread loop or async task, we can await directly
            # Wait, if called from threadsafe we are in async, if from tick we are in async
            asyncio.create_task(self.on_candle(c["symbol"], c))
            
        asyncio.create_task(event_bus.emit(MarketTick(
            symbol=c["symbol"],
            price=c["close"],
            volume=c["volume"],
            source="angel_one",
        )))
        
        self._current_candle = None

    async def disconnect(self) -> None:
        self._running = False
        if self._sws:
            try:
                self._sws.close_connection()
            except:
                pass
        self._connected = False
        logger.info("angel_ws_disconnected", candles=self._candles_received)

    def get_stats(self) -> dict[str, Any]:
        return {
            "connected": self._connected,
            "uptime_s": self.uptime_seconds,
            "candles_received": self._candles_received,
            "ticks_received": self._ticks_received,
            "symbols": self.symbols,
        }
