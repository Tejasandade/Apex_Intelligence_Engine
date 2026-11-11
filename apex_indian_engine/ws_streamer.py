"""
Apex Indian Engine — Real-Time WebSocket Streaming & Candle Aggregator
======================================================================
Streams live sub-second ticks via Angel One SmartWebSocketV2 for:
- NIFTY 50 Spot (Token: 99926000, Exch: 1 / NSE)
- BANKNIFTY Spot (Token: 99926009, Exch: 1 / NSE)
- INDIA VIX (Token: 99926017, Exch: 1 / NSE)
- Active Option Contracts (Exch: 2 / NFO)

Aggregates ticks into 1-minute and 5-minute OHLCV candles with True VWAP.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from SmartApi.smartWebSocketV2 import SmartWebSocketV2

logger = logging.getLogger("apex.indian.ws")

# Callbacks
TickCallback = Callable[[str, float, float], Coroutine[Any, Any, None]]  # (symbol, price, volume)
CandleCallback = Callable[[str, dict[str, Any]], Coroutine[Any, Any, None]]  # (symbol, candle_dict)
VixCallback = Callable[[float], Coroutine[Any, Any, None]]  # (vix_value)


class IndianWebSocketStreamer:
    """
    Asynchronous WebSocket streamer and candle aggregator for Indian markets.
    """

    SPOT_TOKENS = {
        "99926000": "NIFTY",
        "99926009": "BANKNIFTY",
        "99926017": "INDIAVIX",
    }

    def __init__(
        self,
        auth_token: str,
        api_key: str,
        client_id: str,
        feed_token: str,
    ):
        self.auth_token = auth_token
        self.api_key = api_key
        self.client_id = client_id
        self.feed_token = feed_token

        self._sws: Optional[SmartWebSocketV2] = None
        self._connected = False
        self._running = False
        self._ws_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Registered Option Tokens: token -> symbol
        self._option_tokens: dict[str, str] = {}

        # Latest Prices & Tick Timestamps
        self.latest_prices: dict[str, float] = {
            "NIFTY": 0.0,
            "BANKNIFTY": 0.0,
            "INDIAVIX": 14.0,  # Default safe baseline
        }
        self.last_tick_ts: dict[str, float] = {}

        # Candle Aggregation & Cumulative Session VWAP: symbol -> state
        self._candles_5m: dict[str, dict[str, Any]] = {}
        self._candle_start_ts: dict[str, float] = {}
        self._session_date: dict[str, str] = {}
        self._session_tp_sum: dict[str, float] = {}
        self._session_bar_count: dict[str, int] = {}
        self._session_pv_sum: dict[str, float] = {}
        self._session_vol_sum: dict[str, float] = {}
        self._prev_cum_vol: dict[str, float] = {}

        # Callbacks
        self.on_tick: Optional[TickCallback] = None
        self.on_candle_closed: Optional[CandleCallback] = None
        self.on_vix_updated: Optional[VixCallback] = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the WebSocket in a daemon thread."""
        self._loop = loop or asyncio.get_event_loop()
        self._running = True

        self._sws = SmartWebSocketV2(
            auth_token=self.auth_token,
            api_key=self.api_key,
            client_code=self.client_id,
            feed_token=self.feed_token,
        )

        self._sws.on_open = self._on_open
        self._sws.on_data = self._on_data
        self._sws.on_error = self._on_error
        self._sws.on_close = self._on_close

        logger.info(f"Starting Angel One SmartWebSocketV2 for client {self.client_id}...")

        def _run():
            try:
                self._sws.connect()
            except Exception as e:
                logger.error(f"WebSocket connect error: {e}")

        self._ws_thread = threading.Thread(target=_run, daemon=True, name="AngelWSThread")
        self._ws_thread.start()

    def stop(self) -> None:
        """Stops the streaming thread cleanly."""
        self._running = False
        if self._sws:
            try:
                self._sws.close_connection()
            except Exception:
                pass
        self._connected = False
        logger.info("Indian WebSocket Streamer stopped.")

    def subscribe_option_contract(self, token: str, symbol: str) -> None:
        """Subscribes to an active option contract (NFO / Exch: 2)."""
        token_str = str(token)
        self._option_tokens[token_str] = symbol
        if self._connected and self._sws:
            try:
                token_list = [{"exchangeType": 2, "tokens": [token_str]}]
                self._sws.subscribe("apex_opt", 3, token_list)
                logger.info(f"Subscribed to live option feed: {symbol} (Token: {token_str})")
            except Exception as e:
                logger.error(f"Failed to subscribe to option token {token_str}: {e}")

    def unsubscribe_option_contract(self, token: str) -> None:
        """Unsubscribes from an option contract when closed."""
        token_str = str(token)
        if token_str in self._option_tokens:
            del self._option_tokens[token_str]
        if self._connected and self._sws:
            try:
                token_list = [{"exchangeType": 2, "tokens": [token_str]}]
                self._sws.unsubscribe("apex_opt", 3, token_list)
                logger.info(f"Unsubscribed option token {token_str}")
            except Exception as e:
                logger.error(f"Failed to unsubscribe option token {token_str}: {e}")

    def _on_open(self, wsapp):
        self._connected = True
        logger.info("Angel One WebSocket connection OPEN.")

        # Subscribe to Cash Indices (Exchange: 1 = NSE) in Mode 1 (LTP: instantaneous sub-second ticks)
        spot_tokens = list(self.SPOT_TOKENS.keys())
        token_list = [{"exchangeType": 1, "tokens": spot_tokens}]
        self._sws.subscribe("apex_spot", 1, token_list)
        logger.info(f"Subscribed to Spot Indices: NIFTY, BANKNIFTY, INDIA VIX in Mode 1 (LTP) ({spot_tokens})")

        # Subscribe to any registered options in Mode 1 (LTP)
        if self._option_tokens:
            opt_list = [{"exchangeType": 2, "tokens": list(self._option_tokens.keys())}]
            self._sws.subscribe("apex_opt", 1, opt_list)

    def _on_data(self, wsapp, msg: dict[str, Any]):
        if not self._running:
            return

        try:
            token = str(msg.get("token", ""))
            raw_px = msg.get("last_traded_price")
            if raw_px is None:
                return

            # Angel One prices are multiplied by 100 (in paise)
            price = round(float(raw_px) / 100.0, 2)
            vol = float(msg.get("vol_traded", 0.0))

            # Identify Symbol
            symbol = self.SPOT_TOKENS.get(token) or self._option_tokens.get(token)
            if not symbol:
                return

            self.latest_prices[symbol] = price
            self.last_tick_ts[symbol] = time.time()

            if symbol == "INDIAVIX":
                if self.on_vix_updated and self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self.on_vix_updated(price), self._loop)
                return

            # Dispatch tick callback
            if self.on_tick and self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.on_tick(symbol, price, vol), self._loop)

            # Aggregate into 5-minute candles
            self._aggregate_candle(symbol, price, vol)

        except Exception as e:
            logger.debug(f"Tick processing error: {e}")

    def seed_session_vwap(self, symbol: str, candles_today: list[dict[str, Any]]) -> None:
        """Seed daily cumulative session VWAP/TWAP anchor from warmup intraday bars."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        today_bars = [c for c in candles_today if str(c.get("timestamp", "")).startswith(today_str)]
        self._session_date[symbol] = today_str
        tp_sum = 0.0
        pv_sum = 0.0
        vol_sum = 0.0
        for c in today_bars:
            tp = (float(c["high"]) + float(c["low"]) + float(c["close"])) / 3.0
            v = float(c.get("volume", 0.0))
            tp_sum += tp
            if v > 0:
                pv_sum += tp * v
                vol_sum += v
        self._session_tp_sum[symbol] = tp_sum
        self._session_bar_count[symbol] = len(today_bars)
        self._session_pv_sum[symbol] = pv_sum
        self._session_vol_sum[symbol] = vol_sum

    def _compute_live_session_vwap(self, symbol: str, candle: dict[str, Any], price: float, volume: float) -> float:
        """
        Compute cumulative intraday session VWAP (or Typical-Price TWAP anchor when index volume == 0).
        Resets only at the start of a new NSE trading day, never every 5 minutes.
        """
        cur_tp = (float(candle["high"]) + float(candle["low"]) + float(candle["close"])) / 3.0
        if volume > 0:
            prev_v = self._prev_cum_vol.get(symbol, volume)
            delta_v = max(volume - prev_v, 0.0) if volume >= prev_v else volume
            self._prev_cum_vol[symbol] = volume
            self._session_pv_sum[symbol] = self._session_pv_sum.get(symbol, 0.0) + (price * delta_v)
            self._session_vol_sum[symbol] = self._session_vol_sum.get(symbol, 0.0) + delta_v
            if self._session_vol_sum[symbol] > 0:
                return round(self._session_pv_sum[symbol] / self._session_vol_sum[symbol], 2)

        completed_tp_sum = self._session_tp_sum.get(symbol, 0.0)
        completed_bars = self._session_bar_count.get(symbol, 0)
        return round((completed_tp_sum + cur_tp) / max(completed_bars + 1, 1), 2)

    def _aggregate_candle(self, symbol: str, price: float, volume: float) -> None:
        """Aggregates ticks into 5-minute wall-clock aligned candles (:00, :05, :10, :15, etc.) with cumulative session VWAP."""
        now = time.time()
        slot_start = int(now // 300) * 300
        today_str = datetime.fromtimestamp(slot_start).strftime("%Y-%m-%d")

        if self._session_date.get(symbol) != today_str:
            self._session_date[symbol] = today_str
            self._session_tp_sum[symbol] = 0.0
            self._session_bar_count[symbol] = 0
            self._session_pv_sum[symbol] = 0.0
            self._session_vol_sum[symbol] = 0.0
            self._prev_cum_vol[symbol] = volume

        if symbol not in self._candle_start_ts or self._candle_start_ts[symbol] != slot_start:
            if symbol in self._candles_5m:
                # Close previous candle & fold its typical price into cumulative session anchor
                closed_c = self._candles_5m[symbol].copy()
                prev_tp = (float(closed_c["high"]) + float(closed_c["low"]) + float(closed_c["close"])) / 3.0
                self._session_tp_sum[symbol] = self._session_tp_sum.get(symbol, 0.0) + prev_tp
                self._session_bar_count[symbol] = self._session_bar_count.get(symbol, 0) + 1
                closed_c["vwap"] = self._compute_live_session_vwap(symbol, closed_c, float(closed_c["close"]), 0.0)
                closed_c["is_closed"] = True
                if self.on_candle_closed and self._loop and not self._loop.is_closed():
                    asyncio.run_coroutine_threadsafe(self.on_candle_closed(symbol, closed_c), self._loop)

            # Start new candle aligned to NSE 5-minute interval
            now_iso = datetime.fromtimestamp(slot_start).strftime("%Y-%m-%dT%H:%M:%S+05:30")
            self._candle_start_ts[symbol] = slot_start
            new_c = {
                "timestamp": now_iso,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
                "vwap": price,
                "ticks": 1,
            }
            new_c["vwap"] = self._compute_live_session_vwap(symbol, new_c, price, volume)
            self._candles_5m[symbol] = new_c
        else:
            # Update current candle
            c = self._candles_5m[symbol]
            c["high"] = max(c["high"], price)
            c["low"] = min(c["low"], price)
            c["close"] = price
            c["volume"] = max(c["volume"], volume)
            c["ticks"] += 1
            c["vwap"] = self._compute_live_session_vwap(symbol, c, price, volume)

    def update_external_tick(self, symbol: str, price: float, volume: float = 0.0) -> None:
        """Allows engine fallback pollers to update latest prices and candle states."""
        if price <= 0:
            return
        self.latest_prices[symbol] = price
        self._aggregate_candle(symbol, price, volume)

    def _on_error(self, wsapp, error):
        logger.error(f"Angel One WebSocket Error: {error}")

    def _on_close(self, wsapp):
        self._connected = False
        logger.warning("Angel One WebSocket connection CLOSED.")
        # Auto-reconnect if engine is still running
        if self._running:
            logger.info("Attempting WebSocket reconnection in 3 seconds...")
            time.sleep(3.0)
            try:
                self._sws.connect()
            except Exception as e:
                logger.error(f"WebSocket reconnect failed: {e}")

