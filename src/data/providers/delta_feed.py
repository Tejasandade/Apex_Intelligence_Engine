"""
Apex Intelligence Engine — Delta Exchange Gold Order Flow & Microstructure Feed
================================================================================
Adapted from xau_c (Gold Observatory 0.21.0 / delta_depth.py & microstructure.py).

Connects to Delta Exchange public WebSocket (wss://public-socket.india.delta.exchange)
to stream institutional PAXGUSD derivatives microstructure:
1. Top-5 Level-2 Order Book Imbalance (OBI_5): (BidDepth - AskDepth) / (BidDepth + AskDepth)
2. Real-time Aggressor Trade Delta (Market Buys vs Market Sells)
3. Direct institutional order flow feed into Council Momentum & Volume advisors.
Failsafe fallback: If disconnected, Apex automatically uses Binance PAXGUSDT feed.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import time
from decimal import Decimal
from typing import Any, Optional

import websockets

logger = logging.getLogger("apex.data.delta_feed")

DELTA_WS_URL = "wss://public-socket.india.delta.exchange"
PAXG_SYMBOL = "PAXGUSD"


class DeltaOrderBook:
    """Maintains synchronized L2 order book for Delta PAXGUSD."""

    def __init__(self, symbol: str = PAXG_SYMBOL):
        self.symbol = symbol
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.last_update_ts: float = 0.0

    def reset(self):
        self.bids.clear()
        self.asks.clear()

    def apply_update(self, body: dict) -> None:
        try:
            if body.get("type") != "ob_updates" or body.get("sy") != self.symbol:
                return
            action = body.get("action")
            if action == "snapshot":
                self.reset()

            for side_key, book_dict in (("b", self.bids), ("a", self.asks)):
                for item in body.get(side_key, []):
                    price = float(item[0])
                    size = float(item[1])
                    if size == 0:
                        book_dict.pop(price, None)
                    else:
                        book_dict[price] = size

            self.last_update_ts = time.time()
        except Exception as e:
            logger.debug("delta_book_update_error", exc_info=e)

    def get_imbalance_top_n(self, n: int = 5) -> float:
        """Calculate Book Imbalance for top N price levels: (Bids - Asks) / (Bids + Asks)."""
        if not self.bids or not self.asks:
            return 0.0

        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[:n]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[:n]

        bid_vol = sum(size for _, size in sorted_bids)
        ask_vol = sum(size for _, size in sorted_asks)
        total = bid_vol + ask_vol

        if total <= 0:
            return 0.0
        return (bid_vol - ask_vol) / total


class DeltaGoldFeed:
    """
    Background worker that streams Delta Exchange PAXG perpetual order flow.
    """

    def __init__(self):
        self.book = DeltaOrderBook(PAXG_SYMBOL)
        self.trades_history: collections.deque = collections.deque(maxlen=1000)
        self.is_connected: bool = False
        self.last_heartbeat: float = 0.0
        self._running: bool = False

    def get_telemetry(self) -> dict[str, Any]:
        """
        Return instantaneous microstructure telemetry for Gold.
        """
        now = time.time()
        fresh = (now - self.book.last_update_ts) < 10.0 if self.book.last_update_ts > 0 else False

        # Compute rolling 5-minute aggressor volume delta
        five_min_ago = now - 300.0
        recent_trades = [t for t in self.trades_history if t["ts"] >= five_min_ago]

        buy_vol = sum(t["size"] for t in recent_trades if t["side"] == "buy")
        sell_vol = sum(t["size"] for t in recent_trades if t["side"] == "sell")
        trade_total = buy_vol + sell_vol
        aggressor_ratio = (buy_vol - sell_vol) / trade_total if trade_total > 0 else 0.0

        obi_5 = self.book.get_imbalance_top_n(5) if fresh else 0.0

        return {
            "connected": self.is_connected and fresh,
            "fresh": fresh,
            "book_imbalance_5": round(obi_5, 4),
            "aggressor_ratio": round(aggressor_ratio, 4),
            "buy_volume_5m": round(buy_vol, 2),
            "sell_volume_5m": round(sell_vol, 2),
            "trade_count_5m": len(recent_trades),
            "last_ts": self.book.last_update_ts,
        }

    async def run(self):
        """Persistent WebSocket loop with automatic reconnect."""
        self._running = True
        while self._running:
            try:
                logger.info("connecting_to_delta_ws", url=DELTA_WS_URL)
                async with websockets.connect(
                    DELTA_WS_URL,
                    open_timeout=12,
                    ping_interval=20,
                    ping_timeout=15,
                    max_size=2**21,
                ) as ws:
                    # 1. Subscribe to trades
                    sub_trades = {
                        "type": "subscribe",
                        "payload": {"channels": [{"name": "trades", "symbols": [PAXG_SYMBOL]}]},
                    }
                    await ws.send(json.dumps(sub_trades))

                    # 2. Subscribe to L2 book updates
                    sub_book = {
                        "type": "subscribe",
                        "payload": {"channels": [{"name": "ob_updates", "symbols": [PAXG_SYMBOL]}]},
                    }
                    await ws.send(json.dumps(sub_book))

                    # 3. Enable heartbeat
                    await ws.send(json.dumps({"type": "enable_heartbeat"}))
                    self.is_connected = True
                    logger.info("delta_ws_subscribed_successfully", symbol=PAXG_SYMBOL)

                    while self._running:
                        msg_str = await asyncio.wait_for(ws.recv(), timeout=30)
                        data = json.loads(msg_str)
                        msg_type = data.get("type")

                        if msg_type == "heartbeat":
                            self.last_heartbeat = time.time()
                        elif msg_type == "trades":
                            # Process executed market trades
                            # Format: {"type": "trades", "symbol": "PAXGUSD", "trades": [{"price": ..., "size": ..., "side": "buy"}]}
                            for tr in data.get("trades", []):
                                self.trades_history.append({
                                    "ts": time.time(),
                                    "price": float(tr.get("price", 0.0)),
                                    "size": float(tr.get("size", 0.0)),
                                    "side": str(tr.get("side", "")).lower(),
                                })
                        elif msg_type == "ob_updates":
                            # Process full L2 order book updates
                            self.book.apply_update(data)

            except asyncio.CancelledError:
                self._running = False
                break
            except Exception as e:
                self.is_connected = False
                logger.debug("delta_ws_reconnecting", error=str(e))
                await asyncio.sleep(5)


# Global singleton
delta_gold_feed = DeltaGoldFeed()
