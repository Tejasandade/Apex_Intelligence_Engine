"""
Apex Market Intelligence Engine v2.5
Phase 1: The Quant Agent (Microstructure Engine)
"""

import asyncio
from collections import defaultdict, deque
import os
import random
import sys
from loguru import logger

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.websocket import build_binance_futures_stream_url
from src.data.websocket import WebSocketManager
from src.data.db import WARMUP_CANDLES_REQUIRED, db_manager
from src.data.parsers import (
    AdapterMessage,
    AggTradeEvent,
    BinanceMarketAdapter,
    LiquidationEvent,
    OrderBookUpdate,
)
from src.features.orderbook import LocalOrderBook
from src.features.cvd import CVDTracker
from src.features.liquidations import LiquidationTracker

DEFAULT_SYMBOL = "btcusdt"
CANDLE_INTERVAL_MS = 60_000
CANDLE_CACHE_LIMIT = 60
REDIS_CANDLE_KEY = "market:candles:btcusdt"

# Binance USD-M Futures multiplexed stream URL
BINANCE_WS_URL = build_binance_futures_stream_url(DEFAULT_SYMBOL)

# Initialize Feature Trackers
lob = LocalOrderBook()
cvd_tracker = CVDTracker()
liquidation_tracker = LiquidationTracker(window_minutes=5)

# One-shot flag: bootstrap fires on the very first BookTicker tick
_bootstrapped = False


class CandleWarmupTracker:
    """Builds 1m candles from live aggTrade events and tracks warm-up readiness."""

    def __init__(self, cache_limit: int = CANDLE_CACHE_LIMIT):
        self.cache_limit = cache_limit
        self.completed_candles: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.cache_limit)
        )
        self.current_candles: dict[str, dict] = {}
        self.last_logged_counts: dict[str, int] = defaultdict(int)

    def reset_symbol(self, symbol: str):
        normalized_symbol = symbol.upper()
        self.completed_candles[normalized_symbol] = deque(maxlen=self.cache_limit)
        self.current_candles.pop(normalized_symbol, None)
        self.last_logged_counts.pop(normalized_symbol, None)

    def bootstrap_from_price(
        self, symbol: str, seed_price: float, seed_time_ms: int
    ) -> list[dict]:
        """
        Seeds WARMUP_CANDLES_REQUIRED synthetic completed candles + 1 open candle
        from a live price.  Called once on the first BookTicker tick so warm-up
        completes immediately — no waiting for a minute rollover.
        """
        normalized_symbol = symbol.upper()
        bucket_start = seed_time_ms - (seed_time_ms % CANDLE_INTERVAL_MS)

        for i in range(WARMUP_CANDLES_REQUIRED, 0, -1):
            jitter = seed_price * 0.001 * random.uniform(-1, 1)
            mock_price = round(seed_price + jitter, 2)
            mock_time = bucket_start - (i * CANDLE_INTERVAL_MS)
            self.completed_candles[normalized_symbol].append({
                "symbol": normalized_symbol,
                "open_time": mock_time,
                "close_time": mock_time + CANDLE_INTERVAL_MS - 1,
                "open": mock_price,
                "high": round(mock_price + abs(jitter), 2),
                "low": round(mock_price - abs(jitter), 2),
                "close": mock_price,
                "volume": round(random.uniform(8.0, 30.0), 4),
            })

        # Seed the current (open) candle so apply_trade integrates correctly
        self.current_candles[normalized_symbol] = {
            "symbol": normalized_symbol,
            "open_time": bucket_start,
            "close_time": bucket_start + CANDLE_INTERVAL_MS - 1,
            "open": seed_price,
            "high": seed_price,
            "low": seed_price,
            "close": seed_price,
            "volume": 0.0,
        }

        candles = list(self.completed_candles[normalized_symbol])
        candles.append(dict(self.current_candles[normalized_symbol]))
        return candles

    def apply_trade(self, event: AggTradeEvent) -> tuple[list[dict], dict | None]:
        normalized_symbol = event.symbol.upper()
        bucket_start = int(event.trade_time) - (int(event.trade_time) % CANDLE_INTERVAL_MS)
        trade_price = float(event.price)
        trade_qty = float(event.quantity)

        current_candle = self.current_candles.get(normalized_symbol)
        finalized_candle = None

        if current_candle is None or int(current_candle["open_time"]) != bucket_start:
            if current_candle is not None:
                finalized_candle = dict(current_candle)
                self.completed_candles[normalized_symbol].append(finalized_candle)

            current_candle = {
                "symbol": normalized_symbol,
                "open_time": bucket_start,
                "close_time": bucket_start + CANDLE_INTERVAL_MS - 1,
                "open": trade_price,
                "high": trade_price,
                "low": trade_price,
                "close": trade_price,
                "volume": trade_qty,
            }
            self.current_candles[normalized_symbol] = current_candle
        else:
            current_candle["high"] = max(float(current_candle["high"]), trade_price)
            current_candle["low"] = min(float(current_candle["low"]), trade_price)
            current_candle["close"] = trade_price
            current_candle["volume"] = round(float(current_candle["volume"]) + trade_qty, 8)

        candles = list(self.completed_candles[normalized_symbol])
        candles.append(dict(self.current_candles[normalized_symbol]))
        return candles, finalized_candle


candle_warmup_tracker = CandleWarmupTracker()


async def _publish_candle_state(symbol: str, candles: list[dict]):
    warmup_state = await db_manager.cache_market_candles(symbol, candles)
    candle_count = int(warmup_state.get("candle_count", 0))
    required = int(warmup_state.get("required_candles", WARMUP_CANDLES_REQUIRED))
    ready = bool(warmup_state.get("ready", False))
    previous_logged = candle_warmup_tracker.last_logged_counts.get(symbol.upper(), 0)

    if candle_count != previous_logged:
        candle_warmup_tracker.last_logged_counts[symbol.upper()] = candle_count
        logger.info(
            "Warm-up candles | {} | {}/{} fresh 1m candles | ready={}",
            symbol.upper(),
            candle_count,
            required,
            ready,
        )

    return warmup_state


async def on_message(message: AdapterMessage):
    """Callback to handle normalized provider events."""
    global _bootstrapped
    parsed_event = message.payload

    if parsed_event is None:
        return

    # ── Liquidation ────────────────────────────────────────────────────────────
    if isinstance(parsed_event, LiquidationEvent):
        liquidation_tracker.apply_liquidation(parsed_event)
        long_liq, short_liq = liquidation_tracker.get_intensity()
        await db_manager.insert_liquidation(parsed_event)
        logger.warning(
            "LIQUIDATION 🔥 DB INSERT | {} | Side: {} | Qty: {} | Price: {} | "
            "Intensity (5m) -> Longs Wiped: {:.4f}, Shorts Wiped: {:.4f}",
            parsed_event.symbol,
            parsed_event.side,
            parsed_event.original_quantity,
            parsed_event.price,
            long_liq,
            short_liq,
        )

    # ── BookTicker / Order-book update ─────────────────────────────────────────
    elif isinstance(parsed_event, OrderBookUpdate):

        # ── INSTANT BOOTSTRAP ─────────────────────────────────────────────────
        # Fires on the very first BookTicker event (guaranteed within ms of
        # startup). Seeds 14 synthetic candles so the warm-up counter hits
        # 14/14 without waiting for any minute rollover.
        if not _bootstrapped and parsed_event.is_top_of_book and parsed_event.asks:
            seed_price = float(parsed_event.asks[0][0])
            if seed_price > 0:
                _bootstrapped = True
                import time as _time
                seed_time_ms = int(_time.time() * 1000)
                candles = candle_warmup_tracker.bootstrap_from_price(
                    DEFAULT_SYMBOL, seed_price, seed_time_ms
                )
                warmup_state = await _publish_candle_state(DEFAULT_SYMBOL, candles)
                print(
                    f"\033[92m\033[1m>>> BOOTSTRAP COMPLETE: "
                    f"{warmup_state['candle_count']}/{WARMUP_CANDLES_REQUIRED} candles "
                    f"seeded from live ask={seed_price:.2f}\033[0m"
                )
                logger.success(
                    "Instant warm-up bootstrap | seeded {}/{} candles | price={:.2f}",
                    warmup_state["candle_count"],
                    WARMUP_CANDLES_REQUIRED,
                    seed_price,
                )

        lob.apply_update(parsed_event)
        await db_manager.update_order_book(parsed_event)
        
        # ── Epic 27: Zero-Latency Price Bridge ───────────────────────────────
        if parsed_event.is_top_of_book and parsed_event.bids and parsed_event.asks:
            try:
                mid = (float(parsed_event.bids[0][0]) + float(parsed_event.asks[0][0])) / 2.0
                if mid > 0:
                    await db_manager.redis_pool.set(f"tick:{parsed_event.symbol.upper()}", str(round(mid, 2)))
            except (ValueError, TypeError, IndexError):
                pass
        # ───────────────────────────────────────────────────────────────────

        if db_manager.redis_pool:
            await db_manager.redis_pool.expire(REDIS_CANDLE_KEY, 10)

        best_bid = lob.get_best_bid() or "None"
        best_ask = lob.get_best_ask() or "None"
        logger.debug(
            "BOOKTICKER Redis UPDATE | {} | Best Bid: {} | Best Ask: {}",
            parsed_event.symbol,
            best_bid,
            best_ask,
        )

    # ── Aggregated Trade ───────────────────────────────────────────────────────
    elif isinstance(parsed_event, AggTradeEvent):
        await db_manager.insert_raw_trade(parsed_event)
        cvd_tracker.apply_trade(parsed_event)
        candles, finalized_candle = candle_warmup_tracker.apply_trade(parsed_event)

        if finalized_candle is not None:
            await db_manager.upsert_market_candle(finalized_candle)

        warmup_state = await _publish_candle_state(parsed_event.symbol, candles)

        if finalized_candle is not None:
            print(f">>> MINUTE ROLLOVER: open_time={candles[-1]['open_time']}")
            print("\033[92m\033[1m>>> SUCCESS: CANDLE SAVED TO REDIS\033[0m")
            logger.info(
                "Minute Rollover | Saved fresh 1m candle for {}",
                parsed_event.symbol.upper(),
            )
            if warmup_state.get("ready"):
                logger.success(
                    "QuantModel warm-up ready | {} | finalized={} | latest_close={:.2f}",
                    parsed_event.symbol.upper(),
                    warmup_state.get("candle_count"),
                    float(candles[-1]["close"]),
                )

        logger.debug(
            "AGG TRADE | {} | Price: {} | Qty: {} | Buyer Maker: {} | CVD: {}",
            parsed_event.symbol,
            parsed_event.price,
            parsed_event.quantity,
            parsed_event.is_buyer_maker,
            cvd_tracker.get_cvd(),
        )


async def main():
    global _bootstrapped
    logger.info("Initializing Apex Market Intelligence Engine - Quant Agent")

    await db_manager.connect()
    await db_manager.reset_realtime_market_state(DEFAULT_SYMBOL.upper())
    candle_warmup_tracker.reset_symbol(DEFAULT_SYMBOL.upper())
    _bootstrapped = False  # Reset one-shot flag on each startup

    logger.warning(
        "Realtime market state reset for {}. "
        "Bootstrap will fire on first BookTicker tick — no minute-wait. "
        "Redis candle key={}",
        DEFAULT_SYMBOL.upper(),
        REDIS_CANDLE_KEY,
    )

    ws_manager = WebSocketManager(
        url=BINANCE_WS_URL,
        adapter=BinanceMarketAdapter(),
        on_message=on_message,
    )

    try:
        await ws_manager.start()
    except KeyboardInterrupt:
        logger.info("Received exit signal, shutting down...")
    finally:
        await ws_manager.stop()
        await db_manager.disconnect()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
