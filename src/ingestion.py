"""
Apex Market Intelligence Engine v2.5
Phase 1: The Quant Agent (Microstructure Engine)
"""

import asyncio
import sys
import os
from loguru import logger

# Ensure the project root is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.websocket import WebSocketManager
from src.data.parsers import BinanceParser, OrderBookUpdate, LiquidationEvent, AggTradeEvent
from src.data.db import db_manager
from src.features.orderbook import LocalOrderBook
from src.features.cvd import CVDTracker
from src.features.liquidations import LiquidationTracker

# Binance USD-M Futures multiplexed stream URL
BINANCE_WS_URL = "wss://fstream.binance.com/stream?streams=btcusdt@forceOrder/btcusdt@depth@100ms/btcusdt@aggTrade"

# Initialize Feature Trackers
lob = LocalOrderBook()
cvd_tracker = CVDTracker()
liquidation_tracker = LiquidationTracker(window_minutes=5)

async def on_message(raw_message: str):
    """Callback to handle incoming WebSocket messages."""
    parsed_event = BinanceParser.parse_message(raw_message)
    
    if parsed_event is None:
        return

    if isinstance(parsed_event, LiquidationEvent):
        # Update Liquidation Tracker
        liquidation_tracker.apply_liquidation(parsed_event)
        long_liq, short_liq = liquidation_tracker.get_intensity()

        # Route to TimescaleDB
        await db_manager.insert_liquidation(parsed_event)
        
        # Emphasize liquidations in the console
        logger.warning(
            f"LIQUIDATION 🔥 DB INSERT | {parsed_event.symbol} | "
            f"Side: {parsed_event.side} | "
            f"Qty: {parsed_event.original_quantity} | "
            f"Price: {parsed_event.price} | "
            f"Intensity (5m) -> Longs Wiped: {long_liq:.4f}, Shorts Wiped: {short_liq:.4f}"
        )
    elif isinstance(parsed_event, OrderBookUpdate):
        # Apply delta to local order book
        lob.apply_update(parsed_event)
        
        # Route to Redis
        await db_manager.update_order_book(parsed_event)
        
        # Output LOB summary
        best_bid = lob.get_best_bid() or "None"
        best_ask = lob.get_best_ask() or "None"
        logger.debug(
            f"L2 BOOK Redis UPDATE | {parsed_event.symbol} | "
            f"Best Bid: {best_bid} | Best Ask: {best_ask}"
        )
    elif isinstance(parsed_event, AggTradeEvent):
        # Update CVD
        cvd_tracker.apply_trade(parsed_event)
        
        logger.debug(
            f"AGG TRADE | {parsed_event.symbol} | "
            f"Price: {parsed_event.price} | Qty: {parsed_event.quantity} | "
            f"Buyer Maker: {parsed_event.is_buyer_maker} | "
            f"Current CVD: {cvd_tracker.get_cvd()}"
        )

async def main():
    logger.info("Initializing Apex Market Intelligence Engine - Quant Agent")
    
    # Initialize database connections
    await db_manager.connect()
    
    # Initialize and start the WebSocket manager
    ws_manager = WebSocketManager(url=BINANCE_WS_URL, on_message=on_message)
    
    try:
        # Start the ingestion loop
        await ws_manager.start()
    except KeyboardInterrupt:
        logger.info("Received exit signal, shutting down...")
    finally:
        await ws_manager.stop()
        await db_manager.disconnect()

if __name__ == "__main__":
    # Ensure asyncio uses the correct loop policy on Windows (if run locally)
    import sys
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
