import asyncio
import time
from typing import Any, Dict, List
import clickhouse_connect
from src.core.logging import get_logger

logger = get_logger("apex.data.storage.clickhouse")

class ClickHouseClient:
    def __init__(self, host: str = "localhost", port: int = 8123, username: str = "default", password: str = "clickhouse", database: str = "apex", flush_interval_s: float = 1.0):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.flush_interval_s = flush_interval_s
        
        self.client = None
        self._trade_buffer: List[List[Any]] = []
        self._quote_buffer: List[List[Any]] = []
        
        self._running = False
        self._flush_task = None
        
    def connect(self):
        """Connect to ClickHouse and initialize schema."""
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password
            )
            logger.info("clickhouse_connected", host=self.host)
            
            # Create Database
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            
            # Create Trade Table
            self.client.command(f"""
                CREATE TABLE IF NOT EXISTS {self.database}.ticks_trade (
                    symbol String,
                    timestamp UInt64,
                    price Float64,
                    qty Float64,
                    is_buyer_maker UInt8,
                    received_at UInt64
                ) ENGINE = MergeTree()
                ORDER BY (symbol, timestamp)
            """)
            
            # Create Quote Table
            self.client.command(f"""
                CREATE TABLE IF NOT EXISTS {self.database}.ticks_quote (
                    symbol String,
                    timestamp UInt64,
                    best_bid Float64,
                    best_bid_qty Float64,
                    best_ask Float64,
                    best_ask_qty Float64,
                    received_at UInt64
                ) ENGINE = MergeTree()
                ORDER BY (symbol, timestamp)
            """)
            
            logger.info("clickhouse_schema_initialized")
            
        except Exception as e:
            logger.error("clickhouse_connection_failed", error=str(e))
            raise

    async def start_flusher(self):
        """Start the background async flush loop."""
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("clickhouse_flusher_started", interval=self.flush_interval_s)

    async def stop_flusher(self):
        """Stop the background loop and flush remaining ticks."""
        self._running = False
        if self._flush_task:
            await self._flush_task
        await self._flush_buffers()
        logger.info("clickhouse_flusher_stopped")

    def queue_trade(self, tick: Dict[str, Any]):
        """Queue an @aggTrade tick for batch insertion."""
        # Expected tick format from binance_ws: 
        # {'type': 'trade', 'symbol': 'btcusdt', 'price': 65000.0, 'qty': 0.1, 'is_buyer_maker': False, 'timestamp': 1718000000000}
        self._trade_buffer.append([
            tick["symbol"],
            tick["timestamp"],
            tick["price"],
            tick["qty"],
            1 if tick["is_buyer_maker"] else 0,
            int(time.time() * 1000)
        ])

    def queue_quote(self, tick: Dict[str, Any]):
        """Queue a @bookTicker quote for batch insertion."""
        # Format: {'type': 'quote', 'symbol': 'btcusdt', 'bid': 65000.0, 'ask': 65000.1, 'timestamp': 1718000000000}
        self._quote_buffer.append([
            tick["symbol"],
            tick["timestamp"],
            tick.get("bid", 0.0),
            tick.get("bid_qty", 0.0), # Assuming we add qty to binance_ws later
            tick.get("ask", 0.0),
            tick.get("ask_qty", 0.0),
            int(time.time() * 1000)
        ])

    async def _flush_loop(self):
        """Background loop that executes the bulk inserts."""
        while self._running:
            await asyncio.sleep(self.flush_interval_s)
            await self._flush_buffers()

    async def _flush_buffers(self):
        """Perform the actual DB insertion and clear the buffers."""
        if not self.client:
            return
            
        trades = []
        quotes = []
        
        # Atomically drain buffers
        if self._trade_buffer:
            trades = self._trade_buffer[:]
            self._trade_buffer.clear()
            
        if self._quote_buffer:
            quotes = self._quote_buffer[:]
            self._quote_buffer.clear()
            
        if trades:
            # Run blocking DB insert in a separate thread so we don't freeze the async event loop
            try:
                await asyncio.to_thread(
                    self.client.insert,
                    f"{self.database}.ticks_trade",
                    trades,
                    column_names=["symbol", "timestamp", "price", "qty", "is_buyer_maker", "received_at"]
                )
                logger.info("clickhouse_flushed_trades", count=len(trades))
            except Exception as e:
                logger.error("clickhouse_trade_insert_error", error=str(e))
                # Put them back to not lose data
                self._trade_buffer.extend(trades)

        if quotes:
            try:
                await asyncio.to_thread(
                    self.client.insert,
                    f"{self.database}.ticks_quote",
                    quotes,
                    column_names=["symbol", "timestamp", "best_bid", "best_bid_qty", "best_ask", "best_ask_qty", "received_at"]
                )
                logger.info("clickhouse_flushed_quotes", count=len(quotes))
            except Exception as e:
                logger.error("clickhouse_quote_insert_error", error=str(e))
                self._quote_buffer.extend(quotes)
