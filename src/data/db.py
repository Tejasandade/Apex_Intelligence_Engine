import os
import asyncpg
import redis.asyncio as redis
from loguru import logger
import json
from typing import Optional
from src.data.parsers import OrderBookUpdate, LiquidationEvent

# Environment variables (with defaults for local fallback)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TIMESCALE_URL = os.getenv("TIMESCALE_URL", "postgres://user:password@localhost:5432/apex")

class DatabaseManager:
    """Manages asynchronous connections to Redis and TimescaleDB."""
    
    def __init__(self):
        self.redis_pool: Optional[redis.Redis] = None
        self.pg_pool: Optional[asyncpg.Pool] = None

    async def connect(self):
        """Initializes connection pools for both databases."""
        try:
            # Initialize Redis
            self.redis_pool = redis.from_url(REDIS_URL, decode_responses=True)
            await self.redis_pool.ping()
            logger.info("Successfully connected to Redis.")

            # Initialize TimescaleDB (PostgreSQL)
            self.pg_pool = await asyncpg.create_pool(dsn=TIMESCALE_URL)
            logger.info("Successfully connected to TimescaleDB.")
            
            # Setup TimescaleDB tables if they don't exist
            await self._init_timescale_schema()
            
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            raise

    async def disconnect(self):
        """Gracefully closes database connections."""
        if self.redis_pool:
            await self.redis_pool.close()
            logger.info("Disconnected from Redis.")
            
        if self.pg_pool:
            await self.pg_pool.close()
            logger.info("Disconnected from TimescaleDB.")

    async def _init_timescale_schema(self):
        """Ensures the base tables and hypertables exist."""
        if not self.pg_pool:
            return
            
        async with self.pg_pool.acquire() as conn:
            # Example: Create a base table for raw ticks
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_trades (
                    time        TIMESTAMPTZ       NOT NULL,
                    symbol      VARCHAR(20)       NOT NULL,
                    price       DOUBLE PRECISION  NOT NULL,
                    quantity    DOUBLE PRECISION  NOT NULL,
                    is_buyer_maker BOOLEAN        NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS liquidations (
                    time        TIMESTAMPTZ       NOT NULL,
                    symbol      VARCHAR(20)       NOT NULL,
                    side        VARCHAR(10)       NOT NULL,
                    price       DOUBLE PRECISION  NOT NULL,
                    quantity    DOUBLE PRECISION  NOT NULL
                );
            """)
            
            # Try to convert to hypertable (fails silently if already a hypertable in typical setups, 
            # but we use 'IF NOT EXISTS' equivalent logic for Timescale if needed. 
            # For simplicity, we just execute the standard hypertable creation. 
            # In production, you'd check if it's already a hypertable).
            try:
                await conn.execute("SELECT create_hypertable('raw_trades', 'time', if_not_exists => TRUE);")
                await conn.execute("SELECT create_hypertable('liquidations', 'time', if_not_exists => TRUE);")
            except asyncpg.exceptions.UndefinedFunctionError:
                logger.warning("create_hypertable function not found. Ensure TimescaleDB extension is enabled.")
            except Exception as e:
                # Might already be a hypertable
                logger.debug(f"Hypertable initialization note: {e}")

    async def insert_liquidation(self, event: LiquidationEvent):
        """Inserts a LiquidationEvent into TimescaleDB."""
        if not self.pg_pool:
            return
            
        try:
            # event.event_time is in milliseconds
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO liquidations (time, symbol, side, price, quantity)
                    VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5)
                    """,
                    event.event_time,
                    event.symbol,
                    event.side,
                    float(event.price),
                    float(event.original_quantity)
                )
        except Exception as e:
            logger.error(f"Failed to insert liquidation: {e}")

    async def update_order_book(self, event: OrderBookUpdate):
        """Updates the latest order book state in Redis."""
        if not self.redis_pool:
            return
            
        try:
            key = f"book:{event.symbol}"
            # Serialize the Pydantic model to JSON string
            payload = event.model_dump_json()
            # Store in Redis. Using SET for simple latest state representation.
            await self.redis_pool.set(key, payload)
        except Exception as e:
            logger.error(f"Failed to update order book in Redis: {e}")

    async def insert_sentiment(self, score: float):
        """Inserts the aggregated Macro Sentiment Score into TimescaleDB."""
        if not self.pg_pool:
            logger.warning("Postgres pool not initialized. Mocking sentiment insertion.")
            logger.info(f"MOCK DB INSERT | Sentiment Score: {score:.4f}")
            return
            
        try:
            async with self.pg_pool.acquire() as conn:
                # Ensure the macro_sentiment table exists (usually in init_schema, placed here for Phase 2 completion)
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS macro_sentiment (
                        time        TIMESTAMPTZ       NOT NULL,
                        score       DOUBLE PRECISION  NOT NULL
                    );
                    """
                )
                
                await conn.execute(
                    """
                    INSERT INTO macro_sentiment (time, score)
                    VALUES (NOW(), $1)
                    """,
                    float(score)
                )
                logger.info(f"DB INSERT | Macro Sentiment Score: {score:.4f}")
        except Exception as e:
            logger.error(f"Failed to insert sentiment score: {e}")

    async def fetch_latest_sentiment(self) -> float:
        """Fetches the latest Macro Sentiment Score from TimescaleDB."""
        if not self.pg_pool:
            logger.warning("Postgres pool not initialized. Returning default neutral sentiment.")
            return 0.5
            
        try:
            query = """
                SELECT score
                FROM macro_sentiment
                ORDER BY time DESC
                LIMIT 1
            """
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(query)
                if row:
                    return float(row['score'])
                else:
                    return 0.5
        except Exception as e:
            logger.error(f"Failed to fetch latest sentiment: {e}")
            return 0.5

# Global instance
db_manager = DatabaseManager()
