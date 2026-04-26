import os
import asyncpg
import redis.asyncio as redis
from loguru import logger
import json
from typing import Optional
from src.data.parsers import AggTradeEvent, OrderBookUpdate, LiquidationEvent
from src.dashboard.config import DB_SCHEMA_BROKER_CREDENTIALS, DB_SCHEMA_USERS

# Environment variables (with defaults for local fallback)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TIMESCALE_URL = os.getenv("TIMESCALE_URL", "postgres://user:password@localhost:5432/apex")
CANDLE_INTERVAL = "1m"
WARMUP_CANDLES_REQUIRED = 14
REDIS_CANDLE_KEY = "market:candles:btcusdt"


def build_book_key(symbol: str) -> str:
    return f"book:{symbol.upper()}"


def build_candle_cache_key(symbol: str, interval: str = CANDLE_INTERVAL) -> str:
    if symbol.lower() == "btcusdt" and interval == CANDLE_INTERVAL:
        return REDIS_CANDLE_KEY
    return f"market:candles:{symbol.lower()}"


def build_warmup_key(symbol: str, interval: str = CANDLE_INTERVAL) -> str:
    return f"market:warmup:{symbol.lower()}"

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
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()
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
            await conn.execute(DB_SCHEMA_USERS)
            await conn.execute(DB_SCHEMA_BROKER_CREDENTIALS)

            # Example: Create a base table for raw ticks
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_trades (
                    time        TIMESTAMPTZ       NOT NULL,
                    symbol      VARCHAR(20)       NOT NULL,
                    price       DOUBLE PRECISION  NOT NULL,
                    quantity    DOUBLE PRECISION  NOT NULL,
                    is_buyer_maker BOOLEAN        NOT NULL
                );

                CREATE TABLE IF NOT EXISTS market_data (
                    time        TIMESTAMPTZ       NOT NULL,
                    symbol      VARCHAR(20)       NOT NULL,
                    open        DOUBLE PRECISION  NOT NULL,
                    high        DOUBLE PRECISION  NOT NULL,
                    low         DOUBLE PRECISION  NOT NULL,
                    close       DOUBLE PRECISION  NOT NULL,
                    volume      DOUBLE PRECISION  NOT NULL
                );
                
                CREATE TABLE IF NOT EXISTS liquidations (
                    time        TIMESTAMPTZ       NOT NULL,
                    symbol      VARCHAR(20)       NOT NULL,
                    side        VARCHAR(10)       NOT NULL,
                    price       DOUBLE PRECISION  NOT NULL,
                    quantity    DOUBLE PRECISION  NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_broker_credentials_user_id ON broker_credentials(user_id);
                CREATE INDEX IF NOT EXISTS idx_broker_credentials_provider_name ON broker_credentials(provider_name);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, time);
            """)
            
            # Try to convert to hypertable (fails silently if already a hypertable in typical setups, 
            # but we use 'IF NOT EXISTS' equivalent logic for Timescale if needed. 
            # For simplicity, we just execute the standard hypertable creation. 
            # In production, you'd check if it's already a hypertable).
            try:
                await conn.execute("SELECT create_hypertable('raw_trades', 'time', if_not_exists => TRUE);")
                await conn.execute("SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);")
                await conn.execute("SELECT create_hypertable('liquidations', 'time', if_not_exists => TRUE);")
            except asyncpg.exceptions.UndefinedFunctionError:
                logger.warning("create_hypertable function not found. Ensure TimescaleDB extension is enabled.")
            except Exception as e:
                # Might already be a hypertable
                logger.debug(f"Hypertable initialization note: {e}")

    async def ping(self) -> bool:
        """Tests both Postgres and Redis connections."""
        try:
            if self.redis_pool:
                await self.redis_pool.ping()
            if self.pg_pool:
                # pg_pool doesn't have a direct ping, execute simple query
                async with self.pg_pool.acquire() as conn:
                    await conn.execute("SELECT 1")
            return True
        except Exception as e:
            return False

    async def _ensure_connection(self):
        """Checks connections and attempts to reconnect gracefully if dropped."""
        if getattr(self, '_reconnecting', False):
            return False
            
        self._reconnecting = True
        logger.warning("Database drop detected. Attempting to reconnect...")
        try:
            await self.disconnect()
            await self.connect()
            logger.info("Database reconnection successful.")
            return True
        except Exception as reconnect_error:
            logger.error(f"Database reconnection failed: {reconnect_error}")
            return False
        finally:
            self._reconnecting = False
    async def insert_raw_trade(self, event: AggTradeEvent):
        """Inserts a raw aggTrade into TimescaleDB."""
        if not self.pg_pool:
            return

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO raw_trades (time, symbol, price, quantity, is_buyer_maker)
                    VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5)
                    """,
                    event.trade_time,
                    event.symbol,
                    float(event.price),
                    float(event.quantity),
                    bool(event.is_buyer_maker),
                )
        except Exception as e:
            logger.error(f"Failed to insert raw trade: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

    async def upsert_market_candle(self, candle: dict):
        """Persists the latest finalized 1m market candle into TimescaleDB."""
        if not self.pg_pool:
            return

        try:
            async with self.pg_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO market_data (time, symbol, open, high, low, close, volume)
                    VALUES (to_timestamp($1 / 1000.0), $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (symbol, time) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                    """,
                    int(candle["open_time"]),
                    candle["symbol"],
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["volume"]),
                )
        except Exception as e:
            logger.error(f"Failed to upsert market candle: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

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
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

    async def update_order_book(self, event: OrderBookUpdate):
        """Updates the latest order book state in Redis."""
        if not self.redis_pool:
            return
            
        try:
            key = build_book_key(event.symbol)
            # Serialize the Pydantic model to JSON string
            payload = event.model_dump_json()
            # Store in Redis. Using SET for simple latest state representation.
            await self.redis_pool.set(key, payload)
        except Exception as e:
            logger.error(f"Failed to update order book in Redis: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

    async def cache_market_candles(self, symbol: str, candles: list[dict]):
        """Caches the latest 1m candles plus warm-up state in Redis."""
        if not self.redis_pool:
            return {"symbol": symbol.upper(), "ready": False, "candle_count": 0, "required_candles": WARMUP_CANDLES_REQUIRED}

        normalized_symbol = symbol.upper()
        completed_candle_count = max(len(candles) - 1, 0) if candles else 0
        warmup_payload = {
            "symbol": normalized_symbol,
            "ready": completed_candle_count >= WARMUP_CANDLES_REQUIRED,
            "candle_count": completed_candle_count,
            "required_candles": WARMUP_CANDLES_REQUIRED,
        }

        try:
            await self.redis_pool.set(
                build_candle_cache_key(normalized_symbol),
                json.dumps(candles),
            )
            await self.redis_pool.set(
                build_warmup_key(normalized_symbol),
                json.dumps(warmup_payload),
            )
            await self.redis_pool.publish("market:warmup", str(completed_candle_count))
        except Exception as e:
            logger.error(f"Failed to cache market candles in Redis: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

        return warmup_payload

    async def get_warmup_count(self, symbol: str, interval: str = CANDLE_INTERVAL) -> int:
        """Reads the live warm-up candle count directly from Redis for the fastest HUD updates."""
        if not self.redis_pool:
            return 0

        try:
            payload = await self.redis_pool.get(build_candle_cache_key(symbol, interval))
            if not payload:
                return 0

            candles = json.loads(payload)
            if not isinstance(candles, list):
                return 0

            return max(len(candles) - 1, 0)
        except Exception as e:
            logger.error(f"Failed to fetch warm-up count from Redis: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()
            return 0

    async def get_warmup_status(self, symbol: str, interval: str = CANDLE_INTERVAL) -> dict:
        """Builds the warm-up readiness payload from the live Redis candle cache."""
        normalized_symbol = symbol.upper()
        candle_count = await self.get_warmup_count(symbol, interval)
        return {
            "symbol": normalized_symbol,
            "ready": candle_count >= WARMUP_CANDLES_REQUIRED,
            "candle_count": candle_count,
            "required_candles": WARMUP_CANDLES_REQUIRED,
        }

    async def reset_realtime_market_state(self, symbol: str):
        """Clears live Redis state so warm-up always starts from fresh candles."""
        if not self.redis_pool:
            return

        normalized_symbol = symbol.upper()
        try:
            await self.redis_pool.delete(
                build_book_key(normalized_symbol),
                build_candle_cache_key(normalized_symbol),
                build_warmup_key(normalized_symbol),
            )
        except Exception as e:
            logger.error(f"Failed to reset realtime market state: {e}")
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

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
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()

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
            if getattr(self, "_reconnecting", False) is False:
                await self._ensure_connection()
            return 0.5

# Global instance
db_manager = DatabaseManager()
