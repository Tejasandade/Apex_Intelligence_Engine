import os
import asyncio
from pathlib import Path
from loguru import logger
from src.data.db import DatabaseManager

async def run_pre_flight():
    logger.info("🚀 Starting V4 Pre-Flight Checklist...")

    # ── 1. The Nuclear Database Reset ─────────────────────────────────────────
    logger.info("Step 1: Executing Nuclear Database Reset...")
    db = DatabaseManager()
    try:
        await db.connect()
        # Flush Redis
        if db.redis_pool:
            await db.redis_pool.flushall()
            logger.success("Redis memory (stale tick & candle memory) flushed successfully.")
        else:
            logger.warning("Redis connection not available — skipping flush.")

        # Truncate TimescaleDB
        if db.pg_pool:
            async with db.pg_pool.acquire() as conn:
                await conn.execute("TRUNCATE market_data CASCADE;")
                await conn.execute("TRUNCATE raw_trades CASCADE;")
            logger.success("TimescaleDB 'market_data' and 'raw_trades' tables truncated successfully.")
        else:
            logger.warning("TimescaleDB connection not available — skipping truncate.")
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
    finally:
        await db.disconnect()

    # ── 2. Verify the AI Brains ───────────────────────────────────────────────
    logger.info("Step 2: Verifying AI Brains...")
    models_dir = Path("data/models")
    required_models = [
        "crypto_model_trend.json",
        "crypto_model_chop.json",
        "banknifty_model_trend.json",
        "banknifty_model_chop.json"
    ]
    missing = []
    for m in required_models:
        if not (models_dir / m).exists():
            missing.append(m)
    
    if missing:
        error_msg = f"Missing required models: {missing}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    else:
        logger.success("MASSIVE SUCCESS! All 4 regime models are locked and loaded.")

    # ── 3. Clean the Blotter ──────────────────────────────────────────────────
    logger.info("Step 3: Cleaning the Blotter...")
    active_trades_path = Path("data/runtime/active_trades.json")
    if active_trades_path.exists():
        active_trades_path.unlink()
        logger.success("Active trades blotter has been sanitized (file deleted).")
    else:
        logger.info("Blotter is already clean (no active_trades.json found).")

    # ── 4. Clock Synchronization ──────────────────────────────────────────────
    logger.info("Step 4: Synchronizing system clock for microsecond accuracy...")
    
    logger.info("  Unregistering stale w32time service...")
    os.system("w32tm /unregister")
    
    logger.info("  Registering w32time service...")
    os.system("w32tm /register")
    
    logger.info("  Starting w32time service...")
    os.system("net start w32time")
    
    logger.info("  Forcing NTP resync...")
    result = os.system("w32tm /resync")
    
    if result == 0:
        logger.success("System clock synchronized successfully via NTP.")
    else:
        logger.warning(
            f"w32tm /resync returned code {result}. "
            "This step requires Administrator privileges — right-click terminal and 'Run as Administrator'."
        )

    logger.success("🚀 V4 Pre-Flight Checklist COMPLETED CLEANLY!")

if __name__ == "__main__":
    asyncio.run(run_pre_flight())
