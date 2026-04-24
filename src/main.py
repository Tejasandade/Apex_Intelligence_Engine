import asyncio
import sys
import os
import aiohttp
from loguru import logger

# Ensure the project root is in the Python path to resolve 'src' imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.db import db_manager
from src.agents.executor import TradeExecutor
from src.agents.risk_manager import RiskManager

async def fetch_inference(symbol: str) -> float:
    """Fetches the latest bullish probability from the local Inference API."""
    url = f"http://127.0.0.1:8000/inference/{symbol.lower()}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get("bullish_probability", 0.5))
                else:
                    logger.warning(f"Inference API returned status {response.status}")
                    return 0.5
    except Exception as e:
        logger.error(f"Failed to fetch inference from API: {e}")
        return 0.5

async def main_loop():
    logger.info("Initializing Apex Intelligence Engine Main Loop...")
    
    # Initialize Agents
    executor = TradeExecutor(live_trading_enabled=False)  # DRY RUN mode by default
    risk_manager = RiskManager(total_capital=10000.0, max_risk_per_trade=0.05, daily_drawdown_limit=0.05)
    
    symbol = "BTCUSDT"
    
    try:
        # Connect services
        await db_manager.connect()
        await executor.connect()
        
        logger.success("All services connected. Starting operational loop.")
        
        while True:
            logger.info("--- New Evaluation Cycle ---")
            
            # 0. Check Open Positions
            open_positions = await executor.get_open_positions(symbol)
            if open_positions:
                pos = open_positions[0]
                unrealized_pnl = float(pos.get("unRealizedProfit", 0.0))
                pos_amt = float(pos.get("positionAmt", 0.0))
                logger.info(f"Monitoring Active Position | Symbol: {symbol} | Amount: {pos_amt} | Unrealized PnL: ${unrealized_pnl:.2f}")
                logger.info("Skipping new trade evaluations to focus on active position management.")
                
                logger.info("Cycle complete. Sleeping for 60 seconds...")
                await asyncio.sleep(60)
                continue
            
            # 1. Fetch Signal from Quant Agent API
            bullish_probability = await fetch_inference(symbol)
            logger.info(f"Received Signal | Symbol: {symbol} | Bullish Probability: {bullish_probability:.4f}")
            
            # 2. Risk Management: Evaluate Signal
            signal = risk_manager.evaluate_trade_signal(symbol, bullish_probability)
            
            # 3. Execution Routing
            if signal["action"] in ["BUY", "SELL"]:
                size_usd = risk_manager.calculate_position_size(
                    symbol=signal["symbol"],
                    action=signal["action"],
                    confidence=signal["probability"],
                    reward_risk_ratio=1.5,
                    kelly_fraction=0.5
                )
                
                if size_usd > 0:
                    logger.info(f"Risk Approved | Action: {signal['action']} | Size: ${size_usd:.2f} | Executing...")
                    
                    # Convert USD size to asset quantity (mocked conversion for BTC at 70k for now)
                    current_price = 70000.0 
                    quantity = round(size_usd / current_price, 3)
                    
                    if quantity > 0:
                        order = await executor.execute_market_order(symbol, signal["action"], quantity)
                        if order:
                            # Protect position with a trailing stop
                            close_side = "SELL" if signal["action"] == "BUY" else "BUY"
                            await executor.set_trailing_stop(symbol, close_side, quantity, callback_rate=1.0)
                    else:
                        logger.warning("Calculated quantity too small for execution.")
                else:
                    logger.info("Risk Manager rejected the trade size (Size = $0.00).")
            else:
                logger.info("Signal action is HOLD. No execution required.")
                
            # Wait for next cycle
            logger.info("Cycle complete. Sleeping for 60 seconds...")
            await asyncio.sleep(60)
            
    except asyncio.CancelledError:
        logger.info("Main loop task cancelled.")
    except Exception as e:
        logger.exception(f"Fatal error in main loop: {e}")
    finally:
        logger.info("Shutting down Apex Intelligence Engine...")
        await db_manager.disconnect()
        await executor.disconnect()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
