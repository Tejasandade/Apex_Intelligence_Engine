import os
import asyncio
from loguru import logger
from dotenv import load_dotenv
from binance import AsyncClient
from binance import BinanceSocketManager
from binance.enums import ORDER_TYPE_MARKET

load_dotenv()

class TradeExecutor:
    """
    The Trade Execution Integration module.
    Responsible for securely connecting to the exchange API, monitoring websockets,
    and translating Quant Agent signals into live orders.
    """
    def __init__(self, live_trading_enabled: bool = False):
        self.live_trading_enabled = live_trading_enabled
        self.client = None
        self.bm = None
        
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        
        # Testnet configuration
        testnet_flag = os.getenv("BINANCE_USE_TESTNET", "True").lower()
        self.use_testnet = testnet_flag in ["true", "1", "yes"]
        
        if self.live_trading_enabled:
            logger.warning("LIVE TRADING IS ENABLED. Real orders will be executed on the exchange.")
        else:
            logger.info("TradeExecutor initialized in DRY RUN mode. No real orders will be placed.")

    async def connect(self):
        """Initializes the asynchronous Binance client and Socket Manager."""
        if not self.api_key or not self.api_secret:
            logger.error("Binance API keys not found in .env. Execution will fail.")
            return

        logger.info(f"Connecting to Binance API... (Testnet: {self.use_testnet})")
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.use_testnet
        )
        self.bm = BinanceSocketManager(self.client)
        logger.info("Binance AsyncClient and SocketManager connected successfully.")

    async def disconnect(self):
        """Gracefully closes the connection."""
        if self.client:
            await self.client.close_connection()
            logger.info("Binance AsyncClient disconnected.")

    async def execute_market_order(self, symbol: str, side: str, quantity: float):
        """
        Executes a market order on Binance USD-M Futures.
        If live_trading_enabled is False, this is safely mocked as a DRY RUN.
        """
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            logger.error(f"Invalid order side: {side}")
            return None
            
        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] Would execute {side} MARKET order for {quantity} {symbol}")
            return {"status": "DRY_RUN", "symbol": symbol, "side": side, "quantity": quantity}
            
        if not self.client:
            logger.error("Binance client not connected. Call connect() first.")
            return None
            
        try:
            logger.info(f"Executing live {side} MARKET order for {quantity} {symbol}...")
            # For USD-M futures market order
            response = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logger.success(f"Order executed successfully: {response.get('orderId')}")
            return response
        except Exception as e:
            logger.error(f"Failed to execute market order: {e}")
            return None

    async def get_open_positions(self, symbol: str) -> list:
        """Fetches current active positions from Binance USD-M Futures."""
        if not self.client:
            logger.error("Binance client not connected.")
            return []
            
        try:
            positions = await self.client.futures_position_information(symbol=symbol)
            # Filter for open positions (positionAmt != 0)
            open_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
            return open_positions
        except Exception as e:
            logger.error(f"Failed to fetch open positions: {e}")
            return []
            
    async def set_trailing_stop(self, symbol: str, side: str, quantity: float, callback_rate: float = 1.0):
        """
        Sets a trailing stop market order to protect profits.
        Side should be the OPPOSITE of the open position.
        """
        side = side.upper()
        if not self.live_trading_enabled:
            logger.info(f"[DRY RUN] Would set TRAILING STOP ({callback_rate}%) on {symbol} to {side} {quantity}")
            return {"status": "DRY_RUN", "type": "TRAILING_STOP"}
            
        if not self.client:
            logger.error("Binance client not connected.")
            return None
            
        try:
            logger.info(f"Setting {callback_rate}% TRAILING STOP for {side} {quantity} {symbol}...")
            response = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TRAILING_STOP_MARKET",
                quantity=quantity,
                callbackRate=callback_rate
            )
            logger.success(f"Trailing stop set successfully: {response.get('orderId')}")
            return response
        except Exception as e:
            logger.error(f"Failed to set trailing stop: {e}")
            return None

if __name__ == "__main__":
    async def main():
        # Quick test in DRY RUN mode
        executor = TradeExecutor(live_trading_enabled=False)
        await executor.connect()
        await executor.execute_market_order("BTCUSDT", "BUY", 0.01)
        await executor.disconnect()
        
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
