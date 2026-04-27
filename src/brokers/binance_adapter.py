import os
from typing import Optional
from binance import AsyncClient
from binance.enums import ORDER_TYPE_MARKET
from loguru import logger
from .base_adapter import BaseBrokerAdapter


class BinanceAdapter(BaseBrokerAdapter):
    """
    Binance USD-M Futures adapter.
    """

    def __init__(self, mode: str = "DRY_RUN", paper_balance: float = 10000.0):
        super().__init__(mode=mode, paper_balance=paper_balance)
        self.client: Optional[AsyncClient] = None
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")
        testnet_flag = os.getenv("BINANCE_USE_TESTNET", "True").lower()
        self.use_testnet = testnet_flag in ["true", "1", "yes"]

    async def connect(self):
        if self.mode != "LIVE":
            logger.info("BinanceAdapter: DRY_RUN mode. Skipping real API connection unless needed for data.")
            # Even in DRY_RUN, we might need the client for fetching data if we wanted, 
            # but usually for purely mock execution we don't strictly need it to execute.
            # However, for consistency we'll connect it if we want to fetch real positions or balance.
            # For safety, let's still connect to testnet if possible, or just connect.
            # But wait, original code connected if live_trading_enabled.
            pass # Original code skipped if not live.
            # Let's connect anyway if keys are present so we can fetch positions if needed,
            # but we won't execute orders. Wait, original said: `if not self.live_trading_enabled: return`
            return

        if not self.api_key or not self.api_secret:
            logger.error("Binance API keys not found in .env. Execution will fail.")
            return

        logger.info(f"Connecting to Binance API... (Testnet: {self.use_testnet})")
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.use_testnet,
        )
        logger.info("Binance AsyncClient connected successfully.")

    async def disconnect(self):
        if self.client:
            await self.client.close_connection()
            logger.info("Binance AsyncClient disconnected.")

    async def get_account_balance(self) -> float:
        """
        Fetches the current account balance.
        If mode is DRY_RUN, this returns self.paper_balance.
        If LIVE, fetches USDT available margin.
        """
        if self.mode == "DRY_RUN":
            return float(self.paper_balance)
            
        if not self.client:
            logger.error("Binance client not connected. Cannot fetch live balance.")
            return 0.0

        try:
            balances = await self.client.futures_account_balance()
            for asset in balances:
                if asset.get("asset") == "USDT":
                    return float(asset.get("balance", 0.0))
            return 0.0
        except Exception as exc:
            logger.error(f"Failed to fetch Binance account balance: {exc}")
            return 0.0

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        **kwargs
    ) -> Optional[dict]:
        """
        Executes an order on Binance USD-M Futures.
        """
        if self.mode != "LIVE":
            # Just return a mock success response so TradeExecutor knows it "succeeded"
            return {
                "status": "DRY_RUN",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "orderId": f"mock_{symbol}_{side}",
                "avgPrice": kwargs.get("entry_price", 0.0) # mock
            }

        if not self.client:
            logger.error("Binance client not connected. Call connect() first.")
            return None

        # Map generic order_type to Binance specific
        b_type = ORDER_TYPE_MARKET if order_type.upper() == "MARKET" else order_type.upper()
        
        try:
            logger.info(f"Executing live {side} {b_type} order for {quantity} {symbol}...")
            response = await self.client.futures_create_order(
                symbol=symbol,
                side=side.upper(),
                type=b_type,
                quantity=quantity,
                **kwargs
            )
            return response
        except Exception as exc:
            logger.error(f"Failed to execute Binance order: {exc}")
            return None

    async def get_open_positions(self, symbol: str) -> list[dict]:
        """Fetches current active positions from Binance USD-M Futures."""
        if not self.client:
            return []

        try:
            positions = await self.client.futures_position_information(symbol=symbol)
            return [position for position in positions if float(position.get("positionAmt", 0)) != 0]
        except Exception as exc:
            logger.error(f"Failed to fetch open positions from Binance: {exc}")
            return []

    async def set_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        **kwargs
    ) -> Optional[dict]:
        """Sets a trailing stop market order on Binance."""
        if self.mode != "LIVE":
            return {"status": "DRY_RUN", "type": "TRAILING_STOP"}

        if not self.client:
            logger.error("Binance client not connected. Call connect() first.")
            return None

        try:
            logger.info(f"Setting {callback_rate}% TRAILING STOP for {side} {quantity} {symbol}...")
            response = await self.client.futures_create_order(
                symbol=symbol,
                side=side.upper(),
                type="TRAILING_STOP_MARKET",
                quantity=quantity,
                callbackRate=callback_rate,
                **kwargs
            )
            return response
        except Exception as exc:
            logger.error(f"Failed to set trailing stop on Binance: {exc}")
            return None
