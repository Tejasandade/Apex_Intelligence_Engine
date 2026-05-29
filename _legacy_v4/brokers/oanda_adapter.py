"""
OANDA Forex Placeholder Adapter
================================
Stubbed implementation of BaseBrokerAdapter for future OANDA integration.

This adapter satisfies the broker layer interface for the FOREX pod
but does NOT connect to any live API. All methods return safe defaults.
Replace the stub bodies with real OANDA v20 REST calls when the Forex
pod is activated in a future phase.
"""

from typing import Optional
from loguru import logger
from src.brokers.base_adapter import BaseBrokerAdapter


class OandaAdapter(BaseBrokerAdapter):
    """
    Placeholder Forex adapter (OANDA v20).
    Currently a no-op stub — safe to instantiate at startup.
    """

    live_trading_enabled: bool = False   # must remain False until fully implemented

    def __init__(self, paper_balance: float = 10_000.0):
        super().__init__(mode="DRY_RUN", paper_balance=paper_balance)
        logger.info(
            "OandaAdapter created | mode=STUB (not yet implemented) | "
            "paper_balance={:,.2f} USD",
            paper_balance,
        )

    async def connect(self):
        logger.info("OandaAdapter.connect() — stub, no real connection made.")

    async def disconnect(self):
        logger.info("OandaAdapter.disconnect() — stub.")

    async def get_account_balance(self) -> float:
        return self.paper_balance

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        **kwargs,
    ) -> Optional[dict]:
        logger.info(
            "[STUB] OandaAdapter.execute_order | {} {} {} — not implemented",
            side, quantity, symbol,
        )
        return {
            "broker_status": "STUB_NOT_IMPLEMENTED",
            "adapter":       "OandaAdapter",
            "symbol":        symbol,
            "side":          side,
            "quantity":      quantity,
            "live_executed": False,
        }

    async def get_open_positions(self, symbol: str) -> list[dict]:
        return []

    async def set_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        **kwargs,
    ) -> Optional[dict]:
        logger.info(
            "[STUB] OandaAdapter.set_trailing_stop | {} {} — not implemented", side, symbol
        )
        return {"broker_status": "STUB_NOT_IMPLEMENTED", "live_executed": False}
