from abc import ABC, abstractmethod
from typing import Optional


class BaseBrokerAdapter(ABC):
    """
    Abstract Base Class for all broker integrations (Binance, Angel One, OANDA, etc.).
    Decouples the core trading logic from broker-specific SDKs.
    """

    def __init__(self, mode: str = "DRY_RUN", paper_balance: float = 10000.0):
        self.mode = mode.upper()
        self.paper_balance = paper_balance

    @abstractmethod
    async def connect(self):
        """Initializes the connection to the broker API."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Gracefully closes the connection to the broker API."""
        pass

    @abstractmethod
    async def get_account_balance(self) -> float:
        """
        Fetches the current account balance.
        If mode is DRY_RUN, this should return self.paper_balance.
        """
        pass

    @abstractmethod
    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        **kwargs
    ) -> Optional[dict]:
        """
        Executes an order on the exchange.
        Returns the raw broker response dictionary, or None if failed.
        """
        pass

    @abstractmethod
    async def get_open_positions(self, symbol: str) -> list[dict]:
        """
        Fetches active positions from the exchange for a given symbol.
        """
        pass

    @abstractmethod
    async def set_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: float,
        callback_rate: float,
        **kwargs
    ) -> Optional[dict]:
        """
        Sets a trailing stop market order.
        """
        pass
