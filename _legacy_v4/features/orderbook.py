from typing import Dict
from src.data.parsers import OrderBookUpdate

class LocalOrderBook:
    def __init__(self):
        # Maps price level (float) to quantity (float)
        self.bids: Dict[float, float] = {}
        self.asks: Dict[float, float] = {}
        
    def apply_update(self, update: OrderBookUpdate):
        """
        Updates the order book state.
        Removes price levels if the quantity drops to 0.
        """
        if update.is_top_of_book:
            self.bids = {}
            self.asks = {}

        # Apply bid updates
        for price_str, qty_str in update.bids:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                self.bids.pop(price, None)
            else:
                self.bids[price] = qty
                
        # Apply ask updates
        for price_str, qty_str in update.asks:
            price, qty = float(price_str), float(qty_str)
            if qty == 0.0:
                self.asks.pop(price, None)
            else:
                self.asks[price] = qty

    def get_best_bid(self) -> float | None:
        if not self.bids: return None
        return max(self.bids.keys())

    def get_best_ask(self) -> float | None:
        if not self.asks: return None
        return min(self.asks.keys())
