from src.data.parsers import AggTradeEvent

class CVDTracker:
    def __init__(self):
        self.cvd: float = 0.0
        self.buy_volume: float = 0.0
        self.sell_volume: float = 0.0

    def apply_trade(self, trade: AggTradeEvent):
        """
        Updates the Cumulative Volume Delta.
        is_buyer_maker=True means the maker was the buyer, therefore the aggressor was a seller.
        is_buyer_maker=False means the maker was the seller, therefore the aggressor was a buyer.
        """
        qty = float(trade.quantity)
        
        if trade.is_buyer_maker:
            # Aggressive Sell
            self.sell_volume += qty
            self.cvd -= qty
        else:
            # Aggressive Buy
            self.buy_volume += qty
            self.cvd += qty
            
    def get_cvd(self) -> float:
        return self.cvd
