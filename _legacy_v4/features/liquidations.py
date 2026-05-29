from collections import deque
from src.data.parsers import LiquidationEvent

class LiquidationTracker:
    def __init__(self, window_minutes: int = 5):
        self.window_ms = window_minutes * 60 * 1000
        # Store tuples of (timestamp_ms, is_long_liq, volume)
        self.events = deque()
        self.long_liq_vol: float = 0.0
        self.short_liq_vol: float = 0.0

    def apply_liquidation(self, event: LiquidationEvent):
        """
        Updates the rolling window of liquidations.
        A forceOrder with side="SELL" means a long position was liquidated.
        A forceOrder with side="BUY" means a short position was liquidated.
        """
        is_long_liq = (event.side.upper() == "SELL")
        qty = float(event.original_quantity)
        
        # Add to rolling sums
        if is_long_liq:
            self.long_liq_vol += qty
        else:
            self.short_liq_vol += qty
            
        self.events.append((event.event_time, is_long_liq, qty))
        
        # Cleanup old events outside the rolling window
        self._cleanup(event.event_time)

    def _cleanup(self, current_time: int):
        cutoff_time = current_time - self.window_ms
        while self.events and self.events[0][0] < cutoff_time:
            _, is_long, qty = self.events.popleft()
            if is_long:
                self.long_liq_vol -= qty
            else:
                self.short_liq_vol -= qty
                
    def get_intensity(self) -> tuple[float, float]:
        """
        Returns the total volume wiped out in the rolling window.
        Format: (long_liquidation_volume, short_liquidation_volume)
        """
        # Ensure floating point inaccuracies don't result in tiny negative numbers
        return max(0.0, self.long_liq_vol), max(0.0, self.short_liq_vol)
