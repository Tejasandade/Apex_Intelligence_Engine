"""
Apex Intelligence Engine V5 — India Specific Advisors
=====================================================
Specialized advisors for the Indian Equity / Options market.
Provides session timing (IST) and VIX filtering rules.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
import pandas as pd

from src.council.base import BaseAdvisor, Vote, AdvisorVote
from src.core.logging import get_logger

logger = get_logger("apex.council.advisors_india")

IST = ZoneInfo("Asia/Kolkata")

class IndiaSessionAdvisor(BaseAdvisor):
    """
    Session timing rules for Indian Equities (NSE/BSE).
    - Rejects before 09:20 IST (opening volatility)
    - Approves 09:20 to 14:00 IST (prime time)
    - Abstains 14:00 to 14:30 IST (afternoon decay begins)
    - Rejects after 14:30 IST (theta crush, spreads widen)
    - Rejects after 13:00 IST on Wednesdays (BankNifty Expiry day theta death)
    """
    
    def __init__(self, weight: float = 1.0):
        super().__init__("IndiaSessionAdvisor", weight)

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> AdvisorVote:
        now = datetime.now(IST)
        time_str = now.strftime("%H:%M IST")
        
        # Check if weekend
        if now.weekday() >= 5:
            return self._make_vote(Vote.REJECT, 1.0, f"Weekend market closed ({time_str})")
            
        current_minutes = now.hour * 60 + now.minute
        
        # Before 09:20 IST - too much noise
        if current_minutes < (9 * 60 + 20):
            return self._make_vote(Vote.REJECT, 1.0, f"Too early, market opening noise ({time_str})")
            
        # After 14:30 IST - theta crush / auto square-off zone
        if current_minutes > (14 * 60 + 30):
            return self._make_vote(Vote.REJECT, 1.0, f"Late session theta decay / auto-square off zone ({time_str})")
            
        # Wednesday Expiry logic
        if now.weekday() == 2: # Wednesday
            if current_minutes > (13 * 60 + 0):
                # Expiry day after 1pm is pure gambling/theta decay
                return self._make_vote(Vote.REJECT, 1.0, f"BankNifty Wednesday expiry afternoon decay ({time_str})")
        
        # 14:00 to 14:30 IST - caution, lower conviction
        if (14 * 60 + 0) <= current_minutes <= (14 * 60 + 30):
            return self._make_vote(Vote.ABSTAIN, 0.5, f"Caution zone, late afternoon decay ({time_str})")
            
        # 09:20 to 14:00 - Prime time
        return self._make_vote(Vote.APPROVE, 1.0, f"Prime trading session ({time_str})")


class IndiaVIXAdvisor(BaseAdvisor):
    """
    Filters trades based on India VIX (Volatility Index).
    Options premiums are expensive when VIX is high.
    - Approves when VIX is 12-18 (sweet spot)
    - Abstains when VIX is 18-22 (elevated)
    - Rejects when VIX > 22 (too expensive)
    - Rejects when VIX < 10 (dead market)
    """
    
    def __init__(self, broker: Any = None, weight: float = 1.0):
        super().__init__("IndiaVIXAdvisor", weight)
        self.broker = broker
        # If no broker provided, we will just pass. VIX should ideally be fetched from the broker.
        self.last_vix = 14.0 # default safe value

    def evaluate(
        self,
        features: pd.DataFrame,
        direction: str,
        price: float,
        regime: str,
        atr: float,
        **kwargs: Any,
    ) -> AdvisorVote:
        # In a fully connected live system, we would fetch VIX from the broker here.
        # For now, we will assume self.last_vix is updated by a background process or passed in.
        
        if hasattr(self.broker, "get_india_vix"):
            self.last_vix = self.broker.get_india_vix()

        vix = self.last_vix
        
        if vix < 10.0:
            return self._make_vote(Vote.REJECT, 1.0, f"India VIX {vix:.1f} too low (dead market)")
            
        if vix > 22.0:
            return self._make_vote(Vote.REJECT, 1.0, f"India VIX {vix:.1f} too high (expensive premiums)")
            
        if 18.0 < vix <= 22.0:
            return self._make_vote(Vote.ABSTAIN, 0.5, f"India VIX {vix:.1f} elevated (caution)")
            
        # 10.0 to 18.0
        return self._make_vote(Vote.APPROVE, 1.0, f"India VIX {vix:.1f} in sweet spot")
