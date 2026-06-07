"""
Apex Intelligence Engine V5 — Premium Decay Monitor
======================================================
Options-specific exit manager (Replaces TrailingStop for India Phase 4).

Unlike spot crypto where you use price-based trailing stops, options buying
requires monitoring the premium (Delta + Theta). Your max loss is the premium
paid, so strict price stops often get wick-hunted.

This monitor exits based on:
1. Premium Target (e.g. +100% premium)
2. Premium Decay Limit (e.g. -50% premium loss limit)
3. Time Decay (Theta crush: Exit if held > 45 mins with < 10% gain)
4. Auto Square-Off (Forced exit at 15:10 IST)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.core.logging import get_logger

logger = get_logger("apex.execution.premium_monitor")


@dataclass
class PremiumState:
    """State for an open options position."""
    symbol: str
    entry_premium: float
    quantity: float
    entry_time: float = field(default_factory=time.time)
    highest_premium: float = 0.0
    
    # Trackers
    current_premium: float = 0.0
    minutes_held: float = 0.0


class PremiumDecayMonitor:
    """
    Monitors live option premiums to enforce risk limits and targets.
    """

    def __init__(
        self,
        target_pct: float = 1.0,         # Target: +100% gain
        decay_limit_pct: float = 0.5,    # Stop loss: -50% of premium
        time_limit_mins: float = 45.0,   # Max time to wait for a move
        min_gain_for_time: float = 0.1,  # Require 10% gain to avoid time-limit exit
    ):
        self.target_pct = target_pct
        self.decay_limit_pct = decay_limit_pct
        self.time_limit_mins = time_limit_mins
        self.min_gain_for_time = min_gain_for_time
        
        self._states: dict[str, PremiumState] = {}

    def track_position(
        self,
        symbol: str,
        entry_premium: float,
        quantity: float,
    ) -> PremiumState:
        """Register a new option position for monitoring."""
        state = PremiumState(
            symbol=symbol,
            entry_premium=entry_premium,
            highest_premium=entry_premium,
            quantity=quantity,
        )
        self._states[symbol] = state
        
        logger.info(
            "premium_monitor_tracking",
            symbol=symbol,
            entry=f"₹{entry_premium:.2f}",
            qty=quantity
        )
        return state

    def remove_position(self, symbol: str) -> None:
        """Stop tracking a closed position."""
        self._states.pop(symbol, None)

    def update_premium(self, symbol: str, current_premium: float, current_ist_time: datetime) -> str | None:
        """
        Check if the option should be closed.
        
        Args:
            symbol: Option symbol
            current_premium: Current live price of the option
            current_ist_time: Current IST datetime (for square-off check)
            
        Returns:
            Exit reason (str) if it should be closed, else None.
        """
        state = self._states.get(symbol)
        if not state:
            return None
            
        state.current_premium = current_premium
        if current_premium > state.highest_premium:
            state.highest_premium = current_premium
            
        state.minutes_held = (time.time() - state.entry_time) / 60.0
        
        # Calculate % change from entry
        # A premium of 100 dropping to 50 is a -50% change.
        pct_change = (current_premium - state.entry_premium) / state.entry_premium if state.entry_premium > 0 else 0
        
        # 1. Target Hit
        if pct_change >= self.target_pct:
            return "target_hit"
            
        # 2. Premium Decay (Stop Loss)
        if pct_change <= -self.decay_limit_pct:
            return "premium_decay_limit"
            
        # 3. Time Decay (Theta crush avoidance)
        if state.minutes_held >= self.time_limit_mins:
            if pct_change < self.min_gain_for_time:
                return "time_decay_limit"
                
        # 4. Auto Square-Off (15:10 IST)
        if current_ist_time.hour == 15 and current_ist_time.minute >= 10:
            return "auto_square_off"
            
        return None

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return all tracked premiums (for dashboard)."""
        result = {}
        for symbol, state in self._states.items():
            result[symbol] = {
                "entry_premium": state.entry_premium,
                "current_premium": state.current_premium,
                "highest_premium": state.highest_premium,
                "minutes_held": round(state.minutes_held, 1),
                "pct_change": round(((state.current_premium - state.entry_premium) / state.entry_premium * 100), 2) if state.entry_premium > 0 else 0
            }
        return result
