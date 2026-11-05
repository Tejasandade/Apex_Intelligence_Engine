"""
Apex Intelligence Engine V5 — Statistical Alpha Models
======================================================
Mathematical models that generate Alpha based on pure statistical 
deviation (Z-Score) or Structural Breakouts (Donchian Channels).

These replace the black-box ML predictors as the primary "Trigger".
"""

import numpy as np
import pandas as pd
from collections import deque
from src.core.logging import get_logger

logger = get_logger("apex.models.statistical")


class StatisticalAlphaModel:
    """
    Mathematical Alpha Engine for 15m timeframes.
    Combines Trend Following (Donchian Breakout) and 
    Mean Reversion (Volatility-Adjusted Z-Score).
    """

    def __init__(self, name: str = "15m_statistical_alpha"):
        self.name = name
        self.model_name = name
        
        # Parameters
        self.trend_window = 55
        self.z_window = 200
        self.z_threshold = 2.8
        
        # State
        self._buffer = deque(maxlen=max(self.trend_window, self.z_window))
        
    def add_candle(self, candle: dict):
        """Feed raw candles to maintain internal state."""
        self._buffer.append({
            "close": float(candle.get("close", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
        })

    def predict(self, features: pd.DataFrame = None) -> tuple[float, float, str]:
        """
        Calculates mathematical edge.its been
        Returns: (prob_long, prob_short, strategy_type)
        Where probabilities are 1.0 if triggered, 0.5 otherwise.
        strategy_type is either 'BREAKOUT', 'MEAN_REVERSION', or 'UNKNOWN'
        """
        if len(self._buffer) < self.trend_window:
            return 0.5, 0.5, "UNKNOWN"
            
        closes = [c["close"] for c in self._buffer]
        highs = [c["high"] for c in self._buffer]
        lows = [c["low"] for c in self._buffer]
        
        current_close = closes[-1]
        
        prob_long = 0.5
        prob_short = 0.5
        strategy_type = "UNKNOWN"
        
        # --- 1. TREND FOLLOWING (Donchian Breakout) ---
        # Look at the previous N periods (excluding current candle to avoid lookahead)
        recent_highs = highs[-self.trend_window-1:-1]
        recent_lows = lows[-self.trend_window-1:-1]
        
        if len(recent_highs) > 0:
            highest_high = max(recent_highs)
            lowest_low = min(recent_lows)
            
            if current_close > highest_high:
                # Strong structural breakout UP
                prob_long = 0.90
                strategy_type = "BREAKOUT"
            elif current_close < lowest_low:
                # Strong structural breakdown DOWN
                prob_short = 0.90
                strategy_type = "BREAKOUT"
                
        # --- 2. MEAN REVERSION (Z-Score) ---
        if len(closes) >= self.z_window:
            recent_closes = closes[-self.z_window:]
            rolling_mean = np.mean(recent_closes)
            rolling_std = np.std(recent_closes)
            
            if rolling_std > 0:
                z_score = (current_close - rolling_mean) / rolling_std
                
                if z_score <= -self.z_threshold:
                    # Extreme oversold (Snap-back expected)
                    if 0.85 > prob_long:
                        prob_long = 0.85
                        strategy_type = "MEAN_REVERSION"
                elif z_score >= self.z_threshold:
                    # Extreme overbought (Snap-back expected)
                    if 0.85 > prob_short:
                        prob_short = 0.85
                        strategy_type = "MEAN_REVERSION"
                    
        return prob_long, prob_short, strategy_type
