import collections
import numpy as np
import pandas as pd
from typing import Optional
from src.core.logging import get_logger

logger = get_logger("apex.models.stat_arb")

class StatArbEngine:
    def __init__(self, asset_a: str, asset_b: str, max_buffer: int = 5000, z_threshold: float = 2.5):
        """
        Multi-Asset Statistical Arbitrage Engine (Pairs Trading)
        Uses O(1) collections.deque for memory safety against tick floods.
        """
        self.asset_a = asset_a.lower()
        self.asset_b = asset_b.lower()
        self.z_threshold = z_threshold
        
        # O(1) ring buffers to prevent memory leaks from tick floods
        self.buffer_a = collections.deque(maxlen=max_buffer)
        self.buffer_b = collections.deque(maxlen=max_buffer)
        self.timestamps_a = collections.deque(maxlen=max_buffer)
        self.timestamps_b = collections.deque(maxlen=max_buffer)
        
        self.current_beta: float = 1.0
        self.current_z_score: float = 0.0
        
        # AI Feature Buffers
        self.beta_history = collections.deque(maxlen=100)
        self.z_score_history = collections.deque(maxlen=100)
        self.spread_history = collections.deque(maxlen=100)
        self.current_features = None
        
    def add_tick(self, symbol: str, price: float, timestamp: int):
        """Ingest raw ticks into the respective O(1) buffer."""
        symbol = symbol.lower()
        if symbol == self.asset_a:
            self.buffer_a.append(price)
            self.timestamps_a.append(timestamp)
        elif symbol == self.asset_b:
            self.buffer_b.append(price)
            self.timestamps_b.append(timestamp)

    def synchronize_ticks(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Align the two tick streams by their length for regression.
        In a real HFT system, we'd align by timestamp using 'asof' merges,
        but for initial POC, truncating to the shortest length works for highly liquid pairs.
        """
        # Convert deque to C-speed numpy arrays, bypassing GIL for math ops
        arr_a = np.array(self.buffer_a)
        arr_b = np.array(self.buffer_b)
        
        min_len = min(len(arr_a), len(arr_b))
        if min_len < 100:
            return np.array([]), np.array([])
            
        return arr_a[-min_len:], arr_b[-min_len:]

    def update_spread(self) -> Optional[float]:
        """
        Calculates Hedge Ratio (Beta) using OLS, then computes Z-Score of the spread.
        Returns the current Z-Score if enough data exists, otherwise None.
        """
        arr_a, arr_b = self.synchronize_ticks()
        if len(arr_a) < 100:
            return None
            
        # OLS Regression to find Hedge Ratio (Beta)
        # Price_A = Beta * Price_B + Alpha
        # Use np.polyfit for fast 1D linear regression (degree 1)
        # Returns [beta, alpha]
        
        # Institutional Fix: Protect polyfit from zero variance crashes
        if np.var(arr_b) < 1e-8:
            return None
            
        try:
            beta, _ = np.polyfit(arr_b, arr_a, 1)
            self.current_beta = beta
            
            # Calculate the stationary spread
            spread = arr_a - (beta * arr_b)
            
            # Calculate rolling stats
            mean_spread = np.mean(spread)
            std_spread = np.std(spread)
            
            if std_spread == 0:
                return 0.0
                
            # Current Z-Score
            current_spread = spread[-1]
            self.beta_history.append(beta)
            self.spread_history.append(current_spread)
            self.z_score_history.append(self.current_z_score)
            
            # Calculate AI Features if buffers are somewhat warm
            if len(self.z_score_history) >= 50:
                z_vel = self.current_z_score - self.z_score_history[-10]
                beta_slope = self.current_beta - self.beta_history[-50]
                spread_vol = np.std(list(self.spread_history))
                
                self.current_features = pd.DataFrame([{
                    "z_score": self.current_z_score,
                    "z_score_velocity": z_vel,
                    "beta_slope": beta_slope,
                    "spread_volatility": spread_vol
                }])
            
            return self.current_z_score
            
        except Exception as e:
            logger.error("stat_arb_math_error", error=str(e))
            return None

    def get_signal(self) -> str:
        """
        Evaluates Z-score to trigger Dual-Leg execution signals.
        Returns: 'SHORT_A_LONG_B', 'LONG_A_SHORT_B', 'FLAT', or 'HOLD'
        """
        if self.current_z_score > self.z_threshold:
            # Asset A is overvalued relative to Asset B
            return "SHORT_A_LONG_B"
            
        elif self.current_z_score < -self.z_threshold:
            # Asset A is undervalued relative to Asset B
            return "LONG_A_SHORT_B"
            
        elif abs(self.current_z_score) < 0.1:
            # Reverted to mean
            return "FLAT"
            
        return "HOLD"
