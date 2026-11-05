"""
Apex Intelligence Engine V5 — Tick-Level Regime Filter
=======================================================
Classifies the current BTC market regime using raw tick data.
This is the "Master Controller" that decides which strategy gets capital.

Regimes:
- MEAN_REVERTING: Low directional momentum, pairs spread oscillating → StatArb ON
- TRENDING:       Strong directional move in BTC → StatArb OFF, Momentum ON
- HIGH_STRESS:    Extreme volatility spike (flash crash / liquidation cascade) → ALL OFF

The filter operates on a rolling window of BTC prices. It does NOT need
pre-computed candle indicators (ADX/CHOP/ATR). It computes everything
from raw ticks, making it compatible with our WebSocket architecture.

Usage (Live):
    regime_filter = TickRegimeFilter()
    regime_filter.add_tick(price=65000.0)
    current_regime = regime_filter.get_regime()  # "MEAN_REVERTING"

Usage (Backtest):
    regimes = TickRegimeFilter.classify_series(btc_prices, window=500)
"""

from __future__ import annotations

import collections
import math
from enum import Enum
from typing import Optional

import numpy as np

from src.core.logging import get_logger

logger = get_logger("apex.models.tick_regime")


class TickRegime(str, Enum):
    """Market regime classification from tick data."""
    MEAN_REVERTING = "MEAN_REVERTING"  # StatArb is safe to trade
    TRENDING = "TRENDING"              # Momentum strategies should activate
    HIGH_STRESS = "HIGH_STRESS"        # All strategies should halt
    WARMUP = "WARMUP"                  # Not enough data yet


class TickRegimeFilter:
    """
    Stateful regime detector that operates on raw price ticks.
    
    The algorithm uses three metrics computed on a rolling window:
    
    1. Hurst Exponent (H):
       - H < 0.5  → Mean-reverting (prices tend to snap back)
       - H = 0.5  → Random walk (no edge)
       - H > 0.5  → Trending (prices persist in one direction)
       
    2. Directional Strength (DS):
       - Ratio of net price displacement to total distance traveled
       - DS near 0 → choppy / ranging
       - DS near 1 → strong trend
       
    3. Volatility Spike Ratio (VSR):
       - Current short-term volatility vs long-term baseline
       - VSR > 3.0 → extreme stress event (flash crash)
    
    The Hurst Exponent is the gold standard in quantitative finance for
    distinguishing mean-reverting from trending markets. It was originally
    developed for Nile River flood analysis and is used by every major
    quant fund for regime classification.
    """
    
    def __init__(
        self,
        window: int = 500,
        hurst_threshold_low: float = 0.45,
        hurst_threshold_high: float = 0.55,
        ds_threshold: float = 0.15,
        stress_vsr_threshold: float = 3.0,
    ):
        """
        Args:
            window: Rolling window of ticks for regime computation.
            hurst_threshold_low: Below this H value → MEAN_REVERTING.
            hurst_threshold_high: Above this H value → TRENDING.
            ds_threshold: Directional strength above this → confirms trend.
            stress_vsr_threshold: Volatility spike ratio above this → HIGH_STRESS.
        """
        self.window = window
        self.hurst_threshold_low = hurst_threshold_low
        self.hurst_threshold_high = hurst_threshold_high
        self.ds_threshold = ds_threshold
        self.stress_vsr_threshold = stress_vsr_threshold
        
        # Rolling price buffer
        self._prices = collections.deque(maxlen=window * 2)
        
        # State
        self._current_regime = TickRegime.WARMUP
        self._regime_history = collections.deque(maxlen=100)
        self._tick_count = 0
        
    @property
    def current_regime(self) -> TickRegime:
        return self._current_regime
    
    @property
    def is_safe_for_stat_arb(self) -> bool:
        """Returns True if the current regime is favorable for StatArb."""
        return self._current_regime == TickRegime.MEAN_REVERTING
    
    @property
    def is_trending(self) -> bool:
        """Returns True if the market is in a trending regime."""
        return self._current_regime == TickRegime.TRENDING
    
    def add_tick(self, price: float) -> TickRegime:
        """
        Ingest a new BTC price tick and recalculate the regime.
        
        We only recompute every 50 ticks to save CPU, since
        regime doesn't change on a tick-by-tick basis.
        """
        self._prices.append(price)
        self._tick_count += 1
        
        # Wait for enough data
        if len(self._prices) < self.window:
            return TickRegime.WARMUP
        
        # Recalculate every 50 ticks (not every tick — saves CPU)
        if self._tick_count % 50 != 0:
            return self._current_regime
        
        prices = np.array(list(self._prices))[-self.window:]
        
        # 1. Compute Hurst Exponent
        hurst = self._compute_hurst(prices)
        
        # 2. Compute Directional Strength
        ds = self._compute_directional_strength(prices)
        
        # 3. Compute Volatility Spike Ratio
        vsr = self._compute_volatility_spike(prices)
        
        # Classification logic (priority order)
        if vsr > self.stress_vsr_threshold:
            new_regime = TickRegime.HIGH_STRESS
        elif ds > 0.30:
            # Very strong directional move — this IS a trend, regardless of Hurst
            new_regime = TickRegime.TRENDING
        elif hurst > self.hurst_threshold_high and ds > self.ds_threshold:
            # Moderate Hurst + moderate DS = confirmed trend
            new_regime = TickRegime.TRENDING
        elif hurst < self.hurst_threshold_low or ds < 0.05:
            # Low Hurst OR extremely choppy → mean reverting
            new_regime = TickRegime.MEAN_REVERTING
        else:
            # Ambiguous zone (H near 0.5) — default to current regime (sticky)
            new_regime = self._current_regime if self._current_regime != TickRegime.WARMUP else TickRegime.MEAN_REVERTING
        
        # Log regime transitions
        if new_regime != self._current_regime:
            logger.info(
                "regime_transition",
                old=self._current_regime.value,
                new=new_regime.value,
                hurst=round(hurst, 4),
                directional_strength=round(ds, 4),
                volatility_spike_ratio=round(vsr, 2),
            )
        
        self._current_regime = new_regime
        self._regime_history.append(new_regime)
        return new_regime
    
    @staticmethod
    def _compute_hurst(prices: np.ndarray) -> float:
        """
        Compute the Hurst Exponent using the Rescaled Range (R/S) method.
        
        The R/S method divides the series into chunks of increasing size,
        computes the rescaled range for each chunk size, and fits a
        log-log regression. The slope is the Hurst exponent.
        
        Returns:
            H: Hurst exponent (0 to 1).
                H < 0.5 → mean-reverting
                H = 0.5 → random walk
                H > 0.5 → trending
        """
        returns = np.diff(np.log(prices + 1e-10))
        n = len(returns)
        
        if n < 20:
            return 0.5  # Not enough data
        
        # Use chunk sizes from 10 to n/2
        max_chunk = n // 2
        min_chunk = 10
        
        if max_chunk <= min_chunk:
            return 0.5
        
        chunk_sizes = []
        rs_values = []
        
        size = min_chunk
        while size <= max_chunk:
            chunk_sizes.append(size)
            
            # Compute R/S for this chunk size
            num_chunks = n // size
            rs_chunk = []
            
            for i in range(num_chunks):
                chunk = returns[i * size : (i + 1) * size]
                mean_chunk = np.mean(chunk)
                deviations = np.cumsum(chunk - mean_chunk)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(chunk, ddof=1) if np.std(chunk, ddof=1) > 1e-10 else 1e-10
                rs_chunk.append(R / S)
            
            rs_values.append(np.mean(rs_chunk))
            size = int(size * 1.5)  # Geometric spacing
        
        if len(chunk_sizes) < 3:
            return 0.5
        
        # Log-log regression: log(R/S) = H * log(n) + c
        log_sizes = np.log(np.array(chunk_sizes, dtype=float))
        log_rs = np.log(np.array(rs_values, dtype=float) + 1e-10)
        
        # Simple OLS for slope
        try:
            H = np.polyfit(log_sizes, log_rs, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            H = 0.5
        
        # Clamp to valid range
        return max(0.0, min(1.0, H))
    
    @staticmethod
    def _compute_directional_strength(prices: np.ndarray) -> float:
        """
        Compute directional strength: |net displacement| / total distance.
        
        If a price series goes 100 → 105 → 103 → 110:
        - Net displacement = |110 - 100| = 10
        - Total distance = 5 + 2 + 7 = 14
        - DS = 10 / 14 = 0.71 (strong trend)
        
        If it goes 100 → 105 → 100 → 105:
        - Net displacement = |105 - 100| = 5
        - Total distance = 5 + 5 + 5 = 15
        - DS = 5 / 15 = 0.33 (choppy)
        """
        if len(prices) < 2:
            return 0.0
        
        net_displacement = abs(prices[-1] - prices[0])
        total_distance = np.sum(np.abs(np.diff(prices)))
        
        if total_distance < 1e-10:
            return 0.0
        
        return net_displacement / total_distance
    
    @staticmethod
    def _compute_volatility_spike(prices: np.ndarray) -> float:
        """
        Compute ratio of recent volatility to baseline volatility.
        
        Uses the last 10% of the window as "recent" and the full window
        as the baseline. A VSR > 3.0 indicates extreme market stress.
        """
        returns = np.diff(np.log(prices + 1e-10))
        
        if len(returns) < 20:
            return 1.0
        
        # Recent: last 10% of window
        recent_window = max(10, len(returns) // 10)
        recent_vol = np.std(returns[-recent_window:])
        baseline_vol = np.std(returns)
        
        if baseline_vol < 1e-10:
            return 1.0
        
        return recent_vol / baseline_vol
    
    @classmethod
    def classify_series(
        cls,
        prices: np.ndarray,
        window: int = 500,
        step: int = 50,
    ) -> list[tuple[int, str, float, float, float]]:
        """
        Classify regime for an entire price series (for backtesting).
        
        Returns:
            List of (index, regime, hurst, ds, vsr) tuples.
        """
        results = []
        n = len(prices)
        
        for i in range(window, n, step):
            chunk = prices[i - window : i]
            hurst = cls._compute_hurst(chunk)
            ds = cls._compute_directional_strength(chunk)
            vsr = cls._compute_volatility_spike(chunk)
            
            if vsr > 3.0:
                regime = TickRegime.HIGH_STRESS
            elif hurst > 0.55 and ds > 0.15:
                regime = TickRegime.TRENDING
            elif hurst < 0.45:
                regime = TickRegime.MEAN_REVERTING
            else:
                regime = TickRegime.MEAN_REVERTING  # Default for backtest
            
            results.append((i, regime.value, hurst, ds, vsr))
        
        return results
    
    def get_diagnostics(self) -> dict:
        """Return current regime diagnostics for logging/monitoring."""
        if len(self._prices) < self.window:
            return {"regime": "WARMUP", "ticks": len(self._prices), "needed": self.window}
        
        prices = np.array(list(self._prices))[-self.window:]
        return {
            "regime": self._current_regime.value,
            "hurst": round(self._compute_hurst(prices), 4),
            "directional_strength": round(self._compute_directional_strength(prices), 4),
            "volatility_spike_ratio": round(self._compute_volatility_spike(prices), 2),
            "ticks_ingested": self._tick_count,
        }
