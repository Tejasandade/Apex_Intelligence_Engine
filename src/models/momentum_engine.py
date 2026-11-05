"""
Apex Intelligence Engine V5 - Tick Momentum Engine
====================================================
Detects directional momentum using Order Flow Imbalance (OFI)
and Volume-Weighted Price Velocity (VWPV).

Unlike StatArb (which bets prices will revert), Momentum bets that
prices will CONTINUE moving in the same direction. This strategy
activates ONLY during TRENDING regimes detected by the TickRegimeFilter.

The signal is based on two institutional-grade metrics:

1. Order Flow Imbalance (OFI):
   Measures the ratio of aggressive buying vs selling volume.
   OFI = (BuyVolume - SellVolume) / TotalVolume
   - OFI > +0.3 = Strong buy pressure (go LONG)
   - OFI < -0.3 = Strong sell pressure (go SHORT)
   
   In Binance aggTrade data, is_buyer_maker=True means the trade
   was initiated by a seller (they "took" from a buyer's limit order).

2. Volume-Weighted Price Velocity (VWPV):
   How fast the price is moving, weighted by volume.
   Large volume + fast price change = institutional order flow.
   VWPV confirms that the OFI signal isn't just noise.

3. Breakout Confirmation:
   Price must be at a rolling N-tick high (for longs) or low (for shorts).
   This prevents entering during mid-range chop.
"""

from __future__ import annotations

import collections
import time
from typing import Optional

import numpy as np

from src.core.logging import get_logger

logger = get_logger("apex.models.momentum")


class MomentumSignal:
    """Represents a momentum trading signal."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class TickMomentumEngine:
    """
    Tick-level momentum detector for BTC and other trending assets.
    
    Design principles:
    - Uses ONLY tick-level data (price, qty, is_buyer_maker)
    - No candle aggregation needed
    - O(1) memory via ring buffers
    - Recomputes signal every N ticks (configurable)
    """
    
    def __init__(
        self,
        symbol: str,
        ofi_window: int = 200,
        velocity_window: int = 50,
        breakout_window: int = 500,
        ofi_threshold: float = 0.25,
        velocity_threshold: float = 0.0001,
        recompute_interval: int = 10,
    ):
        """
        Args:
            symbol: The symbol this engine tracks (e.g., "btcusdt").
            ofi_window: Number of ticks to compute OFI over.
            velocity_window: Number of ticks for price velocity.
            breakout_window: Number of ticks for high/low breakout.
            ofi_threshold: OFI above this = bullish, below -this = bearish.
            velocity_threshold: Min abs(VWPV) to confirm the move is real.
            recompute_interval: Recompute signal every N ticks.
        """
        self.symbol = symbol.lower()
        self.ofi_window = ofi_window
        self.velocity_window = velocity_window
        self.breakout_window = breakout_window
        self.ofi_threshold = ofi_threshold
        self.velocity_threshold = velocity_threshold
        self.recompute_interval = recompute_interval
        
        # Ring buffers for O(1) memory
        self._prices = collections.deque(maxlen=max(breakout_window, ofi_window) + 10)
        self._volumes = collections.deque(maxlen=ofi_window + 10)
        self._sides = collections.deque(maxlen=ofi_window + 10)  # True = buyer_maker (sell aggressor)
        self._timestamps = collections.deque(maxlen=ofi_window + 10)
        
        # State
        self._tick_count = 0
        self._current_signal = MomentumSignal.FLAT
        self._current_ofi = 0.0
        self._current_vwpv = 0.0
        self._current_price = 0.0
        self._entry_price = 0.0
        self._in_position = False
        
        # P&L tracking
        self._realized_pnl = 0.0
        self._trades = 0
        
    @property
    def current_signal(self) -> str:
        return self._current_signal
    
    @property
    def current_price(self) -> float:
        return self._current_price
    
    def add_tick(self, price: float, qty: float, is_buyer_maker: bool, timestamp: int) -> str:
        """
        Ingest a single aggTrade tick.
        
        Args:
            price: Trade price.
            qty: Trade quantity.
            is_buyer_maker: True if the buyer placed the limit order
                           (meaning the SELLER was the aggressor/taker).
            timestamp: Trade timestamp in ms.
        
        Returns:
            Current signal: "LONG", "SHORT", or "FLAT".
        """
        self._prices.append(price)
        self._volumes.append(qty)
        self._sides.append(is_buyer_maker)
        self._timestamps.append(timestamp)
        self._current_price = price
        self._tick_count += 1
        
        # Wait for buffers to warm up
        if len(self._prices) < self.breakout_window:
            return MomentumSignal.FLAT
        
        # Recompute every N ticks
        if self._tick_count % self.recompute_interval != 0:
            return self._current_signal
        
        # --- Compute OFI ---
        ofi = self._compute_ofi()
        self._current_ofi = ofi
        
        # --- Compute VWPV ---
        vwpv = self._compute_vwpv()
        self._current_vwpv = vwpv
        
        # --- Compute Breakout (Percentile-Based) ---
        # Instead of requiring absolute high/low (too rare in liquid markets),
        # check if price is in the top/bottom 5% of recent range.
        prices_arr = np.array(list(self._prices))[-self.breakout_window:]
        price_range = np.max(prices_arr) - np.min(prices_arr)
        if price_range > 0:
            price_percentile = (price - np.min(prices_arr)) / price_range
        else:
            price_percentile = 0.5
        
        near_high = price_percentile > 0.95  # Top 5% of range
        near_low = price_percentile < 0.05   # Bottom 5% of range
        
        # --- Signal Logic ---
        # LONG: Buy pressure + price accelerating up + near highs
        if ofi > self.ofi_threshold and vwpv > self.velocity_threshold and near_high:
            new_signal = MomentumSignal.LONG
        # SHORT: Sell pressure + price accelerating down + near lows
        elif ofi < -self.ofi_threshold and vwpv < -self.velocity_threshold and near_low:
            new_signal = MomentumSignal.SHORT
        # EXIT: OFI weakens significantly (don't wait for full flip)
        elif self._current_signal != MomentumSignal.FLAT:
            exit_threshold = self.ofi_threshold * 0.3  # Exit when OFI drops to 30% of entry threshold
            if self._current_signal == MomentumSignal.LONG and ofi < exit_threshold:
                new_signal = MomentumSignal.FLAT
            elif self._current_signal == MomentumSignal.SHORT and ofi > -exit_threshold:
                new_signal = MomentumSignal.FLAT
            else:
                new_signal = self._current_signal  # Hold
        else:
            new_signal = MomentumSignal.FLAT
        
        # Track P&L on signal changes
        if new_signal != self._current_signal:
            self._handle_signal_change(new_signal)
        
        self._current_signal = new_signal
        return new_signal
    
    def _compute_ofi(self) -> float:
        """
        Compute Order Flow Imbalance over the OFI window.
        
        OFI = (AggressiveBuyVolume - AggressiveSellVolume) / TotalVolume
        
        In Binance aggTrade:
        - is_buyer_maker = True → SELLER was the aggressor (taker sold)
        - is_buyer_maker = False → BUYER was the aggressor (taker bought)
        """
        volumes = list(self._volumes)[-self.ofi_window:]
        sides = list(self._sides)[-self.ofi_window:]
        
        buy_vol = 0.0
        sell_vol = 0.0
        
        for vol, is_buyer_maker in zip(volumes, sides):
            if is_buyer_maker:
                sell_vol += vol  # Seller was the aggressor
            else:
                buy_vol += vol   # Buyer was the aggressor
        
        total = buy_vol + sell_vol
        if total < 1e-10:
            return 0.0
        
        return (buy_vol - sell_vol) / total
    
    def _compute_vwpv(self) -> float:
        """
        Compute Volume-Weighted Price Velocity.
        
        VWPV = sum(price_change_i * volume_i) / sum(volume_i)
        
        Measures how much the price is moving per unit of volume.
        Large positive VWPV = price rising on heavy volume (bullish).
        Large negative VWPV = price falling on heavy volume (bearish).
        """
        prices = list(self._prices)[-self.velocity_window:]
        volumes = list(self._volumes)[-self.velocity_window:]
        
        if len(prices) < 2:
            return 0.0
        
        price_changes = np.diff(prices)
        vols = np.array(volumes[1:])  # Align with price changes
        
        total_vol = np.sum(vols)
        if total_vol < 1e-10:
            return 0.0
        
        # Normalize by base price to make it percentage-based
        base_price = prices[0] if prices[0] > 0 else 1.0
        vwpv = np.sum(price_changes * vols) / (total_vol * base_price)
        
        return vwpv
    
    def _handle_signal_change(self, new_signal: str):
        """Track P&L when signals change."""
        if self._in_position:
            # Close position
            if self._current_signal == MomentumSignal.LONG:
                pnl = (self._current_price - self._entry_price) / self._entry_price
            elif self._current_signal == MomentumSignal.SHORT:
                pnl = (self._entry_price - self._current_price) / self._entry_price
            else:
                pnl = 0.0
            
            self._realized_pnl += pnl
            self._trades += 1
            self._in_position = False
            
            logger.info(
                "momentum_trade_closed",
                symbol=self.symbol,
                direction=self._current_signal,
                entry=round(self._entry_price, 2),
                exit=round(self._current_price, 2),
                pnl_bps=round(pnl * 10000, 1),
                total_pnl_bps=round(self._realized_pnl * 10000, 1),
                trade_count=self._trades,
            )
        
        if new_signal in (MomentumSignal.LONG, MomentumSignal.SHORT):
            # Open new position
            self._entry_price = self._current_price
            self._in_position = True
            
            logger.info(
                "momentum_trade_opened",
                symbol=self.symbol,
                direction=new_signal,
                price=round(self._current_price, 2),
                ofi=round(self._current_ofi, 4),
                vwpv=round(self._current_vwpv, 6),
            )
    
    def get_diagnostics(self) -> dict:
        """Return current engine diagnostics for monitoring."""
        return {
            "symbol": self.symbol,
            "signal": self._current_signal,
            "ofi": round(self._current_ofi, 4),
            "vwpv": round(self._current_vwpv, 6),
            "price": round(self._current_price, 2),
            "in_position": self._in_position,
            "realized_pnl_bps": round(self._realized_pnl * 10000, 1),
            "trades": self._trades,
            "ticks_ingested": self._tick_count,
        }
