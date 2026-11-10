"""
Apex Indian Engine — India VIX Macro Volatility Regime Core
===========================================================
Conditions option buying strategies, strike selection (ATM vs ITM),
position sizing, and target multipliers based on real-time India VIX.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("apex.indian.vix")


@dataclass
class VixParameters:
    regime_name: str                # LOW_VOL | GOLDEN_ZONE | ELEVATED_VOL | EXTREME_EVENT
    consensus_threshold: float      # Minimum Council consensus score to enter
    prefer_itm: bool                # True -> Select 1-Strike ITM (Delta >= 0.58)
    size_multiplier: float          # Risk allocation multiplier (0.50 to 1.0)
    sl_points_banknifty: float      # Base SL for BankNifty
    sl_points_nifty: float          # Base SL for Nifty
    tp_multiplier: float            # Risk-to-reward target multiplier
    description: str


class VixRegimeManager:
    """
    Evaluates current India VIX level and outputs exact tactical parameters.
    """

    def __init__(self, default_vix: float = 14.5):
        self.current_vix = default_vix

    def update_vix(self, vix_value: float) -> VixParameters:
        """Update current VIX and return tactical regime parameters."""
        if vix_value > 0.0:
            self.current_vix = round(vix_value, 2)
        return self.get_parameters()

    def get_parameters(self) -> VixParameters:
        vix = self.current_vix

        # 1. LOW VOLATILITY (VIX < 12.0) — High Theta burn, low range
        if vix < 12.0:
            return VixParameters(
                regime_name="LOW_VOL",
                consensus_threshold=0.64,  # Stricter entry filter
                prefer_itm=True,           # Buy 1-Strike ITM to protect extrinsic time value
                size_multiplier=0.80,
                sl_points_banknifty=20.0,
                sl_points_nifty=12.0,
                tp_multiplier=1.8,
                description="Low VIX: Compressed range, high theta burn risk — ITM preferred, tight targets",
            )

        # 2. THE GOLDEN ZONE (12.0 <= VIX <= 18.0) — Optimal Option Buying Sweet Spot
        elif 12.0 <= vix <= 18.0:
            return VixParameters(
                regime_name="GOLDEN_ZONE",
                consensus_threshold=0.58,  # Institutional consensus threshold
                prefer_itm=False,          # Standard ATM for optimal gamma
                size_multiplier=1.00,
                sl_points_banknifty=25.0,
                sl_points_nifty=15.0,
                tp_multiplier=2.1,
                description="Golden Zone: Clean trending expansions — ATM options with achievable runner targets",
            )

        # 3. ELEVATED VOLATILITY (18.0 < VIX <= 22.0) — Wide swings, momentum breakouts
        elif 18.0 < vix <= 22.0:
            return VixParameters(
                regime_name="ELEVATED_VOL",
                consensus_threshold=0.60,
                prefer_itm=False,
                size_multiplier=1.00,
                sl_points_banknifty=32.0,  # Wider SL to withstand high ATR wicks
                sl_points_nifty=18.0,
                tp_multiplier=2.4,
                description="Elevated VIX: Fast momentum expansion — Wider stop buffer, realistic momentum targets",
            )

        # 4. EXTREME / EVENT (VIX > 22.0) — High IV crush danger (Budget/Election/RBI Policy)
        else:
            return VixParameters(
                regime_name="EXTREME_EVENT",
                consensus_threshold=0.68,  # High conviction required
                prefer_itm=True,           # ITM only to buffer post-event IV collapse
                size_multiplier=0.50,      # Cut risk in half
                sl_points_banknifty=40.0,
                sl_points_nifty=22.0,
                tp_multiplier=2.0,
                description="Extreme VIX: Severe IV crush risk — Half sizing, ITM contracts, quick harvest",
            )

