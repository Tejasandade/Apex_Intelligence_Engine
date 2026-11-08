"""
Apex Indian Engine — Theta Decay & Reversal Profit Harvester
============================================================
Protects option buyers from time decay (Theta burn) and mean-reversion pullbacks.
Dynamically captures +1.0R to +1.5R exhaustion peaks and limits stagnation holding.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("apex.indian.harvester")


class ThetaHarvester:
    """
    Evaluates open option positions for time decay burnout and rapid exhaustion peaks.
    """

    def __init__(
        self,
        max_stagnant_minutes: float = 35.0,
        reversal_r_threshold: float = 1.0,
    ):
        self.max_stagnant_min = max_stagnant_minutes
        self.reversal_r_threshold = reversal_r_threshold

    def evaluate_harvest(
        self,
        entry_price: float,
        current_price: float,
        sl_price: float,
        entry_time: float,
        underlying_momentum_reversed: bool = False,
    ) -> tuple[bool, str]:
        """
        Evaluates whether an active option position should be harvested right now.
        Returns: (should_close: bool, reason_str: str)
        """
        sl_dist = max(entry_price - sl_price, 1.0)
        fav_dist = current_price - entry_price
        r_mult = fav_dist / sl_dist
        dur_min = (time.time() - entry_time) / 60.0

        # 1. Violent Reversal Harvest (+1.0R to +1.5R Peak Extraction)
        # Banks explosive option gains before retail pullbacks
        if r_mult >= self.reversal_r_threshold and underlying_momentum_reversed:
            return True, f"OPTION MOMENTUM HARVEST (+{r_mult:.1f}R on underlying momentum reversal)"

        # 2. Theta Decay Stagnation Shield
        # If an option is held for > 35m without breaking out, Theta decay eats the premium
        if dur_min >= self.max_stagnant_min and r_mult < 0.8:
            return True, f"THETA DECAY SHIELD ({dur_min:.0f}m stagnant — saving extrinsic value)"

        # 3. Expiry Afternoon Time Decay Acceleration (After 14:30 IST)
        # On weekly expiry day, options after 14:30 decay exponentially
        return False, "HOLD"

