"""
Apex Indian Engine — Live 3-Minute NSE Option Chain & Delta-OI Momentum Tracker
===============================================================================
Polls open interest distributions every 3 minutes (matching NSE's 3-minute OI cycle).
Tracks:
1. Real-time Put-Call Ratio (PCR)
2. Delta-OI Momentum (Change in Call OI vs Change in Put OI)
3. Institutional Short-Covering vs Long-Unwinding flow
4. Dynamic Max Call Resistance & Max Put Support Walls
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from apex_indian_engine.oi_shield import OIShield, OIMetrics

logger = logging.getLogger("apex.indian.oi_tracker")


@dataclass
class OIFlowSnapshot:
    pcr: float
    pcr_change_15m: float
    total_call_oi: int
    total_put_oi: int
    delta_call_oi: int              # Net change in Call OI
    delta_put_oi: int               # Net change in Put OI
    flow_sentiment: str             # SHORT_COVERING_RALLY | LONG_UNWINDING_CRASH | PUT_WRITING_BULLISH | CALL_WRITING_BEARISH | BALANCED
    max_call_wall: float
    max_put_wall: float
    timestamp: float = field(default_factory=lambda: time.time())


class OptionChainOITracker:
    """
    Tracks rolling 3-minute Option Chain snapshots and detects institutional flow shifts.
    """

    def __init__(self, index_name: str = "BANKNIFTY", smart_api_client: Optional[Any] = None):
        self.index_name = index_name.upper()
        self.api = smart_api_client
        self.oi_shield = OIShield(wall_buffer_points=60.0 if "BANK" in self.index_name else 30.0)
        
        # History of snapshots: oldest to newest
        self._history: list[OIFlowSnapshot] = []
        self._last_poll_time = 0.0
        self.poll_interval_sec = 180.0  # 3 minutes

    def update_chain(
        self,
        chain_data: list[dict[str, Any]],
        spot_price: float,
    ) -> OIFlowSnapshot:
        """
        Parses option chain entries: [{strike, option_type, open_interest, change_oi}, ...]
        and generates an institutional OI flow snapshot.
        """
        now = time.time()
        self._last_poll_time = now

        call_oi_by_strike: dict[float, int] = {}
        put_oi_by_strike: dict[float, int] = {}
        delta_call_by_strike: dict[float, int] = {}
        delta_put_by_strike: dict[float, int] = {}

        for c in chain_data:
            s = float(c.get("strike", 0.0))
            opt = str(c.get("option_type", "")).upper()
            oi = int(c.get("open_interest", 0))
            chg_oi = int(c.get("change_oi", 0))

            if opt == "CE":
                call_oi_by_strike[s] = call_oi_by_strike.get(s, 0) + oi
                delta_call_by_strike[s] = delta_call_by_strike.get(s, 0) + chg_oi
            elif opt == "PE":
                put_oi_by_strike[s] = put_oi_by_strike.get(s, 0) + oi
                delta_put_by_strike[s] = delta_put_by_strike.get(s, 0) + chg_oi

        tot_call = sum(call_oi_by_strike.values())
        tot_put = sum(put_oi_by_strike.values())
        tot_delta_call = sum(delta_call_by_strike.values())
        tot_delta_put = sum(delta_put_by_strike.values())

        pcr = round(tot_put / max(tot_call, 1), 2)

        # Max OI Walls
        max_call_wall = max(call_oi_by_strike, key=call_oi_by_strike.get) if call_oi_by_strike else spot_price + 500
        max_put_wall = max(put_oi_by_strike, key=put_oi_by_strike.get) if put_oi_by_strike else spot_price - 500

        # Calculate 15m PCR trend
        pcr_chg_15m = 0.0
        if self._history:
            # 5 snapshots back = 15m
            old_pcr = self._history[-min(5, len(self._history))].pcr
            pcr_chg_15m = round(pcr - old_pcr, 2)

        # Institutional Flow Classification
        if tot_delta_call < 0 and tot_delta_put > 0:
            flow = "SHORT_COVERING_RALLY"  # Call writers covering, Put writers aggressive
        elif tot_delta_put < 0 and tot_delta_call > 0:
            flow = "LONG_UNWINDING_CRASH"  # Put writers trapped/exiting, Call writers piling
        elif tot_delta_put > tot_delta_call * 1.5:
            flow = "PUT_WRITING_BULLISH"
        elif tot_delta_call > tot_delta_put * 1.5:
            flow = "CALL_WRITING_BEARISH"
        else:
            flow = "BALANCED"

        snap = OIFlowSnapshot(
            pcr=pcr,
            pcr_change_15m=pcr_chg_15m,
            total_call_oi=tot_call,
            total_put_oi=tot_put,
            delta_call_oi=tot_delta_call,
            delta_put_oi=tot_delta_put,
            flow_sentiment=flow,
            max_call_wall=max_call_wall,
            max_put_wall=max_put_wall,
            timestamp=now,
        )

        self._history.append(snap)
        if len(self._history) > 100:
            self._history.pop(0)

        logger.info(
            f"NSE OI Flow ({self.index_name}): PCR={pcr:.2f} (Δ15m: {pcr_chg_15m:+.2f}) | "
            f"Flow: {flow} | Call Wall: {max_call_wall:.0f} | Put Wall: {max_put_wall:.0f}"
        )
        return snap

    def get_latest_snapshot(self) -> Optional[OIFlowSnapshot]:
        return self._history[-1] if self._history else None
