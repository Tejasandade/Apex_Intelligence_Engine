"""
Apex Indian Engine — Dynamic Strike & Delta Selection Core
==========================================================
Selects the optimal Option Contract (ATM vs. 1-Strike ITM) based on
India VIX regime, Council conviction, and Black-Scholes Greeks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime
from typing import Any, Optional

from apex_indian_engine.scrip_master import ScripMaster
from apex_indian_engine.greeks import GreeksEngine, OptionGreeks
from apex_indian_engine.vix_regime import VixParameters

logger = logging.getLogger("apex.indian.strikes")


@dataclass
class SelectedContract:
    symbol: str
    token: str
    strike: float
    option_type: str            # "CE" or "PE"
    expiry: str
    lot_size: int
    strike_type: str            # "ATM" or "ITM_1"
    est_premium: float
    greeks: OptionGreeks
    reason: str


class DynamicStrikeSelector:
    """
    Selects between ATM and 1-Strike ITM based on volatility regimes and conviction.
    """

    def __init__(
        self,
        scrip_master: ScripMaster,
        greeks_engine: GreeksEngine,
        smart_api_client: Optional[Any] = None,
    ):
        self.scrip_master = scrip_master
        self.greeks_engine = greeks_engine
        self.api = smart_api_client

    def select_best_strike(
        self,
        index_name: str,
        spot_price: float,
        option_type: str,
        vix_params: VixParameters,
        conviction: float = 0.65,
        days_to_expiry: float = 3.0,
    ) -> Optional[SelectedContract]:
        """
        Resolves the highest-Sharpe strike contract for the current regime.
        """
        idx = index_name.upper()
        opt = option_type.upper()
        expiry = self.scrip_master.get_nearest_expiry(idx)
        if not expiry:
            logger.error(f"No active weekly expiry found for {idx}")
            return None

        # 0DTE Afternoon Theta Shield: After 13:30 IST on expiry day, roll to Next Weekly Expiry
        now_time = datetime.now().time()
        if now_time >= dtime(13, 30):
            next_exp = self.scrip_master.get_next_expiry(idx)
            if next_exp and next_exp != expiry:
                logger.info(f"🛡️ 0DTE AFTERNOON SHIELD: Rolling {idx} from {expiry} to Next Week ({next_exp}) to bypass rapid theta burn.")
                expiry = next_exp

        # 1. Decide ATM vs ITM offset
        # In low VIX or extreme VIX or lower conviction -> 1-Strike ITM
        if vix_params.prefer_itm or conviction < 0.70:
            offset = 1
            strike_label = "ITM_1"
            reason = f"Selected 1-Strike ITM (Delta ~0.60) to protect extrinsic value under {vix_params.regime_name}"
        else:
            offset = 0
            strike_label = "ATM"
            reason = f"Selected pure ATM (Delta ~0.50) for maximum gamma expansion under {vix_params.regime_name}"

        # 2. Resolve contract from Scrip Master
        contract = self.scrip_master.resolve_atm_strike(
            index_name=idx,
            spot_price=spot_price,
            option_type=opt,
            expiry=expiry,
            offset_steps=offset,
        )

        # Fallback to ATM if ITM contract is not indexed
        if not contract and offset != 0:
            logger.warning(f"ITM strike not found for {idx}. Falling back to ATM.")
            contract = self.scrip_master.resolve_atm_strike(
                index_name=idx,
                spot_price=spot_price,
                option_type=opt,
                expiry=expiry,
                offset_steps=0,
            )
            strike_label = "ATM"
            reason = "Fallback to ATM strike (ITM unavailable in chain)"

        if not contract:
            logger.error(f"Failed to resolve any option contract for {idx} {opt}")
            return None

        strike_px = float(contract["strike"])
        lot_sz = int(contract.get("lot_size", 30))

        # 3. Calculate Black-Scholes Greeks & Theoretical Price
        t_years = max(days_to_expiry / 365.0, 0.001)
        sigma = max(vix_params.size_multiplier * 0.15, 0.12)
        
        bs = self.greeks_engine.calc_black_scholes(
            S=spot_price,
            K=strike_px,
            T=t_years,
            r=0.07,
            sigma=sigma,
            option_type=opt,
        )
        est_price = max(bs.price, 45.0)

        # Check live market LTP from Angel One if API is active
        if self.api:
            try:
                ltp_resp = self.api.ltpData("NFO", contract["symbol"], str(contract["token"]))
                live_px = float(ltp_resp.get("data", {}).get("ltp", 0.0))
                if live_px > 1.0:
                    est_price = live_px
                    reason += f" | Live Broker LTP: ₹{live_px:.2f}"
            except Exception as e:
                logger.debug(f"Could not fetch live LTP for strike: {e}")

        greeks = self.greeks_engine.calculate_greeks(
            spot=spot_price,
            strike=strike_px,
            time_to_expiry_years=t_years,
            market_price=est_price,
            option_type=opt,
            initial_iv=sigma,
        )

        return SelectedContract(
            symbol=contract["symbol"],
            token=str(contract["token"]),
            strike=strike_px,
            option_type=opt,
            expiry=expiry,
            lot_size=lot_sz,
            strike_type=strike_label,
            est_premium=est_price,
            greeks=greeks,
            reason=reason,
        )

