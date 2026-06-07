"""
Apex Intelligence Engine V5 — Options Router
=============================================
Translates spot index signals into tradable option contracts.

Key responsibilities:
1. Strike Selection (ATM / OTM based on conviction)
2. Expiry Management (Current week, rolling to next week on Wed PM)
3. Liquidity Filtering (Avoid illiquid strikes)
"""

from datetime import datetime
from typing import Any
from src.core.logging import get_logger

logger = get_logger("apex.execution.options_router")


class OptionsRouter:
    """
    Routes spot signals (e.g. BUY BANKNIFTY) to specific option contracts.
    """

    def __init__(self, broker: Any):
        """
        Args:
            broker: AngelBroker instance (provides access to option chain data)
        """
        self.broker = broker

    def get_tradable_option(
        self,
        symbol: str,
        direction: str,
        spot_price: float,
        council_score: float = 0.0,
    ) -> dict[str, Any] | None:
        """
        Determine the best option contract for the given signal.

        Args:
            symbol: Spot symbol (e.g., BANKNIFTY)
            direction: BUY or SELL
            spot_price: Current spot index price
            council_score: 0.0 to 1.0 (higher conviction = ATM, lower = OTM)

        Returns:
            Dict containing option details:
            {
                "symbol": "BANKNIFTY29MAY2448500CE",
                "token": "12345",
                "type": "CE" or "PE",
                "strike": 48500,
                "expiry": "2024-05-29"
            }
        """
        # Step 1: Determine Option Type
        # Direction BUY -> Call (CE). Direction SELL -> Put (PE)
        option_type = "CE" if direction == "BUY" else "PE"

        # Step 2: Determine Strike Price (BankNifty has 100-pt increments)
        strike_increment = 100 if symbol == "BANKNIFTY" else 50
        
        # Find nearest ATM strike
        atm_strike = round(spot_price / strike_increment) * strike_increment

        if council_score >= 0.80:
            # High conviction: use ATM for maximum delta exposure
            target_strike = atm_strike
            logger.debug(
                "options_router_strike_atm",
                reason="High conviction",
                score=f"{council_score:.2f}",
                strike=target_strike
            )
        else:
            # Normal conviction: use 1 strike OTM to limit premium risk
            # For CE, OTM is higher strike. For PE, OTM is lower strike.
            if option_type == "CE":
                target_strike = atm_strike + strike_increment
            else:
                target_strike = atm_strike - strike_increment
                
            logger.debug(
                "options_router_strike_otm",
                reason="Normal conviction",
                score=f"{council_score:.2f}",
                strike=target_strike
            )

        # Step 3: Fetch expiry calendar & token from broker
        if not hasattr(self.broker, "get_option_contract"):
            logger.error("options_router_broker_missing_method")
            return None

        contract = self.broker.get_option_contract(
            symbol=symbol,
            option_type=option_type,
            strike=target_strike,
        )

        if not contract:
            logger.warning(
                "options_router_contract_not_found",
                symbol=symbol,
                type=option_type,
                strike=target_strike,
            )
            return None

        logger.info(
            "options_router_selected",
            spot=f"${spot_price:,.2f}",
            direction=direction,
            contract=contract["symbol"],
            strike=contract["strike"],
            type=contract["type"]
        )

        return contract
