"""
Apex Intelligence Engine V5 — Crypto Options Router
===================================================
Translates Momentum signals into cheap, high-convexity OTM Options.

Strategy:
When the Regime is TRENDING and Momentum fires a LONG signal:
1. Don't buy Spot/Futures (risk of liquidation from noise).
2. Instead, buy a Weekly OTM Call Option.
3. Your downside is strictly capped to the premium (e.g., $10).
4. Your upside is technically unlimited (1000%+ on big trends).
"""

from datetime import datetime, timedelta
from typing import Any, Optional
from src.core.logging import get_logger

logger = get_logger("apex.execution.options_router")


class CryptoOptionsRouter:
    """
    Routes directional signals to specific Bybit Option contracts.
    """

    def __init__(self, broker: Any):
        """
        Args:
            broker: BybitBroker instance
        """
        self.broker = broker
        
    async def get_tradable_option(
        self,
        base_coin: str,
        direction: str,
        spot_price: float,
        target_otm_pct: float = 0.015, # 1.5% OTM by default
        max_dte: int = 7 # Target weekly options
    ) -> Optional[dict]:
        """
        Find the optimal OTM option contract for maximum convexity.
        
        Args:
            base_coin: "BTC" or "ETH"
            direction: "BUY" (Call) or "SELL" (Put)
            spot_price: Current index price of the base coin
            target_otm_pct: How far Out-Of-The-Money to target
            
        Returns:
            Dict containing the option symbol and details.
        """
        option_type = "C" if direction in ["BUY", "LONG"] else "P"
        
        # Determine Target Strike
        if option_type == "C":
            target_strike_raw = spot_price * (1 + target_otm_pct)
        else:
            target_strike_raw = spot_price * (1 - target_otm_pct)
            
        # Round to nearest standard strike increment (e.g. 500 for BTC, 50 for ETH)
        increment = 500 if base_coin == "BTC" else 50
        target_strike = round(target_strike_raw / increment) * increment
        
        logger.info(
            "options_router_target_strike",
            base=base_coin, spot=spot_price, 
            direction=direction, type=option_type, target=target_strike
        )
        
        # Fetch available chain
        chain = await self.broker.get_option_chain(base_coin)
        if not chain:
            logger.error("options_router_no_chain_data", base=base_coin)
            return None
            
        # Filter chain for matching type and strike
        candidates = []
        now = datetime.utcnow()
        max_expiry_date = now + timedelta(days=max_dte)
        
        for contract in chain:
            sym = contract["symbol"] # e.g. BTC-30JUN24-65000-C-USDT
            parts = sym.split("-")
            if len(parts) >= 4:
                c_base = parts[0]
                c_exp = parts[1]
                c_strike = parts[2]
                c_type = parts[3]
            else:
                continue
            
            if c_type != option_type:
                continue
                
            try:
                # Expiry format: DDMMMYY e.g. 30JUN24
                exp_date = datetime.strptime(c_exp, "%d%b%y")
            except ValueError:
                continue
                
            if exp_date < now or exp_date > max_expiry_date:
                continue
                
            if int(c_strike) == int(target_strike):
                candidates.append({
                    "symbol": sym,
                    "strike": int(c_strike),
                    "expiry": exp_date,
                    "type": c_type
                })
                
        if not candidates:
            logger.warning("options_router_no_exact_strike", target=target_strike)
            # If exact strike is not found, we could implement a nearest-strike fallback here.
            # For strict mathematical convexity, we return None for now.
            return None
            
        # Sort by closest expiry (Gamma explosion)
        candidates.sort(key=lambda x: x["expiry"])
        selected = candidates[0]
        
        logger.info(
            "options_router_contract_selected",
            contract=selected["symbol"],
            expiry=selected["expiry"].strftime("%Y-%m-%d")
        )
        
        return selected
