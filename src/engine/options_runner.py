"""
Apex Intelligence Engine V5 - Options Master Loop
===================================================
Executes the High-Convexity Options Strategy.

Architecture:
1. Feeds live ticks to `TickRegimeFilter`.
2. When regime is `TRENDING`, feeds ticks to `TickMomentumEngine`.
3. When Momentum triggers `LONG` or `SHORT`:
   a. Queries `CryptoOptionsRouter` for the optimal OTM Call/Put.
   b. Queries `BybitBroker` for the current premium (ask price).
   c. If premium is acceptable (e.g., < $20), it executes a Market Buy.
   d. The downside is strictly capped. The upside is unlimited.
"""

import asyncio
from src.core.logging import get_logger
from src.execution.bybit_broker import BybitBroker
from src.execution.options_router import CryptoOptionsRouter
from src.models.tick_regime_filter import TickRegimeFilter
from src.models.momentum_engine import TickMomentumEngine

logger = get_logger("apex.engine.options")

# Strategy Parameters
BASE_COIN = "BTC"
MAX_PREMIUM_USDT = 50.0  # Max dollar amount we are willing to risk per trade
OTM_PERCENTAGE = 0.015   # 1.5% Out-of-the-money

class OptionsRunner:
    def __init__(self, broker: BybitBroker, base_coin: str = BASE_COIN):
        self.broker = broker
        self.base_coin = base_coin.upper()
        self.router = CryptoOptionsRouter(broker)
        
        # We use a larger window for Options to catch macro trends
        self.regime_filter = TickRegimeFilter(window=1000)
        self.momentum_engine = TickMomentumEngine(symbol=self.base_coin, ofi_window=100, ofi_threshold=0.25, velocity_threshold=0.0001)
        
        self.in_position = False
        self.current_option = None

    async def execute_trade(self, signal: str, spot_price: float):
        """Routes the signal and executes the option purchase."""
        logger.info(f"Triggering Options Execution for {signal} at Spot: {spot_price}")
        
        # 1. Get Option Contract
        contract = await self.router.get_tradable_option(
            base_coin=self.base_coin,
            direction=signal,
            spot_price=spot_price,
            target_otm_pct=OTM_PERCENTAGE
        )
        
        if not contract:
            logger.warning("No suitable options contract found.")
            return
            
        sym = contract["symbol"]
        
        # 2. Check Premium
        ticker = await self.broker.get_option_ticker(sym)
        ask_price = ticker.get("ask1Price", 0)
        
        if ask_price <= 0:
            logger.warning(f"No liquidity on ask for {sym}")
            return
            
        # For BTC options, contract size is usually 1 BTC or 0.1 BTC depending on the exchange.
        # Bybit BTC options are settled in USDC and have a multiplier of 1 (1 contract = 1 BTC).
        # We will buy 0.1 contracts.
        qty = 0.1
        cost = ask_price * qty
        
        if cost > MAX_PREMIUM_USDT:
            logger.warning(f"Premium {cost} exceeds max risk {MAX_PREMIUM_USDT} for {sym}")
            return
            
        # 3. Execute
        logger.info(f"Executing Market Buy for {qty} of {sym} at ~{ask_price} (Cost: ${cost:.2f})")
        order_id = await self.broker.place_option_order(
            symbol=sym,
            side="Buy",
            qty=qty
        )
        
        if order_id:
            logger.info(f"OPTIONS TRADE SECURED! Order ID: {order_id}")
            self.in_position = True
            self.current_option = sym

    async def process_tick(self, price: float, qty: float, is_buyer_maker: bool, timestamp: int):
        """Ingest tick and process strategy logic."""
        if self.in_position:
            # In options, we don't manage Stop-Losses tick-by-tick!
            # The position is held until expiration or managed by a separate portfolio process.
            # We will ignore ticks while holding to prevent over-trading.
            return
            
        self.regime_filter.add_tick(price)
        state = self.regime_filter.current_regime
        
        if state == "TRENDING":
            signal = self.momentum_engine.add_tick(price, qty, is_buyer_maker, timestamp)
            
            if signal in ["LONG", "SHORT"]:
                # Execute in background so we don't block the WebSocket loop
                asyncio.create_task(self.execute_trade("BUY" if signal == "LONG" else "SELL", price))
                # Reset engine to prevent instant re-entry
                self.momentum_engine = TickMomentumEngine(symbol=self.base_coin, ofi_window=100, ofi_threshold=0.25, velocity_threshold=0.0001)

async def simulate_options_session():
    """Simulate the Options Runner logic."""
    broker = BybitBroker(api_key="dummy", api_secret="dummy", use_testnet=True)
    runner = OptionsRunner(broker, base_coin="BTC")
    
    print("Simulating Options Runner Logic...")
    # Simulate a fake trend
    price = 60000.0
    for i in range(1500):
        price += 10.0 # Straight line trend up
        await runner.process_tick(price, 1.0, False, i)
        
    print("Simulation complete. If no errors occurred, structural integrity is sound.")
    await broker.close()

if __name__ == "__main__":
    asyncio.run(simulate_options_session())
