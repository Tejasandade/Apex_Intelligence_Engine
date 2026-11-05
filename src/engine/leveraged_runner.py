"""
Apex Intelligence Engine V5 - High Leverage Futures Runner
=========================================================
The definitive live trading engine combining Regime Detection,
Momentum (OFI), StatArb (Heuristic AI), and High Leverage (20x)
via Bybit V5 Futures with strict, exchange-level Stop-Losses.

Architecture:
1. Connect to live exchange stream (Binance for data).
2. Classify market regime (TickRegimeFilter).
3. If Trending: Ask MomentumEngine.
4. If Choppy: Ask StatArb Engine (Transformer/Heuristic).
5. If Signal == GO -> Execute via BybitBroker with 2% Equity Risk & 20x Leverage.
"""

import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.logging import get_logger
from src.models.tick_regime_filter import TickRegimeFilter
from src.models.momentum_engine import TickMomentumEngine
from src.models.transformer_model import ApexTransformerModel
from src.execution.bybit_broker import BybitBroker

logger = get_logger("apex.engine.leveraged")

# Trading Parameters
LEVERAGE = 20
RISK_PER_TRADE_PCT = 0.02 # 2% of equity
STOP_LOSS_PCT = 0.01      # 1% price move against us
TAKE_PROFIT_PCT = 0.03    # 3% price move in our favor (3:1 Reward/Risk)

class LeveragedRunner:
    def __init__(self, broker: BybitBroker, symbol: str):
        self.broker = broker
        self.symbol = symbol.upper()
        
        # Engines
        self.regime_filter = TickRegimeFilter(window_size=1000, volatility_multiplier=2.0)
        self.momentum_engine = TickMomentumEngine(window_size=50)
        self.stat_arb_engine = ApexTransformerModel(
            name="stat_arb_heuristic", 
            feature_columns=['z_score', 'z_score_velocity', 'beta_slope', 'spread_volatility']
        )
        
        self.is_running = False
        
    async def _calculate_position_size(self, current_price: float) -> float:
        """
        Calculate quantity based on 2% equity risk.
        If we lose 1% in price, we lose 2% of equity.
        """
        balance = await self.broker.get_wallet_balance()
        if balance < 10.0:
            logger.warning("insufficient_funds", balance=balance)
            return 0.0
            
        risk_amount = balance * RISK_PER_TRADE_PCT
        # Since Stop Loss is 1% away, we can trade (risk_amount / 0.01) worth of asset
        position_value = risk_amount / STOP_LOSS_PCT
        
        qty = position_value / current_price
        
        # Format qty to 3 decimal places (common for Bybit altcoins)
        return round(qty, 3)

    async def execute_trade(self, side: str, current_price: float):
        """Execute a trade with strict risk management parameters."""
        logger.info("triggering_execution", symbol=self.symbol, side=side)
        
        # 1. Check if already in position
        pos = await self.broker.get_position(self.symbol)
        if pos:
            logger.info("already_in_position", position=pos)
            return
            
        # 2. Set Leverage
        await self.broker.set_leverage(self.symbol, LEVERAGE)
        
        # 3. Calculate Size
        qty = await self._calculate_position_size(current_price)
        if qty <= 0:
            return
            
        # 4. Calculate Stops
        if side.upper() == "BUY":
            sl_price = current_price * (1 - STOP_LOSS_PCT)
            tp_price = current_price * (1 + TAKE_PROFIT_PCT)
        else:
            sl_price = current_price * (1 + STOP_LOSS_PCT)
            tp_price = current_price * (1 - TAKE_PROFIT_PCT)
            
        # Round prices (Bybit usually requires 2-4 decimal places)
        sl_price = round(sl_price, 4)
        tp_price = round(tp_price, 4)
        
        logger.info(
            "sending_order", 
            symbol=self.symbol, qty=qty, 
            leverage=LEVERAGE, sl=sl_price, tp=tp_price
        )
        
        # 5. Place Order
        order_id = await self.broker.place_order_with_stops(
            symbol=self.symbol,
            side=side,
            qty=qty,
            take_profit=tp_price,
            stop_loss=sl_price
        )
        
        if order_id:
            logger.info("trade_executed", order_id=order_id)
            
    def process_tick(self, price: float, qty: float, is_buyer_maker: bool):
        """Simulate the live websocket tick processor."""
        
        # 1. Update Regime
        self.regime_filter.update(price, qty, is_buyer_maker)
        if not self.regime_filter.is_warmed_up():
            return
            
        regime_state = self.regime_filter.get_state()
        
        # 2. Strategy Router
        signal = None
        
        if regime_state.regime == "TRENDING":
            # Pass to Momentum Engine
            self.momentum_engine.update(price, qty, is_buyer_maker)
            if self.momentum_engine.is_warmed_up():
                state = self.momentum_engine.get_state()
                if state.ofi > 0.8: signal = "BUY"
                elif state.ofi < -0.8: signal = "SELL"
                
        elif regime_state.regime == "MEAN_REVERTING":
            # For pure StatArb, we need a 2nd asset (pairs trading).
            # If we are trading a single asset via LeveragedRunner, we can use 
            # Mean Reversion heuristics directly on price z-score, 
            # but for this script we will log and bypass.
            pass
            
        # 3. Execution Phase
        if signal:
            # We must schedule the async execution if we are in a sync loop,
            # or await it if we are async.
            asyncio.create_task(self.execute_trade(signal, price))

async def simulate_live_session():
    """Run a simulated trading session against the Bybit Testnet using dummy ticks."""
    print("=" * 60)
    print("  APEX HIGH-LEVERAGE FUTURES ENGINE")
    print("=" * 60)
    
    # Use dummy keys for safety during the structural verification
    broker = BybitBroker(
        api_key="dummy_api_key",
        api_secret="dummy_api_secret",
        use_testnet=True
    )
    
    symbol = "SOLUSDT"
    runner = LeveragedRunner(broker, symbol)
    
    print(f"  Target: {symbol}")
    print(f"  Leverage: {LEVERAGE}x")
    print(f"  Risk Per Trade: {RISK_PER_TRADE_PCT*100}%")
    print("=" * 60)
    print("  Warming up regime filters...")
    
    # Feed 1000 dummy ticks to warm up the filter
    import random
    base_price = 140.0
    for i in range(1010):
        # Create a trending market artificially to trigger the Momentum Engine
        if i > 900: base_price += random.uniform(0.1, 0.5) 
        else: base_price += random.uniform(-0.1, 0.1)
            
        is_buyer_maker = random.choice([True, False])
        if i > 950: is_buyer_maker = False # Strong buy volume
        
        runner.process_tick(base_price, random.uniform(1, 10), is_buyer_maker)
        
    print("  Simulation complete. Check logs for order intents.")
    await broker.close()

if __name__ == "__main__":
    asyncio.run(simulate_live_session())
