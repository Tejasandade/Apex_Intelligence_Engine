import asyncio
import time
from typing import Any
from src.core.logging import get_logger
from src.data.providers.binance_ws import BinanceWebSocket
from src.models.stat_arb_engine import StatArbEngine
from src.core.events import event_bus
from src.execution.paper_broker import PaperBroker
from src.data.storage.clickhouse_client import ClickHouseClient
from src.models.transformer_model import ApexTransformerModel
from src.models.tick_regime_filter import TickRegimeFilter, TickRegime

logger = get_logger("apex.engine.stat_arb_runner")

class ExecutionState:
    IDLE = "IDLE"
    WAITING_LEG_1_ENTRY = "WAITING_LEG_1_ENTRY"
    IN_POSITION = "IN_POSITION"
    WAITING_EXIT = "WAITING_EXIT"

class StatArbRunner:
    def __init__(self, asset_a: str, asset_b: str):
        self.asset_a = asset_a.lower()
        self.asset_b = asset_b.lower()
        self.engine = StatArbEngine(self.asset_a, self.asset_b)
        
        self.ws = BinanceWebSocket(symbols=[self.asset_a, self.asset_b])
        self.ws.on_tick = self.on_tick
        
        # Paper Broker for testing execution
        self.broker = PaperBroker(initial_balance=10000.0, maker_fee_bps=0.0, taker_fee_bps=5.0)
        
        self.state = ExecutionState.IDLE
        self.leg_1_order_id = None
        self.leg_2_order_id = None
        
        # Current active spread trade direction
        self.active_direction = None # "LONG_A_SHORT_B" or "SHORT_A_LONG_B"
        
        # ClickHouse Client
        self.ch_client = ClickHouseClient()
        
        # Regime Filter — watches BTC as the market's pulse
        self.regime_filter = TickRegimeFilter(window=500)
        self._regime_log_counter = 0
        
        # Load AI Overlay (if it exists)
        self.ai_model = ApexTransformerModel.load("stat_arb_transformer", "models/")
        if self.ai_model:
            logger.info("ai_overlay_active", model="stat_arb_transformer")
        else:
            logger.info("ai_overlay_inactive", reason="Model not found on disk")

    async def setup(self):
        await self.broker.connect()
        try:
            self.ch_client.connect()
            await self.ch_client.start_flusher()
        except Exception as e:
            logger.warning("clickhouse_setup_failed", error=str(e))
        logger.info("stat_arb_runner_setup", asset_a=self.asset_a, asset_b=self.asset_b)

    async def on_tick(self, symbol: str, tick: dict[str, Any]):
        """Called asynchronously on every single tick (trade or quote)"""
        if tick.get("type") == "trade":
            # 1. Update Broker Prices (Crucial for Paper Broker fills)
            self.broker.update_price(symbol, tick["price"], None)
            
            # 2. Persist to ClickHouse
            self.ch_client.queue_trade(tick)
            
            # 3. Feed the Regime Filter (uses Asset A as the market pulse)
            if symbol == self.asset_a:
                self.regime_filter.add_tick(tick["price"])
                # Log regime diagnostics every 1000 ticks
                self._regime_log_counter += 1
                if self._regime_log_counter % 1000 == 0:
                    diag = self.regime_filter.get_diagnostics()
                    logger.info("regime_status", pair=f"{self.asset_a}/{self.asset_b}", **diag)
            
            # 4. Feed the StatArb Engine
            self.engine.add_tick(symbol, tick["price"], tick["timestamp"])
            
            # 5. Recalculate Spread
            z_score = self.engine.update_spread()
            if z_score is not None:
                await self.evaluate_state(z_score)
        
        elif tick.get("type") == "quote":
            # Persist quote to ClickHouse
            self.ch_client.queue_quote(tick)

    async def evaluate_state(self, z_score: float):
        """The Execution State Machine for Leg Risk Management"""
        if self.state == ExecutionState.IDLE:
            signal = self.engine.get_signal()
            if signal in ("LONG_A_SHORT_B", "SHORT_A_LONG_B"):
                # --- LAYER 1: REGIME FILTER ---
                # Block all StatArb entries during trending or high-stress markets
                regime = self.regime_filter.current_regime
                if regime == TickRegime.TRENDING:
                    return  # Market is trending, StatArb would get crushed
                if regime == TickRegime.HIGH_STRESS:
                    return  # Flash crash / liquidation cascade, stay flat
                
                # --- LAYER 2: AI ALPHA OVERLAY ---
                if self.ai_model and self.engine.current_features is not None:
                    ai_prob = self.ai_model.predict(self.engine.current_features)
                    if ai_prob < 0.85:
                        # AI says the spread is drifting, DO NOT TRADE.
                        return
                    else:
                        logger.info("ai_approved_trade", probability=ai_prob, regime=regime.value)
                
                if signal == "LONG_A_SHORT_B":
                    await self.execute_entry("LONG", "SHORT")
                else:
                    await self.execute_entry("SHORT", "LONG")
                
        elif self.state == ExecutionState.IN_POSITION:
            signal = self.engine.get_signal()
            if signal == "FLAT":
                await self.execute_exit()

    async def execute_entry(self, dir_a: str, dir_b: str):
        """Executes Leg 1 as Maker, dynamically sizes Leg 2 as Taker"""
        logger.info("stat_arb_entry_triggered", dir_a=dir_a, dir_b=dir_b, z_score=self.engine.current_z_score)
        self.state = ExecutionState.WAITING_LEG_1_ENTRY
        self.active_direction = f"{dir_a}_A_{dir_b}_B"
        
        beta = self.engine.current_beta
        
        # Risk Sizing: Flat $1000 notional per spread
        current_price_a = await self.broker.get_current_price(self.asset_a)
        qty_a = 1000.0 / current_price_a
        
        # LEG 1: Send Asset A as Maker (Limit)
        logger.info("routing_leg_1_maker", symbol=self.asset_a, side=dir_a, qty=qty_a)
        order_a = await self.broker.place_order(
            symbol=self.asset_a,
            side=dir_a,
            quantity=qty_a,
            order_type="LIMIT",
            price=current_price_a # Place at best bid/ask
        )
        
        if not order_a:
            logger.error("leg_1_rejected")
            self.state = ExecutionState.IDLE
            return

        # SIMULATE: Wait for Leg 1 Fill (Paper broker fills limits instantly for now)
        await asyncio.sleep(0.01) 
        
        # Calculate exactly how much filled
        filled_qty_a = order_a.filled_quantity
        if filled_qty_a <= 0:
            logger.warning("leg_1_missed_fill")
            await self.broker.cancel_order(self.asset_a, order_a.order_id)
            self.state = ExecutionState.IDLE
            return

        # LEG 2: Dynamically size Asset B based on Beta and EXACT fill of Leg 1
        qty_b = filled_qty_a * beta
        
        logger.info("routing_leg_2_taker", symbol=self.asset_b, side=dir_b, qty=qty_b, beta=beta)
        order_b = await self.broker.place_order(
            symbol=self.asset_b,
            side=dir_b,
            quantity=qty_b,
            order_type="MARKET",
            price=0.0
        )
        
        self.state = ExecutionState.IN_POSITION
        logger.info("spread_position_locked", spread_beta=beta)

    async def execute_exit(self):
        """Liquidates both legs at market to close the spread"""
        logger.info("stat_arb_exit_triggered", z_score=self.engine.current_z_score)
        self.state = ExecutionState.WAITING_EXIT
        
        positions = await self.broker.get_positions()
        for pos in positions:
            if pos.side.value != "FLAT":
                close_side = "SELL" if pos.side.value == "LONG" else "BUY"
                await self.broker.place_order(
                    symbol=pos.symbol,
                    side=close_side,
                    quantity=pos.quantity,
                    order_type="MARKET",
                    price=0.0
                )
                
        self.state = ExecutionState.IDLE
        self.active_direction = None
        
        # Log PnL
        balance = await self.broker.get_balance()
        logger.info("spread_closed", total_equity=balance.total_equity)

    async def run(self):
        await self.setup()
        await self.ws.connect()
