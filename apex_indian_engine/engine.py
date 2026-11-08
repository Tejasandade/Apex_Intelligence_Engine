"""
Apex Indian Engine — Autonomous Execution Core
==============================================
Coordinates authentication, scrip master indexing, candle ingestion,
multi-advisor signal evaluation (with 23-feature XGBoost & OI Shield),
risk containment, and order execution.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import pyotp
from SmartApi import SmartConnect

from apex_indian_engine.config import EngineConfig, INDEX_SPECS
from apex_indian_engine.scrip_master import ScripMaster
from apex_indian_engine.greeks import GreeksEngine
from apex_indian_engine.risk_manager import IndianRiskManager
from apex_indian_engine.signal_engine import IndianSignalEngine, OptionSignal
from apex_indian_engine.executor import IndianOrderExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("apex.indian.engine")


class ApexIndianEngine:
    """
    Main Autonomous Engine for Indian Options (NIFTY & BANKNIFTY).
    """

    def __init__(self, mode: str = "PAPER", initial_capital: float = 100_000.0):
        self.config = EngineConfig(mode=mode)
        self.config.risk.initial_capital_inr = initial_capital
        
        self.scrip_master = ScripMaster(cache_path=self.config.scrip_cache_file)
        self.greeks_engine = GreeksEngine()
        self.risk_manager = IndianRiskManager(
            rules=self.config.risk,
            schedule=self.config.session,
            friction=self.config.friction,
            state_file=self.config.state_file,
        )
        self.executor = IndianOrderExecutor(mode=mode)
        
        self.signal_engines: dict[str, IndianSignalEngine] = {
            "NIFTY": IndianSignalEngine("NIFTY"),
            "BANKNIFTY": IndianSignalEngine("BANKNIFTY"),
        }

        self.smart_api: Optional[SmartConnect] = None
        self.feed_token: Optional[str] = None
        self.jwt_token: Optional[str] = None
        self.is_running = False

    def authenticate_angel_one(self) -> bool:
        """Fully automated, headless login via TOTP."""
        api_key = os.getenv("ANGEL_API_KEY")
        client_id = os.getenv("ANGEL_CLIENT_ID")
        pin = os.getenv("ANGEL_PIN")
        totp_key = os.getenv("ANGEL_TOTP_KEY")

        if not all([api_key, client_id, pin, totp_key]):
            logger.error("Angel One credentials missing in .env")
            return False

        try:
            logger.info(f"Connecting to Angel One for client: {client_id}")
            self.smart_api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_key).now()
            sess = self.smart_api.generateSession(client_id, pin, totp)
            
            if sess.get("status"):
                self.jwt_token = sess["data"]["jwtToken"]
                self.feed_token = sess["data"]["feedToken"]
                logger.info(f"Angel One Auth SUCCESS for client: {client_id}")
                self.executor.api = self.smart_api
                return True
            else:
                logger.error(f"Angel One Auth FAILED: {sess.get('message')}")
                return False
        except Exception as e:
            logger.error(f"Angel One Auth Exception: {e}")
            return False

    def initialize(self) -> bool:
        """Download scrip master and load initial market state."""
        print("=" * 70)
        print("  🇮🇳 APEX INDIAN INTELLIGENCE ENGINE — OPTIONS CORE v1.0")
        print(f"  Mode: {self.config.mode} | Capital: ₹{self.risk_manager.capital_inr:,.2f}")
        print("=" * 70)

        # 1. Connect Angel One
        if not self.authenticate_angel_one():
            if self.config.mode == "LIVE":
                logger.critical("Cannot run in LIVE mode without Angel One connection.")
                return False
            else:
                logger.warning("Running in PAPER mode without live Angel One broker session.")

        # 2. Load Scrip Master
        print("  ⏳ Indexing NSE / NFO Option Contracts...")
        if not self.scrip_master.load_or_download():
            logger.critical("Failed to load or download Scrip Master.")
            return False

        nifty_exp = self.scrip_master.get_nearest_expiry("NIFTY")
        bn_exp = self.scrip_master.get_nearest_expiry("BANKNIFTY")
        print(f"  ✓ Active NIFTY Weekly Expiry:     {nifty_exp}")
        print(f"  ✓ Active BANKNIFTY Weekly Expiry: {bn_exp}")
        print(f"  ✓ Risk Rules: Max Daily Loss 3% (₹{self.risk_manager.capital_inr * 0.03:,.0f}) | Max 4 Trades/Day")
        print("=" * 70)
        return True

    def process_candle(
        self,
        index_name: str,
        candle: dict[str, Any],
        oi_chain: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Ingests a 5m candle, runs multi-advisor council + ML + OI shield,
        and triggers option execution if conditions align.
        """
        idx = index_name.upper()
        cur_spot = float(candle.get("close", 0.0))
        cur_vwap = float(candle.get("vwap", cur_spot))

        # 1. Add candle to signal engine
        sig_engine = self.signal_engines.get(idx)
        if not sig_engine:
            return None
        sig_engine.add_candle(candle)

        # 2. Check risk gate before evaluating signals
        skip_time = (self.config.mode == "PAPER")
        can_trade, gate_reason = self.risk_manager.can_trade(skip_time_check=skip_time)
        if not can_trade:
            logger.debug(f"Trading gate locked: {gate_reason}")
            return None

        # 3. Check if already in an active trade
        if self.executor.has_open_position():
            return None

        # 4. Evaluate 4-Advisor Council + ML + OI Shield
        res = sig_engine.evaluate_signal(cur_spot, cur_vwap, oi_chain=oi_chain)
        if not res:
            return None

        direction, opt_type, conviction, score, notes, oi_metrics = res
        logger.info(
            f"⚡ COUNCIL CONSENSUS on {idx}: {direction} ({opt_type}) | "
            f"Score: {score:+.2f} | Conviction: {conviction*100:.0f}%"
        )
        for note in notes:
            logger.info(f"   ↳ {note}")

        # 5. Strike Resolution
        strike = self.scrip_master.get_atm_strike(idx, cur_spot)
        expiry = self.scrip_master.get_nearest_expiry(idx)
        contract = self.scrip_master.get_option_contract(idx, expiry, strike, opt_type)

        if not contract:
            logger.error(f"Could not resolve option contract for {idx} {strike} {opt_type} {expiry}")
            return None

        # 6. Option Premium Estimation (or live LTP)
        # BSM calculation with 3-day proxy to expiry
        bs = self.greeks_engine.calc_black_scholes(
            S=cur_spot,
            K=strike,
            T=3.0 / 365.0,
            r=0.07,
            sigma=0.14,
            option_type=opt_type,
        )
        est_premium = max(bs.price, 45.0)

        # Stop loss points in premium (25 pts for BankNifty, 15 pts for Nifty)
        sl_pts = 25.0 if "BANK" in idx else 15.0

        # 7. Sizing via Risk Manager
        lot_sz = int(contract.get("lot_size", 30))
        sizing = self.risk_manager.calculate_position_size(
            index_name=idx,
            option_premium=est_premium,
            sl_points=sl_pts,
            lot_size=lot_sz,
            conviction=conviction,
        )

        if sizing["lots"] <= 0:
            logger.warning(f"Position sizing rejected: {sizing.get('reason')}")
            return None

        # 8. Execute Order
        trade = self.executor.open_option_trade(
            index_name=idx,
            symbol=str(contract.get("symbol", "")),
            token=str(contract.get("token", "")),
            option_type=opt_type,
            current_option_price=est_premium,
            lots=sizing["lots"],
            quantity=sizing["quantity"],
            sl_points=sl_pts,
            tp_multiplier=2.5,
        )

        if trade:
            return {
                "action": "TRADE_OPENED",
                "trade_id": trade.trade_id,
                "symbol": trade.symbol,
                "direction": trade.direction,
                "lots": trade.lots,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "sl_price": trade.sl_price,
                "tp1_price": trade.tp1_price,
                "tp2_price": trade.tp2_price,
                "score": score,
                "conviction": conviction,
            }

        return None

    def process_tick(
        self,
        current_option_price: float,
        underlying_reversed: bool = False,
    ) -> Optional[dict[str, Any]]:
        """
        Updates active position on new tick, handles trailing stops & theta harvesting.
        """
        if not self.executor.has_open_position():
            return None

        trade = self.executor.active_trade
        closed = self.executor.update_ticks(
            current_option_price=current_option_price,
            underlying_momentum_reversed=underlying_reversed,
        )

        if closed:
            # Settle PnL & friction with Risk Manager
            self.risk_manager.on_trade_closed(
                gross_pnl_inr=closed["gross_pnl"],
                buy_price=trade.entry_price,
                sell_price=closed["exit_price"],
                qty=trade.quantity,
                reason=closed["exit_reason"],
            )
            return closed

        return None

    def get_hud_metrics(self) -> dict[str, Any]:
        """Provides status metrics for Web/Terminal HUD."""
        open_pos = None
        if self.executor.has_open_position():
            t = self.executor.active_trade
            open_pos = {
                "symbol": t.symbol,
                "option_type": t.option_type,
                "direction": t.direction,
                "lots": t.lots,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "sl_price": t.sl_price,
                "tp1_price": t.tp1_price,
                "tp2_price": t.tp2_price,
                "unrealized_pnl": t.unrealized_pnl,
            }

        return {
            "mode": self.config.mode,
            "status": self.risk_manager.status,
            "capital_inr": self.risk_manager.capital_inr,
            "daily_realized_pnl": self.risk_manager.daily_realized_pnl,
            "daily_unrealized_pnl": self.risk_manager.daily_unrealized_pnl,
            "daily_friction_cost": self.risk_manager.daily_friction_cost,
            "net_daily_pnl": self.risk_manager.daily_realized_pnl + self.risk_manager.daily_unrealized_pnl,
            "trades_today": self.risk_manager.trades_today,
            "max_trades": self.config.risk.max_trades_per_day,
            "halt_reason": self.risk_manager.halt_reason,
            "open_position": open_pos,
        }

    async def run_loop(self):
        """Main asynchronous event loop for Indian Options."""
        self.is_running = True
        logger.info("Indian Engine event loop started.")

        while self.is_running:
            try:
                now_dt = datetime.now()
                now_time = now_dt.time()

                # Check 15:15 IST Mandatory Square-Off
                if now_time >= self.config.session.hard_flatten_time:
                    if self.executor.has_open_position():
                        logger.info("Auto-executing Mandatory 15:15 IST Square-Off...")
                        trade = self.executor.active_trade
                        cur_px = trade.entry_price  # or last market tick
                        closed = self.executor.close_trade(cur_px, reason="MANDATORY 15:15 IST SQUARE-OFF")
                        self.risk_manager.on_trade_closed(
                            gross_pnl_inr=closed["gross_pnl"],
                            buy_price=trade.entry_price,
                            sell_price=cur_px,
                            qty=trade.quantity,
                            reason="15:15 IST AUTO FLATTEN",
                        )
                    await asyncio.sleep(10)
                    continue

                # Normal cycle
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Indian Engine loop error: {e}")
                await asyncio.sleep(2.0)
