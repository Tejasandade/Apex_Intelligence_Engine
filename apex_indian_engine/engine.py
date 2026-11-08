"""
Apex Indian Engine — Autonomous Execution Core
==============================================
Coordinates authentication, scrip master indexing, candle ingestion,
multi-advisor signal evaluation, risk containment, and order execution.
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
from apex_indian_engine.signal_engine import IndianSignalEngine
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
                logger.warn("Running in PAPER mode without live Angel One broker session.")

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
        logger.info("indian_engine_loop_started")

        while self.is_running:
            try:
                now_dt = datetime.now()
                now_time = now_dt.time()

                # Check 15:15 IST Mandatory Square-Off
                if now_time >= self.config.session.hard_flatten_time:
                    if self.executor.has_open_position():
                        logger.info("indian_engine_hard_flatten_1515_ist")
                        trade = self.executor.active_trade
                        # Close at current market
                        cur_px = trade.entry_price # or live tick
                        closed = self.executor.close_trade(cur_px, reason="MANDATORY 15:15 IST SQUARE-OFF")
                        self.risk_manager.on_trade_closed(
                            gross_pnl_inr=closed["gross_pnl"],
                            buy_price=trade.entry_price,
                            sell_price=cur_px,
                            qty=trade.quantity,
                            reason="15:15 IST AUTO FLATTEN"
                        )
                    await asyncio.sleep(10)
                    continue

                # Normal cycle
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error("indian_engine_loop_error", error=str(e))
                await asyncio.sleep(2.0)
