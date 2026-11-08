"""
Apex Indian Engine — Autonomous Execution Core
==============================================
Coordinates authentication, scrip master indexing, WebSocket tick ingestion,
India VIX macro volatility conditioning, dynamic strike selection (ATM vs ITM),
3-minute NSE Open Interest tracking, risk containment, marketable limit routing,
dedicated Telegram alerts, and live Terminal ASCII HUD.
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
from apex_indian_engine.vix_regime import VixRegimeManager, VixParameters
from apex_indian_engine.strike_selector import DynamicStrikeSelector, SelectedContract
from apex_indian_engine.oi_tracker import OptionChainOITracker
from apex_indian_engine.ws_streamer import IndianWebSocketStreamer
from apex_indian_engine.telegram_bot import IndianTelegramNotifier
from apex_indian_engine.hud import IndianTerminalHUD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("apex.indian.engine")


class ApexIndianEngine:
    """
    Production-Grade Autonomous Engine for Indian Options (NIFTY & BANKNIFTY).
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

        # Alpha & Protection Systems
        self.vix_manager = VixRegimeManager()
        self.strike_selector = DynamicStrikeSelector(self.scrip_master, self.greeks_engine)
        self.oi_trackers: dict[str, OptionChainOITracker] = {
            "NIFTY": OptionChainOITracker("NIFTY"),
            "BANKNIFTY": OptionChainOITracker("BANKNIFTY"),
        }

        # Telegram, HUD & WebSocket Streamer
        self.telegram = IndianTelegramNotifier()
        self.hud = IndianTerminalHUD()
        self.ws_streamer: Optional[IndianWebSocketStreamer] = None

        # Wire Telegram command hooks
        self.telegram.cmd_get_status = self._format_telegram_status
        self.telegram.cmd_close_position = self._handle_telegram_close
        self.telegram.cmd_pause_trading = self._handle_telegram_pause
        self.telegram.cmd_resume_trading = self._handle_telegram_resume

        self.smart_api: Optional[SmartConnect] = None
        self.feed_token: Optional[str] = None
        self.jwt_token: Optional[str] = None
        self.is_running = False
        self.is_paused = False

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
                
                # Pass client to OI trackers
                for tracker in self.oi_trackers.values():
                    tracker.api = self.smart_api

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
        Ingests a 5m candle, runs multi-advisor council + ML + VIX regime + OI shield,
        and triggers option execution if conditions align.
        """
        if self.is_paused:
            logger.debug("Trading paused by user via Telegram command.")
            return None

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

        # 4. India VIX Regime Conditioning
        vix_params = self.vix_manager.get_parameters()
        sig_engine.min_consensus_threshold = vix_params.consensus_threshold

        # 5. Evaluate 4-Advisor Council + ML + OI Shield
        res = sig_engine.evaluate_signal(cur_spot, cur_vwap, oi_chain=oi_chain)
        if not res:
            return None

        direction, opt_type, conviction, score, notes, oi_metrics = res
        logger.info(
            f"⚡ COUNCIL CONSENSUS on {idx}: {direction} ({opt_type}) | "
            f"Score: {score:+.2f} | Conviction: {conviction*100:.0f}% | VIX: {vix_params.regime_name}"
        )
        for note in notes:
            logger.info(f"   ↳ {note}")

        # 6. Dynamic Strike & Delta Selection (ATM vs. 1-Strike ITM)
        selected: Optional[SelectedContract] = self.strike_selector.select_best_strike(
            index_name=idx,
            spot_price=cur_spot,
            option_type=opt_type,
            vix_params=vix_params,
            conviction=conviction,
        )
        if not selected:
            logger.error(f"Could not resolve optimal strike for {idx} {opt_type}")
            return None

        # Stop loss points in premium
        sl_pts = vix_params.sl_points_banknifty if "BANK" in idx else vix_params.sl_points_nifty

        # 7. Sizing via Risk Manager with VIX scaling
        sizing = self.risk_manager.calculate_position_size(
            index_name=idx,
            option_premium=selected.est_premium,
            sl_points=sl_pts,
            lot_size=selected.lot_size,
            conviction=conviction,
        )

        # Apply VIX size multiplier (e.g. 0.50 on Extreme VIX)
        adj_lots = max(1, int(sizing["lots"] * vix_params.size_multiplier))
        adj_qty = adj_lots * selected.lot_size

        # 8. Execute Marketable Limit Order
        trade = self.executor.open_option_trade(
            index_name=idx,
            symbol=selected.symbol,
            token=selected.token,
            option_type=opt_type,
            current_option_price=selected.est_premium,
            lots=adj_lots,
            quantity=adj_qty,
            sl_points=sl_pts,
            tp_multiplier=vix_params.tp_multiplier,
        )

        if trade:
            # Subscribe live ticks if streamer active
            if self.ws_streamer:
                self.ws_streamer.subscribe_option_contract(trade.token, trade.symbol)

            # Send Telegram Alert
            self.telegram.notify_trade_opened(
                symbol=trade.symbol,
                index_name=idx,
                option_type=opt_type,
                strike=selected.strike,
                lots=trade.lots,
                qty=trade.quantity,
                entry_px=trade.entry_price,
                sl_px=trade.sl_price,
                tp1_px=trade.tp1_price,
                tp2_px=trade.tp2_price,
                score=score,
                conviction=conviction,
                vix=self.vix_manager.current_vix,
                regime=vix_params.regime_name,
                strike_type=selected.strike_type,
            )

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
                "strike_type": selected.strike_type,
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
        prev_tier = trade.trailing_tier

        closed = self.executor.update_ticks(
            current_option_price=current_option_price,
            underlying_momentum_reversed=underlying_reversed,
        )

        # Notify trailing stop ratchet if upgraded
        if self.executor.has_open_position() and trade.trailing_tier > prev_tier:
            sl_dist = trade.entry_price - trade.sl_price
            r_mult = (current_option_price - trade.entry_price) / max(sl_dist, 0.01)
            reason = "Break-Even Lock" if trade.trailing_tier == 2 else "Risk Reduction"
            self.telegram.notify_sl_ratchet(
                symbol=trade.symbol,
                tier=trade.trailing_tier,
                new_sl=trade.sl_price,
                r_mult=r_mult,
                reason=reason,
            )

        if closed:
            # Unsubscribe option ticks
            if self.ws_streamer:
                self.ws_streamer.unsubscribe_option_contract(trade.token)

            # Settle PnL & friction with Risk Manager
            self.risk_manager.on_trade_closed(
                gross_pnl_inr=closed["gross_pnl"],
                buy_price=trade.entry_price,
                sell_price=closed["exit_price"],
                qty=trade.quantity,
                reason=closed["exit_reason"],
            )

            # Notify Telegram
            net_cash = closed["gross_pnl"] - self.risk_manager.daily_friction_cost
            self.telegram.notify_trade_closed(
                symbol=trade.symbol,
                entry_px=trade.entry_price,
                exit_px=closed["exit_price"],
                qty=trade.quantity,
                gross_pnl=closed["gross_pnl"],
                friction_cost=self.risk_manager.daily_friction_cost,
                net_pnl=net_cash,
                reason=closed["exit_reason"],
                today_net_pnl=self.risk_manager.daily_realized_pnl,
                capital_inr=self.risk_manager.capital_inr,
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

        vix_params = self.vix_manager.get_parameters()
        bn_tracker = self.oi_trackers["BANKNIFTY"]
        snap = bn_tracker.get_latest_snapshot()

        pcr = snap.pcr if snap else 1.0
        flow = snap.flow_sentiment if snap else "BALANCED"
        call_wall = snap.max_call_wall if snap else 0.0
        put_wall = snap.max_put_wall if snap else 0.0

        # Spot prices from signal engine or streamer
        bn_candles = self.signal_engines["BANKNIFTY"].candles_5m
        bn_spot = bn_candles[-1]["close"] if bn_candles else 0.0
        bn_vwap = bn_candles[-1].get("vwap", bn_spot) if bn_candles else 0.0

        nifty_candles = self.signal_engines["NIFTY"].candles_5m
        nifty_spot = nifty_candles[-1]["close"] if nifty_candles else 0.0
        nifty_vwap = nifty_candles[-1].get("vwap", nifty_spot) if nifty_candles else 0.0

        return {
            "mode": self.config.mode,
            "status": self.risk_manager.status if not self.is_paused else "PAUSED",
            "capital_inr": self.risk_manager.capital_inr,
            "daily_realized_pnl": self.risk_manager.daily_realized_pnl,
            "daily_unrealized_pnl": self.risk_manager.daily_unrealized_pnl,
            "daily_friction_cost": self.risk_manager.daily_friction_cost,
            "net_daily_pnl": self.risk_manager.daily_realized_pnl + self.risk_manager.daily_unrealized_pnl,
            "trades_today": self.risk_manager.trades_today,
            "max_trades": self.config.risk.max_trades_per_day,
            "halt_reason": self.risk_manager.halt_reason,
            "vix": self.vix_manager.current_vix,
            "vix_regime": vix_params.regime_name,
            "banknifty_spot": bn_spot,
            "banknifty_vwap": bn_vwap,
            "nifty_spot": nifty_spot,
            "nifty_vwap": nifty_vwap,
            "pcr": pcr,
            "oi_flow": flow,
            "call_wall": call_wall,
            "put_wall": put_wall,
            "open_position": open_pos,
        }

    # ── Telegram Remote Control Handlers ──
    def _format_telegram_status(self) -> str:
        m = self.get_hud_metrics()
        sign = "+" if m["net_daily_pnl"] >= 0 else ""
        pos_str = "FLAT"
        if m["open_position"]:
            p = m["open_position"]
            pos_str = f"<b>{p['symbol']}</b> (PnL: ₹{p['unrealized_pnl']:+,.2f})"

        return (
            f"<b>🇮🇳 [APEX INDIAN ENGINE] STATUS REPORT</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Status:</b> {m['status']} | <b>Mode:</b> {m['mode']}\n"
            f"<b>Capital:</b> ₹{m['capital_inr']:,.2f}\n"
            f"<b>Net Daily PnL:</b> <b>{sign}₹{m['net_daily_pnl']:,.2f}</b>\n"
            f"<b>Trades Today:</b> {m['trades_today']}/{m['max_trades']}\n"
            f"<b>India VIX:</b> {m['vix']:.2f} ({m['vix_regime']})\n"
            f"<b>BankNifty PCR:</b> {m['pcr']:.2f} ({m['oi_flow']})\n"
            f"<b>Active Position:</b> {pos_str}\n"
            f"<b>Time:</b> {time.strftime('%H:%M:%S IST')}"
        )

    def _handle_telegram_close(self) -> str:
        if not self.executor.has_open_position():
            return "No active option position to close."
        trade = self.executor.active_trade
        cur_px = trade.highest_fav_price or trade.entry_price
        closed = self.executor.close_trade(cur_px, reason="MANUAL TELEGRAM /CLOSE COMMAND")
        self.risk_manager.on_trade_closed(
            gross_pnl_inr=closed["gross_pnl"],
            buy_price=trade.entry_price,
            sell_price=cur_px,
            qty=trade.quantity,
            reason="TELEGRAM REMOTE CLOSE",
        )
        return f"Closed {trade.symbol} at ₹{cur_px:.2f}. PnL: ₹{closed['gross_pnl']:+,.2f}."

    def _handle_telegram_pause(self) -> str:
        self.is_paused = True
        return "⏸️ Indian Engine trading PAUSED. No new trades will be opened."

    def _handle_telegram_resume(self) -> str:
        self.is_paused = False
        return "▶️ Indian Engine trading RESUMED. Scanning for setups."

    async def run_loop(self):
        """Main asynchronous event loop for Indian Options."""
        self.is_running = True
        logger.info("Indian Engine event loop started.")

        # Start Telegram Command Listener
        self.telegram.start_polling()

        # Start WebSocket Streamer if Angel credentials active
        if self.jwt_token and self.feed_token and self.smart_api:
            try:
                self.ws_streamer = IndianWebSocketStreamer(
                    auth_token=self.jwt_token,
                    api_key=os.getenv("ANGEL_API_KEY", ""),
                    client_id=os.getenv("ANGEL_CLIENT_ID", ""),
                    feed_token=self.feed_token,
                )
                
                # Wire callbacks
                async def _on_vix(vix_val: float):
                    self.vix_manager.update_vix(vix_val)

                async def _on_candle(sym: str, c_dict: dict[str, Any]):
                    if sym in ("NIFTY", "BANKNIFTY"):
                        self.process_candle(sym, c_dict)

                async def _on_tick(sym: str, px: float, vol: float):
                    if self.executor.has_open_position():
                        t = self.executor.active_trade
                        if t.symbol == sym:
                            self.process_tick(current_option_price=px)

                self.ws_streamer.on_vix_updated = _on_vix
                self.ws_streamer.on_candle_closed = _on_candle
                self.ws_streamer.on_tick = _on_tick
                self.ws_streamer.start(asyncio.get_running_loop())
            except Exception as e:
                logger.error(f"WebSocket streamer initialization error: {e}")

        last_hud_render = 0.0

        while self.is_running:
            try:
                now_dt = datetime.now()
                now_time = now_dt.time()

                # 1. Check 15:15 IST Mandatory Square-Off (Penalty prevention)
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

                # 2. Render Live Terminal HUD every 2 seconds
                if (time.time() - last_hud_render) >= 2.0:
                    metrics = self.get_hud_metrics()
                    hud_output = self.hud.render(metrics)
                    # Clear screen and render HUD cleanly
                    print("\033[H\033[J" + hud_output, flush=True)
                    last_hud_render = time.time()

                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Indian Engine loop error: {e}")
                await asyncio.sleep(2.0)
