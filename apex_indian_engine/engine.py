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
import json
import logging
import os
import sys
import time
from datetime import datetime, date, timedelta
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
from apex_indian_engine.web_dashboard import IndianWebServer
from apex_indian_engine.key_levels import InstitutionalLevelsTracker

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
        self.levels_trackers: dict[str, InstitutionalLevelsTracker] = {
            "NIFTY": InstitutionalLevelsTracker("NIFTY"),
            "BANKNIFTY": InstitutionalLevelsTracker("BANKNIFTY"),
        }

        # Telegram, HUD & WebSocket Streamer
        self.telegram = IndianTelegramNotifier()
        self.hud = IndianTerminalHUD()
        self.ws_streamer: Optional[IndianWebSocketStreamer] = None
        self.web_server: Optional[IndianWebServer] = None

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

        self._prev_oi_map: dict[str, dict[Any, int]] = {"NIFTY": {}, "BANKNIFTY": {}}
        self._latest_oi_chains: dict[str, list[dict[str, Any]]] = {"NIFTY": [], "BANKNIFTY": []}
        self.heavyweight_prices: dict[str, float] = {"1333": 0.0, "4963": 0.0, "2885": 0.0}
        self.heavyweight_baselines: dict[str, float] = {}
        self.candle_cache_file = Path("data/indian_candles_cache.json")

    def _get_heavyweights_info(self) -> dict[str, Any]:
        """Compute live Banking Heavyweight flow direction from session baselines."""
        hdfc = self.heavyweight_prices.get("1333", 0.0)
        icici = self.heavyweight_prices.get("4963", 0.0)
        rel = self.heavyweight_prices.get("2885", 0.0)
        flow = "NEUTRAL"

        changes: list[float] = []
        for tok in ("1333", "4963"):
            px = self.heavyweight_prices.get(tok, 0.0)
            base = self.heavyweight_baselines.get(tok, 0.0)
            if px > 0 and base > 0:
                changes.append(((px - base) / base) * 100.0)

        if len(changes) == 2:
            if changes[0] >= 0.25 and changes[1] >= 0.25:
                flow = "BULLISH"
            elif changes[0] <= -0.25 and changes[1] <= -0.25:
                flow = "BEARISH"
        elif changes:
            if changes[0] >= 0.35:
                flow = "BULLISH"
            elif changes[0] <= -0.35:
                flow = "BEARISH"

        return {
            "HDFCBANK": hdfc,
            "ICICIBANK": icici,
            "RELIANCE": rel,
            "BANKING_FLOW": flow,
        }

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
                self.strike_selector.api = self.smart_api
                
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

    def load_cached_candles(self) -> None:
        """Loads cached 5m candles from disk to ensure indicators are instantly warm on boot."""
        if not self.candle_cache_file.exists():
            return
        try:
            with open(self.candle_cache_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            for idx in ["BANKNIFTY", "NIFTY"]:
                candles = cached.get(idx, [])
                if candles:
                    sig_engine = self.signal_engines[idx]
                    sig_engine.candles_5m.clear()
                    for c in candles:
                        sig_engine.add_candle(c)
                    logger.info(f"Loaded {len(candles)} cached 5m candles for {idx} from disk.")
        except Exception as e:
            logger.debug(f"Failed to load candle cache: {e}")

    def save_cached_candles(self) -> None:
        """Saves current 5m candle history to disk."""
        try:
            self.candle_cache_file.parent.mkdir(parents=True, exist_ok=True)
            cached = {
                "BANKNIFTY": self.signal_engines["BANKNIFTY"].candles_5m[-100:],
                "NIFTY": self.signal_engines["NIFTY"].candles_5m[-100:],
            }
            with open(self.candle_cache_file, "w", encoding="utf-8") as f:
                json.dump(cached, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save candle cache: {e}")

    def warmup_historical_candles(self) -> None:
        """Preloads recent 5m historical candles from Angel One so indicators & ML models are ready immediately."""
        self.load_cached_candles()

        if not self.smart_api:
            return

        print("  ⏳ Synchronizing 5-Minute Historical Candles from Angel One...")
        today = datetime.now()
        from_date = (today - timedelta(days=3)).strftime("%Y-%m-%d 09:15")
        to_date = today.strftime("%Y-%m-%d %H:%M")

        configs = [
            ("BANKNIFTY", "99926009"),
            ("NIFTY", "99926000"),
        ]

        for idx, token in configs:
            sig_engine = self.signal_engines[idx]
            for attempt in range(3):
                try:
                    time.sleep(1.8 if attempt == 0 else 2.5)  # Respect Angel historical rate limit
                    params = {
                        "exchange": "NSE",
                        "symboltoken": token,
                        "interval": "FIVE_MINUTE",
                        "fromdate": from_date,
                        "todate": to_date,
                    }
                    res = self.smart_api.getCandleData(params)
                    data = res.get("data", []) if isinstance(res, dict) else []
                    if data:
                        sig_engine.candles_5m.clear()
                        # Compute cumulative intraday Typical-Price VWAP per NSE date
                        cur_day = ""
                        tp_sum = 0.0
                        pv_sum = 0.0
                        vol_sum = 0.0
                        bar_cnt = 0
                        for row in data:
                            ts_str = str(row[0])
                            day_str = ts_str[:10]
                            if day_str != cur_day:
                                cur_day = day_str
                                tp_sum = 0.0
                                pv_sum = 0.0
                                vol_sum = 0.0
                                bar_cnt = 0
                            h, l, c = float(row[2]), float(row[3]), float(row[4])
                            v = float(row[5])
                            tp = (h + l + c) / 3.0
                            bar_cnt += 1
                            tp_sum += tp
                            if v > 0:
                                pv_sum += tp * v
                                vol_sum += v
                                bar_vwap = round(pv_sum / vol_sum, 2)
                            else:
                                bar_vwap = round(tp_sum / bar_cnt, 2)
                            candle_dict = {
                                "timestamp": row[0],
                                "open": float(row[1]),
                                "high": h,
                                "low": l,
                                "close": c,
                                "volume": v,
                                "vwap": bar_vwap,
                            }
                            sig_engine.add_candle(candle_dict)
                        print(f"  ✓ Synchronized {len(data)} candles for {idx} (LTP: ₹{data[-1][4]:,.2f})")
                        break
                    else:
                        logger.warning(f"Candle sync returned empty for {idx} (attempt {attempt+1}): {res.get('message') if isinstance(res, dict) else res}")
                except Exception as e:
                    logger.warning(f"Candle sync attempt {attempt+1} for {idx}: {e}")
                    time.sleep(2.5)

            # If still fewer than 20 candles (e.g. rate limit), synthesize warm-up baseline
            if len(sig_engine.candles_5m) < 20:
                try:
                    mkt = self.smart_api.getMarketData("FULL", {"NSE": [token]})
                    fetched = mkt.get("data", {}).get("fetched", [])
                    if fetched:
                        item = fetched[0]
                        prev_close = float(item.get("close", 0.0))
                        open_px = float(item.get("open", prev_close))
                        high_px = float(item.get("high", max(prev_close, open_px)))
                        low_px = float(item.get("low", min(prev_close, open_px)))
                        ltp = float(item.get("ltp", open_px))

                        import numpy as np
                        prices = np.linspace(prev_close, ltp, 30)
                        for i, p in enumerate(prices):
                            c_dict = {
                                "timestamp": f"warmup_{i}",
                                "open": round(float(p), 2),
                                "high": round(float(max(p, high_px if i == 29 else p + 15)), 2),
                                "low": round(float(min(p, low_px if i == 29 else p - 15)), 2),
                                "close": round(float(p), 2),
                                "volume": 100000.0,
                                "vwap": round(float(p), 2),
                            }
                            sig_engine.add_candle(c_dict)
                        print(f"  ✓ Initialized {len(sig_engine.candles_5m)} baseline candles for {idx} (LTP: ₹{ltp:,.2f})")
                except Exception as e:
                    logger.debug(f"Baseline candle initialization error for {idx}: {e}")

        self.save_cached_candles()

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

        # 3. Warm up Historical Candles for Indicators & ML
        self.warmup_historical_candles()

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

        # 1. Add candle to signal engine & persist live cache (skip overwriting prod cache in test harness)
        sig_engine = self.signal_engines.get(idx)
        if not sig_engine:
            return None
        sig_engine.add_candle(candle)
        if str(self.risk_manager.state_file) == str(Path(self.config.state_file)):
            self.save_cached_candles()

        # 2. Check risk gate before evaluating signals (permit evaluation for high-potential candidate setups)
        skip_time = getattr(self, "ignore_time_checks", False)
        can_trade, gate_reason = self.risk_manager.can_trade(skip_time_check=skip_time, is_high_prob=True)
        if not can_trade:
            logger.debug(f"Trading gate locked: {gate_reason}")
            return None

        # 3. Check if already in an active trade
        if self.executor.has_open_position():
            return None

        # 4. India VIX Regime Conditioning
        vix_params = self.vix_manager.get_parameters()
        sig_engine.min_consensus_threshold = vix_params.consensus_threshold

        # 5. Evaluate Multi-Advisor Council + CPR/ORB Levels + OI Shield
        cur_lvl = self.levels_trackers[idx].calculate_from_candles(sig_engine.candles_5m, cur_spot)
        active_oi = oi_chain if oi_chain is not None else (self._latest_oi_chains.get(idx) or None)
        heavyweights_info = self._get_heavyweights_info()
        res = sig_engine.evaluate_signal(
            cur_spot, cur_vwap, oi_chain=active_oi, key_levels=cur_lvl, is_preview=False, heavyweights=heavyweights_info
        )
        if not res:
            return None

        direction, opt_type, conviction, score, notes, oi_metrics = res
        logger.info(
            f"⚡ COUNCIL CONSENSUS on {idx}: {direction} ({opt_type}) | "
            f"Score: {score:+.2f} | Conviction: {conviction*100:.0f}% | VIX: {vix_params.regime_name}"
        )
        for note in notes:
            logger.info(f"   ↳ {note}")

        # 6. Dynamic Risk Gate Verification based on Signal Potential
        is_high_prob = (conviction >= 0.70 or abs(score) >= 0.65)
        can_trade_final, gate_reason_final = self.risk_manager.can_trade(skip_time_check=skip_time, is_high_prob=is_high_prob)
        if not can_trade_final:
            logger.info(f"Dynamic Risk Gate Locked ({idx} {direction}): {gate_reason_final}")
            return None

        # 7. Dynamic Strike & Delta Selection (ATM vs. 1-Strike ITM)
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

        # Adaptive Risk-Proportional Stop Loss (No micro-pip whipsaws on expensive options)
        prem = float(selected.est_premium)
        if "BANK" in idx:
            # BankNifty option SL: 14% of premium, bounded 30-55 pts (keeps TP2 ~65-115 pts achievable!)
            sl_pts = max(30.0, min(prem * 0.14, 55.0))
        else:
            # Nifty 50 option SL: 18% of premium, bounded 14-26 pts
            sl_pts = max(14.0, min(prem * 0.18, 26.0))
        sl_pts = round(sl_pts, 1)

        # 7a. Institutional No-Chase & Climax Exhaustion Guard
        atr = sig_engine._calc_atr(14)
        live_spot = self.ws_streamer.latest_prices.get(idx, cur_spot) if self.ws_streamer else cur_spot
        drift = (live_spot - cur_spot) if opt_type == "CE" else (cur_spot - live_spot)
        if drift > (0.35 * atr):
            logger.warning(
                f"⚠️ NO-CHASE GATE ({idx} {opt_type}): Spot moved {drift:+.1f} pts past signal close "
                f"(> 0.35 ATR {0.35 * atr:.1f} pts). Rejecting late market entry into overextension."
            )
            return None

        c_body = abs(float(candle.get("close", cur_spot)) - float(candle.get("open", cur_spot)))
        vwap_ext = (cur_spot - cur_vwap) if opt_type == "CE" else (cur_vwap - cur_spot)
        if atr > 0 and vwap_ext > (2.5 * atr) and c_body > (1.4 * atr):
            logger.warning(
                f"⚠️ CLIMAX EXHAUSTION GUARD ({idx} {opt_type}): Spot extended {vwap_ext:.1f} pts (>2.5 ATR) "
                f"from VWAP on huge {c_body:.1f} pt bar. Skipping buying peak!"
            )
            return None

        # 7b. Capital Allocation & Margin Concentration Guard (BankNifty monthly 1-lot ~₹22k-₹32k; SL risk bounded 1-2%)
        cost_per_lot = prem * selected.lot_size
        if cost_per_lot > (self.risk_manager.capital_inr * 0.45):
            if conviction < 0.70 or abs(score) < 0.65:
                logger.warning(
                    f"🛡️ CAPITAL PROTECTION GUARD ({idx}): 1 lot costs ₹{cost_per_lot:,.0f} "
                    f"(> 45% of capital ₹{self.risk_manager.capital_inr:,.0f}). "
                    f"Requires Grade A+ setup (conviction >= 0.70, score >= 0.65). Skipping."
                )
                return None

        # Determine trade potential dynamically based on setup characteristics
        trade_potential = "NORMAL"
        is_breakout = cur_lvl and ("Breakout" in cur_lvl.active_sweep or "Breakdown" in cur_lvl.active_sweep)
        if (conviction >= 0.70 or abs(score) >= 0.65) and is_breakout:
            trade_potential = "BREAKOUT_EXPANSION"
        elif conviction >= 0.65 or abs(score) >= 0.60:
            trade_potential = "GRADE_A"

        # 8. Sizing via Risk Manager with dynamic potential scaling
        sizing = self.risk_manager.calculate_position_size(
            index_name=idx,
            option_premium=selected.est_premium,
            sl_points=sl_pts,
            lot_size=selected.lot_size,
            conviction=conviction,
            council_score=abs(score),
            trade_potential=trade_potential,
            potential_rr=vix_params.tp_multiplier,
        )

        # Apply VIX size multiplier (e.g. 0.50 on Extreme VIX)
        adj_lots = max(1, int(round(sizing["lots"] * vix_params.size_multiplier)))
        adj_qty = adj_lots * selected.lot_size

        # 9. Execute Marketable Limit Order
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
            spot_price=cur_spot,
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
        current_spot_price: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """
        Updates active position on new tick, handles trailing stops & theta harvesting.
        """
        if not self.executor.has_open_position():
            return None

        trade = self.executor.active_trade
        prev_tier = trade.trailing_tier

        # Resolve live spot if not provided
        if current_spot_price <= 0.0 and self.ws_streamer:
            current_spot_price = self.ws_streamer.latest_prices.get(trade.index_name, 0.0)

        # Check underlying momentum reversal dynamically if not passed explicitly
        if not underlying_reversed and current_spot_price > 0.0:
            sig_eng = self.signal_engines.get(trade.index_name)
            if sig_eng and len(sig_eng.candles_5m) >= 5:
                closes = [float(c["close"]) for c in sig_eng.candles_5m]
                ema9 = sig_eng._calc_ema(closes, 9)
                cur_vwap = float(sig_eng.candles_5m[-1].get("vwap", current_spot_price))
                if trade.option_type.upper() == "CE":
                    # For CALLS: Spot falling below EMA9 and VWAP indicates bullish exhaustion
                    if current_spot_price < (ema9 - 5.0) and current_spot_price < cur_vwap:
                        underlying_reversed = True
                elif trade.option_type.upper() == "PE":
                    # For PUTS: Spot rising above EMA9 and VWAP indicates bearish exhaustion
                    if current_spot_price > (ema9 + 5.0) and current_spot_price > cur_vwap:
                        underlying_reversed = True

        closed = self.executor.update_ticks(
            current_option_price=current_option_price,
            underlying_momentum_reversed=underlying_reversed,
            current_spot_price=current_spot_price,
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

    def _compute_mtf_trends(
        self,
        candles: list[dict[str, Any]],
        spot: float,
        pdc: float,
        open_px: float,
    ) -> dict[str, str]:
        """
        Computes 4-timeframe trend alignment (1D, 1H, 15m, 5m).
        Returns dict with "1d", "1h", "15m", "5m" as 'BULL' | 'BEAR' | 'NEUT'.
        """
        mtf = {"1d": "NEUT", "1h": "NEUT", "15m": "NEUT", "5m": "NEUT"}
        if not candles:
            return mtf

        # 1D Trend (Daily): Spot vs Previous Day Close or Day's Open
        if pdc > 0:
            mtf["1d"] = "BULL" if spot >= pdc else "BEAR"
        elif open_px > 0:
            mtf["1d"] = "BULL" if spot >= open_px else "BEAR"

        # 1H Trend (Hourly: 12 x 5m candles)
        closes = [float(c["close"]) for c in candles]
        if len(closes) >= 12:
            sma12 = sum(closes[-12:]) / 12.0
            mtf["1h"] = "BULL" if spot >= sma12 else "BEAR"
        elif len(closes) >= 2:
            mtf["1h"] = "BULL" if closes[-1] >= closes[0] else "BEAR"

        # 15m Trend (15-Minute: 3 x 5m candles)
        if len(closes) >= 3:
            sma3 = sum(closes[-3:]) / 3.0
            mtf["15m"] = "BULL" if spot >= sma3 else "BEAR"

        # 5m Trend (5-Minute: current close vs latest open)
        latest_open = float(candles[-1].get("open", spot))
        if spot > latest_open + 2.0:
            mtf["5m"] = "BULL"
        elif spot < latest_open - 2.0:
            mtf["5m"] = "BEAR"
        else:
            mtf["5m"] = "NEUT"

        return mtf

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
                "initial_lots": t.initial_lots,
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "sl_price": t.sl_price,
                "tp1_price": t.tp1_price,
                "tp2_price": t.tp2_price,
                "unrealized_pnl": t.unrealized_pnl,
                "partial_cash": t.partial_booked_cash,
                "trailing_tier": t.trailing_tier,
                "spot_entry_price": t.spot_entry_price,
                "spot_sl_price": t.spot_sl_price,
            }

        vix_params = self.vix_manager.get_parameters()
        
        # BankNifty OI
        bn_tracker = self.oi_trackers["BANKNIFTY"]
        bn_snap = bn_tracker.get_latest_snapshot()
        bn_pcr = bn_snap.pcr if bn_snap else 1.0
        bn_flow = bn_snap.flow_sentiment if bn_snap else "BALANCED"
        bn_call_wall = bn_snap.max_call_wall if bn_snap else 0.0
        bn_put_wall = bn_snap.max_put_wall if bn_snap else 0.0

        # Nifty OI
        nf_tracker = self.oi_trackers["NIFTY"]
        nf_snap = nf_tracker.get_latest_snapshot()
        nf_pcr = nf_snap.pcr if nf_snap else 1.0
        nf_flow = nf_snap.flow_sentiment if nf_snap else "BALANCED"
        nf_call_wall = nf_snap.max_call_wall if nf_snap else 0.0
        nf_put_wall = nf_snap.max_put_wall if nf_snap else 0.0

        # Live spot prices from streamer or Angel API
        bn_live = self.ws_streamer.latest_prices.get("BANKNIFTY", 0.0) if self.ws_streamer else 0.0
        nifty_live = self.ws_streamer.latest_prices.get("NIFTY", 0.0) if self.ws_streamer else 0.0

        bn_candles = self.signal_engines["BANKNIFTY"].candles_5m
        bn_spot = bn_live if bn_live > 0.0 else (bn_candles[-1]["close"] if bn_candles else 0.0)

        nifty_candles = self.signal_engines["NIFTY"].candles_5m
        nifty_spot = nifty_live if nifty_live > 0.0 else (nifty_candles[-1]["close"] if nifty_candles else 0.0)

        # Dynamic running VWAP from current active candle
        bn_vwap = (
            self.ws_streamer._candles_5m.get("BANKNIFTY", {}).get("vwap")
            if self.ws_streamer and self.ws_streamer._candles_5m.get("BANKNIFTY")
            else (bn_candles[-1].get("vwap", bn_spot) if bn_candles else bn_spot)
        )
        nifty_vwap = (
            self.ws_streamer._candles_5m.get("NIFTY", {}).get("vwap")
            if self.ws_streamer and self.ws_streamer._candles_5m.get("NIFTY")
            else (nifty_candles[-1].get("vwap", nifty_spot) if nifty_candles else nifty_spot)
        )

        # Trailing 20 closes for sparklines
        bn_spark = [float(c["close"]) for c in bn_candles[-20:]] if bn_candles else [bn_spot]
        nf_spark = [float(c["close"]) for c in nifty_candles[-20:]] if nifty_candles else [nifty_spot]

        # Institutional Key Levels & CPR
        bn_lvl = self.levels_trackers["BANKNIFTY"].calculate_from_candles(bn_candles, bn_spot)
        nf_lvl = self.levels_trackers["NIFTY"].calculate_from_candles(nifty_candles, nifty_spot)

        # Advisor Council Consensus & Bias
        hw_info = self._get_heavyweights_info()
        bn_score = 0.0
        bn_conviction = 0.50
        bn_bias = "NEUTRAL"
        if len(bn_candles) >= 20:
            try:
                bn_sig = self.signal_engines["BANKNIFTY"]
                bn_sig.min_consensus_threshold = vix_params.consensus_threshold
                bn_oi = self._latest_oi_chains.get("BANKNIFTY") or None
                res = bn_sig.evaluate_signal(
                    bn_spot, bn_vwap, oi_chain=bn_oi, key_levels=bn_lvl, is_preview=True, heavyweights=hw_info
                )
                if res:
                    bn_bias = f"{res[0]} ({res[1]})"
                    bn_conviction = res[2]
                    bn_score = res[3]
                else:
                    diag = bn_sig.last_diagnostics
                    bn_score = float(diag.get("net_score", 0.0))
                    bn_conviction = min(0.95, round(0.50 + (abs(bn_score) * 0.45), 2))
                    if diag.get("veto"):
                        bn_bias = f"VETO: {str(diag['veto'])[:28]}"
            except Exception:
                pass

        nf_score = 0.0
        nf_conviction = 0.50
        nf_bias = "NEUTRAL"
        if len(nifty_candles) >= 20:
            try:
                nf_sig = self.signal_engines["NIFTY"]
                nf_sig.min_consensus_threshold = vix_params.consensus_threshold
                nf_oi = self._latest_oi_chains.get("NIFTY") or None
                res = nf_sig.evaluate_signal(
                    nifty_spot, nifty_vwap, oi_chain=nf_oi, key_levels=nf_lvl, is_preview=True, heavyweights=hw_info
                )
                if res:
                    nf_bias = f"{res[0]} ({res[1]})"
                    nf_conviction = res[2]
                    nf_score = res[3]
                else:
                    diag = nf_sig.last_diagnostics
                    nf_score = float(diag.get("net_score", 0.0))
                    nf_conviction = min(0.95, round(0.50 + (abs(nf_score) * 0.45), 2))
                    if diag.get("veto"):
                        nf_bias = f"VETO: {str(diag['veto'])[:28]}"
            except Exception:
                pass

        # Multi-Timeframe (MTF) Trend Alignment
        bn_mtf = self._compute_mtf_trends(bn_candles, bn_spot, bn_lvl.pdc, bn_lvl.pdh)
        nf_mtf = self._compute_mtf_trends(nifty_candles, nifty_spot, nf_lvl.pdc, nf_lvl.pdh)

        # Inter-Market SMT Flow & Heavyweight Momentum
        bn_chg = round(((bn_spot - bn_lvl.pdc) / max(bn_lvl.pdc, 1.0)) * 100, 2) if bn_lvl.pdc > 0 else 0.0
        nf_chg = round(((nifty_spot - nf_lvl.pdc) / max(nf_lvl.pdc, 1.0)) * 100, 2) if nf_lvl.pdc > 0 else 0.0
        
        if bn_chg > 0.20 and nf_chg > 0.20:
            smt_flow = "CONVERGENT_BULLISH"
        elif bn_chg < -0.20 and nf_chg < -0.20:
            smt_flow = "CONVERGENT_BEARISH"
        elif bn_chg > 0.15 and nf_chg < -0.10:
            smt_flow = "SMT DIVERGENT (Bank Leading)"
        elif bn_chg < -0.15 and nf_chg > 0.10:
            smt_flow = "SMT DIVERGENT (Bank Dragging)"
        else:
            smt_flow = "BALANCED_COUPLED"

        hdfc_px = self.heavyweight_prices.get("1333", 0.0)
        icici_px = self.heavyweight_prices.get("4963", 0.0)

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
            "banknifty_sparkline": bn_spark,
            "banknifty_score": bn_score,
            "banknifty_conviction": bn_conviction,
            "banknifty_bias": bn_bias,
            "banknifty_mtf": bn_mtf,
            "nifty_spot": nifty_spot,
            "nifty_vwap": nifty_vwap,
            "nifty_sparkline": nf_spark,
            "nifty_score": nf_score,
            "nifty_conviction": nf_conviction,
            "nifty_bias": nf_bias,
            "nifty_mtf": nf_mtf,
            "smt_flow": smt_flow,
            "bn_chg_pct": bn_chg,
            "nf_chg_pct": nf_chg,
            "hdfc_ltp": hdfc_px,
            "icici_ltp": icici_px,
            "pcr": bn_pcr,
            "oi_flow": bn_flow,
            "call_wall": bn_call_wall,
            "put_wall": bn_put_wall,
            "nifty_pcr": nf_pcr,
            "nifty_oi_flow": nf_flow,
            "nifty_call_wall": nf_call_wall,
            "nifty_put_wall": nf_put_wall,
            "bn_levels": {
                "pdh": bn_lvl.pdh,
                "pdl": bn_lvl.pdl,
                "pivot": bn_lvl.pivot,
                "tc": bn_lvl.tc,
                "bc": bn_lvl.bc,
                "cpr_width": bn_lvl.cpr_width,
                "cpr_regime": bn_lvl.cpr_regime,
                "r1": bn_lvl.r1,
                "s1": bn_lvl.s1,
                "orh": bn_lvl.orh,
                "orl": bn_lvl.orl,
                "active_sweep": bn_lvl.active_sweep,
            },
            "nifty_levels": {
                "pdh": nf_lvl.pdh,
                "pdl": nf_lvl.pdl,
                "pivot": nf_lvl.pivot,
                "tc": nf_lvl.tc,
                "bc": nf_lvl.bc,
                "cpr_width": nf_lvl.cpr_width,
                "cpr_regime": nf_lvl.cpr_regime,
                "r1": nf_lvl.r1,
                "s1": nf_lvl.s1,
                "orh": nf_lvl.orh,
                "orl": nf_lvl.orl,
                "active_sweep": nf_lvl.active_sweep,
            },
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
            f"<b>Nifty PCR:</b> {m['nifty_pcr']:.2f} ({m['nifty_oi_flow']})\n"
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

    async def _poll_live_ticks_loop(self):
        """Continuous non-blocking background poller for fallback spot LTP, heavyweights, and active option ticks."""
        logger.info("Started background Angel One live LTP poller loop (non-blocking thread offload).")
        # Seed baseline previous closes for BankNifty heavyweights once
        if self.smart_api:
            try:
                base_res = await asyncio.to_thread(
                    self.smart_api.getMarketData, "FULL", {"NSE": ["1333", "4963", "2885"]}
                )
                for item in (base_res or {}).get("data", {}).get("fetched", []):
                    tok = str(item.get("symbolToken", ""))
                    base_px = float(item.get("close", 0.0)) or float(item.get("open", 0.0))
                    if tok and base_px > 0:
                        self.heavyweight_baselines[tok] = base_px
            except Exception as e:
                logger.debug(f"Heavyweight baseline fetch error: {e}")

        last_hw_poll = 0.0
        while self.is_running:
            try:
                now_t = time.time()
                has_pos = self.executor.has_open_position()
                ws_ok = bool(self.ws_streamer and self.ws_streamer.is_connected)
                bn_age = now_t - (self.ws_streamer.last_tick_ts.get("BANKNIFTY", 0.0) if self.ws_streamer else 0.0)
                nf_age = now_t - (self.ws_streamer.last_tick_ts.get("NIFTY", 0.0) if self.ws_streamer else 0.0)
                need_spot_poll = (not ws_ok) or bn_age > 3.0 or nf_age > 3.0
                need_hw_poll = (now_t - last_hw_poll) >= 2.5

                if self.smart_api and (has_pos or need_spot_poll or need_hw_poll):
                    exchange_tokens: dict[str, list[str]] = {}
                    nse_toks: list[str] = []
                    if need_spot_poll:
                        nse_toks.extend(["99926009", "99926000", "99926017"])
                    if need_hw_poll:
                        nse_toks.extend(["1333", "4963", "2885"])
                        last_hw_poll = now_t
                    if nse_toks:
                        exchange_tokens["NSE"] = nse_toks
                    if has_pos:
                        trade = self.executor.active_trade
                        exchange_tokens["NFO"] = [str(trade.token)]

                    if exchange_tokens:
                        mkt_res = await asyncio.to_thread(self.smart_api.getMarketData, "LTP", exchange_tokens)
                        fetched = (mkt_res or {}).get("data", {}).get("fetched", [])
                        for item in fetched:
                            tok = str(item.get("symbolToken", ""))
                            ltp = float(item.get("ltp", 0.0))
                            if ltp <= 0:
                                continue

                            if tok == "99926009" and self.ws_streamer:
                                self.ws_streamer.update_external_tick("BANKNIFTY", ltp)
                            elif tok == "99926000" and self.ws_streamer:
                                self.ws_streamer.update_external_tick("NIFTY", ltp)
                            elif tok == "99926017":
                                self.vix_manager.update_vix(ltp)
                            elif tok in ("1333", "4963", "2885"):
                                self.heavyweight_prices[tok] = ltp
                                if tok not in self.heavyweight_baselines:
                                    self.heavyweight_baselines[tok] = ltp
                            elif self.executor.has_open_position() and tok == str(self.executor.active_trade.token):
                                spot_px = self.ws_streamer.latest_prices.get(self.executor.active_trade.index_name, 0.0) if self.ws_streamer else 0.0
                                self.process_tick(current_option_price=ltp, current_spot_price=spot_px)

            except Exception as e:
                logger.debug(f"Live tick poller exception: {e}")

            await asyncio.sleep(0.75 if self.executor.has_open_position() else 1.0)

    async def _poll_oi_loop(self):
        """Periodic background poller (every 60s) for institutional Option Chain & Delta-OI."""
        logger.info("Started background Option Chain OI poller loop.")
        await asyncio.sleep(5.0)  # Initial grace delay
        while self.is_running:
            try:
                if self.smart_api:
                    for idx in ["BANKNIFTY", "NIFTY"]:
                        expiry = self.scrip_master.get_nearest_expiry(idx)
                        if not expiry:
                            continue

                        cur_spot = (
                            self.ws_streamer.latest_prices.get(idx, 0.0)
                            if self.ws_streamer else 0.0
                        )
                        if cur_spot <= 0:
                            candles = self.signal_engines[idx].candles_5m
                            cur_spot = candles[-1]["close"] if candles else (59000.0 if "BANK" in idx else 25000.0)

                        step = 100 if "BANK" in idx else 50
                        base_strike = round(cur_spot / step) * step
                        strikes = [base_strike + (i * step) for i in range(-8, 9)]

                        tokens_map = {}
                        tokens_list = []
                        for s in strikes:
                            for opt in ["CE", "PE"]:
                                c = self.scrip_master.get_option_contract(idx, expiry, s, opt)
                                if c:
                                    tok = str(c["token"])
                                    tokens_list.append(tok)
                                    tokens_map[tok] = {"strike": s, "option_type": opt, "symbol": c["symbol"]}

                        if tokens_list:
                            res = await asyncio.to_thread(self.smart_api.getMarketData, "FULL", {"NFO": tokens_list})
                            fetched = (res or {}).get("data", {}).get("fetched", [])
                            
                            prev_map = self._prev_oi_map.get(idx, {})
                            curr_map = {}
                            chain_data = []

                            for item in fetched:
                                tok = str(item.get("symbolToken", ""))
                                meta = tokens_map.get(tok)
                                if meta:
                                    s = meta["strike"]
                                    opt = meta["option_type"]
                                    oi = int(item.get("opnInterest", 0))
                                    key = (s, opt)
                                    curr_map[key] = oi
                                    delta_oi = oi - prev_map.get(key, oi)

                                    chain_data.append({
                                        "strike": s,
                                        "option_type": opt,
                                        "open_interest": oi,
                                        "change_oi": delta_oi,
                                        "ltp": float(item.get("ltp", 0.0)),
                                    })

                            self._prev_oi_map[idx] = curr_map

                            if chain_data:
                                self._latest_oi_chains[idx] = chain_data
                                tracker = self.oi_trackers.get(idx)
                                if tracker:
                                    tracker.update_chain(chain_data, cur_spot)

                        await asyncio.sleep(2.0)

            except Exception as e:
                logger.error(f"Option Chain OI poller exception: {e}")

            await asyncio.sleep(60.0)

    async def run_loop(self):
        """Main asynchronous event loop for Indian Options."""
        self.is_running = True
        logger.info("Indian Engine event loop started.")

        # Start Telegram Command Listener
        self.telegram.start_polling()

        # Start Indian Web Dashboard on port 8081
        try:
            self.web_server = IndianWebServer(self, host="0.0.0.0", port=8081)
            await self.web_server.start()
            logger.info("🌐 Indian Web Dashboard live at http://localhost:8081")
        except Exception as e:
            logger.error(f"Failed to start Indian Web Dashboard: {e}")

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
                        oi_ch = self._latest_oi_chains.get(sym) or None
                        self.process_candle(sym, c_dict, oi_chain=oi_ch)

                async def _on_tick(sym: str, px: float, vol: float):
                    if self.executor.has_open_position():
                        t = self.executor.active_trade
                        if t.symbol == sym:
                            spot_px = self.ws_streamer.latest_prices.get(t.index_name, 0.0) if self.ws_streamer else 0.0
                            self.process_tick(current_option_price=px, current_spot_price=spot_px)

                for seed_sym in ("NIFTY", "BANKNIFTY"):
                    self.ws_streamer.seed_session_vwap(seed_sym, self.signal_engines[seed_sym].candles_5m)
                self.ws_streamer.on_vix_updated = _on_vix
                self.ws_streamer.on_candle_closed = _on_candle
                self.ws_streamer.on_tick = _on_tick
                self.ws_streamer.start(asyncio.get_running_loop())
            except Exception as e:
                logger.error(f"WebSocket streamer initialization error: {e}")

        # Spawn Live LTP Poller and OI Tracker tasks
        asyncio.create_task(self._poll_live_ticks_loop())
        asyncio.create_task(self._poll_oi_loop())

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

                # 2. Render Live Terminal HUD with high frequency (every 0.5s)
                if (time.time() - last_hud_render) >= 0.5:
                    metrics = self.get_hud_metrics()
                    hud_output = self.hud.render(metrics)
                    # Clear screen and render HUD cleanly
                    print("\033[H\033[J" + hud_output, flush=True)
                    last_hud_render = time.time()

                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Indian Engine loop error: {e}")
                await asyncio.sleep(2.0)
