"""
Apex Indian Engine — Order Execution & Position Life-Cycle Core
===============================================================
Executes option trades in PAPER mode (full tick-level simulation with STT/GST friction)
and LIVE mode (direct Angel One SmartAPI order routing).
"""

from __future__ import annotations

import logging
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from apex_indian_engine.config import FrictionModel
from apex_indian_engine.theta_harvester import ThetaHarvester

logger = logging.getLogger("apex.indian.executor")


@dataclass
class ActiveOptionTrade:
    trade_id: str
    index_name: str
    symbol: str                     # e.g. "BANKNIFTY26NOV52400CE"
    token: str
    option_type: str                # "CE" or "PE"
    direction: str                  # "LONG" (we buy the option)
    lots: int
    quantity: int
    entry_price: float
    entry_time: float
    sl_price: float                 # Hard stop loss price on option premium
    tp1_price: float                # Take Profit 1 (+1.5R)
    tp2_price: float                # Take Profit 2 (+3.0R)
    trailing_tier: int = 0          # 0: Initial, 1: Tightened, 2: Break-Even, 3: Partial
    highest_fav_price: float = 0.0
    status: str = "OPEN"            # OPEN | CLOSED
    broker_order_id: str = ""
    exit_price: float = 0.0
    exit_time: float = 0.0
    exit_reason: str = ""
    unrealized_pnl: float = 0.0
    partial_booked_cash: float = 0.0 # Realized cash banked on partial harvest
    initial_lots: int = 1
    initial_quantity: int = 30
    spot_entry_price: float = 0.0
    spot_sl_price: float = 0.0
    initial_sl_dist: float = 0.0


class IndianOrderExecutor:
    """
    Manages active option positions, dynamic trailing stops, and execution routing.
    """

    def __init__(self, mode: str = "PAPER", smart_api_client: Optional[Any] = None):
        self.mode = mode.upper()
        self.api = smart_api_client
        self.active_trade: Optional[ActiveOptionTrade] = None
        self.journal_file = Path("data/indian_trade_journal.json")
        self.trade_history: list[dict[str, Any]] = []
        self._load_journal()
        self.friction = FrictionModel()
        self.theta_harvester = ThetaHarvester(max_stagnant_minutes=35.0, reversal_r_threshold=1.0)

    def _load_journal(self) -> None:
        try:
            if self.journal_file.exists():
                with open(self.journal_file, "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} closed trades from {self.journal_file}")
        except Exception as e:
            logger.error(f"Failed to load Indian trade journal: {e}")

    def _save_journal(self) -> None:
        try:
            self.journal_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.journal_file, "w") as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save Indian trade journal: {e}")

    def has_open_position(self) -> bool:
        return self.active_trade is not None and self.active_trade.status == "OPEN"

    def check_margin_available(self, required_inr: float) -> tuple[bool, float]:
        """
        Queries Angel One RMS Limits to ensure sufficient trading cash exists.
        Returns: (has_sufficient_margin, available_cash_inr)
        """
        if self.mode != "LIVE" or not self.api:
            return True, 1_000_000.0  # Paper mode simulation has unlimited virtual capital

        try:
            rms = self.api.rmsLimit()
            if rms.get("status"):
                net_avail = float(rms.get("data", {}).get("net", 0.0))
                if net_avail < required_inr:
                    logger.error(f"RMS MARGIN REJECTION: Required ₹{required_inr:,.2f} > Available ₹{net_avail:,.2f}")
                    return False, net_avail
                return True, net_avail
        except Exception as e:
            logger.warning(f"Could not verify RMS margin via SmartAPI: {e}. Proceeding cautiously.")

        return True, 0.0

    def _verify_order_fill(self, order_id: str, timeout_sec: float = 3.0) -> tuple[bool, float]:
        """
        Polls Angel One order book to confirm the order executed and get exact filled price.
        """
        if self.mode != "LIVE" or not self.api or not order_id:
            return True, 0.0

        start = time.time()
        while (time.time() - start) < timeout_sec:
            try:
                book = self.api.orderBook()
                if book.get("status"):
                    for ord_entry in book.get("data", []):
                        if str(ord_entry.get("orderid")) == str(order_id):
                            status = str(ord_entry.get("orderstatus", "")).upper()
                            if status in ("COMPLETE", "FILLED"):
                                avg_px = float(ord_entry.get("averageprice", 0.0))
                                return True, avg_px
                            elif status in ("REJECTED", "CANCELLED"):
                                rej_reason = ord_entry.get("text", "Unknown rejection")
                                logger.error(f"Angel One Order {order_id} was {status}: {rej_reason}")
                                return False, 0.0
            except Exception as e:
                logger.debug(f"Order poll error: {e}")
            time.sleep(0.5)

        logger.warning(f"Order {order_id} fill verification timed out after {timeout_sec}s.")
        return True, 0.0

    def open_option_trade(
        self,
        index_name: str,
        symbol: str,
        token: str,
        option_type: str,
        current_option_price: float,
        lots: int,
        quantity: int,
        sl_points: float = 25.0,
        tp_multiplier: float = 2.0,
        spot_price: float = 0.0,
    ) -> Optional[ActiveOptionTrade]:
        """
        Executes a Marketable Limit Buy order for the specified Option contract.
        """
        if self.has_open_position():
            logger.warning(f"Cannot open new trade: {self.active_trade.symbol} already active.")
            return None

        # Add 1-tick marketable buffer (₹0.50) to guarantee fill without freak slippage
        fill_price = round(current_option_price + 0.50, 2)
        total_order_val = fill_price * quantity

        # RMS Margin Guard
        has_margin, avail_cash = self.check_margin_available(total_order_val)
        if not has_margin:
            logger.error(f"TRADE ABORTED: Insufficient Angel One margin for {symbol} (Requires ₹{total_order_val:,.2f})")
            return None

        sl_price = max(1.0, round(fill_price - sl_points, 2))
        tp1_price = round(fill_price + (sl_points * 1.25), 2)
        tp2_price = round(fill_price + (sl_points * tp_multiplier), 2)

        # Structural Underlying Spot Invalidation (Nifty: 30 pts, BankNifty: 90 pts)
        # Note: Check "BANK" explicitly because "NIFTY" is a substring of "BANKNIFTY"!
        spot_dist = 90.0 if "BANK" in index_name.upper() else 30.0
        if option_type.upper() == "CE":
            spot_sl_px = round(spot_price - spot_dist, 2) if spot_price > 0 else 0.0
        else:
            spot_sl_px = round(spot_price + spot_dist, 2) if spot_price > 0 else 0.0

        trade_id = f"OPT_{datetime.now().strftime('%H%M%S')}_{token}"

        broker_id = ""
        if self.mode == "LIVE" and self.api:
            try:
                # Marketable Limit Order: Caps maximum adverse slippage
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": symbol,
                    "symboltoken": str(token),
                    "transactiontype": "BUY",
                    "exchange": "NFO",
                    "ordertype": "LIMIT",
                    "price": str(fill_price),
                    "producttype": "INTRADAY",  # MIS auto-square-off
                    "duration": "DAY",
                    "quantity": str(quantity),
                }
                res = self.api.placeOrder(order_params)
                broker_id = str(res.get("data", {}).get("orderid", ""))
                logger.info(f"Angel SmartAPI Marketable Limit Buy placed: {symbol} @ ₹{fill_price:.2f} (Order ID: {broker_id})")

                # Verify Fill Status
                filled, real_fill_px = self._verify_order_fill(broker_id)
                if not filled:
                    logger.error(f"Angel SmartAPI order {broker_id} was rejected by exchange.")
                    return None
                if real_fill_px > 0.0:
                    fill_price = real_fill_px
                    sl_price = max(1.0, round(fill_price - sl_points, 2))
                    tp1_price = round(fill_price + (sl_points * 1.25), 2)
                    tp2_price = round(fill_price + (sl_points * tp_multiplier), 2)
            except Exception as e:
                logger.error(f"Angel SmartAPI order failed: {e}")
                return None

        init_sl_dist = round(max(fill_price - sl_price, 1.0), 2)
        trade = ActiveOptionTrade(
            trade_id=trade_id,
            index_name=index_name,
            symbol=symbol,
            token=token,
            option_type=option_type,
            direction="LONG",
            lots=lots,
            quantity=quantity,
            initial_lots=lots,
            initial_quantity=quantity,
            entry_price=fill_price,
            entry_time=time.time(),
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            highest_fav_price=fill_price,
            broker_order_id=broker_id,
            spot_entry_price=spot_price,
            spot_sl_price=spot_sl_px,
            initial_sl_dist=init_sl_dist,
        )

        self.active_trade = trade
        logger.info(
            f"OPENED {self.mode} {symbol}: {lots} lots ({quantity} qty) @ ₹{fill_price:.2f} | "
            f"SL: ₹{sl_price:.2f} | TP1: ₹{tp1_price:.2f} | TP2: ₹{tp2_price:.2f} | "
            f"Spot: ₹{spot_price:.1f} (Spot SL: ₹{spot_sl_px:.1f})"
        )
        return trade

    def update_ticks(
        self,
        current_option_price: float,
        underlying_momentum_reversed: bool = False,
        current_spot_price: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """
        Process tick updates for the active position.
        Evaluates Underlying Spot Invalidation, Stop Loss, Break-Even Lock (+1.0R), TP1, TP2, and Theta Decay / Reversal Harvesting.
        Returns closed trade dict if exited, else None.
        """
        if not self.has_open_position():
            return None

        trade = self.active_trade
        cur = current_option_price
        entry = trade.entry_price
        qty = trade.quantity
        # Anchor R-multiple to original planned SL risk so ratcheting SL never collapses denominator
        sl_dist = trade.initial_sl_dist if trade.initial_sl_dist > 0 else max(entry - trade.sl_price, 1.0)

        # Track unrealized PnL (including banked partial cash)
        trade.unrealized_pnl = round((cur - entry) * qty + trade.partial_booked_cash, 2)
        trade.highest_fav_price = max(trade.highest_fav_price, cur)
        r_mult = (cur - entry) / max(sl_dist, 0.01)

        # 0. Structural Underlying Spot Invalidation Check
        # If the underlying index itself breaches the market structure invalidation boundary, exit immediately!
        if current_spot_price > 0.0 and trade.spot_sl_price > 0.0:
            if trade.option_type.upper() == "CE" and current_spot_price <= trade.spot_sl_price:
                logger.warning(
                    f"🛑 [UNDERLYING INVALIDATION] {trade.symbol} CE: Spot index (₹{current_spot_price:,.1f}) "
                    f"breached support invalidation (₹{trade.spot_sl_price:,.1f}). Exiting option!"
                )
                return self.close_trade(cur, reason=f"SPOT INVALIDATION ({current_spot_price:.0f} <= {trade.spot_sl_price:.0f})")
            elif trade.option_type.upper() == "PE" and current_spot_price >= trade.spot_sl_price:
                logger.warning(
                    f"🛑 [UNDERLYING INVALIDATION] {trade.symbol} PE: Spot index (₹{current_spot_price:,.1f}) "
                    f"breached resistance invalidation (₹{trade.spot_sl_price:,.1f}). Exiting option!"
                )
                return self.close_trade(cur, reason=f"SPOT INVALIDATION ({current_spot_price:.0f} >= {trade.spot_sl_price:.0f})")

        # 1. Hard Stop Loss Check
        if cur <= trade.sl_price:
            return self.close_trade(cur, reason="STOP LOSS HIT")

        # 2. Theta Decay & Reversal Harvesting Check
        should_harvest, harvest_reason = self.theta_harvester.evaluate_harvest(
            entry_price=trade.entry_price,
            current_price=cur,
            sl_price=round(entry - sl_dist, 2),
            entry_time=trade.entry_time,
            underlying_momentum_reversed=underlying_momentum_reversed,
        )
        if should_harvest:
            return self.close_trade(cur, reason=harvest_reason)

        # 3. Dynamic Tier 1: Risk Reduction (+0.7R) -> Tighten SL to -0.20R
        if trade.trailing_tier < 1 and r_mult >= 0.7:
            new_sl = round(entry - (sl_dist * 0.20), 2)
            trade.sl_price = new_sl
            trade.trailing_tier = 1
            logger.info(f"{trade.symbol} TIER 1 (+0.7R): SL tightened to ₹{new_sl:.2f}")

        # 4. Dynamic Tier 2: Break-Even Lock (+1.0R) -> Lock SL to Entry + Fees
        if trade.trailing_tier < 2 and r_mult >= 1.0:
            fee_points = 2.0  # ₹2 premium points to cover STT/brokerage
            new_sl = round(entry + fee_points, 2)
            trade.sl_price = new_sl
            trade.trailing_tier = 2
            logger.info(f"{trade.symbol} TIER 2 (+1.0R): SL locked to Break-Even @ ₹{new_sl:.2f}")

        # 5. Take Profit 1 (+1.25R) -> Partial Harvest or +0.50R Profit Floor Lock on 1-Lot
        if trade.trailing_tier < 3 and cur >= trade.tp1_price:
            trade.trailing_tier = 3
            if trade.lots > 1:
                close_lots = trade.lots // 2
                close_qty = (trade.quantity // trade.lots) * close_lots
                cash_banked = round((cur - entry) * close_qty, 2)
                trade.partial_booked_cash += cash_banked
                trade.lots -= close_lots
                trade.quantity -= close_qty
                logger.info(
                    f"🏆 [PARTIAL CASH HARVEST] {trade.symbol}: Banked {close_lots} lots cash (+₹{cash_banked:,.2f}) at TP1! "
                    f"Keeping {trade.lots} lots runner open to trail!"
                )
                if self.mode == "LIVE" and self.api:
                    try:
                        self.api.placeOrder({
                            "variety": "NORMAL",
                            "tradingsymbol": trade.symbol,
                            "symboltoken": str(trade.token),
                            "transactiontype": "SELL",
                            "exchange": "NFO",
                            "ordertype": "MARKET",
                            "producttype": "INTRADAY",
                            "duration": "DAY",
                            "quantity": str(close_qty),
                        })
                    except Exception as e:
                        logger.error(f"Live partial sell failed: {e}")
                trade.sl_price = round(entry + (sl_dist * 0.30), 2)
                logger.info(f"{trade.symbol} TP1 Hit: Runner SL locked to +0.3R @ ₹{trade.sl_price:.2f}")
            else:
                # Single-lot protection: lock guaranteed +0.50R real profit floor instead of mere BE
                trade.sl_price = round(entry + (sl_dist * 0.50), 2)
                logger.info(f"{trade.symbol} TP1 Hit (1-Lot): Guaranteed +0.50R Profit Lock @ ₹{trade.sl_price:.2f}")

        # 6. Full Take Profit 2 -> Full Exit of Remaining Position
        if cur >= trade.tp2_price:
            return self.close_trade(cur, reason=f"FULL TARGET TP2 (+{r_mult:.1f}R)")

        return None

    def close_trade(self, exit_price: float, reason: str = "") -> dict[str, Any]:
        """Closes the active option position and records statistics."""
        if not self.has_open_position():
            return {}

        trade = self.active_trade
        trade.status = "CLOSED"
        trade.exit_price = round(exit_price, 2)
        trade.exit_time = time.time()
        trade.exit_reason = reason

        # Total PnL = remaining quantity PnL + already banked cash profit!
        gross_pnl = round((trade.exit_price - trade.entry_price) * trade.quantity + trade.partial_booked_cash, 2)

        # Execute Live Sell on Broker if in LIVE mode
        if self.mode == "LIVE" and self.api and trade.quantity > 0:
            try:
                order_params = {
                    "variety": "NORMAL",
                    "tradingsymbol": trade.symbol,
                    "symboltoken": str(trade.token),
                    "transactiontype": "SELL",
                    "exchange": "NFO",
                    "ordertype": "MARKET",
                    "producttype": "INTRADAY",
                    "duration": "DAY",
                    "quantity": str(trade.quantity),
                }
                res = self.api.placeOrder(order_params)
                logger.info(f"Angel SmartAPI Exit order placed: {trade.symbol}")
            except Exception as e:
                logger.error(f"Angel SmartAPI exit order failed: {e}")

        closed_record = {
            "trade_id": trade.trade_id,
            "index_name": trade.index_name,
            "symbol": trade.symbol,
            "option_type": trade.option_type,
            "lots": trade.initial_lots,
            "quantity": trade.initial_quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "gross_pnl": gross_pnl,
            "partial_cash": trade.partial_booked_cash,
            "exit_reason": reason,
            "dur_min": round((trade.exit_time - trade.entry_time) / 60.0, 1),
        }

        self.trade_history.append(closed_record)
        self._save_journal()
        self.active_trade = None
        s = "+" if gross_pnl >= 0 else ""
        logger.info(f"CLOSED {trade.symbol} @ ₹{exit_price:.2f} | Total PnL: {s}₹{gross_pnl:,.2f} | Reason: {reason}")
        return closed_record
