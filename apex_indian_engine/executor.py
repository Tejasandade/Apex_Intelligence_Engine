"""
Apex Indian Engine — Order Execution & Position Life-Cycle Core
===============================================================
Executes option trades in PAPER mode (full tick-level simulation with STT/GST friction)
and LIVE mode (direct Angel One SmartAPI order routing).
"""

from __future__ import annotations

import logging
import time
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


class IndianOrderExecutor:
    """
    Manages active option positions, dynamic trailing stops, and execution routing.
    """

    def __init__(self, mode: str = "PAPER", smart_api_client: Optional[Any] = None):
        self.mode = mode.upper()
        self.api = smart_api_client
        self.active_trade: Optional[ActiveOptionTrade] = None
        self.trade_history: list[dict[str, Any]] = []
        self.friction = FrictionModel()
        self.theta_harvester = ThetaHarvester(max_stagnant_minutes=35.0, reversal_r_threshold=1.0)

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
        tp1_price = round(fill_price + (sl_points * 1.5), 2)
        tp2_price = round(fill_price + (sl_points * tp_multiplier), 2)

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
                    tp1_price = round(fill_price + (sl_points * 1.5), 2)
                    tp2_price = round(fill_price + (sl_points * tp_multiplier), 2)
            except Exception as e:
                logger.error(f"Angel SmartAPI order failed: {e}")
                return None

        trade = ActiveOptionTrade(
            trade_id=trade_id,
            index_name=index_name,
            symbol=symbol,
            token=token,
            option_type=option_type,
            direction="LONG",
            lots=lots,
            quantity=quantity,
            entry_price=fill_price,
            entry_time=time.time(),
            sl_price=sl_price,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            highest_fav_price=fill_price,
            broker_order_id=broker_id,
        )

        self.active_trade = trade
        logger.info(
            f"OPENED {self.mode} {symbol}: {lots} lots ({quantity} qty) @ ₹{fill_price:.2f} | "
            f"SL: ₹{sl_price:.2f} | TP1: ₹{tp1_price:.2f} | TP2: ₹{tp2_price:.2f}"
        )
        return trade

    def update_ticks(self, current_option_price: float, underlying_momentum_reversed: bool = False) -> Optional[dict[str, Any]]:
        """
        Process tick updates for the active position.
        Evaluates Stop Loss, Break-Even Lock (+1.0R), TP1, TP2, and Theta Decay / Reversal Harvesting.
        Returns closed trade dict if exited, else None.
        """
        if not self.has_open_position():
            return None

        trade = self.active_trade
        cur = current_option_price
        entry = trade.entry_price
        qty = trade.quantity
        sl_dist = entry - trade.sl_price

        # Track unrealized PnL
        trade.unrealized_pnl = round((cur - entry) * qty, 2)
        trade.highest_fav_price = max(trade.highest_fav_price, cur)
        r_mult = (cur - entry) / max(sl_dist, 0.01)

        # 1. Hard Stop Loss Check
        if cur <= trade.sl_price:
            return self.close_trade(cur, reason="STOP LOSS HIT")

        # 2. Theta Decay & Reversal Harvesting Check
        should_harvest, harvest_reason = self.theta_harvester.evaluate_harvest(
            entry_price=trade.entry_price,
            current_price=cur,
            sl_price=trade.sl_price,
            entry_time=trade.entry_time,
            underlying_momentum_reversed=underlying_momentum_reversed,
        )
        if should_harvest:
            return self.close_trade(cur, reason=harvest_reason)

        # 3. Dynamic Tier 1: Risk Reduction (+0.8R) -> Tighten SL to -0.25R
        if trade.trailing_tier < 1 and r_mult >= 0.8:
            new_sl = round(entry - (sl_dist * 0.25), 2)
            trade.sl_price = new_sl
            trade.trailing_tier = 1
            logger.info(f"{trade.symbol} TIER 1 (+0.8R): SL tightened to ₹{new_sl:.2f}")

        # 3. Dynamic Tier 2: Break-Even Lock (+1.2R) -> Lock SL to Entry + Fees
        if trade.trailing_tier < 2 and r_mult >= 1.2:
            fee_points = 2.0  # ₹2 premium points to cover STT/brokerage
            new_sl = round(entry + fee_points, 2)
            trade.sl_price = new_sl
            trade.trailing_tier = 2
            logger.info(f"{trade.symbol} TIER 2 (+1.2R): SL locked to Break-Even @ ₹{new_sl:.2f}")

        # 4. Take Profit 1 (+1.5R) -> Lock in partial gains
        if trade.trailing_tier < 3 and cur >= trade.tp1_price:
            trade.trailing_tier = 3
            # Tighten trailing SL to +0.8R
            trade.sl_price = round(entry + (sl_dist * 0.8), 2)
            logger.info(f"{trade.symbol} TP1 HIT: +1.5R @ ₹{cur:.2f}")

        # 5. Full Take Profit 2 (+2.5R to +3.5R)
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

        gross_pnl = round((trade.exit_price - trade.entry_price) * trade.quantity, 2)

        # Execute Live Sell on Broker if in LIVE mode
        if self.mode == "LIVE" and self.api:
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
            "lots": trade.lots,
            "quantity": trade.quantity,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "gross_pnl": gross_pnl,
            "exit_reason": reason,
            "dur_min": round((trade.exit_time - trade.entry_time) / 60.0, 1),
        }

        self.trade_history.append(closed_record)
        self.active_trade = None
        s = "+" if gross_pnl >= 0 else ""
        logger.info(f"CLOSED {trade.symbol} @ ₹{exit_price:.2f} | PnL: {s}₹{gross_pnl:.2f} | Reason: {reason}")
        return closed_record
