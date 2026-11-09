"""
Apex Indian Engine — Risk & Capital Protection Manager
======================================================
Strict capital preservation, daily loss circuit breakers, position sizing,
and session timing enforcement for Indian derivative accounts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, date, time as dtime
from pathlib import Path
from typing import Any, Optional

from apex_indian_engine.config import RiskRules, SessionSchedule, FrictionModel

logger = logging.getLogger("apex.indian.risk")


class IndianRiskManager:
    """
    Manages capital allocation, sizing, and loss containment in INR (₹).
    """

    def __init__(
        self,
        rules: Optional[RiskRules] = None,
        schedule: Optional[SessionSchedule] = None,
        friction: Optional[FrictionModel] = None,
        state_file: str = "data/indian_engine_state.json",
    ):
        self.rules = rules or RiskRules()
        self.schedule = schedule or SessionSchedule()
        self.friction = friction or FrictionModel()
        self.state_file = Path(state_file)

        # State tracking
        self.today_date = date.today().isoformat()
        self.capital_inr = self.rules.initial_capital_inr
        self.day_start_capital = self.capital_inr
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.daily_friction_cost = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.status = "READY"              # READY | IN_PROGRESS | DAILY_HALT | CLOSED
        self.halt_reason = ""
        self.halt_until_ts = 0.0

        self.load_state()

    def check_day_reset(self) -> None:
        """Reset daily stats at the start of each new calendar day."""
        today_str = date.today().isoformat()
        if self.today_date != today_str:
            logger.info(f"Indian Engine Day Reset: {self.today_date} -> {today_str}")
            self.today_date = today_str
            self.day_start_capital = self.capital_inr
            self.daily_realized_pnl = 0.0
            self.daily_unrealized_pnl = 0.0
            self.daily_friction_cost = 0.0
            self.trades_today = 0
            self.consecutive_losses = 0
            self.status = "READY"
            self.halt_reason = ""
            self.halt_until_ts = 0.0
            self.save_state()

    def force_reset(self, capital: Optional[float] = None) -> None:
        """Force manual reset of capital and intraday trading stats to start fresh."""
        if capital is not None:
            self.capital_inr = float(capital)
            self.day_start_capital = float(capital)
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.daily_friction_cost = 0.0
        self.trades_today = 0
        self.consecutive_losses = 0
        self.status = "READY"
        self.halt_reason = ""
        self.halt_until_ts = 0.0
        self.save_state()
        logger.info(f"Indian Engine Force Reset: Capital=₹{self.capital_inr:,.2f}, Trades=0/4, Status=READY")

    def can_enter_trade(self, now_time: Optional[dtime] = None, skip_time_check: bool = False) -> tuple[bool, str]:
        """
        Validates whether opening a new option trade is permitted right now.
        Checks IST market hours, killzones, daily loss limits, and trade caps.
        """
        self.check_day_reset()

        if not skip_time_check:
            # 1. Trading Schedule & Kill Zone Check
            cur_t = now_time or datetime.now().time()
            if cur_t < self.schedule.trade_start:
                return False, f"Pre-market / opening discovery window (Trading starts at {self.schedule.trade_start.strftime('%H:%M')} IST)"

            if cur_t >= self.schedule.afternoon_killzone_end:
                return False, f"Afternoon session cutoff passed ({self.schedule.afternoon_killzone_end.strftime('%H:%M')} IST) — No new entries"

            # Check Lunch Chop
            if self.schedule.lunch_chop_start <= cur_t < self.schedule.lunch_chop_end:
                return False, "European open lunch chop (11:30-13:00 IST) — Trading paused to avoid sideways Theta burn"

        # 2. Cooling pause from consecutive losses
        if time.time() < self.halt_until_ts:
            rem_m = (self.halt_until_ts - time.time()) / 60.0
            return False, f"Consecutive loss cooling pause active ({rem_m:.0f}m remaining)"

        # 3. Maximum trades per day
        if self.trades_today >= self.rules.max_trades_per_day:
            return False, f"Maximum daily trades limit reached ({self.trades_today}/{self.rules.max_trades_per_day})"

        # 4. Daily Loss Circuit Breaker
        current_net_pnl = self.daily_realized_pnl + self.daily_unrealized_pnl
        hard_limit = self.day_start_capital * (self.rules.max_daily_loss_pct / 100.0)
        soft_buffer = self.day_start_capital * (self.rules.daily_safety_buffer_pct / 100.0)

        if current_net_pnl <= -soft_buffer:
            self.status = "DAILY_HALT"
            self.halt_reason = f"Daily safety buffer hit (-₹{abs(current_net_pnl):,.2f} / -₹{hard_limit:,.2f})"
            return False, self.halt_reason

        # 5. Daily Profit Target Lock (Bank profits and lock session)
        profit_lock_inr = self.day_start_capital * (self.rules.daily_profit_lock_pct / 100.0)
        if self.daily_realized_pnl >= profit_lock_inr:
            return False, f"Daily profit target secured (+₹{self.daily_realized_pnl:,.2f} banked today) — Trading concluded for the session"

        return True, "CLEAR"

    can_trade = can_enter_trade

    def calculate_position_size(
        self,
        index_name: str,
        option_premium: float,
        sl_points: float,
        lot_size: int,
        conviction: float = 0.60,
        council_score: float = 0.60,
        trade_potential: str = "NORMAL",
        potential_rr: float = 2.0,
    ) -> dict[str, Any]:
        """Calculates sizing dictionary compatible with engine order execution."""
        lots, qty, risk_inr = self.calculate_lot_size(
            sl_points=sl_points,
            lot_size=lot_size,
            conviction=conviction,
            council_score=council_score,
            trade_potential=trade_potential,
            potential_rr=potential_rr,
        )
        return {
            "lots": lots,
            "quantity": qty,
            "risk_inr": risk_inr,
            "premium": option_premium,
        }

    def calculate_lot_size(
        self,
        sl_points: float,
        lot_size: int,
        conviction: float = 0.60,
        council_score: float = 0.60,
        trade_potential: str = "NORMAL",
        potential_rr: float = 2.0,
    ) -> tuple[int, int, float]:
        """
        Calculates exact number of lots and quantity anchored to risk in INR.
        Scales risk dynamically according to trade setup potential:
        - Grade A+ Breakout Expansion (High conviction, high RR): 2.0% risk (enables 2-lot runner harvest)
        - Solid Momentum (conviction >= 0.65 or potential_rr >= 2.5): 1.4% risk
        - Defensive post-loss: 0.65% risk
        - Standard Scalp: 1.0% risk
        Returns: (lots_count, total_quantity, planned_risk_inr)
        """
        if sl_points <= 0 or lot_size <= 0:
            return 1, lot_size, round(sl_points * lot_size, 2)

        cost_per_lot = sl_points * lot_size

        is_grade_a_plus = (
            trade_potential == "BREAKOUT_EXPANSION"
            or (conviction >= 0.70 and council_score >= 0.65)
            or potential_rr >= 3.0
        )
        is_grade_a = (conviction >= 0.65 or council_score >= 0.60 or potential_rr >= 2.5)

        if self.consecutive_losses >= 1:
            risk_pct = 0.65  # Conservative post-loss scale-down
        elif is_grade_a_plus:
            # 2.0% risk budget allows taking 2 lots on BankNifty/Nifty to enable partial cash harvest at +1.5R
            risk_pct = 2.0
        elif is_grade_a:
            risk_pct = 1.4
        else:
            risk_pct = self.rules.max_risk_per_trade_pct

        budget_inr = self.capital_inr * (risk_pct / 100.0)

        # Ensure budget does not violate daily safety buffer (max 2.25% daily loss)
        daily_loss_budget_left = (self.day_start_capital * (self.rules.daily_safety_buffer_pct / 100.0)) + self.daily_realized_pnl
        if daily_loss_budget_left > 0:
            budget_inr = min(budget_inr, daily_loss_budget_left * 0.90)

        raw_lots = int(round(budget_inr / max(cost_per_lot, 1.0)))

        # On Grade A+ setups with sufficient daily budget, take 2 lots so TP1 partial cash harvest activates
        if is_grade_a_plus and self.consecutive_losses == 0 and budget_inr >= (cost_per_lot * 1.5):
            raw_lots = max(2, raw_lots)

        lots = max(1, min(raw_lots, 10))  # Max 10 lots for retail protection
        qty = lots * lot_size
        planned_risk = round(qty * sl_points, 2)

        return lots, qty, planned_risk

    def on_trade_closed(
        self,
        gross_pnl_inr: float,
        buy_price: float,
        sell_price: float,
        qty: int,
        reason: str = "",
    ) -> dict[str, Any]:
        """Record trade completion, deduct taxes & friction, and update capital."""
        self.check_day_reset()

        friction_cost = self.friction.calculate_roundtrip_cost(buy_price, sell_price, qty)
        net_pnl = gross_pnl_inr - friction_cost

        self.capital_inr += net_pnl
        self.daily_realized_pnl += net_pnl
        self.daily_friction_cost += friction_cost
        self.daily_unrealized_pnl = 0.0
        self.trades_today += 1

        is_win = (net_pnl > 0)
        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.rules.consecutive_loss_limit:
                self.halt_until_ts = time.time() + 3600.0  # 60m cooldown

        self.save_state()

        logger.info(
            f"Trade Logged | Gross: ₹{gross_pnl_inr:+.2f} | Taxes/Fees: ₹{friction_cost:.2f} | "
            f"Net: ₹{net_pnl:+.2f} | New Capital: ₹{self.capital_inr:,.2f} | Reason: {reason}"
        )

        return {
            "gross_pnl": round(gross_pnl_inr, 2),
            "friction": round(friction_cost, 2),
            "net_pnl": round(net_pnl, 2),
            "capital_inr": round(self.capital_inr, 2),
            "is_win": is_win,
            "trades_today": self.trades_today,
        }

    def save_state(self) -> None:
        """Persist state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "today_date": self.today_date,
            "capital_inr": self.capital_inr,
            "day_start_capital": self.day_start_capital,
            "daily_realized_pnl": self.daily_realized_pnl,
            "daily_unrealized_pnl": self.daily_unrealized_pnl,
            "daily_friction_cost": self.daily_friction_cost,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "status": self.status,
            "halt_reason": self.halt_reason,
            "last_updated": time.time(),
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self) -> None:
        """Load state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                if data.get("today_date") == self.today_date:
                    self.capital_inr = float(data.get("capital_inr", self.capital_inr))
                    self.day_start_capital = float(data.get("day_start_capital", self.capital_inr))
                    self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
                    self.daily_friction_cost = float(data.get("daily_friction_cost", 0.0))
                    self.trades_today = int(data.get("trades_today", 0))
                    self.consecutive_losses = int(data.get("consecutive_losses", 0))
                    self.status = data.get("status", "READY")
                    self.halt_reason = data.get("halt_reason", "")
            except Exception:
                pass
