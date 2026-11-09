"""
Apex Indian Engine — Central Configuration & Market Specification
===================================================================
Institutional configuration for NSE NIFTY, BANKNIFTY & FINNIFTY options trading.
Covers IST session schedules, risk buffers, option strike filters, and Indian tax models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from typing import Any


@dataclass
class IndexSpec:
    name: str                       # e.g. "NIFTY", "BANKNIFTY"
    spot_symbol: str                # e.g. "Nifty 50", "Nifty Bank"
    exchange: str                   # "NSE"
    fno_exchange: str               # "NFO"
    spot_token: str                 # Angel One token for spot index
    step_size: int                  # Strike step size: Nifty=50, BankNifty=100
    weekly_expiry_day: int          # 3 = Thursday (Nifty), 2 = Wednesday (BankNifty)
    default_lot_size: int           # Default lot size (dynamically overridden by scrip master)


INDEX_SPECS: dict[str, IndexSpec] = {
    "NIFTY": IndexSpec(
        name="NIFTY",
        spot_symbol="Nifty 50",
        exchange="NSE",
        fno_exchange="NFO",
        spot_token="99926000",       # NSE Nifty 50 Index token
        step_size=50,
        weekly_expiry_day=3,         # Thursday
        default_lot_size=75,
    ),
    "BANKNIFTY": IndexSpec(
        name="BANKNIFTY",
        spot_symbol="Nifty Bank",
        exchange="NSE",
        fno_exchange="NFO",
        spot_token="99926009",       # NSE BankNifty Index token
        step_size=100,
        weekly_expiry_day=2,         # Wednesday
        default_lot_size=30,
    ),
    "FINNIFTY": IndexSpec(
        name="FINNIFTY",
        spot_symbol="Nifty Fin Services",
        exchange="NSE",
        fno_exchange="NFO",
        spot_token="99926037",
        step_size=50,
        weekly_expiry_day=1,         # Tuesday
        default_lot_size=65,
    ),
}


@dataclass
class SessionSchedule:
    """Trading session timings in Indian Standard Time (IST)."""
    market_open: time = time(9, 15)
    trade_start: time = time(9, 25)          # Skip initial 10m price discovery
    morning_killzone_end: time = time(10, 45) # Peak morning momentum
    lunch_chop_start: time = time(11, 30)     # European open consolidation
    lunch_chop_end: time = time(13, 15)       # European open consolidation ends at 13:15
    afternoon_killzone_start: time = time(13, 15) # Afternoon trend & expiry expansion
    afternoon_killzone_end: time = time(15, 0)
    hard_flatten_time: time = time(15, 15)    # Compulsory intraday square-off
    market_close: time = time(15, 30)


@dataclass
class RiskRules:
    """Institutional capital preservation rules for Indian derivative trading."""
    initial_capital_inr: float = 100_000.0   # Default ₹1,00,000 account
    max_daily_loss_pct: float = 3.0          # Max loss 3.0% (₹3,000 on ₹1 Lakh)
    daily_safety_buffer_pct: float = 2.25    # Circuit breaker halts at 2.25% (₹2,250)
    max_risk_per_trade_pct: float = 1.0      # 1.0% risk per trade (₹1,000)
    max_trades_per_day: int = 3              # Strictly prevents overtrading (3 sniper setups max)
    daily_profit_lock_pct: float = 2.5       # +2.5% (₹2,500) banked halts trading to secure day's profit
    consecutive_loss_limit: int = 2          # 2 back-to-back losses = 45m cooling pause
    min_trade_duration_sec: int = 45         # Anti-tick scalping hold
    target_profit_rr: float = 2.5            # Minimum 1:2.5 Risk-Reward ratio
    trail_sl_trigger_r: float = 1.0          # Move to break-even at +1.0R


@dataclass
class OptionSelectorRules:
    """Parameters for selecting the optimal strike contract."""
    target_delta_min: float = 0.48           # At-The-Money / 1 strike In-The-Money
    target_delta_max: float = 0.62
    min_volume: int = 15_000                 # Avoid illiquid contracts
    min_open_interest: int = 50_000
    prefer_weekly: bool = True
    max_bid_ask_spread_pct: float = 1.5      # Max 1.5% spread between bid and ask


@dataclass
class FrictionModel:
    """Real-world transaction cost model for Indian Options (NSE/BSE)."""
    brokerage_per_order: float = 20.0        # ₹20 flat per order (₹40 roundtrip)
    stt_rate_sell: float = 0.001             # 0.1% STT on sell turnover (premium value)
    exchange_txn_charge: float = 0.000505    # 0.0505% NSE turnover charge
    sebi_turnover_charge: float = 0.000001   # ₹10 per crore
    gst_rate: float = 0.18                   # 18% GST on (Brokerage + Exchange fees)
    stamp_duty_buy: float = 0.00003          # 0.003% Stamp duty on buy value

    def calculate_roundtrip_cost(self, buy_price: float, sell_price: float, qty: int) -> float:
        """Calculate exact friction cost in INR for a completed option trade."""
        buy_val = buy_price * qty
        sell_val = sell_price * qty
        total_turnover = buy_val + sell_val

        brokerage = self.brokerage_per_order * 2
        stt = sell_val * self.stt_rate_sell
        exchange_fee = total_turnover * self.exchange_txn_charge
        sebi_fee = total_turnover * self.sebi_turnover_charge
        gst = (brokerage + exchange_fee) * self.gst_rate
        stamp_duty = buy_val * self.stamp_duty_buy

        total_cost = brokerage + stt + exchange_fee + sebi_fee + gst + stamp_duty
        return round(total_cost, 2)

    def calculate_total_friction(self, buy_val: float, sell_val: float):
        """Calculate friction using turnover values directly."""
        total_turnover = buy_val + sell_val
        brokerage = self.brokerage_per_order * 2
        stt = sell_val * self.stt_rate_sell
        exchange_fee = total_turnover * self.exchange_txn_charge
        sebi_fee = total_turnover * self.sebi_turnover_charge
        gst = (brokerage + exchange_fee) * self.gst_rate
        stamp_duty = buy_val * self.stamp_duty_buy
        total_cost = round(brokerage + stt + exchange_fee + sebi_fee + gst + stamp_duty, 2)

        class _FrictionResult:
            def __init__(self, cost):
                self.total_friction = cost
        return _FrictionResult(total_cost)


@dataclass
class EngineConfig:
    active_indices: list[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])
    session: SessionSchedule = field(default_factory=SessionSchedule)
    risk: RiskRules = field(default_factory=RiskRules)
    options: OptionSelectorRules = field(default_factory=OptionSelectorRules)
    friction: FrictionModel = field(default_factory=FrictionModel)
    mode: str = "PAPER"                     # "PAPER" or "LIVE"
    state_file: str = "data/indian_engine_state.json"
    scrip_cache_file: str = "data/angel_scrip_master.json"

