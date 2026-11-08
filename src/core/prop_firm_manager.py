"""
Apex Intelligence Engine — Prop Firm Manager & Challenge Simulator
==================================================================
Simulates institutional prop firm challenge rules (SharkFunded, FTMO, FundedNext)
with real execution mapped directly to your broker (MT5 / Bybit).

Features:
- Daily drawdown circuit breaker: automatically halts trading if daily loss reaches safety buffer.
- Maximum overall drawdown safety floor.
- Dynamic lot sizing anchored to prop firm virtual balance and exact risk %.
- Daily reset at 00:00 UTC (re-anchoring daily equity baseline).
- Challenge progress tracking (target %, daily loss, total drawdown).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("apex.core.prop_firm")

# ── Preset Challenge Profiles ──────────────────────────────────────────────────
PROFILES: dict[str, dict[str, Any]] = {
    "sharkfunded_5k_lite_1step": {
        "name": "SharkFunded Lite 5K (1-Step)",
        "initial_balance": 5000.0,
        "profit_target_pct": 9.0,         # $450.00 target (target balance: $5,450.00)
        "max_daily_loss_pct": 3.0,        # $150.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $112.50 circuit breaker (halt before 3% breach!)
        "max_total_loss_pct": 6.0,        # $300.00 max overall loss (hard floor: $4,700.00)
        "total_safety_buffer_pct": 4.8,   # $240.00 emergency stop (halt before 6% breach!)
        "base_risk_pct": 0.5,             # 0.5% risk ($25.00 per trade)
        "strong_risk_pct": 0.65,          # 0.65% risk ($32.50) on high conviction
        "min_risk_pct": 0.35,             # 0.35% risk ($17.50) conservative/post-loss
        "max_concurrent_trades": 1,
        "news_blackout_mins": 10,
    },
    "sharkfunded_10k_lite_1step": {
        "name": "SharkFunded Lite 10K (1-Step)",
        "initial_balance": 10000.0,
        "profit_target_pct": 9.0,         # $900.00 target
        "max_daily_loss_pct": 3.0,        # $300.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $225.00 circuit breaker
        "max_total_loss_pct": 6.0,        # $600.00 max overall loss
        "total_safety_buffer_pct": 4.8,   # $480.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($50.00 per trade)
        "strong_risk_pct": 0.65,          # 0.65% risk ($65.00)
        "min_risk_pct": 0.35,             # 0.35% risk ($35.00)
        "max_concurrent_trades": 1,
        "news_blackout_mins": 10,
    },
    "sharkfunded_25k_lite_1step": {
        "name": "SharkFunded Lite 25K (1-Step)",
        "initial_balance": 25000.0,
        "profit_target_pct": 9.0,         # $2,250.00 target
        "max_daily_loss_pct": 3.0,        # $750.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $562.50 circuit breaker
        "max_total_loss_pct": 6.0,        # $1,500.00 max overall loss
        "total_safety_buffer_pct": 4.8,   # $1,200.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($125.00 per trade)
        "strong_risk_pct": 0.65,          # 0.65% risk ($162.50)
        "min_risk_pct": 0.35,             # 0.35% risk ($87.50)
        "max_concurrent_trades": 2,
        "news_blackout_mins": 10,
    },
    "ftmo_10k_2step": {
        "name": "FTMO 10K (Phase 1)",
        "initial_balance": 10000.0,
        "profit_target_pct": 10.0,        # $1,000.00 target
        "max_daily_loss_pct": 5.0,        # $500.00 max daily loss
        "daily_safety_buffer_pct": 4.0,   # $400.00 circuit breaker
        "max_total_loss_pct": 10.0,       # $1,000.00 max overall loss
        "total_safety_buffer_pct": 8.0,   # $800.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($50.00)
        "strong_risk_pct": 0.75,          # 0.75% risk ($75.00)
        "min_risk_pct": 0.35,             # 0.35% risk ($35.00)
        "max_concurrent_trades": 1,
        "news_blackout_mins": 5,
    },
    "ftmo_25k_2step": {
        "name": "FTMO 25K (Phase 1)",
        "initial_balance": 25000.0,
        "profit_target_pct": 10.0,        # $2,500.00 target
        "max_daily_loss_pct": 5.0,        # $1,250.00 max daily loss
        "daily_safety_buffer_pct": 4.0,   # $1,000.00 circuit breaker
        "max_total_loss_pct": 10.0,       # $2,500.00 max overall loss
        "total_safety_buffer_pct": 8.0,   # $2,000.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($125.00)
        "strong_risk_pct": 0.75,          # 0.75% risk ($187.50)
        "min_risk_pct": 0.35,             # 0.35% risk ($87.50)
        "max_concurrent_trades": 2,
        "news_blackout_mins": 5,
    },
    "ftmo_50k_2step": {
        "name": "FTMO 50K (Phase 1)",
        "initial_balance": 50000.0,
        "profit_target_pct": 10.0,        # $5,000.00 target
        "max_daily_loss_pct": 5.0,        # $2,500.00 max daily loss
        "daily_safety_buffer_pct": 4.0,   # $2,000.00 circuit breaker
        "max_total_loss_pct": 10.0,       # $5,000.00 max overall loss
        "total_safety_buffer_pct": 8.0,   # $4,000.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($250.00)
        "strong_risk_pct": 0.75,          # 0.75% risk ($375.00)
        "min_risk_pct": 0.35,             # 0.35% risk ($175.00)
        "max_concurrent_trades": 2,
        "news_blackout_mins": 5,
    },
    "ftmo_100k_2step": {
        "name": "FTMO 100K (Phase 1)",
        "initial_balance": 100000.0,
        "profit_target_pct": 10.0,        # $10,000.00 target
        "max_daily_loss_pct": 5.0,        # $5,000.00 max daily loss
        "daily_safety_buffer_pct": 4.0,   # $4,000.00 circuit breaker
        "max_total_loss_pct": 10.0,       # $10,000.00 max overall loss
        "total_safety_buffer_pct": 8.0,   # $8,000.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($500.00)
        "strong_risk_pct": 0.75,          # 0.75% risk ($750.00)
        "min_risk_pct": 0.35,             # 0.35% risk ($350.00)
        "max_concurrent_trades": 3,
        "news_blackout_mins": 5,
    },
    "fundednext_6k_stellar": {
        "name": "FundedNext 6K (1-Step Stellar)",
        "initial_balance": 6000.0,
        "profit_target_pct": 10.0,        # $600.00 target
        "max_daily_loss_pct": 3.0,        # $180.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $135.00 circuit breaker
        "max_total_loss_pct": 6.0,        # $360.00 max overall loss
        "total_safety_buffer_pct": 4.8,   # $288.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($30.00)
        "strong_risk_pct": 0.65,          # 0.65% risk ($39.00)
        "min_risk_pct": 0.35,             # 0.35% risk ($21.00)
        "max_concurrent_trades": 1,
        "news_blackout_mins": 10,
    },
    "fundednext_25k_stellar": {
        "name": "FundedNext 25K (1-Step Stellar)",
        "initial_balance": 25000.0,
        "profit_target_pct": 10.0,        # $2,500.00 target
        "max_daily_loss_pct": 3.0,        # $750.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $562.50 circuit breaker
        "max_total_loss_pct": 6.0,        # $1,500.00 max overall loss
        "total_safety_buffer_pct": 4.8,   # $1,200.00 emergency stop
        "base_risk_pct": 0.4,             # 0.4% risk ($100.00)
        "strong_risk_pct": 0.6,           # 0.6% risk ($150.00)
        "min_risk_pct": 0.25,             # 0.25% risk ($62.50)
        "max_concurrent_trades": 2,
        "news_blackout_mins": 10,
    },
    "the5ers_5k_bootcamp": {
        "name": "The5ers 5K (Bootcamp)",
        "initial_balance": 5000.0,
        "profit_target_pct": 6.0,         # $300.00 target
        "max_daily_loss_pct": 3.0,        # $150.00 max daily loss
        "daily_safety_buffer_pct": 2.25,  # $112.50 circuit breaker
        "max_total_loss_pct": 5.0,        # $250.00 max overall loss
        "total_safety_buffer_pct": 4.0,   # $200.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($25.00)
        "strong_risk_pct": 0.65,          # 0.65% risk ($32.50)
        "min_risk_pct": 0.35,             # 0.35% risk ($17.50)
        "max_concurrent_trades": 1,
        "news_blackout_mins": 10,
    },
    "funding_pips_5k_2step": {
        "name": "Funding Pips 5K (2-Step Phase 1)",
        "initial_balance": 5000.0,
        "profit_target_pct": 8.0,         # $400.00 target
        "max_daily_loss_pct": 5.0,        # $250.00 max daily loss
        "daily_safety_buffer_pct": 4.0,   # $200.00 circuit breaker
        "max_total_loss_pct": 10.0,       # $500.00 max overall loss
        "total_safety_buffer_pct": 8.0,   # $400.00 emergency stop
        "base_risk_pct": 0.5,             # 0.5% risk ($25.00)
        "strong_risk_pct": 0.75,          # 0.75% risk ($37.50)
        "min_risk_pct": 0.35,             # 0.35% risk ($17.50)
        "max_concurrent_trades": 1,
        "news_blackout_mins": 5,
    },
}

# ── Dynamic Firm & Multi-Phase Templates (Phase 1, Phase 2, Funded, 1-Step) ───
FIRM_TEMPLATES: dict[str, dict[str, Any]] = {
    "sharkfunded": {
        "name": "SharkFunded",
        "sizes": [5000, 10000, 25000, 50000, 100000],
        "default_size": 5000,
        "phases": {
            "1-step": {
                "name": "1-Step Evaluation",
                "profit_target_pct": 9.0,
                "max_daily_loss_pct": 3.0,
                "daily_safety_buffer_pct": 2.25,
                "max_total_loss_pct": 6.0,
                "total_safety_buffer_pct": 4.8,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 1,
            },
            "phase1": {
                "name": "Phase 1 (Assessment)",
                "profit_target_pct": 8.0,
                "max_daily_loss_pct": 4.0,
                "daily_safety_buffer_pct": 3.0,
                "max_total_loss_pct": 8.0,
                "total_safety_buffer_pct": 6.5,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase2": {
                "name": "Phase 2 (Verification)",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 4.0,
                "daily_safety_buffer_pct": 3.0,
                "max_total_loss_pct": 8.0,
                "total_safety_buffer_pct": 6.5,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "funded": {
                "name": "Funded Master Account",
                "profit_target_pct": 0.0,
                "max_daily_loss_pct": 3.0,
                "daily_safety_buffer_pct": 2.25,
                "max_total_loss_pct": 6.0,
                "total_safety_buffer_pct": 4.8,
                "base_risk_pct": 0.4,
                "max_concurrent_trades": 2,
            }
        }
    },
    "ftmo": {
        "name": "FTMO",
        "sizes": [10000, 25000, 50000, 100000, 200000],
        "default_size": 10000,
        "phases": {
            "phase1": {
                "name": "Phase 1 (Challenge)",
                "profit_target_pct": 10.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase2": {
                "name": "Phase 2 (Verification)",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "funded": {
                "name": "FTMO Trader (Funded)",
                "profit_target_pct": 0.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            }
        }
    },
    "fundednext": {
        "name": "FundedNext",
        "sizes": [6000, 15000, 25000, 50000, 100000, 200000],
        "default_size": 6000,
        "phases": {
            "1-step": {
                "name": "Stellar 1-Step",
                "profit_target_pct": 10.0,
                "max_daily_loss_pct": 3.0,
                "daily_safety_buffer_pct": 2.25,
                "max_total_loss_pct": 6.0,
                "total_safety_buffer_pct": 4.8,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase1": {
                "name": "Stellar 2-Step (Phase 1)",
                "profit_target_pct": 8.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase2": {
                "name": "Stellar 2-Step (Phase 2)",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "funded": {
                "name": "FundedNext Master (Funded)",
                "profit_target_pct": 0.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            }
        }
    },
    "the5ers": {
        "name": "The 5%ers",
        "sizes": [5000, 10000, 20000, 60000, 100000],
        "default_size": 5000,
        "phases": {
            "bootcamp": {
                "name": "Bootcamp (Phase 1)",
                "profit_target_pct": 6.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 6.0,
                "total_safety_buffer_pct": 4.8,
                "base_risk_pct": 0.4,
                "max_concurrent_trades": 1,
            },
            "hyper": {
                "name": "Hyper Growth (Phase 1)",
                "profit_target_pct": 10.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase2": {
                "name": "Phase 2 Verification",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 8.0,
                "total_safety_buffer_pct": 6.5,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "funded": {
                "name": "The5ers Funded Trader",
                "profit_target_pct": 0.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 6.0,
                "total_safety_buffer_pct": 4.8,
                "base_risk_pct": 0.4,
                "max_concurrent_trades": 2,
            }
        }
    },
    "funding_pips": {
        "name": "Funding Pips",
        "sizes": [5000, 10000, 25000, 50000, 100000],
        "default_size": 5000,
        "phases": {
            "phase1": {
                "name": "Student (Phase 1)",
                "profit_target_pct": 8.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "phase2": {
                "name": "Practitioner (Phase 2)",
                "profit_target_pct": 5.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            },
            "funded": {
                "name": "Master (Funded)",
                "profit_target_pct": 0.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            }
        }
    },
    "custom": {
        "name": "Custom Challenge",
        "sizes": [5000, 10000, 25000, 50000, 100000, 200000],
        "default_size": 10000,
        "phases": {
            "custom": {
                "name": "Custom Parameters",
                "profit_target_pct": 10.0,
                "max_daily_loss_pct": 5.0,
                "daily_safety_buffer_pct": 4.0,
                "max_total_loss_pct": 10.0,
                "total_safety_buffer_pct": 8.0,
                "base_risk_pct": 0.5,
                "max_concurrent_trades": 2,
            }
        }
    }
}


class PropFirmManager:
    """
    Manages and simulates a prop firm evaluation challenge.
    Tracks virtual challenge equity, enforces strict drawdown caps, and auto-scales lots.
    """

    def __init__(self, profile_key: str = "sharkfunded_5k_lite_1step", state_file: str = "data/prop_firm_state.json"):
        self.profile_key = profile_key if profile_key in PROFILES else "sharkfunded_5k_lite_1step"
        self.config = PROFILES[self.profile_key]
        self.state_file = Path(state_file)

        self.initial_balance: float = float(self.config["initial_balance"])
        self.virtual_balance: float = self.initial_balance
        self.peak_balance: float = self.initial_balance
        self.day_start_balance: float = self.initial_balance
        self.today_date: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        self.daily_realized_pnl: float = 0.0
        self.daily_unrealized_pnl: float = 0.0
        self.total_realized_pnl: float = 0.0

        self.wins: int = 0
        self.losses: int = 0
        self.total_trades: int = 0
        self.status: str = "IN_PROGRESS"  # IN_PROGRESS | PHASE_1_PASSED | DAILY_HALT | BREACHED
        self.halt_reason: str = ""

        self.load_state()
        self.check_day_rollover()

    @property
    def cumulative_profit(self) -> float:
        """Current cumulative profit above the challenge starting balance."""
        return self.virtual_balance - self.initial_balance

    def check_day_rollover(self) -> None:
        """Check for 00:00 UTC day rollover to reset daily loss tracker."""
        current_utc_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if current_utc_date != self.today_date:
            logger.info(
                "prop_firm_day_rollover",
                old_date=self.today_date,
                new_date=current_utc_date,
                closing_daily_pnl=self.daily_realized_pnl,
            )
            self.today_date = current_utc_date
            self.day_start_balance = self.virtual_balance
            self.daily_realized_pnl = 0.0
            self.daily_unrealized_pnl = 0.0
            if self.status == "DAILY_HALT":
                self.status = "IN_PROGRESS"
                self.halt_reason = ""
            self.save_state()

    def update_floating_pnl(self, unrealized_usd: float) -> None:
        """Update floating PnL from the active open position."""
        self.check_day_rollover()
        self.daily_unrealized_pnl = float(unrealized_usd)

    def can_trade(self) -> tuple[bool, str]:
        """
        Evaluate if a new trade is permitted under Prop Firm Challenge rules.
        """
        self.check_day_rollover()

        # 1. Target already cleared!
        target_usd = self.initial_balance * (self.config["profit_target_pct"] / 100.0)
        current_profit = self.virtual_balance - self.initial_balance
        if current_profit >= target_usd:
            self.status = "PHASE_1_PASSED"
            return False, f"🎉 CHALLENGE PASSED! Profit target reached (+${current_profit:.2f} / +${target_usd:.2f})."

        # 2. Daily Loss Limit Check
        daily_loss_limit_usd = self.day_start_balance * (self.config["max_daily_loss_pct"] / 100.0)
        daily_safety_stop_usd = self.day_start_balance * (self.config["daily_safety_buffer_pct"] / 100.0)
        current_daily_pnl = self.daily_realized_pnl + self.daily_unrealized_pnl

        if current_daily_pnl <= -daily_safety_stop_usd:
            self.status = "DAILY_HALT"
            self.halt_reason = (
                f"Daily Safety Stop Hit: -${abs(current_daily_pnl):.2f} / -${daily_loss_limit_usd:.2f}. "
                f"Halted to protect {self.config['name']} daily limit."
            )
            return False, self.halt_reason

        # 3. Maximum Overall Drawdown Check
        max_total_loss_usd = self.initial_balance * (self.config["max_total_loss_pct"] / 100.0)
        total_safety_stop_usd = self.initial_balance * (self.config["total_safety_buffer_pct"] / 100.0)
        hard_floor = self.initial_balance - max_total_loss_usd
        safety_floor = self.initial_balance - total_safety_stop_usd

        current_equity = self.virtual_balance + self.daily_unrealized_pnl
        if current_equity <= safety_floor:
            self.status = "BREACHED" if current_equity <= hard_floor else "DAILY_HALT"
            self.halt_reason = (
                f"Total Drawdown Protection: Equity ${current_equity:.2f} near floor ${hard_floor:.2f} "
                f"(-{((self.initial_balance - current_equity)/self.initial_balance)*100:.1f}%)."
            )
            return False, self.halt_reason

        return True, "CLEAR"

    def can_open_multi_asset_trade(
        self,
        symbol: str,
        direction: str,
        active_positions: dict[str, dict],
        estimated_risk_usd: float = 25.0,
        max_concurrent: int = 2
    ) -> tuple[bool, str]:
        """
        Evaluate if a new trade can be opened in a multi-asset environment
        while strictly complying with SharkFunded prop firm risk caps.
        """
        # 1. Base challenge gate
        ok, reason = self.can_trade()
        if not ok:
            return False, reason

        sym_key = symbol.upper()
        dir_upper = direction.upper()

        # 2. Per-symbol isolation: Is this exact symbol already open?
        for sym, pos in active_positions.items():
            if sym.upper() == sym_key and pos.get("position") == "OPEN":
                return False, f"Symbol {sym_key} already has an OPEN position"

        # 3. Maximum concurrent positions cap across portfolio
        open_count = sum(1 for pos in active_positions.values() if pos.get("position") == "OPEN")
        if open_count >= max_concurrent:
            return False, f"Portfolio concurrent position limit reached ({open_count}/{max_concurrent})"

        # 4. Correlation Shield: BTC and ETH are strongly correlated
        if "ETH" in sym_key:
            btc_pos = active_positions.get("BTCUSDT", {})
            if btc_pos.get("position") == "OPEN" and btc_pos.get("direction") == dir_upper:
                return False, f"Correlation Shield: BTC already {dir_upper} — blocking same-direction {sym_key}"
        elif "BTC" in sym_key:
            eth_pos = active_positions.get("ETHUSDT", {})
            if eth_pos.get("position") == "OPEN" and eth_pos.get("direction") == dir_upper:
                return False, f"Correlation Shield: ETH already {dir_upper} — blocking same-direction {sym_key}"

        # 5. Combined Portfolio Daily Risk Budget Pool
        # Sum of maximum potential loss if all active trades hit stop loss
        total_open_risk = sum(
            float(pos.get("risk_usd", 25.0))
            for pos in active_positions.values()
            if pos.get("position") == "OPEN"
        )
        daily_safety_stop_usd = self.day_start_balance * (self.config["daily_safety_buffer_pct"] / 100.0)
        remaining_daily_budget = daily_safety_stop_usd + self.daily_realized_pnl

        projected_combined_risk = total_open_risk + float(estimated_risk_usd)
        if projected_combined_risk > remaining_daily_budget:
            return False, (
                f"Combined Risk Pool: Projected risk (${projected_combined_risk:.2f}) "
                f"exceeds remaining daily safety buffer (${remaining_daily_budget:.2f})"
            )

        return True, "CLEAR"

    def update_multi_floating_pnl(self, active_positions: dict[str, dict]) -> None:
        """Sum unrealized PnL across all open positions."""
        total_unrealized = sum(
            float(pos.get("unrealized_pnl", 0.0))
            for pos in active_positions.values()
            if pos.get("position") == "OPEN"
        )
        self.update_floating_pnl(total_unrealized)

    def calculate_lot_size(
        self,
        sl_distance: float,
        price: float,
        conviction: float = 0.60,
        council_score: float = 0.65,
        consecutive_losses: int = 0,
        symbol: str = "BTCUSDT",
    ) -> tuple[float, float]:
        """
        Calculate exact lot size anchored to the Prop Firm challenge rules.
        Dynamically adapts contract size and minimum volume for BTC, ETH, and GOLD.
        Returns: (lots, risk_usd)
        """
        if sl_distance <= 0 or price <= 0:
            return 0.01, 25.0

        # Dynamic risk scaling based on conviction and streak
        if conviction >= 0.65 or council_score >= 0.75:
            risk_pct = float(self.config["strong_risk_pct"])
        elif consecutive_losses >= 1:
            risk_pct = float(self.config["min_risk_pct"])
        else:
            risk_pct = float(self.config["base_risk_pct"])

        risk_usd = self.virtual_balance * (risk_pct / 100.0)

        # ── SOFT-LANDING RISK DECELERATION ──
        # When approaching challenge victory (within $100 of profit target),
        # smoothly taper down risk so banked profits are protected and victory is locked in.
        target_usd = self.initial_balance * (self.config["profit_target_pct"] / 100.0)
        current_profit = self.virtual_balance - self.initial_balance
        remaining_target = target_usd - current_profit
        if 0 < remaining_target <= 100.0:
            soft_risk = max(10.0, remaining_target * 0.35)
            if soft_risk < risk_usd:
                logger.info(
                    "prop_firm_soft_landing_deceleration",
                    remaining_target=round(remaining_target, 2),
                    original_risk=round(risk_usd, 2),
                    soft_landing_risk=round(soft_risk, 2)
                )
                risk_usd = soft_risk

        # Ensure risk_usd does not exceed remaining daily budget
        daily_safety_stop_usd = self.day_start_balance * (self.config["daily_safety_buffer_pct"] / 100.0)
        remaining_daily_budget = daily_safety_stop_usd + (self.daily_realized_pnl + self.daily_unrealized_pnl)
        if remaining_daily_budget > 0:
            risk_usd = min(risk_usd, remaining_daily_budget * 0.70)  # Max 70% of remaining daily budget on 1 trade

        # Asset-specific contract size & minimum volume defaults
        sym_upper = str(symbol).upper()
        contract_size = 1.0
        step = 0.01
        min_v = 0.01

        if "GOLD" in sym_upper or "XAU" in sym_upper or "PAXG" in sym_upper:
            contract_size = 100.0   # 1 standard lot = 100 oz
            min_v = 0.01
        elif "ETH" in sym_upper:
            contract_size = 1.0
            min_v = 0.02           # XM minimum volume for ETHUSD# is 0.02
        else:
            contract_size = 1.0    # BTCUSD#
            min_v = 0.01

        # Query live MT5 broker specs if terminal is connected
        try:
            import MetaTrader5 as mt5
            term = mt5.terminal_info()
            if term and term.connected:
                if "BTC" in sym_upper:
                    cands = ["BTCUSD#", "BTCUSD", "BTCUSD.i", "BTCUSDm", symbol, f"{symbol}#"]
                elif "ETH" in sym_upper:
                    cands = ["ETHUSD#", "ETHUSD", "ETHUSD.i", "ETHUSDm", symbol, f"{symbol}#"]
                elif "GOLD" in sym_upper or "XAU" in sym_upper or "PAXG" in sym_upper:
                    cands = ["GOLD.i#", "GOLD#", "GOLD", "XAUUSD#", "XAUUSD", symbol, f"{symbol}#"]
                else:
                    cands = [symbol, f"{symbol}#"]

                for cand in cands:
                    s_info = mt5.symbol_info(cand)
                    if s_info:
                        contract_size = float(getattr(s_info, "trade_contract_size", contract_size))
                        step = float(getattr(s_info, "volume_step", step))
                        min_v = float(getattr(s_info, "volume_min", min_v))
                        break
        except Exception:
            pass

        raw_lots = risk_usd / max(sl_distance * contract_size, 0.0001)

        # Clamp to broker volume step
        lots = max(min_v, round(raw_lots / step) * step)
        lots = round(lots, 2)

        actual_risk_usd = round(lots * sl_distance * contract_size, 2)
        return lots, actual_risk_usd

    def on_trade_closed(self, pnl_usd: float, exit_reason: str = "") -> None:
        """Update virtual balance, daily stats, and target progression upon trade close."""
        self.check_day_rollover()
        pnl = float(pnl_usd)

        self.virtual_balance += pnl
        self.daily_realized_pnl += pnl
        self.total_realized_pnl += pnl
        self.daily_unrealized_pnl = 0.0
        self.total_trades += 1

        if pnl > 0:
            self.wins += 1
            self.peak_balance = max(self.peak_balance, self.virtual_balance)
        else:
            self.losses += 1

        # Check if target cleared!
        target_usd = self.initial_balance * (self.config["profit_target_pct"] / 100.0)
        current_profit = self.virtual_balance - self.initial_balance
        if current_profit >= target_usd:
            self.status = "PHASE_1_PASSED"
            self.halt_reason = f"TARGET REACHED: +${current_profit:.2f} (+{(current_profit/self.initial_balance)*100:.1f}%)"

        self.save_state()

    def get_hud_metrics(self) -> dict[str, Any]:
        """Extract formatted metrics for Web Dashboard & Terminal."""
        self.check_day_rollover()
        current_profit = self.virtual_balance - self.initial_balance
        target_usd = self.initial_balance * (self.config["profit_target_pct"] / 100.0)
        progress_pct = max(0.0, min(100.0, (current_profit / target_usd) * 100.0)) if target_usd > 0 else 0.0

        daily_limit_usd = self.day_start_balance * (self.config["max_daily_loss_pct"] / 100.0)
        current_daily_pnl = self.daily_realized_pnl + self.daily_unrealized_pnl
        daily_loss_used_pct = min(100.0, (abs(min(0.0, current_daily_pnl)) / daily_limit_usd) * 100.0)

        max_loss_usd = self.initial_balance * (self.config["max_total_loss_pct"] / 100.0)
        total_dd_usd = max(0.0, self.initial_balance - (self.virtual_balance + self.daily_unrealized_pnl))
        total_dd_pct = (total_dd_usd / self.initial_balance) * 100.0

        win_rate = round((self.wins / self.total_trades) * 100.0, 1) if self.total_trades > 0 else 0.0
        remaining_target_usd = max(0.0, target_usd - current_profit)
        soft_landing_active = (0 < remaining_target_usd <= 100.0) and (current_profit > 0)

        return {
            "challenge_name": self.config["name"],
            "name": self.config["name"],
            "firm": self.config.get("firm", "Prop Firm"),
            "phase": self.config.get("phase", "Phase 1"),
            "profile_key": self.profile_key,
            "initial_balance": float(self.initial_balance),
            "virtual_balance": round(self.virtual_balance, 2),
            "virtual_equity": round(self.virtual_balance, 2),
            "target_balance": round(self.initial_balance + target_usd, 2),
            "current_profit_usd": round(current_profit, 2),
            "cumulative_profit": round(current_profit, 2),
            "target_usd": round(target_usd, 2),
            "profit_target_usd": round(target_usd, 2),
            "profit_target_pct": float(self.config.get("profit_target_pct", 10.0)),
            "remaining_target_usd": round(remaining_target_usd, 2),
            "progress_pct": round(progress_pct, 1),
            "soft_landing_active": soft_landing_active,
            "daily_pnl_usd": round(current_daily_pnl, 2),
            "daily_limit_usd": round(daily_limit_usd, 2),
            "max_daily_loss_usd": round(daily_limit_usd, 2),
            "max_daily_loss_pct": float(self.config.get("max_daily_loss_pct", 5.0)),
            "daily_safety_buffer_pct": float(self.config.get("daily_safety_buffer_pct", 4.0)),
            "daily_loss_used_usd": round(abs(min(0.0, current_daily_pnl)), 2),
            "daily_loss_used_pct": round(daily_loss_used_pct, 1),
            "total_dd_usd": round(total_dd_usd, 2),
            "max_dd_used_usd": round(total_dd_usd, 2),
            "max_dd_usd": round(max_loss_usd, 2),
            "max_loss_usd": round(max_loss_usd, 2),
            "total_dd_pct": round(total_dd_pct, 2),
            "max_dd_limit_pct": float(self.config.get("max_total_loss_pct", 10.0)),
            "total_safety_buffer_pct": float(self.config.get("total_safety_buffer_pct", 8.0)),
            "base_risk_pct": float(self.config.get("base_risk_pct", 0.5)),
            "max_concurrent_trades": int(self.config.get("max_concurrent_trades", 1)),
            "status": self.status,
            "halt_reason": self.halt_reason,
            "trades_count": self.total_trades,
            "win_rate": win_rate,
        }

    def save_state(self) -> None:
        """Persist virtual prop firm challenge state to disk."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "profile_key": self.profile_key,
            "config": self.config,
            "initial_balance": self.initial_balance,
            "virtual_balance": self.virtual_balance,
            "peak_balance": self.peak_balance,
            "day_start_balance": self.day_start_balance,
            "today_date": self.today_date,
            "daily_realized_pnl": self.daily_realized_pnl,
            "total_realized_pnl": self.total_realized_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "total_trades": self.total_trades,
            "status": self.status,
            "halt_reason": self.halt_reason,
            "last_updated": time.time(),
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_custom_config(self, custom_cfg: dict, reset: bool = False) -> bool:
        """
        Dynamically update or override challenge configuration with user-customized values:
        firm, phase, initial_balance, profit_target_pct, max_daily_loss_pct,
        daily_safety_buffer_pct, max_total_loss_pct, total_safety_buffer_pct,
        base_risk_pct, max_concurrent_trades, etc.
        """
        old_initial = self.initial_balance
        profit_offset = self.virtual_balance - old_initial

        for k, v in custom_cfg.items():
            if k == "initial_balance":
                self.config[k] = float(v)
            elif k in ("profit_target_pct", "max_daily_loss_pct", "daily_safety_buffer_pct",
                       "max_total_loss_pct", "total_safety_buffer_pct", "base_risk_pct"):
                self.config[k] = float(v)
            elif k == "max_concurrent_trades":
                self.config[k] = int(v)
            else:
                self.config[k] = v

        if "initial_balance" in custom_cfg:
            self.initial_balance = float(custom_cfg["initial_balance"])

        # Format display name
        firm_str = self.config.get("firm", custom_cfg.get("firm_name", "Prop Firm"))
        size_str = f"${int(self.initial_balance):,}" if self.initial_balance >= 1000 else f"${self.initial_balance}"
        phase_str = self.config.get("phase", "")
        self.config["name"] = f"{firm_str} {size_str} ({phase_str})".strip() if phase_str else f"{firm_str} {size_str}"
        self.profile_key = f"custom_{int(self.initial_balance)}"

        if reset or self.virtual_balance <= 0:
            self.reset_challenge()
        else:
            self.virtual_balance = self.initial_balance + profit_offset
            self.peak_balance = max(self.virtual_balance, self.initial_balance)
            self.day_start_balance = self.initial_balance
            self.save_state()

        logger.info("prop_firm_custom_updated", config=self.config)
        return True

    def switch_profile(self, profile_key: str, reset: bool = False) -> bool:
        """Dynamically switch challenge profile (e.g. SharkFunded -> FTMO -> FundedNext)."""
        if profile_key not in PROFILES:
            return False
        self.profile_key = profile_key
        self.config = PROFILES[profile_key]
        old_initial = self.initial_balance
        profit_offset = self.virtual_balance - old_initial
        self.initial_balance = float(self.config["initial_balance"])
        if reset or self.virtual_balance <= 0:
            self.reset_challenge()
        else:
            self.virtual_balance = self.initial_balance + profit_offset
            self.peak_balance = max(self.virtual_balance, self.initial_balance)
            self.day_start_balance = self.initial_balance
            self.save_state()
        logger.info("prop_firm_profile_switched", profile=profile_key, name=self.config["name"])
        return True

    def reset_challenge(self) -> None:
        """Reset the challenge metrics to initial state (e.g. starting new challenge)."""
        self.initial_balance = float(self.config["initial_balance"])
        self.virtual_balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.day_start_balance = self.initial_balance
        self.today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.daily_realized_pnl = 0.0
        self.daily_unrealized_pnl = 0.0
        self.total_realized_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.total_trades = 0
        self.status = "IN_PROGRESS"
        self.halt_reason = ""
        self.save_state()
        logger.info("prop_firm_challenge_reset", profile=self.profile_key, balance=self.virtual_balance)

    def load_state(self) -> bool:
        """Load virtual prop firm challenge state from disk."""
        if not self.state_file.exists():
            return False
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self.profile_key = data.get("profile_key", self.profile_key)
            if "config" in data and isinstance(data["config"], dict):
                self.config = data["config"]
                self.initial_balance = float(self.config.get("initial_balance", self.initial_balance))
            elif self.profile_key in PROFILES:
                self.config = PROFILES[self.profile_key]
                self.initial_balance = float(self.config["initial_balance"])
            self.virtual_balance = float(data.get("virtual_balance", self.initial_balance))
            self.peak_balance = float(data.get("peak_balance", self.initial_balance))
            self.day_start_balance = float(data.get("day_start_balance", self.initial_balance))
            self.today_date = data.get("today_date", self.today_date)
            self.daily_realized_pnl = float(data.get("daily_realized_pnl", 0.0))
            self.total_realized_pnl = float(data.get("total_realized_pnl", 0.0))
            self.wins = int(data.get("wins", 0))
            self.losses = int(data.get("losses", 0))
            self.total_trades = int(data.get("total_trades", 0))
            self.status = data.get("status", "IN_PROGRESS")
            self.halt_reason = data.get("halt_reason", "")
            return True
        except Exception as e:
            logger.warn("prop_firm_state_load_failed", error=str(e))
            return False

