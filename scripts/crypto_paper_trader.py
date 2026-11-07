"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        APEX INTELLIGENCE ENGINE v5.0 — INSTITUTIONAL QUANT LAB               ║
║         Multi-Alpha Consensus • Macro Trend Lock • 1:2.5+ Asymmetric R:R     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Autonomous Institutional Trading Engine with Survival Objective:
1. Capital is Life (HP): $1,000 bankroll. Max drawdown circuit breaker at 15%.
   - HP gauge dynamically controls risk exposure (Thriving -> Defensive -> Bunker).
2. Multi-Alpha Tournament Arena:
   - Alpha 1: Order Flow Momentum Breakout (OF-MOM)
   - Alpha 2: Liquidity Sweep & Mitigation (LIQ-REV)
   - Alpha 3: Structural Order Book Wall Absorption (OB-ABSORB)
   - Performance-Weighted Consensus: Alphas with WR < 35% are automatically QUARANTINED.
3. Macro Trend Lockdown (Zero Falling Knives):
   - 15m/1h Trend dictates directional permission:
     * Bearish regime -> Longs strictly prohibited (No counter-trend dip buying).
     * Bullish regime -> Shorts strictly prohibited.
4. Institutional Asymmetric Payoff Engineering (1:2.5+ R:R):
   - Sizing strictly accounts for roundtrip taker fees + slippage (14 bps).
   - TP1 (2.5R target): Locks 70% of position profit.
   - Runner (30% position): Protected at Breakeven, trailing with wide 2.5x ATR buffer to 4.5R.
5. Anti-Whipsaw Directional Streak Protection:
   - 2 consecutive losses in same direction triggers 45m directional lockout.
6. Executive Decongested Quant Cockpit:
   - Crystal-clear visual hierarchy, spacious cards, zero flicker ANSI layout.
"""

import asyncio
import json
import websockets
import datetime
import time
import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

if os.name == 'nt':
    os.system('')  # Enable ANSI color escape processing in Windows Console

import numpy as np
import requests
from collections import deque

# ---------------------------------------------------------
# ANSI COLOR PALETTE (EXECUTIVE QUANT TERMINAL)
# ---------------------------------------------------------
C_RESET   = "\033[0m"
C_BOLD    = "\033[1m"
C_DIM     = "\033[2m"
C_RED     = "\033[91m"
C_GREEN   = "\033[92m"
C_YELLOW  = "\033[93m"
C_BLUE    = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN    = "\033[96m"
C_WHITE   = "\033[97m"
C_GRAY    = "\033[90m"

# ---------------------------------------------------------
# CONSTANTS & EXECUTION MODEL
# ---------------------------------------------------------
SYMBOL                   = "BTCUSDT"
INITIAL_CAPITAL          = 1000.0
MAX_DRAWDOWN_LIMIT_PCT   = 15.0   # Survival fail-safe at 15% drawdown ($150 buffer)
TAKER_FEE_BPS            = 5.0
SLIPPAGE_BPS             = 2.0
ROUNDTRIP_COST_BPS       = (TAKER_FEE_BPS + SLIPPAGE_BPS) * 2  # 14.0 bps

TRADE_JOURNAL_FILE       = "trade_journal.json"

# ---------------------------------------------------------
# GLOBAL STATE & TELEMETRY
# ---------------------------------------------------------
market_state = {
    "symbol":              SYMBOL,
    "ws_connected":        False,
    "ltp":                 0.0,
    "bid":                 0.0,
    "ask":                 0.0,
    "spread_bps":          0.0,
    "vpin":                0.0,
    "vpin_percentile":     50,
    "cofi_z":              0.0,
    "cofi_z_ema":          0.0,
    "cofi_direction":      0,      # +1 buy, -1 sell, 0 neutral
    "obi":                 0.0,    # L2 Order Book Imbalance (-1.0 to +1.0)
    "ticks_today":         0,
    "buckets_completed":   0,
    "bucket_velocity_sec": 0.0,
    "atr_14":              35.0,
    "atr_bps":             4.0,
    "current_minute_ts":   0,
}

macro_structure = {
    "hvns":       [],
    "poc":        0.0,
    "micro_hvns": [],
}

cvd_state = {
    "cvd":            0.0,
    "cvd_history":    deque(maxlen=60),
    "price_history":  deque(maxlen=60),
    "divergence":     "NONE",
    "delta_momentum": 0.0,
    "5min_buy_vol":   0.0,
    "5min_sell_vol":  0.0,
}

sweep_state = {
    "recent_highs":      deque(maxlen=25),
    "recent_lows":       deque(maxlen=25),
    "last_sweep":        "NONE",
    "sweep_price":       0.0,
    "sweep_candles_ago": 0,
}

ob_state = {
    "bullish_obs":  [],
    "bearish_obs":  [],
}

fvg_state = {
    "bullish_fvgs": [],
    "bearish_fvgs": [],
}

mtf_state = {
    "ema_20":   0.0,
    "ema_50":   0.0,
    "trend":    "NEUTRAL",
    "slope":    0.0,
    "strength": 0.0,
}

session_state = {
    "current":           "UNKNOWN",
    "execution_allowed": True,
    "size_multiplier":   1.0,
}

# ---------------------------------------------------------
# SURVIVAL & HEALTH SYSTEM (Capital as Life / HP)
# ---------------------------------------------------------
survival = {
    "initial_capital":       INITIAL_CAPITAL,
    "account_balance":       INITIAL_CAPITAL,
    "peak_balance":          INITIAL_CAPITAL,
    "max_drawdown_limit":    MAX_DRAWDOWN_LIMIT_PCT,
    "current_drawdown_pct":  0.0,
    "hp_pct":                100.0,
    "mode":                  "THRIVING",   # THRIVING (>70%), DEFENSIVE (40-70%), BUNKER (<40%), ELIMINATED (0%)
    "consecutive_losses":    0,
    "circuit_breaker":       False,
    "global_cooldown_until": 0.0,          # Global halt across ALL alphas
    "session_halt_until":    0.0,          # Rolling multi-loss circuit breaker
    "cooldown_reason":       "",           # Reason for active halt
    "rolling_losses":        deque(maxlen=20), # (timestamp, pnl_usd)
    "directional_lock":      {"direction": None, "until": 0.0}, # Anti-whipsaw lockout
}

# ---------------------------------------------------------
# MULTI-ALPHA TOURNAMENT ARENA
# ---------------------------------------------------------
alphas = {
    "OF_MOM": {
        "name":        "OF-MOM (Order Flow Momentum)",
        "tag":         "OF-MOM",
        "weight":      0.50,
        "conviction":  0.0,      # -1.0 to +1.0
        "direction":   "FLAT",   # LONG, SHORT, FLAT
        "confidence":  0,        # 0 to 100%
        "rationale":   "Initializing...",
        "trades":      0,
        "wins":        0,
        "losses":      0,
        "pnl":         0.0,
        "rolling_wr":  50.0,
        "status":      "ACTIVE", # ACTIVE, PROBATION, QUARANTINED
    },
    "LIQ_REV": {
        "name":        "LIQ-REV (Liquidity Sweep Fade)",
        "tag":         "LIQ-REV",
        "weight":      0.25,
        "conviction":  0.0,
        "direction":   "FLAT",
        "confidence":  0,
        "rationale":   "Initializing...",
        "trades":      0,
        "wins":        0,
        "losses":      0,
        "pnl":         0.0,
        "rolling_wr":  50.0,
        "status":      "ACTIVE",
    },
    "OB_ABSORB": {
        "name":        "OB-ABSORB (Order Book Wall Defense)",
        "tag":         "OB-ABSORB",
        "weight":      0.25,
        "conviction":  0.0,
        "direction":   "FLAT",
        "confidence":  0,
        "rationale":   "Initializing...",
        "trades":      0,
        "wins":        0,
        "losses":      0,
        "pnl":         0.0,
        "rolling_wr":  50.0,
        "status":      "ACTIVE",
    }
}

# Active position state
portfolio = {
    "position":              "FLAT",     # FLAT, LONG, SHORT
    "phase":                 0,          # 0=flat, 1=initial full, 2=runner post 70% scale
    "entry_price":           0.0,
    "sl_distance":           0.0,
    "active_sl_price":       0.0,
    "tp1_price":             0.0,        # 2.5R target (70% scale-out)
    "active_tp_price":       0.0,        # 4.5R target (30% runner)
    "notional_size_usd":     0.0,
    "initial_notional_usd":  0.0,
    "leverage_used":         0.0,
    "realized_pnl":          0.0,
    "banked_pnl":            0.0,
    "unrealized_pnl":        0.0,
    "highest_unrealized":    0.0,
    "trade_open_time":       0.0,
    "lead_alpha":            "",
    "be_moved":              False,
    "entry_reason":          "",
}

alpha_cooldowns = {
    "OF_MOM": 0.0,
    "LIQ_REV": 0.0,
    "OB_ABSORB": 0.0,
}

# Trade history & rolling structures
trade_journal         = []
recent_trades         = deque(maxlen=8)
recent_post_mortems   = deque(maxlen=6)

current_bucket        = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0, "start_time": time.time()}
vpin_history          = deque(maxlen=500)
order_flow_imbalances = deque(maxlen=500)
cofi_z_history        = deque(maxlen=50)

minute_candles          = deque(maxlen=120)
live_candle             = {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}
five_min_candles        = deque(maxlen=100)
five_min_accumulator    = []
fifteen_min_candles     = deque(maxlen=60)
fifteen_min_accumulator = []

liquidation_events      = deque()
liquidations            = {"longs_usd": 0.0, "shorts_usd": 0.0}

DYNAMIC_PARAMS = {
    "BUCKET_VOLUME_SIZE": 10.0,
    "TARGET_RR_RATIO":    2.5,   # TP1 target (70% position exit)
    "SCALE_OUT_FRAC":     0.70,  # Lock 70% profit at TP1
    "RUNNER_RR_RATIO":    4.5,   # TP2 target (30% runner exit)
    "RUNNER_FRAC":        0.30,  # 30% runner position
}

# ---------------------------------------------------------
# DATABASE & JOURNAL STORAGE (REHYDRATION ON RESTART)
# ---------------------------------------------------------
def save_trade_journal():
    try:
        with open(TRADE_JOURNAL_FILE, "w") as f:
            json.dump(trade_journal, f, indent=2)
    except Exception:
        pass


def update_alpha_quarantine_status():
    """
    Evaluates rolling win rates and benches underperforming alphas (< 35% WR).
    Guarantees at least 1 viable alpha by keeping the highest performer on probation.
    """
    candidates = []
    for k, a in alphas.items():
        tr = a.get("trades", 0)
        wr = a.get("rolling_wr", 50.0)
        if tr >= 3 and wr < 35.0:
            a["status"] = "QUARANTINED"
            a["weight"] = 0.05
        else:
            a["status"] = "ACTIVE"
        candidates.append((k, wr, tr))

    active_count = sum(1 for a in alphas.values() if a["status"] == "ACTIVE")
    if active_count == 0:
        # Keep best performing alpha on PROBATION so engine can operate with strict trend locks
        best_k = max(candidates, key=lambda x: (x[1], -x[2]))[0]
        alphas[best_k]["status"] = "PROBATION"
        alphas[best_k]["weight"] = 0.70

    # Normalize weights across active and probationary alphas
    active_keys = [k for k, a in alphas.items() if a["status"] in ["ACTIVE", "PROBATION"]]
    if active_keys:
        tot_w = sum(alphas[k]["weight"] for k in active_keys)
        for k in active_keys:
            alphas[k]["weight"] = round(alphas[k]["weight"] / tot_w, 3)
    
    # Quarantined alphas have weight 0.00 in consensus
    for k, a in alphas.items():
        if a["status"] == "QUARANTINED":
            a["weight"] = 0.00


def load_trade_journal():
    global trade_journal
    try:
        if os.path.exists(TRADE_JOURNAL_FILE):
            with open(TRADE_JOURNAL_FILE, "r") as f:
                trade_journal = json.load(f)

            # Clean up stale dangling open trades if more than 1 exists
            open_trades = [t for t in trade_journal if t.get("outcome") == "OPEN"]
            if len(open_trades) > 1:
                for t in open_trades[:-1]:
                    t["outcome"] = "CANCELLED"
                    t["exit_reason"] = "STALE_ABANDONED"
                    t["exit_ts"] = datetime.datetime.now().isoformat()
                    t["post_mortem"] = "Voided: Abandoned open trade from prior process crash."
                save_trade_journal()

            closed = [t for t in trade_journal if t.get("outcome") in ["WIN", "LOSS"]]
            if closed:
                current_run_trades = [t for t in closed if t.get("id", 0) >= 8]
                target_trades = current_run_trades if current_run_trades else closed
                session_pnl = sum(t.get("pnl_usd", 0.0) for t in target_trades)

                survival["account_balance"] = round(INITIAL_CAPITAL + session_pnl, 2)

                # Reconstruct peak balance
                running_bal = INITIAL_CAPITAL
                peak_bal = INITIAL_CAPITAL
                for t in target_trades:
                    running_bal += t.get("pnl_usd", 0.0)
                    if running_bal > peak_bal:
                        peak_bal = running_bal
                survival["peak_balance"] = round(peak_bal, 2)

                # Reconstruct stats for each alpha
                for k in alphas:
                    alpha_trades = [t for t in closed if t.get("lead_alpha") == k]
                    if alpha_trades:
                        wins = sum(1 for t in alpha_trades if t.get("outcome") == "WIN")
                        losses = sum(1 for t in alpha_trades if t.get("outcome") == "LOSS")
                        pnl = sum(t.get("pnl_usd", 0.0) for t in alpha_trades)
                        alphas[k]["trades"] = len(alpha_trades)
                        alphas[k]["wins"] = wins
                        alphas[k]["losses"] = losses
                        alphas[k]["pnl"] = round(pnl, 2)
                        alphas[k]["rolling_wr"] = round((wins / len(alpha_trades)) * 100.0, 1)

                update_alpha_quarantine_status()

                # Calculate consecutive losses streak on current tournament run (within last 3 hours)
                streak = 0
                now_epoch = time.time()
                for t in reversed(target_trades):
                    if t.get("outcome") == "LOSS":
                        # Time decay preservation scratches do not count towards consecutive loss streaks
                        if "TIME DECAY" in t.get("exit_reason", ""):
                            continue
                        ex_ts = t.get("exit_ts")
                        ts_epoch = 0.0
                        try:
                            if isinstance(ex_ts, str):
                                ts_epoch = datetime.datetime.fromisoformat(ex_ts).timestamp()
                            elif isinstance(ex_ts, list) and ex_ts:
                                ts_epoch = datetime.datetime.fromisoformat(ex_ts[0]).timestamp()
                        except Exception:
                            ts_epoch = 0.0
                        if ts_epoch > 0 and (now_epoch - ts_epoch) > 10800.0:
                            break
                        streak += 1
                    else:
                        break
                survival["consecutive_losses"] = min(streak, 3)

                # Rehydrate rolling losses
                survival["rolling_losses"].clear()
                for t in closed:
                    if t.get("outcome") == "LOSS":
                        ex_ts = t.get("exit_ts")
                        ts_epoch = 0.0
                        try:
                            if isinstance(ex_ts, str):
                                ts_epoch = datetime.datetime.fromisoformat(ex_ts).timestamp()
                            elif isinstance(ex_ts, list) and ex_ts:
                                ts_epoch = datetime.datetime.fromisoformat(ex_ts[0]).timestamp()
                        except Exception:
                            ts_epoch = 0.0
                        if ts_epoch > 0 and (now_epoch - ts_epoch) <= 7200.0:
                            survival["rolling_losses"].append((ts_epoch, t.get("pnl_usd", -5.0)))

                # Populate recent trades and post-mortems
                recent_trades.clear()
                for t in closed[-5:]:
                    action = t.get("action", "BUY")
                    pnl_val = t.get("pnl_usd", 0.0)
                    sign = "+" if pnl_val >= 0 else ""
                    lead_tag = t.get("lead_alpha", "")
                    reason_tag = t.get("exit_reason", "CLOSED")
                    msg = (f"[{t.get('open_ts','')[11:19]}] {action:4} | Entry: {t.get('entry_price',0):,.1f} | "
                           f"Exit: {t.get('exit_price',0):,.1f} | PnL: ${sign}{pnl_val:.2f} | {reason_tag} [{lead_tag}]")
                    recent_trades.append(msg)

                recent_post_mortems.clear()
                for t in closed[-4:]:
                    if t.get("post_mortem"):
                        pm = f"[{t.get('exit_ts','')[11:19] if isinstance(t.get('exit_ts'), str) else ''}] [{t.get('lead_alpha','')}] {t.get('post_mortem')} (${t.get('pnl_usd',0):+.2f})"
                        recent_post_mortems.append(pm)

            # Rehydrate active trade if open
            if open_trades:
                latest_open = open_trades[-1]
                pos_dir = "LONG" if latest_open.get("action") == "BUY" else "SHORT"
                ep = latest_open.get("entry_price", 0.0)
                sl_p = latest_open.get("sl_price", 0.0)
                sl_dist = abs(ep - sl_p) if ep > 0 and sl_p > 0 else 150.0
                tp1_p = latest_open.get("tp_price", 0.0)
                tp2_p = latest_open.get("tp2_price", 0.0)
                if tp2_p == 0.0 or tp2_p == tp1_p:
                    tp2_p = ep + sl_dist * 4.5 if pos_dir == "LONG" else ep - sl_dist * 4.5
                portfolio.update({
                    "position":             pos_dir,
                    "phase":                1,
                    "entry_price":          ep,
                    "sl_distance":          sl_dist,
                    "active_sl_price":      sl_p,
                    "tp1_price":            tp1_p,
                    "active_tp_price":      tp2_p,
                    "notional_size_usd":    latest_open.get("notional_usd", 0.0),
                    "initial_notional_usd": latest_open.get("notional_usd", 0.0),
                    "leverage_used":        round(latest_open.get("notional_usd", 0.0) / max(1.0, survival["account_balance"]), 1),
                    "realized_pnl":         0.0,
                    "banked_pnl":           0.0,
                    "unrealized_pnl":       0.0,
                    "highest_unrealized":   0.0,
                    "trade_open_time":      time.time(),
                    "lead_alpha":           latest_open.get("lead_alpha", ""),
                    "be_moved":             False,
                    "entry_reason":         latest_open.get("reason", ""),
                })

            update_survival_health()
    except Exception as e:
        print(f"Error loading journal: {e}")


# ---------------------------------------------------------
# SURVIVAL HP CONTROLLER
# ---------------------------------------------------------
def update_survival_health():
    bal = survival["account_balance"]
    peak = survival["peak_balance"]
    if bal > peak:
        survival["peak_balance"] = bal
        peak = bal

    dd_usd = peak - bal
    dd_pct = (dd_usd / peak * 100.0) if peak > 0 else 0.0
    survival["current_drawdown_pct"] = dd_pct

    max_dd = survival["max_drawdown_limit"]
    hp = max(0.0, 100.0 * (1.0 - (dd_pct / max_dd)))
    survival["hp_pct"] = round(hp, 1)

    if hp <= 0.0:
        survival["mode"] = "ELIMINATED"
        survival["circuit_breaker"] = True
    elif hp < 15.0:
        survival["mode"] = "SURVIVAL BUNKER (HALTED)"
    elif hp < 45.0:
        survival["mode"] = "DEFENSIVE RECOVERY 🛡️"
    elif hp < 70.0:
        survival["mode"] = "DEFENSIVE 🟡"
    else:
        survival["mode"] = "THRIVING 🟢"


def get_survival_risk_scalar():
    """Returns dynamic risk scaling based on current HP. Hard freeze only under 15% HP."""
    hp = survival["hp_pct"]
    if survival["circuit_breaker"] or hp < 15.0:
        return 0.0     # Hard capital preservation freeze only near actual 15% DD limit
    elif hp < 45.0:
        return 0.40    # Defensive recovery: conservative sizing ($2.50-$3.50 risk)
    elif hp < 70.0:
        return 0.65    # Defensive sizing with active recovery capacity
    else:
        return 1.00    # Full capital deployment


def check_circuit_breakers():
    """Checks session limits and rolling losses to enforce institutional cooldowns."""
    now_t = time.time()
    if survival["circuit_breaker"] or survival["hp_pct"] <= 0.0:
        return False, "Circuit breaker tripped: Max Drawdown hit."
    if survival["hp_pct"] < 15.0:
        return False, "SURVIVAL BUNKER: Trading halted (Capital buffer exhausted <15% HP)."
    if now_t < survival.get("session_halt_until", 0.0):
        rem = int(survival["session_halt_until"] - now_t)
        return False, f"Session halt active: {rem}s cooling ({survival.get('cooldown_reason','')})."
    if now_t < survival.get("global_cooldown_until", 0.0):
        rem = int(survival["global_cooldown_until"] - now_t)
        return False, f"Global cooldown: {rem}s cooling ({survival.get('cooldown_reason','')})."

    # Check rolling losses in last 2 hours (7200 seconds)
    two_hours_ago = now_t - 7200.0
    recent_losses = [pnl for t, pnl in survival.get("rolling_losses", []) if t >= two_hours_ago]
    if len(recent_losses) >= 3:
        survival["session_halt_until"] = now_t + 2700.0  # 45 minute halt
        survival["cooldown_reason"] = "3 losses in 2h"
        return False, "Session halt: 3 losses in 2h (45m cooling)."

    return True, "CLEAR"


# ---------------------------------------------------------
# DYNAMIC MULTI-ARMED BANDIT META-ALLOCATOR
# ---------------------------------------------------------
def reward_alpha(lead_alpha_key, is_win, pnl_usd):
    """Reinforces winning alphas and penalizes losing alphas dynamically."""
    if lead_alpha_key not in alphas:
        return

    a = alphas[lead_alpha_key]
    a["trades"] += 1
    a["pnl"]    += pnl_usd
    if is_win:
        a["wins"] += 1
        a["weight"] *= 1.30
    else:
        a["losses"] += 1
        a["weight"] *= 0.75

    tot = a["wins"] + a["losses"]
    a["rolling_wr"] = (a["wins"] / tot * 100.0) if tot > 0 else 50.0
    update_alpha_quarantine_status()


# ---------------------------------------------------------
# ALPHA 1: ORDER FLOW MOMENTUM BREAKOUT (OF-MOM)
# ---------------------------------------------------------
def evaluate_alpha_of_mom():
    cofi_z_ema = market_state.get("cofi_z_ema", 0.0)
    vpin_pct  = market_state["vpin_percentile"]
    obi       = market_state["obi"]
    mtf_trend = mtf_state["trend"]
    mtf_str   = mtf_state["strength"]
    atr_bps   = market_state.get("atr_bps", 4.0)

    score_long  = 0.0
    score_short = 0.0
    notes       = []

    # Short-term price velocity over last 3 minutes (bps return)
    if len(minute_candles) >= 3:
        p_base = minute_candles[-3]["close"]
        price_vel_bps = ((market_state["ltp"] - p_base) / p_base) * 10000.0 if p_base > 0 else 0.0
    else:
        price_vel_bps = 0.0

    is_mtf_bull = mtf_str > 5.0 or mtf_trend in ["BULLISH", "BULLISH_EXPANSION", "BULLISH_SLOPE"]
    is_mtf_bear = mtf_str < -5.0 or mtf_trend in ["BEARISH", "BEARISH_BREAKDOWN", "BEARISH_SLOPE"]
    is_cascade_down = mtf_str < -10.0 or mtf_trend in ["BEARISH", "BEARISH_BREAKDOWN", "BEARISH_SLOPE"]
    is_cascade_up   = mtf_str > 10.0 or mtf_trend in ["BULLISH", "BULLISH_EXPANSION", "BULLISH_SLOPE"]

    # Flow-Price Concordance & Trend Expansion using Institutional Z-Scores & Volatility
    vol_adjusted_velocity = max(2.0, atr_bps * 0.25)
    
    if cofi_z_ema >= 1.5 and (price_vel_bps >= vol_adjusted_velocity or (is_mtf_bull and cofi_z_ema >= 2.5)):
        if not is_cascade_down:
            score_long += 0.40
            notes.append(f"COFI Z {cofi_z_ema:+.1f}σ")
            if is_mtf_bull:
                score_long += 0.30
                notes.append("MTF Aligned")
    elif cofi_z_ema <= -1.5 and (price_vel_bps <= -vol_adjusted_velocity or (is_mtf_bear and cofi_z_ema <= -2.5)):
        if not is_cascade_up:
            score_short += 0.40
            notes.append(f"COFI Z {cofi_z_ema:+.1f}σ")
            if is_mtf_bear:
                score_short += 0.30
                notes.append("MTF Aligned")

    # VPIN Confirmation
    if vpin_pct >= 75:
        if score_long > 0:
            score_long += 0.15
            notes.append(f"VPIN {vpin_pct}th")
        elif score_short > 0:
            score_short += 0.15
            notes.append(f"VPIN {vpin_pct}th")

    # L2 OBI Confirmation
    if obi > 0.35 and cofi_z_ema > 0:
        score_long += 0.15
        notes.append(f"L2 OBI +{obi:.2f}")
    elif obi < -0.35 and cofi_z_ema < 0:
        score_short += 0.15
        notes.append(f"L2 OBI {obi:.2f}")

    # Strict Regime Veto
    if is_cascade_down:
        score_long = 0.0
    if is_cascade_up:
        score_short = 0.0

    net_conviction = score_long - score_short
    net_conviction = max(-1.0, min(1.0, net_conviction))

    direction = "LONG" if net_conviction >= 0.40 else ("SHORT" if net_conviction <= -0.40 else "FLAT")
    conf      = int(abs(net_conviction) * 100)

    alphas["OF_MOM"].update({
        "conviction": round(net_conviction, 2),
        "direction":  direction,
        "confidence": conf,
        "rationale":  " | ".join(notes) or "Consolidating",
    })
    return net_conviction


# ---------------------------------------------------------
# ALPHA 2: LIQUIDITY SWEEP & MITIGATION MEAN REVERSION (LIQ-REV)
# ---------------------------------------------------------
def evaluate_alpha_liq_rev():
    ltp         = market_state["ltp"]
    sweep       = sweep_state["last_sweep"]
    sweep_fresh = sweep_state["sweep_candles_ago"] <= 3 and sweep not in ["NONE", "FAILED", "INVALIDATED"]
    cvd_div     = cvd_state["divergence"]
    cofi_ema    = market_state["cofi_z_ema"]
    atr         = market_state["atr_14"]
    mtf_str     = mtf_state["strength"]
    mtf_trend   = mtf_state["trend"]

    score_long  = 0.0
    score_short = 0.0
    notes       = []

    is_cascade_down = mtf_str < -10.0 or mtf_trend in ["BEARISH", "BEARISH_BREAKDOWN", "BEARISH_SLOPE"]
    is_cascade_up   = mtf_str > 10.0 or mtf_trend in ["BULLISH", "BULLISH_EXPANSION", "BULLISH_SLOPE"]

    # Only take sweeps IN THE DIRECTION of the macro trend
    if sweep == "BULLISH_SWEEP" and sweep_fresh:
        sweep_px = sweep_state["sweep_price"]
        if is_cascade_down:
            notes.append("Bull Sweep (Vetoed: Bear Trend)")
        elif ltp < sweep_px:
            notes.append("Low Broken (No Reclaim)")
        elif ltp > sweep_px + atr * 1.5:
            sweep_state["last_sweep"] = "FAILED"
            notes.append("Sweep Stale / Extended")
        else:
            score_long += 0.45
            notes.append(f"Reclaimed Low @ {sweep_px:,.0f}")
    elif sweep == "BEARISH_SWEEP" and sweep_fresh:
        sweep_px = sweep_state["sweep_price"]
        if is_cascade_up:
            notes.append("Bear Sweep (Vetoed: Bull Trend)")
        elif ltp > sweep_px:
            notes.append("High Broken (No Reclaim)")
        elif ltp < sweep_px - atr * 1.5:
            sweep_state["last_sweep"] = "FAILED"
            notes.append("Sweep Stale / Extended")
        else:
            score_short += 0.45
            notes.append(f"Reclaimed High @ {sweep_px:,.0f}")

    # Delta Divergence Confirmation
    if cvd_div == "BULLISH" and not is_cascade_down:
        score_long += 0.25
        notes.append("Bullish CVD Div")
    elif cvd_div == "BEARISH" and not is_cascade_up:
        score_short += 0.25
        notes.append("Bearish CVD Div")

    # Volume Profile Proximity
    all_levels = macro_structure["hvns"] + macro_structure["micro_hvns"] + [macro_structure["poc"]]
    near_struct = any(abs(ltp - lvl) <= atr * 0.8 for lvl in all_levels if lvl > 0)
    if near_struct:
        if score_long > 0:
            score_long += 0.20
            notes.append("At VP Structure")
        elif score_short > 0:
            score_short += 0.20
            notes.append("At VP Structure")

    # Hard Regime Veto
    if is_cascade_down:
        score_long = 0.0
    if is_cascade_up:
        score_short = 0.0

    net_conviction = score_long - score_short
    net_conviction = max(-1.0, min(1.0, net_conviction))

    direction = "LONG" if net_conviction >= 0.40 else ("SHORT" if net_conviction <= -0.40 else "FLAT")
    conf      = int(abs(net_conviction) * 100)

    alphas["LIQ_REV"].update({
        "conviction": round(net_conviction, 2),
        "direction":  direction,
        "confidence": conf,
        "rationale":  " | ".join(notes) or "No Sweep Trigger",
    })
    return net_conviction


# ---------------------------------------------------------
# ALPHA 3: STRUCTURAL ORDER BOOK WALL ABSORPTION (OB-ABSORB)
# ---------------------------------------------------------
def evaluate_alpha_ob_absorb():
    ltp       = market_state["ltp"]
    obi       = market_state["obi"]
    mtf_str   = mtf_state["strength"]
    mtf_trend = mtf_state["trend"]
    cvd_mom   = cvd_state["delta_momentum"]

    score_long  = 0.0
    score_short = 0.0
    notes       = []

    is_cascade_down = mtf_str < -10.0 or mtf_trend in ["BEARISH", "BEARISH_BREAKDOWN", "BEARISH_SLOPE"]
    is_cascade_up   = mtf_str > 10.0 or mtf_trend in ["BULLISH", "BULLISH_EXPANSION", "BULLISH_SLOPE"]

    # Bullish OB mitigation
    for ob in ob_state["bullish_obs"]:
        if ob["low"] <= ltp <= ob["high"]:
            if is_cascade_down:
                notes.append("Bull OB (Bypassed: Bear Trend)")
            else:
                score_long += 0.40
                notes.append(f"Bull OB [{ob['low']:,.0f}-{ob['high']:,.0f}]")
            break

    # Bearish OB mitigation
    for ob in ob_state["bearish_obs"]:
        if ob["low"] <= ltp <= ob["high"]:
            if is_cascade_up:
                notes.append("Bear OB (Bypassed: Bull Trend)")
            else:
                score_short += 0.40
                notes.append(f"Bear OB [{ob['low']:,.0f}-{ob['high']:,.0f}]")
            break

    # Order Book Wall Defense with delta anti-spoofing
    if obi >= 0.45:
        if cvd_mom < -25.0:
            notes.append("Bid Wall (Wall Smashed by Sells)")
        elif not is_cascade_down:
            score_long += 0.35
            notes.append(f"Bid Wall ({obi:+.2f})")
    elif obi <= -0.45:
        if cvd_mom > 25.0:
            notes.append("Ask Wall (Wall Smashed by Buys)")
        elif not is_cascade_up:
            score_short += 0.35
            notes.append(f"Ask Wall ({obi:+.2f})")

    # Hard Regime Veto
    if is_cascade_down:
        score_long = 0.0
    if is_cascade_up:
        score_short = 0.0

    net_conviction = score_long - score_short
    net_conviction = max(-1.0, min(1.0, net_conviction))

    direction = "LONG" if net_conviction >= 0.40 else ("SHORT" if net_conviction <= -0.40 else "FLAT")
    conf      = int(abs(net_conviction) * 100)

    alphas["OB_ABSORB"].update({
        "conviction": round(net_conviction, 2),
        "direction":  direction,
        "confidence": conf,
        "rationale":  " | ".join(notes) or "Open Space",
    })
    return net_conviction


# ---------------------------------------------------------
# MULTI-ALPHA CONTINUOUS EVALUATION
# ---------------------------------------------------------
def update_alpha_evaluations():
    """Evaluates all 3 Alphas and calculates Performance-Weighted Consensus."""
    c1 = evaluate_alpha_of_mom()
    c2 = evaluate_alpha_liq_rev()
    c3 = evaluate_alpha_ob_absorb()

    convictions = {"OF_MOM": c1, "LIQ_REV": c2, "OB_ABSORB": c3}

    # Only ACTIVE or PROBATION alphas contribute to consensus
    active_alphas = {k: v for k, v in alphas.items() if v.get("status") in ["ACTIVE", "PROBATION"]}
    if active_alphas:
        tot_w = sum(a["weight"] for a in active_alphas.values())
        consensus = sum(convictions[k] * alphas[k]["weight"] for k in active_alphas) / tot_w if tot_w > 0 else 0.0
    else:
        consensus = 0.0

    return c1, c2, c3, consensus


# ---------------------------------------------------------
# META-ALLOCATOR DECISION ENGINE
# ---------------------------------------------------------
def run_meta_decision(c1=None, c2=None, c3=None, consensus_conviction=None):
    if portfolio["position"] != "FLAT":
        return

    # 1. Execution session gate
    if not session_state["execution_allowed"] or session_state["size_multiplier"] <= 0.0:
        return

    # 2. Circuit Breaker & Survival Gates
    can_trade, reason = check_circuit_breakers()
    if not can_trade:
        return

    if c1 is None or consensus_conviction is None:
        c1, c2, c3, consensus_conviction = update_alpha_evaluations()

    convictions = {"OF_MOM": c1, "LIQ_REV": c2, "OB_ABSORB": c3}

    # 3. Dynamic Consensus Entry Threshold
    sess = session_state["current"]
    # High threshold in Asian chop / Pacific lull (0.58) stops whipsaw bleed
    base_threshold = 0.58 if sess in ["ASIAN", "PACIFIC_LULL"] else 0.48

    streak = survival.get("consecutive_losses", 0)
    streak_penalty = min(0.12, streak * 0.03)
    entry_threshold = base_threshold + streak_penalty

    # Spread gate: only enter if spread is tight (< 8.0 bps)
    if market_state["spread_bps"] > 8.0:
        return

    # Trigger based on WEIGHTED CONSENSUS
    is_long  = consensus_conviction >= entry_threshold
    is_short = consensus_conviction <= -entry_threshold

    if not (is_long or is_short):
        return

    # 4. HARD MACRO REGIME GATE (15m/1h Trend Lockdown)
    # Never buy into a downtrend; never short an uptrend
    macro_trend = mtf_state["trend"]
    if macro_trend in ["BEARISH", "BEARISH_BREAKDOWN", "BEARISH_SLOPE"]:
        if is_long:
            return  # Strict veto: Longs blocked during bear regime!
    elif macro_trend in ["BULLISH", "BULLISH_EXPANSION", "BULLISH_SLOPE"]:
        if is_short:
            return  # Strict veto: Shorts blocked during bull regime!

    # 5. DIRECTIONAL STREAK LOCKOUT (Anti-whipsaw defense)
    now_t = time.time()
    dir_lock = survival.get("directional_lock", {})
    if now_t < dir_lock.get("until", 0.0):
        locked_dir = dir_lock.get("direction")
        if (is_long and locked_dir == "LONG") or (is_short and locked_dir == "SHORT"):
            return  # Directional cooldown active!

    # 6. Active Lead Alpha Selection (must be ACTIVE or PROBATION)
    active_candidates = {
        k: conv for k, conv in convictions.items()
        if alphas[k]["status"] in ["ACTIVE", "PROBATION"]
        and ((is_long and conv > 0) or (is_short and conv < 0))
        and now_t >= alpha_cooldowns.get(k, 0.0)
    }
    if not active_candidates:
        return

    lead_key   = max(active_candidates, key=lambda k: abs(active_candidates[k]))
    lead_conv  = active_candidates[lead_key]
    lead_alpha = alphas[lead_key]

    entry_px = market_state["ask"] if is_long else market_state["bid"]
    if entry_px <= 0:
        return

    # 7. Survival Risk Sizing & Asymmetric R:R Structure
    risk_hp_scalar = get_survival_risk_scalar()
    if risk_hp_scalar <= 0.0:
        return

    # Base risk: 0.75% of capital (institutional conservative risk per trade)
    base_risk_pct = 0.0075
    final_risk_mult = max(0.50, risk_hp_scalar * session_state["size_multiplier"])
    risk_usd = survival["account_balance"] * base_risk_pct * final_risk_mult

    atr_val = max(market_state["atr_14"], 40.0)

    # Overextension Guard: Never buy the peak of parabolic expansion or short the floor
    ema20 = mtf_state.get("ema_20", 0.0)
    if ema20 > 0:
        if is_long and entry_px > ema20 + (atr_val * 6.0):
            return  # Overextended: Wait for pullback towards EMA20!
        elif is_short and entry_px < ema20 - (atr_val * 6.0):
            return  # Overextended: Wait for bounce towards EMA20!

    # Structural Volatility Stop Distance (Dynamic Institutional Risk Sizing)
    structural_buffer = atr_val * 1.5
    if is_long:
        recent_lows = list(sweep_state["recent_lows"])
        recent_swing_low = min(recent_lows[-10:]) if len(recent_lows) >= 10 else (entry_px - atr_val * 4.0)
        struct_dist = max(0.0, entry_px - recent_swing_low) + structural_buffer
        sl_dist = max(atr_val * 3.5, struct_dist)
    else:
        recent_highs = list(sweep_state["recent_highs"])
        recent_swing_high = max(recent_highs[-10:]) if len(recent_highs) >= 10 else (entry_px + atr_val * 4.0)
        struct_dist = max(0.0, recent_swing_high - entry_px) + structural_buffer
        sl_dist = max(atr_val * 3.5, struct_dist)

    sl_fraction = sl_dist / entry_px
    fee_fraction = ROUNDTRIP_COST_BPS / 10000.0
    total_loss_fraction = sl_fraction + fee_fraction

    # Sizing: notional chosen so that price SL + roundtrip fee = risk_usd exactly!
    notional = risk_usd / total_loss_fraction
    max_not  = survival["account_balance"] * 7.0  # Cap at 7x max leverage
    notional = min(notional, max_not)

    if notional < 150.0:
        return  # Reject tiny sub-threshold orders

    leverage = notional / survival["account_balance"]

    # Asymmetric Targets:
    # TP1: 2.5R target (scales out 70% of position)
    # TP2: 4.5R target (for 30% runner)
    tp1_dist = sl_dist * DYNAMIC_PARAMS["TARGET_RR_RATIO"]
    tp2_dist = sl_dist * DYNAMIC_PARAMS["RUNNER_RR_RATIO"]

    if is_long:
        sl_price  = entry_px - sl_dist
        tp1_price = entry_px + tp1_dist
        tp2_price = entry_px + tp2_dist
    else:
        sl_price  = entry_px + sl_dist
        tp1_price = entry_px - tp1_dist
        tp2_price = entry_px - tp2_dist

    portfolio.update({
        "position":             "LONG" if is_long else "SHORT",
        "phase":                1,
        "entry_price":          entry_px,
        "sl_distance":          sl_dist,
        "active_sl_price":      sl_price,
        "tp1_price":            tp1_price,
        "active_tp_price":      tp2_price,
        "notional_size_usd":    notional,
        "initial_notional_usd": notional,
        "leverage_used":        leverage,
        "realized_pnl":         0.0,
        "banked_pnl":           0.0,
        "unrealized_pnl":       0.0,
        "highest_unrealized":   0.0,
        "trade_open_time":      now_t,
        "lead_alpha":           lead_key,
        "be_moved":             False,
        "entry_reason":         f"[{lead_alpha['tag']}] Weighted Consensus: {consensus_conviction:+.2f} | Macro: {macro_trend}",
    })

    # Log to recent trades
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    action  = "BUY" if is_long else "SELL"
    msg     = (f"[{now_str}] OPEN  {action:4} | Entry: {entry_px:,.1f} | SL: {sl_price:,.1f} | "
               f"2.5R TP: {tp1_price:,.1f} | Risk: ${risk_usd:.2f} ({leverage:.1f}x) | Lead: {lead_alpha['tag']}")
    recent_trades.append(msg)

    # Append to journal
    trade_journal.append({
        "id":           len(trade_journal) + 1,
        "open_ts":      datetime.datetime.now().isoformat(),
        "action":       action,
        "entry_price":  entry_px,
        "sl_price":     sl_price,
        "tp_price":     tp1_price,
        "tp2_price":    tp2_price,
        "lead_alpha":   lead_key,
        "consensus":    round(consensus_conviction, 3),
        "hp_at_entry":  survival["hp_pct"],
        "notional_usd": round(notional, 2),
        "reason":       portfolio["entry_reason"],
        "outcome":      "OPEN",
        "exit_price":   0.0,
        "pnl_usd":      0.0,
        "post_mortem":  "",
    })
    save_trade_journal()


# ---------------------------------------------------------
# EXIT ENGINE & AUTONOMOUS POST-MORTEM
# ---------------------------------------------------------
def check_exits(consensus_conviction=0.0):
    if portfolio["position"] == "FLAT":
        return

    now_t   = time.time()
    dur_min = (now_t - portfolio["trade_open_time"]) / 60.0
    entry   = portfolio["entry_price"]
    pos     = portfolio["position"]
    sl_dist = portfolio.get("sl_distance", 150.0)

    def close_position(price, reason, post_mortem_tag):
        if pos == "LONG":
            gross_pct = (price - entry) / entry * 10000.0
        else:
            gross_pct = (entry - price) / entry * 10000.0

        net_pct = gross_pct - ROUNDTRIP_COST_BPS
        net_usd = (net_pct / 10000.0) * portfolio["notional_size_usd"]
        total_trade_pnl = portfolio.get("banked_pnl", 0.0) + net_usd

        survival["account_balance"] += net_usd
        update_survival_health()

        is_win = total_trade_pnl > 0
        lead = portfolio["lead_alpha"]

        # Track consecutive losses & directional lockout
        is_time_decay = "TIME DECAY" in reason
        if is_win:
            survival["consecutive_losses"] = 0
        else:
            survival["rolling_losses"].append((time.time(), total_trade_pnl))

            if is_time_decay:
                # Time Decay is a capital preservation scratch, NOT an adverse stop loss!
                # Lock only the timed-out direction for 15m; brief 120s global pause
                survival["directional_lock"] = {"direction": pos, "until": time.time() + 900.0}
                survival["global_cooldown_until"] = time.time() + 120.0
                survival["cooldown_reason"] = "Time decay scratch (2m pause)"
                alpha_cooldowns[lead] = time.time() + 120.0
            else:
                survival["consecutive_losses"] += 1
                streak = survival["consecutive_losses"]
                # Directional lockout: lock the failed direction for 30 minutes!
                survival["directional_lock"] = {"direction": pos, "until": time.time() + 1800.0}

                # Global cooldown: only 180s (3m) on 1st loss, 450s (7.5m) on 2nd loss
                cooldown_sec = 180.0 if streak == 1 else 450.0
                survival["global_cooldown_until"] = time.time() + cooldown_sec
                survival["cooldown_reason"] = f"{reason} ({streak}L streak)"
                alpha_cooldowns[lead] = time.time() + cooldown_sec + 60.0

        reward_alpha(lead, is_win, total_trade_pnl)

        # Log post-mortem diagnostic
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        action_label = "CLOSE LONG " if pos == "LONG" else "CLOSE SHORT"
        pnl_str = f"${total_trade_pnl:+.2f}"
        banked = portfolio.get("banked_pnl", 0.0)
        pnl_breakdown = f"Total: {pnl_str} (70% TP: +${banked:.2f} | Runner: ${net_usd:+.2f})" if banked > 0 else f"PnL: {pnl_str}"

        msg = f"[{now_str}] {action_label} | Exit: {price:,.1f} | {pnl_breakdown} | {reason} [{lead}]"
        recent_trades.append(msg)
        recent_post_mortems.append(f"[{now_str}] [{lead}] {post_mortem_tag} ({pnl_breakdown})")

        # Update last journal entry
        for t in reversed(trade_journal):
            if t.get("outcome") == "OPEN":
                t["exit_ts"]       = datetime.datetime.now().isoformat()
                t["exit_price"]    = price
                t["pnl_usd"]       = round(total_trade_pnl, 2)
                t["scale_out_pnl"] = round(banked, 2)
                t["runner_pnl"]    = round(net_usd, 2)
                t["outcome"]       = "WIN" if is_win else "LOSS"
                t["exit_reason"]   = reason
                t["post_mortem"]   = post_mortem_tag
                break
        save_trade_journal()

        portfolio.update({
            "position":           "FLAT",
            "phase":              0,
            "banked_pnl":         0.0,
            "unrealized_pnl":     0.0,
            "highest_unrealized": 0.0,
            "active_sl_price":    0.0,
            "active_tp_price":    0.0,
            "tp1_price":          0.0,
            "be_moved":           False,
        })

    # Long Position Exit Checks
    if pos == "LONG":
        cur = market_state["bid"]
        if cur <= 0:
            cur = market_state["ltp"]
        if cur <= 0:
            return  # Safety guard: Never exit on invalid price

        gross_pct = (cur - entry) / entry * 10000.0
        net_pct   = gross_pct - ROUNDTRIP_COST_BPS
        unreal    = (net_pct / 10000.0) * portfolio["notional_size_usd"]
        portfolio["unrealized_pnl"]   = unreal
        portfolio["highest_unrealized"] = max(portfolio["highest_unrealized"], unreal)

        if portfolio["phase"] == 1:
            # Stop loss hit
            if cur <= portfolio["active_sl_price"]:
                close_position(cur, "STOP LOSS HIT", "SL hit: Price reached adverse risk threshold.")
                return

            # 2.5R Partial Scale Out: lock 70% of position profit!
            if cur >= portfolio.get("tp1_price", entry + sl_dist * 2.5):
                scale_frac = DYNAMIC_PARAMS["SCALE_OUT_FRAC"]
                part_unreal = unreal * scale_frac
                portfolio["banked_pnl"] = portfolio.get("banked_pnl", 0.0) + part_unreal
                survival["account_balance"] += part_unreal
                update_survival_health()
                portfolio["notional_size_usd"] *= (1.0 - scale_frac)
                # Move SL to Breakeven (+ 1.5x fee buffer)
                be_sl = entry + ((ROUNDTRIP_COST_BPS * 1.5) / 10000.0) * entry
                portfolio["active_sl_price"] = be_sl
                portfolio["be_moved"]        = True
                portfolio["phase"]           = 2
                recent_post_mortems.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 2.5R TP Hit (+${part_unreal:.2f}) | 70% locked, SL to BE")

            # Time Decay Preservation (90 minutes, adverse drift only)
            if dur_min > 90.0 and cur < entry - market_state["atr_14"] * 1.0 and cvd_state["delta_momentum"] < 0:
                close_position(cur, "TIME DECAY (90m)", "Consolidating adverse with negative delta.")
                return

        elif portfolio["phase"] == 2:
            if cur <= portfolio["active_sl_price"]:
                close_position(cur, "BE / TRAIL STOP", "30% Runner protected at Breakeven / ATR trailing stop.")
                return

            # ATR Trailing Stop (gives 2.5x ATR room to breathe)
            trail_buf = max(market_state["atr_14"] * 2.5, entry * 0.0035)
            struct_trail = cur - trail_buf
            if struct_trail > portfolio["active_sl_price"]:
                portfolio["active_sl_price"] = struct_trail

            if cur >= portfolio["active_tp_price"]:
                close_position(cur, "FULL TP HIT (4.5R)", "Runner hit maximum institutional target.")
                return

    # Short Position Exit Checks
    elif pos == "SHORT":
        cur = market_state["ask"]
        if cur <= 0:
            cur = market_state["ltp"]
        if cur <= 0:
            return  # Safety guard: Never exit on invalid price

        gross_pct = (entry - cur) / entry * 10000.0
        net_pct   = gross_pct - ROUNDTRIP_COST_BPS
        unreal    = (net_pct / 10000.0) * portfolio["notional_size_usd"]
        portfolio["unrealized_pnl"]   = unreal
        portfolio["highest_unrealized"] = max(portfolio["highest_unrealized"], unreal)

        if portfolio["phase"] == 1:
            if cur >= portfolio["active_sl_price"]:
                close_position(cur, "STOP LOSS HIT", "SL hit: Price reached adverse risk threshold.")
                return

            # 2.5R Partial Scale Out: lock 70% of position profit!
            if cur <= portfolio.get("tp1_price", entry - sl_dist * 2.5):
                scale_frac = DYNAMIC_PARAMS["SCALE_OUT_FRAC"]
                part_unreal = unreal * scale_frac
                portfolio["banked_pnl"] = portfolio.get("banked_pnl", 0.0) + part_unreal
                survival["account_balance"] += part_unreal
                update_survival_health()
                portfolio["notional_size_usd"] *= (1.0 - scale_frac)
                be_sl = entry - ((ROUNDTRIP_COST_BPS * 1.5) / 10000.0) * entry
                portfolio["active_sl_price"] = be_sl
                portfolio["be_moved"]        = True
                portfolio["phase"]           = 2
                recent_post_mortems.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 2.5R TP Hit (+${part_unreal:.2f}) | 70% locked, SL to BE")

            if dur_min > 90.0 and cur > entry + market_state["atr_14"] * 1.0 and cvd_state["delta_momentum"] > 0:
                close_position(cur, "TIME DECAY (90m)", "Consolidating adverse with positive delta.")
                return

        elif portfolio["phase"] == 2:
            if cur >= portfolio["active_sl_price"]:
                close_position(cur, "BE / TRAIL STOP", "30% Runner protected at Breakeven / ATR trailing stop.")
                return

            trail_buf = max(market_state["atr_14"] * 2.5, entry * 0.0035)
            struct_trail = cur + trail_buf
            if struct_trail < portfolio["active_sl_price"]:
                portfolio["active_sl_price"] = struct_trail

            if cur <= portfolio["active_tp_price"]:
                close_position(cur, "FULL TP HIT (4.5R)", "Runner hit maximum institutional target.")
                return


# ---------------------------------------------------------
# CANDLE & TECHNICAL AGGREGATION
# ---------------------------------------------------------
def calculate_atr(period=14):
    if len(minute_candles) < period + 1:
        return 35.0
    candles = list(minute_candles)
    trs = []
    for i in range(1, len(candles)):
        h, l, pc = candles[i]['high'], candles[i]['low'], candles[i-1]['close']
        tr = max(h - l, abs(h - pc), abs(l - pc))
        if 0 < tr < 5000:
            trs.append(tr)
    return float(np.mean(trs[-period:])) if trs else 35.0


def update_dynamic_parameters():
    if len(minute_candles) < 14 or market_state["ltp"] <= 0:
        return
    atr = max(calculate_atr(14), 30.0)
    market_state["atr_14"] = atr
    market_state["atr_bps"] = (atr / market_state["ltp"]) * 10000.0 if market_state["ltp"] > 0 else 0.0


def calc_ema(closes, period):
    if not closes:
        return 0.0
    k   = 2.0 / (period + 1)
    ema = closes[0]
    for price in closes[1:]:
        ema = price * k + ema * (1 - k)
    return ema


def update_mtf_trend():
    if len(fifteen_min_candles) < 3:
        mtf_state["trend"] = "NEUTRAL"
        return
    closes = [c["close"] for c in fifteen_min_candles]
    p = len(closes)
    ema20 = calc_ema(closes, min(20, p))
    ema50 = calc_ema(closes, min(50, p))
    mtf_state["ema_20"] = ema20
    mtf_state["ema_50"] = ema50
    cur = market_state["ltp"] if market_state["ltp"] > 0 else closes[-1]

    if p >= 4:
        old20 = calc_ema(closes[:-3], min(20, p-3))
        mtf_state["slope"] = ((ema20 - old20) / old20) * 10000 if old20 > 0 else 0.0

    strength_bps = ((cur - ema20) / ema20) * 10000 if ema20 > 0 else 0.0
    mtf_state["strength"] = strength_bps

    # Strict multi-timeframe regime classification
    if strength_bps < -25.0:
        mtf_state["trend"] = "BEARISH_BREAKDOWN"
    elif strength_bps > 25.0:
        mtf_state["trend"] = "BULLISH_EXPANSION"
    elif cur < ema20 and ema20 <= ema50:
        mtf_state["trend"] = "BEARISH"
    elif cur > ema20 and ema20 >= ema50:
        mtf_state["trend"] = "BULLISH"
    elif strength_bps < -10.0 or (cur < ema20 and mtf_state["slope"] < 0):
        mtf_state["trend"] = "BEARISH_SLOPE"
    elif strength_bps > 10.0 or (cur > ema20 and mtf_state["slope"] > 0):
        mtf_state["trend"] = "BULLISH_SLOPE"
    else:
        mtf_state["trend"] = "NEUTRAL"


def detect_fvgs():
    if len(five_min_candles) < 3:
        return
    candles = list(five_min_candles)
    ltp = market_state["ltp"]
    fvg_state["bullish_fvgs"] = [f for f in fvg_state["bullish_fvgs"] if ltp > f["bottom"]]
    fvg_state["bearish_fvgs"] = [f for f in fvg_state["bearish_fvgs"] if ltp < f["top"]]
    c0, c1, c2 = candles[-3], candles[-2], candles[-1]

    if c2["low"] > c0["high"] and c1["close"] > c1["open"]:
        gb, gt = c0["high"], c2["low"]
        spb = ((gt - gb) / c1["close"]) * 10000 if c1["close"] > 0 else 0
        if spb > 5 and not any(abs(f["bottom"] - gb) < 2.0 for f in fvg_state["bullish_fvgs"]):
            fvg_state["bullish_fvgs"].append({"bottom": gb, "top": gt, "size_bps": spb})

    if c2["high"] < c0["low"] and c1["close"] < c1["open"]:
        gt, gb = c0["low"], c2["high"]
        spb = ((gt - gb) / c1["close"]) * 10000 if c1["close"] > 0 else 0
        if spb > 5 and not any(abs(f["top"] - gt) < 2.0 for f in fvg_state["bearish_fvgs"]):
            fvg_state["bearish_fvgs"].append({"bottom": gb, "top": gt, "size_bps": spb})


def detect_order_blocks():
    if len(five_min_candles) < 15:
        return
    candles = list(five_min_candles)
    ltp     = market_state["ltp"]
    bodies  = [abs(c["close"] - c["open"]) for c in candles]
    is_bull = [c["close"] > c["open"] for c in candles]
    n       = len(candles)

    ob_state["bullish_obs"] = [o for o in ob_state["bullish_obs"] if ltp > o["low"] - (o["high"]-o["low"])*0.5]
    ob_state["bearish_obs"] = [o for o in ob_state["bearish_obs"] if ltp < o["high"] + (o["high"]-o["low"])*0.5]

    for i in range(n-1, max(n-4, 12), -1):
        avg_body = float(np.mean(bodies[max(0, i-10):i])) if i > 0 else 0
        if avg_body <= 0 or bodies[i] < 2.0 * avg_body:
            continue
        disp = bodies[i] / avg_body
        if is_bull[i]:
            for j in range(i-1, max(0, i-6), -1):
                if not is_bull[j]:
                    ob = {"low": candles[j]["low"], "high": candles[j]["high"], "disp": disp}
                    if not any(abs(o["low"] - ob["low"]) < 2.0 for o in ob_state["bullish_obs"]):
                        ob_state["bullish_obs"].append(ob)
                        if len(ob_state["bullish_obs"]) > 5:
                            ob_state["bullish_obs"].pop(0)


def track_swing_levels(five_min_candle):
    sweep_state["recent_highs"].append(five_min_candle["high"])
    sweep_state["recent_lows"].append(five_min_candle["low"])


def detect_liquidity_sweep(five_min_candle):
    if len(sweep_state["recent_highs"]) < 5:
        return
    h, l, c = five_min_candle["high"], five_min_candle["low"], five_min_candle["close"]
    prev_highs = list(sweep_state["recent_highs"])[:-1]
    prev_lows  = list(sweep_state["recent_lows"])[:-1]

    recent_swing_high = max(prev_highs[-10:]) if len(prev_highs) >= 10 else max(prev_highs)
    recent_swing_low  = min(prev_lows[-10:])  if len(prev_lows)  >= 10 else min(prev_lows)

    if h > recent_swing_high and c < recent_swing_high:
        sweep_state["last_sweep"]        = "BEARISH_SWEEP"
        sweep_state["sweep_price"]       = recent_swing_high
        sweep_state["sweep_candles_ago"] = 0

    elif l < recent_swing_low and c > recent_swing_low:
        sweep_state["last_sweep"]        = "BULLISH_SWEEP"
        sweep_state["sweep_price"]       = recent_swing_low
        sweep_state["sweep_candles_ago"] = 0


def detect_cvd_divergence():
    if len(cvd_state["cvd_history"]) < 5 or len(cvd_state["price_history"]) < 5:
        cvd_state["divergence"] = "NONE"
        return
    prices = list(cvd_state["price_history"])[-5:]
    cvds   = list(cvd_state["cvd_history"])[-5:]
    ps = prices[-1] - prices[0]
    cs = cvds[-1]   - cvds[0]
    if ps < 0 and cs > 0 and abs(cs) > 0.1:
        cvd_state["divergence"] = "BULLISH"
    elif ps > 0 and cs < 0 and abs(cs) > 0.1:
        cvd_state["divergence"] = "BEARISH"
    else:
        cvd_state["divergence"] = "NONE"


def aggregate_5min_candle(one_min_candle):
    five_min_accumulator.append(one_min_candle)
    if len(five_min_accumulator) >= 5:
        o = five_min_accumulator[0]["open"]
        h = max(c["high"] for c in five_min_accumulator)
        l = min(c["low"]  for c in five_min_accumulator)
        c = five_min_accumulator[-1]["close"]
        v = sum(c_["volume"] for c_ in five_min_accumulator)
        c5 = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        five_min_candles.append(c5)
        five_min_accumulator.clear()

        cvd_state["cvd_history"].append(cvd_state["cvd"])
        cvd_state["price_history"].append(c)
        if len(cvd_state["cvd_history"]) >= 3:
            cvds = list(cvd_state["cvd_history"])
            cvd_state["delta_momentum"] = float(np.mean(np.diff(cvds[-3:])))
        cvd_state["cvd"] = 0.0

        detect_fvgs()
        detect_order_blocks()
        detect_cvd_divergence()
        track_swing_levels(c5)
        detect_liquidity_sweep(c5)

        # 15m candle aggregation
        fifteen_min_accumulator.append(c5)
        if len(fifteen_min_accumulator) >= 3:
            o15 = fifteen_min_accumulator[0]["open"]
            h15 = max(x["high"] for x in fifteen_min_accumulator)
            l15 = min(x["low"]  for x in fifteen_min_accumulator)
            c15 = fifteen_min_accumulator[-1]["close"]
            v15 = sum(x["volume"] for x in fifteen_min_accumulator)
            fifteen_min_candles.append({"open": o15, "high": h15, "low": l15, "close": c15, "volume": v15})
            fifteen_min_accumulator.clear()
            update_mtf_trend()

        if sweep_state["last_sweep"] not in ["NONE", "FAILED", "INVALIDATED"]:
            sweep_state["sweep_candles_ago"] += 1
            if sweep_state["sweep_candles_ago"] > 6:
                sweep_state["last_sweep"] = "NONE"


# ---------------------------------------------------------
# VOLUME PROFILES & PRE-WARM
# ---------------------------------------------------------
def build_volume_profiles():
    try:
        # Macro 30d profile
        res = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=720", timeout=10)
        data = res.json()
        highs = [float(k[2]) for k in data]
        lows  = [float(k[3]) for k in data]
        mn, mx = min(lows), max(highs)
        n_bins = 60
        bin_size = (mx - mn) / n_bins
        profile  = np.zeros(n_bins)
        for k in data:
            cl, ch, cv = float(k[3]), float(k[2]), float(k[5])
            sb = int(max(0, min(n_bins-1, (cl-mn)/bin_size)))
            eb = int(max(0, min(n_bins-1, (ch-mn)/bin_size)))
            if sb == eb:
                profile[sb] += cv
            else:
                vpb = cv / (eb - sb + 1)
                for b in range(sb, eb+1):
                    profile[b] += vpb
        top_idx = np.argsort(profile)[-6:]
        macro_structure["hvns"] = sorted([mn + (i*bin_size) + bin_size/2 for i in top_idx])
        poc_idx = int(np.argmax(profile))
        macro_structure["poc"] = mn + poc_idx*bin_size + bin_size/2

        # Session 24h profile
        res24 = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=96", timeout=10)
        data24 = res24.json()
        h24 = [float(k[2]) for k in data24]
        l24 = [float(k[3]) for k in data24]
        mn24, mx24 = min(l24), max(h24)
        n_bins24 = 30
        bs24 = (mx24 - mn24) / n_bins24
        p24 = np.zeros(n_bins24)
        for k in data24:
            cl, ch, cv = float(k[3]), float(k[2]), float(k[5])
            sb = int(max(0, min(n_bins24-1, (cl-mn24)/bs24)))
            eb = int(max(0, min(n_bins24-1, (ch-mn24)/bs24)))
            if sb == eb:
                p24[sb] += cv
            else:
                vpb = cv / (eb - sb + 1)
                for b in range(sb, eb+1):
                    p24[b] += vpb
        top_idx24 = np.argsort(p24)[-4:]
        macro_structure["micro_hvns"] = sorted([mn24 + (i*bs24) + bs24/2 for i in top_idx24])
    except Exception as e:
        print(f"Volume profile error: {e}")


def prewarm_engine():
    print("=" * 70)
    print("  INITIALIZING APEX SURVIVAL QUANT LAB v5.0...")
    print("=" * 70)
    build_volume_profiles()

    res = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=120", timeout=10)
    data = res.json()
    for k in data:
        minute_candles.append({
            "open": float(k[1]), "high": float(k[2]),
            "low":  float(k[3]), "close": float(k[4]),
            "volume": float(k[5])
        })
    market_state["ltp"] = float(data[-1][4])
    market_state["bid"] = market_state["ltp"]
    market_state["ask"] = market_state["ltp"]
    update_dynamic_parameters()
    market_state["current_minute_ts"] = int(time.time() // 60) * 60
    live_candle.update({"open": market_state["ltp"], "high": market_state["ltp"],
                         "low": market_state["ltp"], "close": market_state["ltp"], "volume": 0.0})

    res5 = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=100", timeout=10)
    data5 = res5.json()
    for k in data5:
        c5 = {"open": float(k[1]), "high": float(k[2]),
              "low":  float(k[3]), "close": float(k[4]), "volume": float(k[5])}
        five_min_candles.append(c5)
        bv = float(k[9])
        sv = float(k[5]) - bv
        cvd_state["cvd_history"].append(bv - sv)
        cvd_state["price_history"].append(float(k[4]))

    detect_fvgs()
    detect_order_blocks()
    detect_cvd_divergence()

    c5_list = list(five_min_candles)
    for i in range(0, len(c5_list) - 2, 3):
        grp = c5_list[i:i+3]
        if len(grp) == 3:
            fifteen_min_candles.append({
                "open":   grp[0]["open"],
                "high":   max(g["high"] for g in grp),
                "low":    min(g["low"] for g in grp),
                "close":  grp[-1]["close"],
                "volume": sum(g["volume"] for g in grp),
            })
    update_mtf_trend()
    for c in c5_list:
        track_swing_levels(c)

    load_trade_journal()
    update_session()
    update_alpha_evaluations()
    print(f"  POC: {macro_structure['poc']:,.1f} | HVNs: {[round(x,0) for x in macro_structure['hvns']]}")
    print(f"  15m MTF: {mtf_state['trend']} (EMA20: {mtf_state['ema_20']:,.1f} EMA50: {mtf_state['ema_50']:,.1f})")
    print(f"  Survival HP: {survival['hp_pct']}% | Bankroll: ${survival['account_balance']:,.2f}")
    print("=" * 70)
    print("  ENGINE ONLINE — INSTITUTIONAL QUANT LAB ACTIVE")
    print("=" * 70)


# ---------------------------------------------------------
# SESSION TRACKER (24/7 ACTIVE TRADING)
# ---------------------------------------------------------
def update_session():
    utc_hour = datetime.datetime.now(datetime.timezone.utc).hour
    if 7 <= utc_hour < 13:
        session_state.update({"current": "LONDON",       "execution_allowed": True, "size_multiplier": 1.0})
    elif 13 <= utc_hour < 21:
        session_state.update({"current": "NEW_YORK",     "execution_allowed": True, "size_multiplier": 1.0})
    elif 0 <= utc_hour < 7:
        session_state.update({"current": "ASIAN",        "execution_allowed": True, "size_multiplier": 0.7})
    else:
        session_state.update({"current": "PACIFIC_LULL", "execution_allowed": True, "size_multiplier": 0.5})


# ---------------------------------------------------------
# TICK & BUCKET PROCESSING
# ---------------------------------------------------------
def process_trade_tick(price, qty, is_buyer_maker):
    market_state["ltp"] = price
    market_state["ticks_today"] += 1
    if price <= 0 or qty <= 0:
        return

    ts_now = time.time()
    if market_state["current_minute_ts"] == 0:
        market_state["current_minute_ts"] = int(ts_now // 60) * 60
        live_candle.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0})

    if ts_now > market_state["current_minute_ts"] + 60:
        completed = dict(live_candle)
        minute_candles.append(completed)
        update_dynamic_parameters()
        aggregate_5min_candle(completed)
        market_state["current_minute_ts"] = int(ts_now // 60) * 60
        live_candle.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0})
    else:
        if live_candle["high"] == 0:
            live_candle.update({"open": price, "high": price, "low": price})
        live_candle["high"]   = max(live_candle["high"], price)
        live_candle["low"]    = min(live_candle["low"],  price)
        live_candle["close"]  = price
        live_candle["volume"] += qty

    if market_state["bid"] > 0 and market_state["ask"] > 0:
        market_state["spread_bps"] = ((market_state["ask"] - market_state["bid"]) / price) * 10000.0

    direction = "SELL" if is_buyer_maker else "BUY"
    if direction == "BUY":
        cvd_state["cvd"] += qty
    else:
        cvd_state["cvd"] -= qty

    # COFI Z-score
    signed_vol = qty if direction == "BUY" else -qty
    order_flow_imbalances.append(signed_vol)
    if len(order_flow_imbalances) > 20:
        arr  = np.array(order_flow_imbalances)
        mean = float(np.mean(arr))
        std  = float(np.std(arr)) + 1e-9
        market_state["cofi_z"] = (signed_vol - mean) / std

        k = 2.0 / 21.0
        market_state["cofi_z_ema"] = market_state["cofi_z"] * k + market_state["cofi_z_ema"] * (1.0 - k)
        c_ema = market_state["cofi_z_ema"]
        market_state["cofi_direction"] = +1 if c_ema > 1.2 else (-1 if c_ema < -1.2 else 0)

    # VPIN Volume Bucketing
    global current_bucket
    if direction == "BUY":
        current_bucket["buy_vol"] += qty
    else:
        current_bucket["sell_vol"] += qty
    current_bucket["total_vol"] += qty

    if current_bucket["total_vol"] >= DYNAMIC_PARAMS["BUCKET_VOLUME_SIZE"]:
        imb  = abs(current_bucket["buy_vol"] - current_bucket["sell_vol"])
        vpin = imb / current_bucket["total_vol"]
        vpin_history.append(vpin)
        market_state["bucket_velocity_sec"] = time.time() - current_bucket["start_time"]
        market_state["buckets_completed"]  += 1
        current_bucket = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0, "start_time": time.time()}

        varr = np.array(vpin_history)
        market_state["vpin"] = float(varr[-1])
        if len(varr) > 10:
            market_state["vpin_percentile"] = int(float(np.sum(varr <= varr[-1]) / len(varr) * 100))


# ---------------------------------------------------------
# REAL-TIME EXECUTIVE QUANT DASHBOARD (v5.0 COCKPIT)
# ---------------------------------------------------------
first_render = True

def render_survival_dashboard():
    global first_render
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if first_render:
        os.system('cls' if os.name == 'nt' else 'clear')
        first_render = False

    hp        = survival["hp_pct"]
    hp_blocks = int(hp // 10)

    # HP color coding
    if hp >= 70.0:
        hp_color = C_GREEN
        mode_pill = f"{C_GREEN}{C_BOLD}THRIVING 🟢{C_RESET}"
    elif hp >= 45.0:
        hp_color = C_YELLOW
        mode_pill = f"{C_YELLOW}{C_BOLD}DEFENSIVE 🟡{C_RESET}"
    elif hp >= 15.0:
        hp_color = C_YELLOW
        mode_pill = f"{C_YELLOW}{C_BOLD}DEFENSIVE RECOVERY 🛡️{C_RESET}"
    else:
        hp_color = C_RED
        mode_pill = f"{C_RED}{C_BOLD}SURVIVAL BUNKER 🔴{C_RESET}"

    hp_bar = f"{hp_color}{'█' * hp_blocks}{'░' * (10 - hp_blocks)}{C_RESET}"

    ltp  = market_state["ltp"]
    sp   = market_state["spread_bps"]
    obi  = market_state["obi"]
    vpin = market_state["vpin_percentile"]
    cmom = cvd_state["delta_momentum"]
    mtf  = mtf_state["trend"]
    str_bps = mtf_state["strength"]

    if "BULLISH" in mtf:
        mtf_badge = f"{C_GREEN}{C_BOLD}BULLISH 🟢 ({str_bps:+.1f} bps){C_RESET}"
        gate_badge = f"{C_GREEN}{C_BOLD}LONGS ONLY (Shorts blocked by Bullish trend){C_RESET}"
    elif "BEARISH" in mtf:
        mtf_badge = f"{C_RED}{C_BOLD}BEARISH 🔴 ({str_bps:+.1f} bps){C_RESET}"
        gate_badge = f"{C_RED}{C_BOLD}SHORTS ONLY (Longs blocked by Bearish trend){C_RESET}"
    else:
        mtf_badge = f"{C_YELLOW}NEUTRAL 🟡 ({str_bps:+.1f} bps){C_RESET}"
        gate_badge = f"{C_YELLOW}CONCORDANT ONLY (High threshold required){C_RESET}"

    obi_color = C_GREEN if obi > 0.3 else (C_RED if obi < -0.3 else C_GRAY)
    obi_str = "BUY WALL" if obi > 0.3 else ("SELL WALL" if obi < -0.3 else "BALANCED")

    bal  = survival["account_balance"]
    peak = survival["peak_balance"]
    dd   = survival["current_drawdown_pct"]
    pnl_today = bal - INITIAL_CAPITAL
    pnl_color = C_GREEN if pnl_today >= 0 else C_RED
    pnl_sign  = "+" if pnl_today >= 0 else "-"

    sess = session_state["current"]
    sess_mult = session_state["size_multiplier"]

    import re
    ANSI_RE = re.compile(r'\033\[[0-9;]*[mHJK]')

    def visible_len(s):
        return len(ANSI_RE.sub('', s))

    def truncate_ansi(s, max_len):
        cur_len = 0
        res = []
        tokens = re.split(r'(\033\[[0-9;]*[mHJK])', s)
        for t in tokens:
            if not t:
                continue
            if t.startswith('\033['):
                res.append(t)
            else:
                rem = max_len - cur_len
                if rem <= 0:
                    break
                if len(t) > rem:
                    res.append(t[:rem - 1] + '…')
                    cur_len += rem
                    break
                else:
                    res.append(t)
                    cur_len += len(t)
        res.append(C_RESET)
        return ''.join(res)

    BOX_WIDTH = 90

    def row(content):
        vlen = visible_len(content)
        if vlen > BOX_WIDTH:
            content = truncate_ansi(content, BOX_WIDTH)
            vlen = visible_len(content)
        pad = max(0, BOX_WIDTH - vlen)
        return f"{C_CYAN}║{C_RESET} {content}{' ' * pad} {C_CYAN}║{C_RESET}\033[K"

    border_top = f"{C_CYAN}╔{'═' * (BOX_WIDTH + 2)}╗{C_RESET}\033[K"
    border_mid = f"{C_CYAN}╠{'═' * (BOX_WIDTH + 2)}╣{C_RESET}\033[K"
    border_div = f"{C_CYAN}╟{'─' * (BOX_WIDTH + 2)}╢{C_RESET}\033[K"
    border_bot = f"{C_CYAN}╚{'═' * (BOX_WIDTH + 2)}╝{C_RESET}\033[K"

    buf = []
    buf.append("\033[H")  # Move to top-left without flicker
    buf.append(border_top)
    buf.append(row(f"{C_BOLD}{C_WHITE}⚡ APEX INTELLIGENCE ENGINE v5.0 — INSTITUTIONAL QUANT LAB ⚡{C_RESET}"))
    buf.append(row(f"{C_DIM}Time:{C_RESET} {now_str} IST  │  {C_DIM}Session:{C_RESET} {C_CYAN}{sess} ({sess_mult:.1f}x){C_RESET}  │  {C_DIM}Binance Feed:{C_RESET} {C_GREEN}CONNECTED ●{C_RESET}"))
    buf.append(border_mid)

    # 1. Capital & HP Card
    buf.append(row(f"{C_BOLD}{C_CYAN}CAPITAL & HEALTH (HP):{C_RESET}"))
    buf.append(row(f" Bankroll: {C_WHITE}${bal:,.2f}{C_RESET} [{hp_bar}] {hp_color}{hp:5.1f}% HP{C_RESET}  │  Peak: ${peak:,.2f}  │  Drawdown: {C_RED}-{dd:4.1f}%{C_RESET}"))
    buf.append(row(f" Mode: {mode_pill}  │  Net PnL: {pnl_color}{pnl_sign}${abs(pnl_today):.2f}{C_RESET}  │  Asymmetry: {C_GREEN}{C_BOLD}1 : 2.50 Target R:R (70% Scale){C_RESET}"))

    # Active Cooldown & Defense Telemetry
    now_t = time.time()
    circuit_status = ""
    if survival["circuit_breaker"]:
        circuit_status = f"{C_RED}{C_BOLD}CIRCUIT BREAKER: TRIPPED (Max DD Reached){C_RESET}"
    elif survival["hp_pct"] < 40.0:
        circuit_status = f"{C_RED}{C_BOLD}BUNKER LOCKDOWN: TRADING HALTED (Capital Defense){C_RESET}"
    elif now_t < survival.get("session_halt_until", 0.0):
        rem = int(survival["session_halt_until"] - now_t)
        circuit_status = f"{C_YELLOW}{C_BOLD}SESSION HALT: {rem}s cooling ({survival.get('cooldown_reason','')}){C_RESET}"
    elif now_t < survival.get("global_cooldown_until", 0.0):
        rem = int(survival["global_cooldown_until"] - now_t)
        circuit_status = f"{C_YELLOW}COOLDOWN: {rem}s remaining ({survival.get('cooldown_reason','')}){C_RESET}"
    else:
        circuit_status = f"{C_GREEN}ACTIVE DEFENSE: ALL GATES CLEAR{C_RESET}"

    dir_lock = survival.get("directional_lock", {})
    if now_t < dir_lock.get("until", 0.0):
        rem = int(dir_lock["until"] - now_t)
        circuit_status += f"  │  {C_YELLOW}Streak Lock: {dir_lock.get('direction')} ({rem}s){C_RESET}"

    buf.append(row(f" Defense Status: {circuit_status}"))
    buf.append(border_div)

    # 2. Market Microstructure Card
    buf.append(row(f"{C_BOLD}{C_CYAN}MARKET MICROSTRUCTURE & MACRO REGIME:{C_RESET}"))
    buf.append(row(f" BTC Price: {C_BOLD}${ltp:,.1f}{C_RESET}  │  Spread: {sp:3.1f} bps  │  15m Trend: {mtf_badge}"))
    buf.append(row(f" Flow: L2 OBI: {obi_color}{obi:+.2f} ({obi_str}){C_RESET}  │  VPIN: {vpin}th pct  │  CVD Mom: {cmom:+.1f}"))
    buf.append(row(f" Macro Gate: {gate_badge}"))
    buf.append(border_div)

    # 3. Multi-Alpha Arena Tournament Table
    buf.append(row(f"{C_BOLD}{C_CYAN}MULTI-ALPHA TOURNAMENT ARENA:{C_RESET}"))
    buf.append(row(f" {C_DIM}STRATEGY    WEIGHT   STATUS           WR      CONVICTION       RATIONALE{C_RESET}"))
    for k, a in alphas.items():
        w_pct  = int(a["weight"] * 100)
        conv   = a["conviction"]
        d_str  = a["direction"]
        st     = a["status"]
        wr     = f"{a['rolling_wr']:.0f}%" if a["trades"] > 0 else "--%"

        if st == "ACTIVE":
            st_badge = f"{C_GREEN}ACTIVE 🟢   {C_RESET}"
        elif st == "PROBATION":
            st_badge = f"{C_YELLOW}PROBATION 🟡{C_RESET}"
        else:
            st_badge = f"{C_RED}QUARANTINE 🔒{C_RESET}"

        if d_str == "LONG":
            dir_pill = f"{C_GREEN}{C_BOLD}[+{conv:.2f} LONG ]{C_RESET}"
        elif d_str == "SHORT":
            dir_pill = f"{C_RED}{C_BOLD}[{conv:.2f} SHORT]{C_RESET}"
        else:
            dir_pill = f"{C_GRAY}[ {conv:+.2f} FLAT ]{C_RESET}"

        rat_short = a["rationale"][:26].ljust(26)
        buf.append(row(f" {C_BOLD}{a['tag']:9}{C_RESET}  {w_pct:3d}%    {st_badge}  {wr:>4}   {dir_pill}   {C_DIM}{rat_short}{C_RESET}"))

    buf.append(border_div)

    # 4. Position & Order Cockpit
    buf.append(row(f"{C_BOLD}{C_CYAN}EXECUTION & ACTIVE POSITION:{C_RESET}"))
    pos = portfolio["position"]
    if pos != "FLAT":
        unreal = portfolio["unrealized_pnl"]
        u_color = C_GREEN if unreal >= 0 else C_RED
        u_sign  = "+" if unreal >= 0 else ""
        be_str = f"{C_GREEN}BE LOCKED{C_RESET}" if portfolio["be_moved"] else f"{C_YELLOW}PHASE 1 (100% SIZE){C_RESET}"
        pos_color = C_GREEN if pos == "LONG" else C_RED

        buf.append(row(f" Position: {pos_color}{C_BOLD}{pos}{C_RESET} @ ${portfolio['entry_price']:,.1f}  │  PnL: {u_color}{C_BOLD}{u_sign}${unreal:.2f}{C_RESET}  │  Status: {be_str}"))
        buf.append(row(f" Orders:   SL: ${portfolio['active_sl_price']:,.1f}  │  70% TP1: ${portfolio['tp1_price']:,.1f} (+2.5R)  │  30% TP2: ${portfolio['active_tp_price']:,.1f} (+4.5R)"))
        buf.append(row(f" Exposure: Size: ${portfolio['notional_size_usd']:,.0f} ({portfolio['leverage_used']:.1f}x leverage)  │  Lead: {portfolio['lead_alpha']}"))
    else:
        flat_desc = circuit_status if "COOLDOWN" in circuit_status or "HALT" in circuit_status else f"Scanning for Consensus Setup >= {0.58 if sess in ['ASIAN','PACIFIC_LULL'] else 0.48:.2f}"
        buf.append(row(f" Position: {C_GRAY}FLAT{C_RESET} @ ${ltp:,.1f}  │  {flat_desc}"))

    buf.append(border_div)

    # 5. Recent Execution Audit Log
    buf.append(row(f"{C_BOLD}{C_CYAN}RECENT EXECUTION AUDIT LOG:{C_RESET}"))
    buf.append(row(f" {C_DIM}TIME     DIR   ENTRY      EXIT       NET PNL    R-MULT   OUTCOME / STRATEGY{C_RESET}"))
    if trade_journal:
        for t in trade_journal[-4:]:
            ts = t.get("open_ts", "")[11:19]
            act = t.get("action", "BUY")
            act_str = f"{C_GREEN}L{C_RESET}" if act == "BUY" else f"{C_RED}S{C_RESET}"
            ep = f"{t.get('entry_price', 0):,.0f}"
            out = t.get("outcome")
            if out == "OPEN":
                xp = " ---  "
                pnl_str = f"{C_YELLOW}ACTIVE{C_RESET}"
                r_mult  = " +0.0R"
                reas = f"PHASE {portfolio.get('phase', 1)} IN PLAY"
            else:
                xp = f"{t.get('exit_price', 0):,.0f}"
                pnl_val = t.get("pnl_usd", 0.0)
                if pnl_val >= 0:
                    pnl_str = f"{C_GREEN}+${pnl_val:.2f}{C_RESET}"
                    r_mult  = f"{C_GREEN}+2.5R{C_RESET}" if pnl_val > 5.0 else f"{C_GREEN}+0.4R{C_RESET}"
                else:
                    pnl_str = f"{C_RED}-${abs(pnl_val):.2f}{C_RESET}"
                    r_mult  = f"{C_RED}-1.0R{C_RESET}"
                reas = t.get("exit_reason", "CLOSED")

            lead = t.get("lead_alpha", "")
            lead_tag = f"[{lead}]" if lead else ""
            log_line = f" {ts}   {act_str}   {ep:>6}    {xp:>6}    {pnl_str:>16}   {r_mult:>14}   {reas} {lead_tag}"
            buf.append(row(log_line))
    else:
        buf.append(row(f" {C_DIM}No trades recorded in current tournament yet.{C_RESET}"))

    buf.append(border_bot)
    buf.append("\033[J")

    print("\n".join(buf), end="", flush=True)


# ---------------------------------------------------------
# WEBSOCKET STREAM HANDLERS
# ---------------------------------------------------------
async def binance_ws():
    url = ("wss://fstream.binance.com/stream?streams="
           "btcusdt@trade/btcusdt@bookTicker/btcusdt@depth20@100ms/btcusdt@forceOrder")
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                market_state["ws_connected"] = True
                while True:
                    msg  = await ws.recv()
                    data = json.loads(msg)
                    st   = data.get("stream")
                    pay  = data.get("data")

                    if st == "btcusdt@trade":
                        process_trade_tick(float(pay["p"]), float(pay["q"]), pay["m"])
                        update_session()
                        c1, c2, c3, consensus = update_alpha_evaluations()
                        run_meta_decision(c1, c2, c3, consensus)
                        check_exits(consensus)

                    elif st == "btcusdt@bookTicker":
                        market_state["bid"] = float(pay["b"])
                        market_state["ask"] = float(pay["a"])

                    elif st == "btcusdt@depth20@100ms":
                        try:
                            bids = pay.get("b", [])
                            asks = pay.get("a", [])
                            tb = sum(float(b[1]) for b in bids)
                            ta = sum(float(a[1]) for a in asks)
                            if tb + ta > 0:
                                market_state["obi"] = (tb - ta) / (tb + ta)
                        except Exception:
                            pass

                    elif st == "btcusdt@forceOrder":
                        try:
                            o = pay.get("o", {})
                            s = o.get("S")
                            uv = float(o.get("q", 0)) * float(o.get("p", 0))
                            if s == "SELL":
                                liquidations["longs_usd"]  += uv
                            else:
                                liquidations["shorts_usd"] += uv
                        except Exception:
                            pass

        except Exception as e:
            market_state["ws_connected"] = False
            await asyncio.sleep(3)


async def ui_loop():
    while True:
        if market_state["ws_connected"] and market_state["ticks_today"] > 0:
            render_survival_dashboard()
        await asyncio.sleep(1)


async def main():
    prewarm_engine()
    update_survival_health()
    await asyncio.gather(binance_ws(), ui_loop())


if __name__ == "__main__":
    asyncio.run(main())
