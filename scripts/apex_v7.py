"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  APEX INTELLIGENCE ENGINE v7.1 — FULL STACK EXECUTION SHELL                 ║
║  Uses ALL existing infrastructure: SignalEngine + Council + Ensemble         ║
╚══════════════════════════════════════════════════════════════════════════════╝

What this does differently from everything before:
  - LiveSignalEngine handles feature computation, regime routing, ensemble vote
  - Council (5 advisors) must approve every trade before execution
  - PerformanceMonitor tracks live model accuracy, halts if decay detected
  - WalkForwardRetrainer periodically updates models with new data
  - TickMomentumEngine runs in parallel for short-lived momentum entries
  - This script is ONLY an execution shell + dashboard. Brain = existing src/
"""

import asyncio
import json
import sys
import os
import re
import time
import datetime
from collections import deque
from pathlib import Path

# APEX v7.1 — Track A: load secrets from .env (never hardcode credentials)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd
import requests
import websockets

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if os.name == 'nt':
    os.system('')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.live_store import LiveFeatureStore
from src.models.xgboost_model import ApexXGBoostModel
from src.models.regime import RegimeDetector, MarketRegime
from src.models.ensemble import ModelEnsemble, WeightingMethod
from src.models.momentum_engine import TickMomentumEngine
from src.engine.signal_engine import LiveSignalEngine, TradingSignal

# ─── APEX v7.2 — Market Intelligence Providers ───────────────────────────────
from src.data.providers.news_guard import NewsGuard
from src.data.providers.funding_rate import FundingRateMonitor
from src.data.providers.sentiment import SentimentMonitor
from src.core.state_manager import StateManager
from src.models.training.retrain_scheduler import RetrainScheduler

# ─── ANSI ────────────────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"
BLU="\033[94m"; MAG="\033[95m"; CYN="\033[96m"
WHT="\033[97m"; GRY="\033[90m"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SYMBOL          = os.getenv("APEX_SYMBOL", "BTCUSDT")  # APEX v7.1 — Track A
INITIAL_CAPITAL = float(os.getenv("APEX_INITIAL_CAPITAL", "1000.0"))  # APEX v7.1 — Track A
# Multi-symbol support — add symbols here, models will be loaded per-symbol
SYMBOLS = ["BTCUSDT"]  # APEX v7.1 — Track A: Extend: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
MODEL_DIR       = Path("data/models")

# ─── RISK CONFIG (loaded from configs/risk.yaml, with safe defaults) ─────────────────────
def _load_risk_yaml():
    # Load risk.yaml; returns empty dict on any failure.
    try:
        import yaml
        p = Path(__file__).parent.parent / "configs" / "risk.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

_risk_cfg = _load_risk_yaml()

def _rc(section, key, default):
    # Safe accessor: _rc('sizing', 'max_allocation_pct', 0.05)
    return _risk_cfg.get(section, {}).get(key, default)

JOURNAL_FILE    = "trade_journal_v7.json"

# ─── EXECUTION MODE ───────────────────────────────────────────────────────────
# DEMO_MODE=true  → real orders on Bybit TESTNET (play money, real execution)
# DEMO_MODE=false → paper-only (no broker, no orders placed)
DEMO_MODE    = os.getenv("APEX_DEMO_MODE", "false").lower() == "true"
HEADLESS     = os.getenv("APEX_HEADLESS", "false").lower() == "true"  # server/cloud mode

# Bybit Testnet keys (only used when DEMO_MODE=true)
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")

TAKER_FEE_BPS   = 5.0
SLIPPAGE_BPS    = 2.0
ROUNDTRIP_BPS   = (TAKER_FEE_BPS + SLIPPAGE_BPS) * 2   # 14 bps

# ─── TELEGRAM ALERTS (optional) ───────────────────────────────────────────────
# APEX v7.1 — Track A: secrets loaded from .env, never hardcoded
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def telegram(msg: str):
    """Send a non-blocking Telegram alert. Silent if token not set."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception:
        pass

# Entry gates
MIN_PROB        = _rc("conviction", "min_signal_threshold", 0.58)   # from risk.yaml
STRONG_PROB     = 0.65      # High-conviction → full risk
MIN_PROB_DIFF   = 0.06      # Long vs Short separation
MIN_RR          = _rc("trade_management", "default_rr_target", 3.0)   # from risk.yaml
MIN_SL_USD      = 250.0     # Never tighter than $250 stop
MAX_SL_USD      = 800.0     # Never wider than $800

# Risk
BASE_RISK_PCT   = _rc("sizing", "min_allocation_pct", 0.010)   # from risk.yaml
MAX_RISK_PCT   = _rc("sizing", "max_allocation_pct", 0.015)   # from risk.yaml
MIN_RISK_PCT    = 0.005
MAX_LEVERAGE    = 5.0
MAX_TRADES_DAY  = _rc("sizing", "max_concurrent_trades", 3)   # from risk.yaml
MIN_INTERVAL    = 1800      # 30min between trades
LOSS_COOLDOWN   = 7200      # 2hr after loss
SESSION_LOSSES  = 3         # halt after N consecutive losses

# ─── BROKER (Bybit Testnet — only active when DEMO_MODE=true) ─────────────────
_bybit_broker = None   # instantiated in prewarm() when DEMO_MODE=true

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
market = {"ltp": 0.0, "bid": 0.0, "ask": 0.0,
          "spread_bps": 0.0, "obi": 0.0, "ws_connected": False, "ticks": 0}

# ─── MARKET INTELLIGENCE (APEX v7.2) ─────────────────────────────────────────
news_guard       = NewsGuard()
funding_monitor  = FundingRateMonitor()
sentiment_monitor = SentimentMonitor()
state_manager    = StateManager()
retrain_scheduler: RetrainScheduler | None = None   # set in prewarm()

# Candle buffers
candles_5m  = deque(maxlen=210)
candles_1h  = deque(maxlen=168)
live_candle = {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0, "timestamp": 0}
cur_min_ts  = 0
acc_1m      = []   # accumulate into 5m
acc_5m_15m  = []   # accumulate 5m into 15m
acc_15m_1h  = []   # accumulate 15m into 1h

# Async queue for safely passing 5m candles from sync tick handler → async ML loop
# Instantiated inside main() after the event loop is running
_candle_queue: asyncio.Queue | None = None

# CVD
cvd_bucket = 0.0

# Live model probabilities (updated every 5m candle)
live_probs = {"long": 0.5, "short": 0.5, "updated": False}

# Signal engine (full stack)
signal_engine: LiveSignalEngine | None = None

# Momentum engine (tick-level)
momentum_eng = TickMomentumEngine(symbol=SYMBOL, ofi_window=300,
                                   velocity_window=60, breakout_window=600,
                                   ofi_threshold=0.28, velocity_threshold=0.00008,
                                   recompute_interval=15)

# ─── PORTFOLIO ────────────────────────────────────────────────────────────────
portfolio = {
    "position": "FLAT", "direction": None,
    "entry_price": 0.0, "entry_time": 0.0,
    "sl_price": 0.0, "sl_distance": 0.0,
    "tp1_price": 0.0, "tp2_price": 0.0,
    "notional": 0.0, "initial_notional": 0.0,
    "unrealized_pnl": 0.0, "banked_pnl": 0.0,
    "phase": 0,
    "signal_id": "",
    "prob": 0.0,
    "council_score": 0.0,
    "regime_entry": "",
    "source": "",         # "ML" or "MOMENTUM"
    "highest_fav": 0.0,
}

# ─── ACCOUNT ──────────────────────────────────────────────────────────────────
account = {
    "balance": INITIAL_CAPITAL, "peak": INITIAL_CAPITAL,
    "drawdown_pct": 0.0, "net_pnl": 0.0,
    "total": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
    "profit_factor": 0.0,
    "avg_win": 0.0, "avg_loss": 0.0, "best": 0.0, "worst": 0.0,
    "total_wins_usd": 0.0, "total_losses_usd": 0.0,
    "consecutive_losses": 0,
    "max_consec_wins": 0,
    "avg_dur_min": 0.0,
    "best_day_usd": 0.0, "worst_day_usd": 0.0,
    "sharpe": 0.0,
    "trades_today": 0, "today_date": "",
    "last_trade_time": 0.0, "last_loss_time": 0.0,
    "halted": False, "halt_reason": "", "halt_until": 0.0,
}

session = {"name": "UNKNOWN", "risk_mult": 1.0}
last_signal: TradingSignal | None = None
regime_info = {"current": "UNKNOWN", "adx": 0.0, "chop": 0.0, "atr": 0.0}
trade_journal: list[dict] = []
trade_log = deque(maxlen=12)
# P3: last gate rejection — shown on dashboard when a signal is approved but not executed
last_gate_rejection = {"reason": "", "signal": "", "ts": 0.0}

# ─── PERSISTENCE ──────────────────────────────────────────────────────────────
def save_journal():
    try:
        with open(JOURNAL_FILE, "w") as f:
            json.dump(trade_journal, f, indent=2)
    except Exception:
        pass

def load_journal():
    global trade_journal
    try:
        if not os.path.exists(JOURNAL_FILE):
            return
        with open(JOURNAL_FILE) as f:
            trade_journal = json.load(f)
        closed = [t for t in trade_journal if t.get("outcome") in ["WIN","LOSS"]]
        if not closed:
            return
        wins = [t for t in closed if t["outcome"] == "WIN"]
        losses = [t for t in closed if t["outcome"] == "LOSS"]
        total_pnl = sum(t.get("pnl_usd", 0) for t in closed)
        account["balance"] = round(INITIAL_CAPITAL + total_pnl, 2)

        running = INITIAL_CAPITAL; peak = INITIAL_CAPITAL
        for t in closed:
            running += t.get("pnl_usd", 0)
            peak = max(peak, running)
        account["peak"] = round(peak, 2)

        account["total"]  = len(closed)
        account["wins"]   = len(wins)
        account["losses"] = len(losses)
        account["win_rate"] = round(len(wins)/len(closed)*100, 1) if closed else 0
        account["total_wins_usd"]   = sum(t.get("pnl_usd",0) for t in wins)
        account["total_losses_usd"] = abs(sum(t.get("pnl_usd",0) for t in losses))
        account["avg_win"]  = round(account["total_wins_usd"]/len(wins),   2) if wins   else 0
        account["avg_loss"] = round(account["total_losses_usd"]/len(losses),2) if losses else 0
        account["profit_factor"] = round(account["total_wins_usd"]/account["total_losses_usd"],3) if account["total_losses_usd"] > 0 else 999.0
        account["best"]  = max(t.get("pnl_usd",0) for t in closed)
        account["worst"] = min(t.get("pnl_usd",0) for t in closed)
        streak = 0
        for t in reversed(closed):
            if t["outcome"] == "LOSS": streak += 1
            else: break
        account["consecutive_losses"] = streak

        # Max consecutive wins
        win_streak = 0; best_win_streak = 0; cur_win = 0
        for t in closed:
            if t["outcome"] == "WIN":
                cur_win += 1; best_win_streak = max(best_win_streak, cur_win)
            else:
                cur_win = 0
        account["max_consec_wins"] = best_win_streak

        # Average trade duration (minutes)
        durations = [t.get("dur_min", 0) for t in closed if t.get("dur_min", 0) > 0]
        account["avg_dur_min"] = round(sum(durations) / len(durations), 1) if durations else 0.0

        # Best / worst day
        from collections import defaultdict
        daily_pnl: dict = defaultdict(float)
        for t in closed:
            day = t.get("open_ts", "")[:10]
            if day:
                daily_pnl[day] += t.get("pnl_usd", 0)
        if daily_pnl:
            account["best_day_usd"]  = round(max(daily_pnl.values()), 2)
            account["worst_day_usd"] = round(min(daily_pnl.values()), 2)
        else:
            account["best_day_usd"]  = 0.0
            account["worst_day_usd"] = 0.0

        # Sharpe ratio (annualized, requires >=20 trades with daily PnL)
        if len(daily_pnl) >= 5:
            daily_vals = list(daily_pnl.values())
            mean_d = sum(daily_vals) / len(daily_vals)
            var_d  = sum((x - mean_d)**2 for x in daily_vals) / len(daily_vals)
            std_d  = var_d**0.5
            account["sharpe"] = round((mean_d / std_d) * (252**0.5), 2) if std_d > 0 else 0.0
        else:
            account["sharpe"] = 0.0

        for t in closed[-8:]:
            pnl = t.get("pnl_usd", 0)
            s = "+" if pnl >= 0 else ""
            trade_log.append(f"[{t.get('open_ts','?')[11:19]}] {t.get('action','?')} "
                             f"@{t.get('entry_price',0):,.0f}→{t.get('exit_price',0):,.0f} "
                             f"PnL:{s}${pnl:.2f} ({t.get('exit_reason','?')})")

        for t in trade_journal:
            if t.get("outcome") == "OPEN":
                t["outcome"] = "CANCELLED"; t["exit_reason"] = "STALE"; t["pnl_usd"] = 0.0
        save_journal()
        update_health()
    except Exception as e:
        print(f"Journal load error: {e}")

def update_health():
    bal  = account["balance"]; peak = account["peak"]
    if bal > peak: account["peak"] = bal; peak = bal
    dd = (peak - bal) / peak * 100 if peak > 0 else 0
    account["drawdown_pct"] = round(dd, 2)
    account["net_pnl"]      = round(bal - INITIAL_CAPITAL, 2)
    if dd >= 20.0 and not account["halted"]:
        account["halted"] = True
        account["halt_reason"] = "MAX DRAWDOWN 20%"
        account["halt_until"] = float("inf")

# ─── SESSION ──────────────────────────────────────────────────────────────────
def update_session():
    utc_h = datetime.datetime.now(datetime.timezone.utc).hour
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if today != account["today_date"]:
        account["trades_today"] = 0; account["today_date"] = today
        if "MAX TRADES" in account.get("halt_reason","") or "DAY" in account.get("halt_reason",""):
            account["halted"] = False; account["halt_reason"] = ""; account["halt_until"] = 0.0
    session["name"] = ("LONDON" if 7<=utc_h<13 else "NEW_YORK" if 13<=utc_h<21
                        else "ASIAN" if 0<=utc_h<7 else "PACIFIC")
    session["risk_mult"] = 1.0 if session["name"] in ("LONDON","NEW_YORK") else (0.75 if session["name"]=="ASIAN" else 0.6)

def can_trade() -> tuple[bool, str]:
    now = time.time()
    if account["halted"]:
        if account["halt_until"] <= now:
            account["halted"] = False; account["halt_reason"] = ""
        else:
            rem = int(account["halt_until"] - now) if account["halt_until"] < float("inf") else 999999
            return False, f"HALTED: {account['halt_reason']} ({rem//60}m left)"
    if account["trades_today"] >= MAX_TRADES_DAY:
        return False, f"DAY LIMIT ({MAX_TRADES_DAY} done)"
    if account["last_loss_time"] > 0 and (now - account["last_loss_time"]) < LOSS_COOLDOWN:
        rem = int(LOSS_COOLDOWN - (now - account["last_loss_time"]))
        return False, f"LOSS COOLDOWN {rem//60}m{rem%60:02d}s"
    if account["last_trade_time"] > 0 and (now - account["last_trade_time"]) < MIN_INTERVAL:
        rem = int(MIN_INTERVAL - (now - account["last_trade_time"]))
        return False, f"INTERVAL {rem}s"
    if market["spread_bps"] > 8.0:
        return False, f"SPREAD {market['spread_bps']:.1f}bps"
    # APEX v7.2: News blackout gate
    blackout, blackout_reason = news_guard.is_blackout(SYMBOL)
    if blackout:
        return False, f"NEWS BLACKOUT: {blackout_reason}"
    return True, "CLEAR"

# ─── STRUCTURAL SL ────────────────────────────────────────────────────────────
def find_sl(direction: str, entry: float) -> tuple[float, float]:
    atr = max(regime_info["atr"], 80.0)
    min_sl = max(MIN_SL_USD, atr * 2.0)
    max_sl = min(MAX_SL_USD, atr * 5.0)

    candidates = []
    data = list(candles_5m)[-40:]
    if len(data) >= 10:
        if direction == "LONG":
            candidates.append(min(c["low"] for c in data[-20:]))
            candidates.append(min(c["low"] for c in data[-10:]))
        else:
            candidates.append(max(c["high"] for c in data[-20:]))
            candidates.append(max(c["high"] for c in data[-10:]))

    if direction == "LONG":
        valid = [c for c in candidates if c < entry and (entry - c) >= min_sl]
        sl_price = max(valid) - atr * 0.3 if valid else entry - min_sl
        sl_dist = entry - sl_price
    else:
        valid = [c for c in candidates if c > entry and (c - entry) >= min_sl]
        sl_price = min(valid) + atr * 0.3 if valid else entry + min_sl
        sl_dist = sl_price - entry

    sl_dist = float(np.clip(sl_dist, min_sl, max_sl))
    sl_price = entry - sl_dist if direction == "LONG" else entry + sl_dist
    return round(sl_price, 1), round(sl_dist, 1)

# ─── SIZING ───────────────────────────────────────────────────────────────────
def size_position(sl_dist: float, prob: float, entry: float, council_score: float = 0.7, direction: str = "LONG") -> tuple[float, float]:
    if prob >= STRONG_PROB or council_score >= 0.80:
        risk_pct = MAX_RISK_PCT
    elif prob >= MIN_PROB + 0.04:
        risk_pct = BASE_RISK_PCT
    else:
        risk_pct = MIN_RISK_PCT

    streak = account["consecutive_losses"]
    if streak >= 2:   risk_pct *= 0.50
    elif streak >= 1: risk_pct *= 0.70

    risk_pct *= session["risk_mult"]
    dd = account["drawdown_pct"]
    if dd >= 10: risk_pct *= 0.40
    elif dd >= 5: risk_pct *= 0.65

    # APEX v7.2: Sentiment & funding rate modifiers
    sentiment_mult = sentiment_monitor.get_regime_modifier()
    funding_bias   = funding_monitor.get_signal_bias(SYMBOL)
    # Reduce size if crowd is strongly against our direction
    if direction == "LONG" and funding_bias < -0.2:
        risk_pct *= 0.75
    elif direction == "SHORT" and funding_bias > 0.2:
        risk_pct *= 0.75
    # Apply funding size multiplier (reduces size in extreme crowding)
    risk_pct *= funding_monitor.size_multiplier(SYMBOL, direction)
    # Apply sentiment modifier
    risk_pct *= sentiment_mult

    risk_usd = account["balance"] * risk_pct
    sl_frac  = sl_dist / entry
    fee_frac = ROUNDTRIP_BPS / 10000.0
    total_loss = sl_frac + fee_frac
    if total_loss <= 0: return 0.0, 0.0

    notional = min(risk_usd / total_loss, account["balance"] * MAX_LEVERAGE)
    if notional < 80: return 0.0, 0.0
    return round(notional, 2), round(risk_usd, 2)

# ─── ENTRY EXECUTION ──────────────────────────────────────────────────────────
def execute_signal(signal: TradingSignal):
    """Execute a TradingSignal from the LiveSignalEngine."""
    global last_signal
    if portfolio["position"] != "FLAT": return

    ok, reason = can_trade()
    if not ok: return

    direction = "LONG" if signal.direction == "BUY" else "SHORT"
    prob      = signal.conviction
    council   = signal.council_score

    entry = market["ask"] if direction == "LONG" else market["bid"]
    if entry <= 0: return

    sl_price, sl_dist = find_sl(direction, entry)
    if sl_dist <= 0: return

    rr = 3.5 if council >= 0.85 else (3.0 if council >= 0.70 else MIN_RR)
    tp1 = entry + sl_dist * 2.0 if direction == "LONG" else entry - sl_dist * 2.0
    tp2 = entry + sl_dist * rr  if direction == "LONG" else entry - sl_dist * rr

    # APEX v7.2: Snap TP2 to nearest Volume Profile HVN / VPOC level
    try:
        from src.features.indicators.volume_profile import compute_volume_profile_snapshot
        vp = compute_volume_profile_snapshot(pd.DataFrame(list(candles_5m)[-100:]))
        if vp["profile_ready"]:
            candidates = [vp["poc"], vp["vah"]] + vp["hvn_prices"]
            if direction == "LONG":
                valid_tps = [p for p in candidates if p > entry + sl_dist * 2.0]
                if valid_tps:
                    nearest = min(valid_tps)
                    if (nearest - entry) / sl_dist >= MIN_RR:
                        tp2 = nearest
            else:
                valid_tps = [p for p in candidates if p < entry - sl_dist * 2.0]
                if valid_tps:
                    nearest = max(valid_tps)
                    if (entry - nearest) / sl_dist >= MIN_RR:
                        tp2 = nearest
    except Exception:
        pass

    notional, risk_usd = size_position(sl_dist, prob, entry, council, direction)
    if notional <= 0: return

    now_t   = time.time()
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    action  = "BUY" if direction == "LONG" else "SELL"

    portfolio.update({
        "position": "OPEN", "direction": direction,
        "entry_price": entry, "entry_time": now_t,
        "sl_price": sl_price, "sl_distance": sl_dist,
        "tp1_price": tp1, "tp2_price": tp2,
        "notional": notional, "initial_notional": notional,
        "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 1,
        "signal_id": signal.signal_id,
        "prob": prob, "council_score": council,
        "regime_entry": signal.regime, "source": "ML",
        "highest_fav": entry,
    })

    account["last_trade_time"] = now_t
    account["trades_today"]   += 1
    last_gate_rejection["reason"] = ""  # clear last rejection on successful entry
    last_signal = signal

    council_str = f"Council:{council:.2f}" if signal.council_approved else f"Council:OVERRIDE({council:.2f})"
    log = (f"[{now_str}] OPEN {action} @${entry:,.1f} | "
           f"SL:${sl_price:,.1f}(${sl_dist:.0f}) TP2:${tp2:,.1f}({rr:.1f}R) | "
           f"Prob:{prob:.3f} {council_str} Risk:${risk_usd:.2f}")
    trade_log.append(log)
    if HEADLESS: print(f"\n🔥 TRADE EVENT: {log}", flush=True)
    telegram(f"APEX OPEN {action} @${entry:,.0f} | SL:${sl_price:,.0f} TP2:${tp2:,.0f} | p={prob:.3f} Council:{council:.2f} | Risk:${risk_usd:.2f}")

    trade_journal.append({
        "id": len(trade_journal)+1,
        "open_ts": datetime.datetime.now().isoformat(),
        "action": action, "entry_price": entry,
        "sl_price": sl_price, "tp1_price": tp1, "tp2_price": tp2,
        "sl_dist": sl_dist, "notional": notional, "risk_usd": risk_usd,
        "leverage": round(notional/account["balance"], 2),
        "prob": prob, "council_score": council,
        "regime": signal.regime, "signal_id": signal.signal_id,
        "council_votes": signal.council_votes,
        "outcome": "OPEN", "exit_price": 0.0, "pnl_usd": 0.0, "exit_reason": ""
    })
    save_journal()
    state_manager.save_state(portfolio, account)   # APEX v7.2: crash-resume

    # ── DEMO MODE: place real order on Bybit Testnet ──────────────────────────
    if DEMO_MODE and _bybit_broker is not None:
        import asyncio as _aio
        async def _place():
            try:
                qty = round(notional / entry, 3)   # BTC quantity
                side = "Buy" if direction == "LONG" else "Sell"
                oid = await _bybit_broker.place_order_with_stops(
                    symbol=SYMBOL,
                    side=side,
                    qty=qty,
                    stop_loss=round(sl_price, 1),
                    take_profit=round(tp2, 1),
                )
                if oid:
                    trade_log.append(f"[BYBIT] Order placed: {oid}")
                    telegram(f"BYBIT TESTNET ORDER {side} {qty} BTC | SL:{sl_price:.1f} TP:{tp2:.1f} | id:{oid}")
            except Exception as _be:
                trade_log.append(f"[BYBIT] Order error: {_be}")
        try:
            loop = _aio.get_event_loop()
            loop.create_task(_place())
        except Exception:
            pass


# ─── EXIT ENGINE ──────────────────────────────────────────────────────────────
def check_exits():
    if portfolio["position"] != "OPEN": return

    direction = portfolio["direction"]
    entry     = portfolio["entry_price"]
    now_t     = time.time()
    dur_min   = (now_t - portfolio["entry_time"]) / 60.0

    cur = market["bid"] if direction == "LONG" else market["ask"]
    if cur <= 0: cur = market["ltp"]
    if cur <= 0: return

    # Unrealized PnL
    gbps = ((cur-entry)/entry if direction=="LONG" else (entry-cur)/entry) * 10000.0
    portfolio["unrealized_pnl"] = (gbps - ROUNDTRIP_BPS) / 10000.0 * portfolio["notional"]

    if direction == "LONG":
        portfolio["highest_fav"] = max(portfolio["highest_fav"], cur)
    elif portfolio["highest_fav"] == 0 or cur < portfolio["highest_fav"]:
        portfolio["highest_fav"] = cur

    def close(price: float, reason: str):
        if direction == "LONG":
            nbps = ((price-entry)/entry * 10000.0) - ROUNDTRIP_BPS
        else:
            nbps = ((entry-price)/entry * 10000.0) - ROUNDTRIP_BPS

        net_usd = (nbps/10000.0) * portfolio["notional"]
        total   = round(portfolio["banked_pnl"] + net_usd, 2)
        is_win  = total > 0

        account["balance"] += net_usd
        update_health()

        if is_win:
            account["wins"]             += 1
            account["consecutive_losses"] = 0
            account["total_wins_usd"]   += total
        else:
            account["losses"]           += 1
            account["consecutive_losses"] += 1
            account["total_losses_usd"] += abs(total)
            account["last_loss_time"]    = time.time()
            if signal_engine and portfolio["signal_id"]:
                signal_engine.resolve_signal(portfolio["signal_id"], False, total)
            # Anti-spiral
            if account["consecutive_losses"] >= SESSION_LOSSES:
                account["halted"]     = True
                account["halt_reason"] = f"{SESSION_LOSSES} consecutive losses"
                account["halt_until"] = time.time() + 4*3600
            elif account["consecutive_losses"] >= 2:
                account["halted"]     = True
                account["halt_reason"] = "2 consecutive losses — cooling"
                account["halt_until"] = time.time() + LOSS_COOLDOWN

        if is_win and signal_engine and portfolio["signal_id"]:
            signal_engine.resolve_signal(portfolio["signal_id"], True, total)

        account["total"]   += 1
        account["win_rate"] = round(account["wins"]/account["total"]*100, 1) if account["total"] > 0 else 0
        account["avg_win"]  = round(account["total_wins_usd"]/account["wins"],   2) if account["wins"]   > 0 else 0
        account["avg_loss"] = round(account["total_losses_usd"]/account["losses"],2) if account["losses"] > 0 else 0
        account["profit_factor"] = round(account["total_wins_usd"]/account["total_losses_usd"],3) if account["total_losses_usd"] > 0 else 999.0
        account["best"]  = max(account["best"], total)
        account["worst"] = min(account["worst"], total)

        s = "+" if total >= 0 else ""
        trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                         f"CLOSE @${price:,.1f} PnL:{s}${total:.2f} | {reason}")
        # P4: Telegram on close
        emoji = "✅" if is_win else "❌"
        telegram(f"{emoji} APEX CLOSE @${price:,.0f} | PnL: {s}${total:.2f} | {reason} | Balance: ${account['balance']:,.2f}")

        for t in reversed(trade_journal):
            if t.get("outcome") == "OPEN":
                t.update({"exit_ts": datetime.datetime.now().isoformat(),
                          "exit_price": price, "pnl_usd": total,
                          "outcome": "WIN" if is_win else "LOSS",
                          "exit_reason": reason, "dur_min": round(dur_min,1)})
                break
        save_journal()
        state_manager.save_state(portfolio, account)   # APEX v7.2: crash-resume

        portfolio.update({"position": "FLAT", "direction": None,
                          "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0})

    sl = portfolio["sl_price"]
    tp1 = portfolio["tp1_price"]
    tp2 = portfolio["tp2_price"]
    atr = max(regime_info["atr"], 80.0)

    # 1. SL
    if direction == "LONG" and cur <= sl: close(cur, "STOP LOSS"); return
    if direction == "SHORT" and cur >= sl: close(cur, "STOP LOSS"); return

    # 2. Partial TP at 2R
    if portfolio["phase"] == 1:
        hit = (direction=="LONG" and cur>=tp1) or (direction=="SHORT" and cur<=tp1)
        if hit:
            partial = portfolio["unrealized_pnl"] * 0.50
            portfolio["banked_pnl"] += partial
            account["balance"]      += partial
            update_health()
            portfolio["notional"]   *= 0.50
            fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
            portfolio["sl_price"] = entry + fee_buf if direction=="LONG" else entry - fee_buf
            portfolio["phase"] = 2
            trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                             f"2R TP1 +${partial:.2f} | 50% locked, SL→BE")

    # 3. Runner trail + full TP
    if portfolio["phase"] == 2:
        trail_buf = atr * 2.5
        if direction == "LONG":
            new_sl = cur - trail_buf
            if new_sl > portfolio["sl_price"]: portfolio["sl_price"] = new_sl
            if cur >= tp2: close(cur, "FULL TP"); return
        else:
            new_sl = cur + trail_buf
            if new_sl < portfolio["sl_price"]: portfolio["sl_price"] = new_sl
            if cur <= tp2: close(cur, "FULL TP"); return

    # 4. Time decay
    if dur_min > 360: close(cur, "MAX DURATION 6h"); return
    if dur_min > 240 and portfolio["phase"] == 1 and portfolio["unrealized_pnl"] < 0:
        close(cur, "TIME DECAY 4h adverse"); return

    # 5. Model-halt exit — if engine halted due to decay, get out
    if signal_engine and signal_engine.is_halted and portfolio["unrealized_pnl"] > 0:
        close(cur, "ENGINE HALTED — locking profits"); return

# ─── CANDLE PIPELINE ──────────────────────────────────────────────────────────
async def _candle_worker():
    """Async worker: drains _candle_queue and processes each 5m candle through the ML engine."""
    global last_signal
    while True:
        c5 = await _candle_queue.get()
        if not signal_engine:
            _candle_queue.task_done()
            continue
        try:
            sig = await signal_engine.process_candle(c5)

            # APEX v7.2: Feed candle to retrain scheduler (non-blocking)
            if retrain_scheduler:
                try:
                    retrain_scheduler.on_candle(c5)
                    retrain_scheduler.check_pending_swap()   # pick up new models if ready
                except Exception:
                    pass

            # Always extract live probabilities from the ensemble — even if no signal
            try:
                feats = signal_engine.live_store.update(c5)  # already called inside; uses cached
                if feats is not None and signal_engine.ensemble_long and signal_engine.ensemble_short:
                    re = signal_engine.regime_detector.current_regime
                    pl, _, _ = signal_engine.ensemble_long.predict(feats, current_regime=re.value)
                    ps, _, _ = signal_engine.ensemble_short.predict(feats, current_regime=re.value)
                    live_probs["long"]    = round(pl, 4)
                    live_probs["short"]   = round(ps, 4)
                    live_probs["updated"] = True
                    regime_info["current"] = re.value
            except Exception:
                pass

            if sig is not None:
                last_signal = sig
                regime_info["current"] = sig.regime
                regime_info["atr"]     = sig.atr

                ok, reason = can_trade()
                if portfolio["position"] == "FLAT" and ok:
                    if sig.council_approved and sig.direction in ("BUY", "SELL"):
                        execute_signal(sig)
                elif portfolio["position"] == "FLAT" and sig.council_approved and sig.direction in ("BUY", "SELL"):
                    # P3: Approved but gated — track for dashboard
                    last_gate_rejection["reason"] = reason
                    last_gate_rejection["signal"] = ("LONG" if sig.direction == "BUY" else "SHORT") + f" p={sig.conviction:.3f}"
                    last_gate_rejection["ts"] = time.time()
        except Exception:
            pass
        finally:
            _candle_queue.task_done()


def process_tick(price: float, qty: float, is_buyer_maker: bool):
    global cur_min_ts, cvd_bucket

    market["ltp"] = price
    market["ticks"] += 1
    if price <= 0 or qty <= 0: return

    now = time.time()

    # Build 1m candle
    if cur_min_ts == 0:
        cur_min_ts = int(now // 60) * 60
        live_candle.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0, "timestamp": cur_min_ts * 1000})

    if now >= cur_min_ts + 60:
        # Minute closed → batch into 5m
        acc_1m.append(dict(live_candle))
        if len(acc_1m) >= 5:
            c5 = {"open": acc_1m[0]["open"], "high": max(x["high"] for x in acc_1m),
                  "low": min(x["low"] for x in acc_1m), "close": acc_1m[-1]["close"],
                  "volume": sum(x["volume"] for x in acc_1m), "timestamp": int(now * 1000)}
            acc_1m.clear()
            candles_5m.append(dict(c5))
            if _candle_queue is not None:
                try:
                    _candle_queue.put_nowait(c5)
                except asyncio.QueueFull:
                    pass  # Drop if queue backed up (shouldn't happen at 5m intervals)

        cur_min_ts = int(now // 60) * 60
        live_candle.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0, "timestamp": cur_min_ts * 1000})
    else:
        if live_candle["high"] == 0: live_candle.update({"open": price, "high": price, "low": price})
        live_candle["high"]   = max(live_candle["high"], price)
        live_candle["low"]    = min(live_candle["low"],  price)
        live_candle["close"]  = price
        live_candle["volume"] += qty

    if market["bid"] > 0 and market["ask"] > 0:
        market["spread_bps"] = ((market["ask"] - market["bid"]) / price) * 10000.0

    # CVD
    cvd_bucket += -qty if is_buyer_maker else qty

    # Momentum engine (tick-level signals for quick entries)
    mom_sig = momentum_eng.add_tick(price, qty, is_buyer_maker, int(now * 1000))

    # On every tick: check exits, manage positions
    check_exits()


# ─── PREWARM ──────────────────────────────────────────────────────────────────

def fetch_klines(symbol: str, interval: str, limit: int = 210) -> list[dict]:
    """
    Fetch historical klines with multi-exchange fallback:
    1. Try Binance Futures (fapi.binance.com)
    2. Try Binance Spot (api.binance.com)
    3. Fallback to Bybit Linear (api.bybit.com) - unblocked in US cloud datacenters
    Returns list of dicts: [{'timestamp': int, 'open': float, 'high': float, 'low': float, 'close': float, 'volume': float}, ...]
    sorted oldest to newest.
    """
    # 1. Try Binance Futures
    try:
        r = requests.get(
            f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}",
            timeout=8
        )
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return [
                {"timestamp": int(x[0]), "open": float(x[1]), "high": float(x[2]),
                 "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])}
                for x in data
            ]
    except Exception:
        pass

    # 2. Try Binance Spot
    try:
        r = requests.get(
            f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}",
            timeout=8
        )
        data = r.json()
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            return [
                {"timestamp": int(x[0]), "open": float(x[1]), "high": float(x[2]),
                 "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])}
                for x in data
            ]
    except Exception:
        pass

    # 3. Fallback to Bybit Linear (works globally without geo-restrictions)
    interval_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '2h': '120', '4h': '240', '1d': 'D'}
    bybit_interval = interval_map.get(interval, '5')
    try:
        r = requests.get(
            f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}",
            timeout=8
        )
        data = r.json()
        if data.get("retCode") == 0:
            rows = data.get("result", {}).get("list", [])
            rows = rows[::-1]  # Bybit is newest-first; reverse to oldest-first
            return [
                {"timestamp": int(x[0]), "open": float(x[1]), "high": float(x[2]),
                 "low": float(x[3]), "close": float(x[4]), "volume": float(x[5])}
                for x in rows
            ]
    except Exception:
        pass

    return []

def prewarm():
    global signal_engine

    print("=" * 72)
    print("  APEX INTELLIGENCE ENGINE v7.1 — FULL STACK INITIALIZATION")
    print("=" * 72)

    # 1. Load all XGBoost models
    print("  Loading XGBoost ensemble models...")
    models: dict[str, ApexXGBoostModel] = {}
    regimes_dir = ["ranging_high_vol", "ranging_low_vol", "trending_low_vol"]
    for regime in regimes_dir:
        for direction in ["LONG", "SHORT"]:
            for bag_n in [1, 2, 3]:
                name = f"btcusdt_crypto_{regime}_{direction}_bag{bag_n}"
                meta_path = MODEL_DIR / f"{name}_metadata.json"
                model_path = MODEL_DIR / f"{name}.json"
                if model_path.exists() and meta_path.exists():
                    try:
                        meta = json.load(open(meta_path))
                        m = ApexXGBoostModel(name=name, feature_columns=meta["feature_columns"])
                        m.load(path=MODEL_DIR)
                        key = f"btcusdt_crypto_{regime}_{direction}_bag{bag_n}"
                        models[key] = m
                    except Exception as e:
                        print(f"    WARN: Could not load {name}: {e}")

    # Also try to load transformer model if available
    try:
        from src.models.transformer_model import ApexTransformerModel
        tf_path = MODEL_DIR / "btcusdt_crypto_transformer.pth"
        if tf_path.exists():
            tf_model = ApexTransformerModel.load("btcusdt_crypto_transformer", str(MODEL_DIR))
            if tf_model:
                models["btcusdt_crypto_transformer"] = tf_model
                print(f"  ✓ Transformer model loaded")
    except Exception:
        pass

    print(f"  ✓ {len(models)} model files loaded")

    # 2. Build LiveSignalEngine with the full stack
    signal_engine = LiveSignalEngine(
        symbol=SYMBOL,
        market_type="crypto",
        models=models,
        conviction_threshold=MIN_PROB,
        risk_per_trade=account["balance"] * BASE_RISK_PCT,
        default_rr=MIN_RR,
        max_concurrent=1,
        enable_ensemble=True,
        enable_monitoring=True,
        enable_walk_forward=True,
        enable_council=True,
        enable_statistical_alpha=False,  # We use ML, not the Donchian fallback
        retrain_interval=2000,
        consensus_threshold=0.60,
    )
    print(f"  ✓ LiveSignalEngine created (ensemble + council + monitor)")

    # 3. Warm up with historical 5m data (robust multi-exchange fetch)
    print("  Fetching historical 5m candles (210 × 5m)...")
    raw5m = fetch_klines(SYMBOL, "5m", 210)
    if not raw5m:
        print("  ⚠️ Warning: Could not fetch 5m candles from any exchange, starting with empty buffer")
        prewarm.replay_candles = []
    else:
        hist = pd.DataFrame(raw5m)
        signal_engine.warm_up(hist[["timestamp","open","high","low","close","volume"]])
        print(f"  ✓ Signal engine warmed up: {signal_engine.live_store.buffer_length} candles")
        for c in raw5m:
            candles_5m.append(c)
        prewarm.replay_candles = raw5m[-3:]

    # 4. Load 1h for structural SL
    raw1h = fetch_klines(SYMBOL, "1h", 72)
    for c in raw1h:
        candles_1h.append(c)
    print(f"  ✓ 1H candles loaded: {len(candles_1h)}")

    if raw5m:
        market["ltp"] = float(raw5m[-1]["close"])
        market["bid"] = market["ltp"]; market["ask"] = market["ltp"]
    regime_info["atr"] = 200.0  # Default until first signal

    update_session()
    load_journal()
    # APEX v7.2: restore account state after crash/restart
    state_manager.load_state(portfolio, account)
    update_health()

    # APEX v7.2: Set up hot-swap retrain scheduler
    global retrain_scheduler

    def _on_models_ready(new_models):
        """Hot-swap callback — atomically replaces ensemble members."""
        if signal_engine and new_models:
            try:
                signal_engine.hot_swap_models(new_models)
                telegram(f"🔄 APEX: {len(new_models)} models retrained + hot-swapped")
                print(f"\n  ✓ HOT-SWAP: {len(new_models)} new models loaded into ensemble")
            except Exception as e:
                print(f"  WARN: hot_swap error: {e}")

    retrain_scheduler = RetrainScheduler(
        symbol=SYMBOL,
        retrain_every_n_candles=2000,
        swap_callback=_on_models_ready,
    )

    print(f"\n  ── Current State ──")
    print(f"  BTC:     ${market['ltp']:,.1f}")
    print(f"  Account: ${account['balance']:,.2f} | PnL: ${account['net_pnl']:+.2f} | W{account['wins']}/L{account['losses']}")
    print(f"  Engine:  {len(models)} models | Council=ON | Monitor=ON | WalkForward=ON")
    print("=" * 72)
    print("  ALL SYSTEMS GO — FULL INTELLIGENCE STACK ACTIVE")
    print("=" * 72)

    # ── Broker init (Bybit Testnet — DEMO_MODE only) ──────────────────────────
    global _bybit_broker
    if DEMO_MODE:
        if BYBIT_API_KEY and BYBIT_API_SECRET:
            from src.execution.bybit_broker import BybitBroker
            _bybit_broker = BybitBroker(
                api_key=BYBIT_API_KEY,
                api_secret=BYBIT_API_SECRET,
                use_testnet=True,
            )
            print(f"  ✓ Bybit TESTNET broker ready (key: ...{BYBIT_API_KEY[-6:]})")
        else:
            print("  ⚠  DEMO_MODE=true but BYBIT_API_KEY not set → paper-only")
    else:
        print("  ► PAPER TRADE MODE — no broker orders")

    if HEADLESS:
        import os as _os
        _os.makedirs("logs", exist_ok=True)
        print("  ► HEADLESS MODE — logs/apex.log")


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
ANSI_RE = re.compile(r'\033\[[0-9;]*[mHJK]')
_first = True

def vlen(s): return len(ANSI_RE.sub('', s))
def trunc(s, n):
    cur, res = 0, []
    for t in re.split(r'(\033\[[0-9;]*[mHJK])', s):
        if not t: continue
        if t.startswith('\033['): res.append(t)
        else:
            rem = n - cur
            if rem <= 0: break
            res.append(t[:rem-1]+'…' if len(t)>rem else t); cur+=min(len(t),rem)
    res.append(R); return ''.join(res)

def render():
    global _first
    if _first: os.system('cls' if os.name=='nt' else 'clear'); _first=False

    W = 94
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def row(c):
        vl = vlen(c)
        if vl>W: c=trunc(c,W); vl=vlen(c)
        return f"{CYN}║{R} {c}{' '*(W-vl)} {CYN}║{R}\033[K"

    TOP=f"{CYN}╔{'═'*(W+2)}╗{R}\033[K"; MID=f"{CYN}╠{'═'*(W+2)}╣{R}\033[K"
    DIV=f"{CYN}╟{'─'*(W+2)}╢{R}\033[K"; BOT=f"{CYN}╚{'═'*(W+2)}╝{R}\033[K"

    buf=["\033[H", TOP]
    buf.append(row(f"{BOLD}{WHT}⚡ APEX INTELLIGENCE ENGINE v7.1 — FULL ML STACK (Council + Ensemble + Monitor) ⚡{R}"))
    ws = f"{GRN}LIVE ●{R}" if market["ws_connected"] else f"{RED}DISCONNECTED{R}"
    buf.append(row(f"{DIM}Time:{R} {now} IST  │  {CYN}{session['name']} ({session['risk_mult']:.0%}){R}  │  WS: {ws}"))
    buf.append(MID)

    # Account
    pnl_c = GRN if account["net_pnl"]>=0 else RED
    dd_c  = GRN if account["drawdown_pct"]<3 else (YLW if account["drawdown_pct"]<8 else RED)
    pf_c  = GRN if account["profit_factor"]>=1.5 else (YLW if account["profit_factor"]>=1.0 else RED)
    buf.append(row(f"{BOLD}{CYN}ACCOUNT:{R}"))
    buf.append(row(f" Balance: {BOLD}${account['balance']:,.2f}{R}  │  PnL: {pnl_c}{BOLD}${account['net_pnl']:+.2f}{R}  │  DD: {dd_c}{account['drawdown_pct']:.1f}%{R}  │  {account['wins']}W/{account['losses']}L ({account['win_rate']:.0f}%)  │  PF: {pf_c}{account['profit_factor']:.2f}{R}"))
    if account["avg_win"]>0 or account["avg_loss"]>0:
        buf.append(row(f" AvgW: {GRN}+${account['avg_win']:.2f}{R}  AvgL: {RED}-${account['avg_loss']:.2f}{R}  │  Best: {GRN}+${account['best']:.2f}{R}  Worst: {RED}${account['worst']:.2f}{R}  │  Streak: {account['consecutive_losses']}L"))

    ok, reason = can_trade()
    if account["halted"]: buf.append(row(f" {RED}{BOLD}⛔ {account['halt_reason']}{R}"))
    elif not ok:          buf.append(row(f" {YLW}⏸  {reason}{R}"))
    else:                 buf.append(row(f" {GRN}✓  ALL GATES CLEAR  │  Trades today: {account['trades_today']}/{MAX_TRADES_DAY}{R}"))
    buf.append(DIV)

    # Signal Engine status
    buf.append(row(f"{BOLD}{CYN}ML SIGNAL ENGINE:{R}"))
    if signal_engine:
        reg = regime_info["current"]
        atr = regime_info["atr"]
        reg_c = GRN if "TRENDING" in reg else YLW
        halted_str = f" {RED}[MODEL DECAY — HALTED]{R}" if signal_engine.is_halted else ""
        supp = signal_engine._suppression_counter
        supp_str = f"  {YLW}[SUPPRESSED {supp} candles]{R}" if supp > 0 else ""
        buf.append(row(f" Regime: {reg_c}{reg}{R}{halted_str}{supp_str}  │  ATR:${atr:,.0f}  │  Candles:{signal_engine._candles_processed}  │  Signals:{signal_engine._signals_generated}"))

        # Live probability bars (updated every 5m close)
        pl = live_probs["long"]; ps = live_probs["short"]
        pl_c = GRN if pl >= MIN_PROB else (YLW if pl >= 0.52 else GRY)
        ps_c = RED if ps >= MIN_PROB else (YLW if ps >= 0.52 else GRY)
        pl_bar = "█" * int(pl * 12) + "░" * (12 - int(pl * 12))
        ps_bar = "█" * int(ps * 12) + "░" * (12 - int(ps * 12))
        thr_tag = f"threshold {MIN_PROB}"
        buf.append(row(f" LONG  p:{pl_c}{pl_bar} {pl:.4f}{R}  │  SHORT p:{ps_c}{ps_bar} {ps:.4f}{R}  [{thr_tag}]"))

        # Last signal details
        ls = last_signal
        if ls:
            ls_dir = "LONG" if ls.direction == "BUY" else "SHORT"
            ls_c = GRN if ls.direction == "BUY" else RED
            council_c = GRN if ls.council_approved else RED
            buf.append(row(f" Last: {ls_c}{ls_dir}{R} p={ls.conviction:.3f}  Council: {council_c}{'APPROVED' if ls.council_approved else 'BLOCKED'}{R} score={ls.council_score:.2f}  [{ls.council_summary}]"))
            # P3: Show if approved but gated
            gr = last_gate_rejection
            if gr["reason"] and gr["signal"] and (time.time() - gr["ts"]) < 600:
                age = int(time.time() - gr["ts"])
                buf.append(row(f" {YLW}⚠  Approved {gr['signal']} NOT executed — {gr['reason']} ({age}s ago){R}"))
        else:
            buf.append(row(f" {GRY}Waiting for first live 5m candle close...{R}"))

        # Momentum engine
        mom_d = momentum_eng.get_diagnostics()
        mom_sig_str = mom_d["signal"]
        mom_c = GRN if mom_sig_str=="LONG" else (RED if mom_sig_str=="SHORT" else GRY)
        buf.append(row(f" Momentum: {mom_c}{mom_sig_str}{R}  OFI:{mom_d['ofi']:+.3f}  Ticks:{mom_d['ticks_ingested']}  OBI:{market['obi']:+.3f}"))
    else:
        buf.append(row(f" {GRY}Engine not initialized{R}"))
    buf.append(DIV)

    # ─── MARKET INTELLIGENCE (APEX v7.2) ──────────────────────────────────────
    buf.append(row(f"{BOLD}{CYN}MARKET INTEL:{R}"))
    # Fear & Greed
    fg_val   = sentiment_monitor.get_fear_greed()
    fg_label = sentiment_monitor.get_label()
    fg_col   = sentiment_monitor.get_color_tag()
    # Funding rate
    fr_pct   = funding_monitor.get_funding_pct(SYMBOL)
    fr_label = funding_monitor.get_label(SYMBOL)
    fr_col   = RED if funding_monitor.is_extreme(SYMBOL) else (YLW if abs(fr_pct) > 0.03 else GRN)
    # Next event
    nxt = news_guard.get_next_event()
    if nxt:
        mins_to = (nxt["time"] - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0
        evt_str = f"{nxt['name'][:35]} in {mins_to:.0f}m"
        evt_col = RED if mins_to < 30 else (YLW if mins_to < 90 else GRY)
    else:
        evt_str, evt_col = "No events scheduled", GRY
    # Blackout status
    is_blk, blk_rsn = news_guard.is_blackout(SYMBOL)
    news_str = f"{RED}⛔ BLACKOUT{R}" if is_blk else f"{GRN}✓ CLEAR{R}"
    buf.append(row(f" F&G: {fg_col}{fg_val} ({fg_label}){R}  │  Funding: {fr_col}{fr_pct:+.4f}% ({fr_label}){R}  │  News: {news_str}"))
    buf.append(row(f" Next Event: {evt_col}{evt_str}{R}"))
    if retrain_scheduler:
        rs = retrain_scheduler.get_status()
        rt_str = f"{YLW}RETRAINING...{R}" if rs["is_retraining"] else f"{GRY}Next in {rs['next_retrain_in']} candles{R}"
        buf.append(row(f" Retrain: {rt_str}  │  Count: {rs['retrain_count']}  │  Candles: {rs['candles_accumulated']}"))
    buf.append(DIV)

    # Position
    buf.append(row(f"{BOLD}{CYN}POSITION:{R}"))
    if portfolio["position"] == "OPEN":
        d = portfolio["direction"]
        d_c = GRN if d=="LONG" else RED
        ur_c = GRN if portfolio["unrealized_pnl"]>=0 else RED
        ph = "FULL" if portfolio["phase"]==1 else "RUNNER(50%)"
        dur = (time.time() - portfolio["entry_time"]) / 60.0
        buf.append(row(f" {d_c}{BOLD}{d}{R} @${portfolio['entry_price']:,.1f}  │  PnL: {ur_c}{BOLD}${portfolio['unrealized_pnl']:+.2f}{R}  │  Phase:{ph}  │  Age:{dur:.0f}m"))
        buf.append(row(f" SL:${portfolio['sl_price']:,.1f}(${portfolio['sl_distance']:.0f})  │  TP1:${portfolio['tp1_price']:,.1f}  │  TP2:${portfolio['tp2_price']:,.1f}  │  P:{portfolio['prob']:.3f}  Cncl:{portfolio['council_score']:.2f}"))
    else:
        buf.append(row(f" {GRY}FLAT @${market['ltp']:,.1f}  │  Waiting for Council-approved ML signal{R}"))
    buf.append(DIV)

    # Trade log
    buf.append(row(f"{BOLD}{CYN}TRADE LOG:{R}"))
    logs = list(trade_log)[-7:]
    if logs:
        for l in logs:
            c = GRN if "+$" in l or "TP" in l else (RED if "-$" in l or "STOP" in l or "LOSS" in l else GRY)
            buf.append(row(f" {c}{l}{R}"))
    else:
        buf.append(row(f" {DIM}No trades yet — waiting for high-conviction Council vote...{R}"))
    buf.append(BOT)
    buf.append("\033[J")
    print("\n".join(buf), end="", flush=True)


# ─── WEBSOCKET ────────────────────────────────────────────────────────────────
async def binance_ws():
    url = (f"wss://fstream.binance.com/stream?streams="
           f"{SYMBOL.lower()}@trade/{SYMBOL.lower()}@bookTicker"
           f"/{SYMBOL.lower()}@depth20@100ms/{SYMBOL.lower()}@kline_5m")
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                market["ws_connected"] = True
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    st, pay = data.get("stream",""), data.get("data",{})
                    if "@trade" in st:
                        try:
                            process_tick(float(pay["p"]), float(pay["q"]), pay["m"])
                            update_session()
                        except Exception as _te:
                            pass   # never crash WS loop on a bad tick
                    elif "@bookTicker" in st:
                        market["bid"] = float(pay["b"]); market["ask"] = float(pay["a"])
                    elif "@depth20" in st:
                        try:
                            bids=pay.get("b",[]); asks=pay.get("a",[])
                            tb=sum(float(b[1]) for b in bids); ta=sum(float(a[1]) for a in asks)
                            if tb+ta>0: market["obi"] = round((tb-ta)/(tb+ta), 3)
                        except Exception: pass
                    elif "@kline_5m" in st:
                        # ── APEX v7.3: Direct kline stream — no manual accumulation ──
                        k = pay.get("k", {})
                        if k.get("x"):   # x=True means candle is closed
                            c5 = {
                                "open":      float(k["o"]),
                                "high":      float(k["h"]),
                                "low":       float(k["l"]),
                                "close":     float(k["c"]),
                                "volume":    float(k["v"]),
                                "timestamp": int(k["t"]),
                            }
                            candles_5m.append(dict(c5))
                            if _candle_queue is not None:
                                try:
                                    _candle_queue.put_nowait(c5)
                                except asyncio.QueueFull:
                                    pass
        except Exception:
            market["ws_connected"] = False
            await asyncio.sleep(3)

async def ui_loop():
    _boot_ts = __import__("time").time()
    _log_file = open("logs/apex.log", "a", buffering=1) if HEADLESS else None
    while True:
        if HEADLESS:
            # Server mode: write status line to log file and stdout every 30s
            import time as _t
            if int(_t.time()) % 30 == 0:
                ltp = market.get("ltp", 0)
                bal = account.get("balance", 0)
                pos = portfolio.get("position", "FLAT")
                ticks = market.get("ticks", 0)
                pnl = account.get("pnl", 0.0)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                line = f"[{ts}] BTC: ${ltp:,.1f} | Bal: ${bal:,.2f} | PnL: ${pnl:+,.2f} | Pos: {pos} | Ticks: {ticks}"
                print(line, flush=True)
                try:
                    _log_file.write(line + "\n")
                except Exception:
                    pass
        else:
            # Terminal mode: render dashboard
            if market["ws_connected"] and market["ticks"] > 0:
                render()
            elif __import__("time").time() - _boot_ts > 4:
                render()
        await asyncio.sleep(1)

# ─── MARKET INTELLIGENCE LOOP (APEX v7.2) ────────────────────────────────────
async def _market_intel_loop():
    """Background loop: updates news, funding rate, and sentiment every 5 minutes."""
    # Initial fetch at startup — fire-and-forget with 8s timeout per provider
    # Using create_task so a hanging DNS call never blocks WS or dashboard
    async def _safe_fetch(coro, name):
        try:
            await asyncio.wait_for(coro, timeout=8)
        except Exception:
            pass   # DNS failure, timeout, etc. — silently skip

    asyncio.create_task(_safe_fetch(news_guard.update_events(), "news"))
    asyncio.create_task(_safe_fetch(funding_monitor.update(SYMBOL), "funding"))
    asyncio.create_task(_safe_fetch(sentiment_monitor.update(), "sentiment"))
    # Don't await — let them run in background while WS + dashboard start immediately
    while True:
        await asyncio.sleep(300)   # 5 minutes
        try:
            await asyncio.gather(
                asyncio.wait_for(news_guard.update_events(), timeout=10),
                asyncio.wait_for(funding_monitor.update(SYMBOL), timeout=10),
                asyncio.wait_for(sentiment_monitor.update(), timeout=10),
                return_exceptions=True,
            )
            # Push fresh sentiment data into Council's SentimentAdvisor
            if signal_engine and signal_engine.council:
                try:
                    fg_val = int(sentiment_monitor.get_fear_greed())
                    fund_pct = float(funding_monitor.get_funding_pct(SYMBOL))
                    signal_engine.council.update_sentiment(fg_val, fund_pct)
                except Exception:
                    pass
        except Exception:
            pass

async def main():
    global _candle_queue
    _candle_queue = asyncio.Queue(maxsize=50)  # Must be created inside running event loop
    prewarm()
    update_health()

    # ── Replay last few historical candles to populate regime/ATR ──
    replay = getattr(prewarm, "replay_candles", [])
    if replay and signal_engine:
        print(f"  Replaying {len(replay)} historical candles to initialize ML state...")
        for c in replay:
            try:
                sig = await signal_engine.process_candle(c)
                if sig is not None:
                    global last_signal
                    last_signal = sig
                    regime_info["current"] = sig.regime
                    regime_info["atr"]     = sig.atr
            except Exception:
                pass
        # ── CRITICAL: reset signal engine state so replay doesn't block live signals ──
        # process_candle() sets _suppression_counter=5 and increments _open_position_count
        # for replay signals that were never actually executed. Clear both.
        signal_engine._suppression_counter = 0
        signal_engine._open_position_count = 0
        signal_engine._pending_signals.clear()
        signal_engine._signals_generated   = 0   # reset so dashboard counter is clean
        # Sync regime/ATR from the engine's internal state
        if hasattr(signal_engine, "regime_detector"):
            regime_info["current"] = signal_engine.regime_detector.current_regime.value
        try:
            buf = signal_engine.live_store.get_buffer_as_df()
            if not buf.empty:
                feats = signal_engine.live_store._feature_store.build_features(buf)
                if not feats.empty:
                    regime_info["atr"]  = float(feats["ATR"].iloc[-1])
                    regime_info["adx"]  = float(feats["ADX"].iloc[-1])
                    regime_info["chop"] = float(feats["CHOP"].iloc[-1])
        except Exception:
            pass
        print(f"  Replay complete — Regime: {regime_info['current']} | ATR: ${regime_info['atr']:,.0f} | Engine state cleared for live trading")

    await asyncio.gather(
        binance_ws(),
        ui_loop(),
        _candle_worker(),
        _market_intel_loop(),   # APEX v7.2: news + funding + sentiment
    )

if __name__ == "__main__":
    asyncio.run(main())
