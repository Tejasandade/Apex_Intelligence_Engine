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
from src.data.providers.cross_market import CrossMarketMonitor
from src.core.state_manager import StateManager
from src.core.prop_firm_manager import PropFirmManager
from src.models.training.retrain_scheduler import RetrainScheduler

# ─── ANSI ────────────────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
RED="\033[91m"; GRN="\033[92m"; YLW="\033[93m"
BLU="\033[94m"; MAG="\033[95m"; CYN="\033[96m"
WHT="\033[97m"; GRY="\033[90m"

# ─── MULTI-ASSET CONFIG ────────────────────────────────────────────────────────
DEFAULT_SYMBOLS   = ["BTCUSDT", "ETHUSDT", "PAXGUSDT"]
env_syms          = os.getenv("APEX_SYMBOLS", "")
if env_syms:
    TRADED_SYMBOLS = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
else:
    single_sym = os.getenv("APEX_SYMBOL", "")
    TRADED_SYMBOLS = [single_sym.upper()] if single_sym else DEFAULT_SYMBOLS

PRIMARY_SYMBOL  = TRADED_SYMBOLS[0]
SYMBOL          = PRIMARY_SYMBOL
INITIAL_CAPITAL = float(os.getenv("APEX_INITIAL_CAPITAL", "1000.0"))
SYMBOLS         = TRADED_SYMBOLS
MODEL_DIR       = Path("data/models")

# ─── PROP FIRM MODE (Pluggable Challenge Module) ──────────────────────────────
# Enabled when APEX_PROP_FIRM_MODE=true. If false, runs in standard personal account mode.
PROP_FIRM_MODE    = os.getenv("APEX_PROP_FIRM_MODE", "true").lower() == "true"
PROP_FIRM_PROFILE = os.getenv("APEX_PROP_FIRM_PROFILE", "sharkfunded_5k_lite_1step")
prop_firm_manager = PropFirmManager(profile_key=PROP_FIRM_PROFILE) if PROP_FIRM_MODE else None

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
BROKER_TYPE  = os.getenv("APEX_BROKER", "bybit").lower()  # "bybit" or "mt5"
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

def send_challenge_alert(event_type: str, details: dict):
    """
    Send rich mobile notification with Prop Firm Challenge HUD to Telegram and Web UI.
    """
    if PROP_FIRM_MODE and prop_firm_manager:
        pf = prop_firm_manager.get_hud_metrics()
        cur_prof = pf.get("current_profit_usd", 0.0)
        tgt_usd = pf.get("target_usd", 450.0)
        prog_pct = pf.get("progress_pct", 0.0)
        v_bal = pf.get("virtual_balance", 5000.0)
        d_pnl = pf.get("daily_pnl_usd", 0.0)
        d_limit = pf.get("daily_limit_usd", 150.0)

        icon = "🔥" if event_type == "OPEN" else ("🎯" if "TP" in event_type else ("✅" if "CLOSED" in event_type and details.get("pnl_usd", 0) > 0 else ("❌" if "CLOSED" in event_type else "🛡️")))
        txt = (
            f"{icon} APEX: {event_type}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Asset: {details.get('symbol', SYMBOL)} | Action: {details.get('action', '')} @ ${details.get('price', 0.0):,.1f}\n"
        )
        if details.get("lots"):
            txt += f"Size: {details.get('lots', 0.0)} lots | Risk: ${details.get('risk_usd', 0.0):.2f}\n"
        if details.get("sl"):
            txt += f"SL: ${details.get('sl', 0.0):,.1f} | TP2: ${details.get('tp2', 0.0):,.1f}\n"
        if details.get("pnl_usd") is not None:
            pnl_s = f"+${details['pnl_usd']:.2f}" if details['pnl_usd'] >= 0 else f"-${abs(details['pnl_usd']):.2f}"
            txt += f"Result: {pnl_s} ({details.get('reason', '')})\n"
        if details.get("new_sl"):
            txt += f"Updated SL: ${details.get('new_sl', 0.0):,.1f}\n"

        txt += (
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🏆 SHARKFUNDED 5K CHALLENGE\n"
            f"• Virtual Equity: ${v_bal:,.2f}\n"
            f"• Target: {cur_prof:+.2f} / +${tgt_usd:.0f} ({prog_pct:.1f}%)\n"
            f"• Daily PnL: {d_pnl:+.2f} / -${d_limit:.0f}\n"
            f"• Status: {pf.get('status', 'IN_PROGRESS')}"
        )
    else:
        txt = f"APEX {event_type}: {details.get('action', '')} @ ${details.get('price', 0.0):,.1f} | PnL: ${details.get('pnl_usd', 0.0):+.2f}"

    telegram(txt)

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
MIN_INTERVAL    = 300       # 5min (1 candle) between trades (was 30m)
LOSS_COOLDOWN_1 = 600       # 10min (2 candles) after 1 loss (was 2hr!)
LOSS_COOLDOWN_2 = 1200      # 20min (4 candles) after 2 consecutive losses
LOSS_COOLDOWN_3 = 3600      # 60min anti-spiral after 3 consecutive losses
SESSION_LOSSES  = 3         # halt after N consecutive losses

# ─── BROKER (Bybit Testnet — only active when DEMO_MODE=true) ─────────────────
_bybit_broker = None   # instantiated in prewarm() when DEMO_MODE=true
_active_broker = None  # unified broker (BybitBroker or MT5Broker)

# ─── MULTI-ASSET STATE CONTAINERS ─────────────────────────────────────────────
def _create_symbol_portfolio(sym: str) -> dict:
    return {
        "symbol": sym, "position": "FLAT", "direction": None,
        "trade_mode": "SCALP", # SCALP (precision 5m) | SWING (macro 1D/4H/1H aligned)
        "entry_price": 0.0, "entry_time": 0.0,
        "sl_price": 0.0, "sl_distance": 0.0,
        "tp1_price": 0.0, "tp2_price": 0.0,
        "tp1_rr": 1.8, "tp2_rr": 3.5,
        "max_duration_min": 360.0, "adverse_decay_min": 240.0,
        "risk_tier": 0, "entry_lots": 0.0, "risk_usd": 0.0,
        "notional": 0.0, "initial_notional": 0.0,
        "unrealized_pnl": 0.0, "banked_pnl": 0.0,
        "phase": 0, "signal_id": "", "prob": 0.0,
        "council_score": 0.0, "regime_entry": "",
        "source": "", "highest_fav": 0.0,
        "broker_ticket": "", "broker_pnl": 0.0,
    }

portfolios: dict[str, dict] = {sym: _create_symbol_portfolio(sym) for sym in TRADED_SYMBOLS}
portfolio = portfolios[PRIMARY_SYMBOL]

def _create_symbol_market(sym: str) -> dict:
    return {"symbol": sym, "ltp": 0.0, "bid": 0.0, "ask": 0.0, "spread_bps": 0.0, "obi": 0.0, "ws_connected": False, "ticks": 0}

symbol_markets: dict[str, dict] = {sym: _create_symbol_market(sym) for sym in TRADED_SYMBOLS}
market = symbol_markets[PRIMARY_SYMBOL]

# Per-symbol candle buffers (5m execution, 15m structural, 1h trend, 4h intermediate, 1d macro)
symbol_candles_5m: dict[str, deque]  = {sym: deque(maxlen=210) for sym in TRADED_SYMBOLS}
symbol_candles_15m: dict[str, deque] = {sym: deque(maxlen=100) for sym in TRADED_SYMBOLS}
symbol_candles_1h: dict[str, deque]  = {sym: deque(maxlen=168) for sym in TRADED_SYMBOLS}
symbol_candles_4h: dict[str, deque]  = {sym: deque(maxlen=60) for sym in TRADED_SYMBOLS}
symbol_candles_1d: dict[str, deque]  = {sym: deque(maxlen=60) for sym in TRADED_SYMBOLS}

symbol_live_candle: dict[str, dict]  = {
    sym: {"open": 0.0, "high": 0.0, "low": 0.0, "close": 0.0, "volume": 0.0, "timestamp": 0}
    for sym in TRADED_SYMBOLS
}
symbol_cur_min_ts: dict[str, int]    = {sym: 0 for sym in TRADED_SYMBOLS}
symbol_acc_1m: dict[str, list]       = {sym: [] for sym in TRADED_SYMBOLS}
symbol_acc_5m_15m: dict[str, list]   = {sym: [] for sym in TRADED_SYMBOLS}
symbol_acc_15m_1h: dict[str, list]   = {sym: [] for sym in TRADED_SYMBOLS}
symbol_acc_1h_4h: dict[str, list]    = {sym: [] for sym in TRADED_SYMBOLS}
symbol_acc_4h_1d: dict[str, list]    = {sym: [] for sym in TRADED_SYMBOLS}

# Multi-timeframe Macro Structure Store (1D, 4H, 1H)
symbol_macro_structure: dict[str, dict] = {
    sym: {
        "symbol": sym,
        "trend_1d": "NEUTRAL",
        "ema20_1d": 0.0, "ema50_1d": 0.0, "rsi_1d": 50.0,
        "trend_4h": "NEUTRAL",
        "ema20_4h": 0.0, "ema50_4h": 0.0, "structure_4h": "RANGE",
        "trend_1h": "NEUTRAL",
        "ema20_1h": 0.0, "ema50_1h": 0.0,
        "macro_bias": "NEUTRAL",
        "score": 0,
        "swing_eligible_long": False,
        "swing_eligible_short": False,
    }
    for sym in TRADED_SYMBOLS
}

def _calc_ema(prices: list[float], period: int) -> float:
    if not prices or len(prices) < period:
        return prices[-1] if prices else 0.0
    k = 2.0 / (period + 1.0)
    ema = float(prices[0])
    for p in prices[1:]:
        ema = float(p) * k + ema * (1.0 - k)
    return ema

def _calc_rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) <= period:
        return 50.0
    diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0.0 for d in diffs]
    losses = [-d if d < 0 else 0.0 for d in diffs]
    if len(gains) < period:
        return 50.0
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(diffs)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def compute_macro_structure(symbol: str) -> dict:
    """
    Computes higher-timeframe market structure (1D, 4H, 1H) for a given symbol.
    Extracts EMA alignments, RSI, and structural swing pivots.
    Determines overall Macro Bias (STRONG_BULL, MOD_BULL, NEUTRAL, MOD_BEAR, STRONG_BEAR)
    and evaluates Swing Trade eligibility.
    """
    sym = symbol.upper()
    c_1d = list(symbol_candles_1d.get(sym, []))
    c_4h = list(symbol_candles_4h.get(sym, []))
    c_1h = list(symbol_candles_1h.get(sym, []))

    res = {
        "symbol": sym,
        "trend_1d": "NEUTRAL",
        "ema20_1d": 0.0, "ema50_1d": 0.0, "rsi_1d": 50.0,
        "trend_4h": "NEUTRAL",
        "ema20_4h": 0.0, "ema50_4h": 0.0, "structure_4h": "RANGE",
        "trend_1h": "NEUTRAL",
        "ema20_1h": 0.0, "ema50_1h": 0.0,
        "macro_bias": "NEUTRAL",
        "score": 0,
        "swing_eligible_long": False,
        "swing_eligible_short": False,
    }

    score = 0

    # 1. Daily (1D) Analysis
    if len(c_1d) >= 15:
        closes_1d = [float(c["close"]) for c in c_1d]
        cur_1d = closes_1d[-1]
        ema20_1d = _calc_ema(closes_1d, 20)
        ema50_1d = _calc_ema(closes_1d, min(50, len(closes_1d)))
        rsi_1d = _calc_rsi(closes_1d, 14)

        res["ema20_1d"] = round(ema20_1d, 2)
        res["ema50_1d"] = round(ema50_1d, 2)
        res["rsi_1d"] = round(rsi_1d, 1)

        if cur_1d > ema50_1d and ema20_1d >= ema50_1d:
            res["trend_1d"] = "BULLISH"
            score += 2
        elif cur_1d < ema50_1d and ema20_1d <= ema50_1d:
            res["trend_1d"] = "BEARISH"
            score -= 2
        elif cur_1d > ema20_1d:
            res["trend_1d"] = "MILD_BULL"
            score += 1
        elif cur_1d < ema20_1d:
            res["trend_1d"] = "MILD_BEAR"
            score -= 1

    # 2. 4-Hour (4H) Intermediate Structure
    if len(c_4h) >= 12:
        closes_4h = [float(c["close"]) for c in c_4h]
        cur_4h = closes_4h[-1]
        ema20_4h = _calc_ema(closes_4h, 20)
        ema50_4h = _calc_ema(closes_4h, min(50, len(closes_4h)))

        res["ema20_4h"] = round(ema20_4h, 2)
        res["ema50_4h"] = round(ema50_4h, 2)

        highs = [float(c["high"]) for c in c_4h[-12:]]
        lows = [float(c["low"]) for c in c_4h[-12:]]
        recent_h = max(highs[-6:])
        prev_h = max(highs[:6])
        recent_l = min(lows[-6:])
        prev_l = min(lows[:6])

        if recent_h > prev_h and recent_l > prev_l:
            res["structure_4h"] = "BULL_EXPANSION"
            score += 1
        elif recent_h < prev_h and recent_l < prev_l:
            res["structure_4h"] = "BEAR_EXPANSION"
            score -= 1

        if cur_4h > ema20_4h > ema50_4h:
            res["trend_4h"] = "BULLISH"
            score += 1
        elif cur_4h < ema20_4h < ema50_4h:
            res["trend_4h"] = "BEARISH"
            score -= 1

    # 3. 1-Hour (1H) Setup Analysis
    if len(c_1h) >= 12:
        closes_1h = [float(c["close"]) for c in c_1h]
        cur_1h = closes_1h[-1]
        ema20_1h = _calc_ema(closes_1h, 20)
        ema50_1h = _calc_ema(closes_1h, min(50, len(closes_1h)))

        res["ema20_1h"] = round(ema20_1h, 2)
        res["ema50_1h"] = round(ema50_1h, 2)

        if cur_1h > ema20_1h:
            res["trend_1h"] = "BULLISH"
            score += 1
        elif cur_1h < ema20_1h:
            res["trend_1h"] = "BEARISH"
            score -= 1

    res["score"] = score
    if score >= 3:
        res["macro_bias"] = "STRONG_BULL"
        res["swing_eligible_long"] = True
    elif score in (1, 2):
        res["macro_bias"] = "MOD_BULL"
        res["swing_eligible_long"] = (res["trend_1d"] in ("BULLISH", "MILD_BULL") and res["trend_4h"] == "BULLISH")
    elif score in (-1, -2):
        res["macro_bias"] = "MOD_BEAR"
        res["swing_eligible_short"] = (res["trend_1d"] in ("BEARISH", "MILD_BEAR") and res["trend_4h"] == "BEARISH")
    elif score <= -3:
        res["macro_bias"] = "STRONG_BEAR"
        res["swing_eligible_short"] = True
    else:
        res["macro_bias"] = "NEUTRAL"

    symbol_macro_structure[sym] = res
    return res

# Backward compatibility pointers
candles_5m  = symbol_candles_5m[PRIMARY_SYMBOL]
candles_15m = symbol_candles_15m[PRIMARY_SYMBOL]
candles_1h  = symbol_candles_1h[PRIMARY_SYMBOL]
candles_4h  = symbol_candles_4h[PRIMARY_SYMBOL]
candles_1d  = symbol_candles_1d[PRIMARY_SYMBOL]
live_candle = symbol_live_candle[PRIMARY_SYMBOL]
cur_min_ts  = 0
acc_1m      = symbol_acc_1m[PRIMARY_SYMBOL]
acc_5m_15m  = symbol_acc_5m_15m[PRIMARY_SYMBOL]
acc_15m_1h  = symbol_acc_15m_1h[PRIMARY_SYMBOL]
acc_1h_4h   = symbol_acc_1h_4h[PRIMARY_SYMBOL]
acc_4h_1d   = symbol_acc_4h_1d[PRIMARY_SYMBOL]

# Per-symbol signal engines and momentum engines
signal_engines: dict[str, LiveSignalEngine] = {}
momentum_engines: dict[str, TickMomentumEngine] = {
    sym: TickMomentumEngine(symbol=sym, ofi_window=300, velocity_window=60, breakout_window=600,
                            ofi_threshold=0.28, velocity_threshold=0.00008, recompute_interval=15)
    for sym in TRADED_SYMBOLS
}
momentum_eng = momentum_engines[PRIMARY_SYMBOL]
signal_engine: LiveSignalEngine | None = None
_loaded_model_count: int = 24

symbol_live_probs: dict[str, dict] = {
    sym: {"long": 0.5, "short": 0.5, "updated": False}
    for sym in TRADED_SYMBOLS
}
live_probs = symbol_live_probs[PRIMARY_SYMBOL]
symbol_regime_info: dict[str, dict] = {
    sym: {"current": "UNKNOWN", "adx": 0.0, "chop": 0.0, "atr": 80.0 if "BTC" in sym else (10.0 if "ETH" in sym else 5.0)}
    for sym in TRADED_SYMBOLS
}
regime_info = symbol_regime_info[PRIMARY_SYMBOL]

# ─── MARKET INTELLIGENCE (APEX v7.2) ─────────────────────────────────────────
news_guard           = NewsGuard()
funding_monitor      = FundingRateMonitor()
sentiment_monitor    = SentimentMonitor()
cross_market_monitor = CrossMarketMonitor()
state_manager        = StateManager()
retrain_scheduler: RetrainScheduler | None = None   # set in prewarm()

# Async queue for safely passing 5m candles from sync tick handler → async ML loop
# Instantiated inside main() after the event loop is running
_candle_queue: asyncio.Queue | None = None

# CVD
cvd_bucket = 0.0

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

def is_rollover_or_maintenance() -> tuple[bool, str]:
    """
    Detects daily broker server rollover spikes and maintenance windows.
    Protects against artificial spread widening (e.g. 23:55-00:10 UTC on XM).
    Configurable via APEX_WEEKEND_SHIELD (default true).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # 1. Daily Broker Rollover window: 23:55 UTC to 00:10 UTC
    if (now_utc.hour == 23 and now_utc.minute >= 55) or (now_utc.hour == 0 and now_utc.minute <= 10):
        return True, "XM DAILY ROLLOVER WINDOW (23:55-00:10 UTC) — SPREAD SHIELD"
    # 2. Weekend Maintenance window on XM: Saturday 06:00 to 12:00 UTC
    weekend_shield_enabled = os.getenv("APEX_WEEKEND_SHIELD", "true").lower() == "true"
    if weekend_shield_enabled and now_utc.weekday() == 5 and (6 <= now_utc.hour < 12):
        return True, "WEEKEND SERVER MAINTENANCE WINDOW (06:00-12:00 UTC) — TRADING PAUSED"
    # 3. Abnormal Broker Spread (> 12.0 bps for BTC)
    current_spread = market.get("spread_bps", 0.0)
    if current_spread > 12.0:
        return True, f"ABNORMAL SPREAD ({current_spread:.1f} bps > 12.0 bps)"
    return False, "CLEAR"

def can_trade_symbol(symbol: str = SYMBOL, direction: str = "LONG") -> tuple[bool, str]:
    now = time.time()
    sym = symbol.upper()
    dir_upper = direction.upper()

    # Weekend & Daily Rollover Spread Shield (XM MT5 Protection)
    shielded, shield_reason = is_rollover_or_maintenance()
    if shielded:
        return False, f"SPREAD SHIELD: {shield_reason}"

    # ── Modular Prop Firm Gate (SharkFunded, FTMO, FundedNext, etc.) ──
    if PROP_FIRM_MODE and prop_firm_manager:
        pf_ok, pf_reason = prop_firm_manager.can_open_multi_asset_trade(
            symbol=sym,
            direction=dir_upper,
            active_positions=portfolios,
            estimated_risk_usd=25.0,
            max_concurrent=2,
        )
        if not pf_ok:
            return False, f"PROP_FIRM: {pf_reason}"
    else:
        # ── Standard Account Multi-Asset Risk Gate (Personal Account / Demo) ──
        # 1. Per-symbol isolation
        if portfolios.get(sym, {}).get("position") == "OPEN":
            return False, f"{sym} already has an OPEN position"

        # 2. Maximum concurrent trades across portfolio
        max_c = int(_rc("sizing", "max_concurrent_trades", 3))
        active_count = sum(1 for p in portfolios.values() if p.get("position") == "OPEN")
        if active_count >= max_c:
            return False, f"Max concurrent trades reached ({active_count}/{max_c})"

        # 3. Crypto Correlation Shield
        if "ETH" in sym:
            btc_pos = portfolios.get("BTCUSDT", {})
            if btc_pos.get("position") == "OPEN" and btc_pos.get("direction") == dir_upper:
                return False, f"Correlation Shield: BTC already {dir_upper}"
        elif "BTC" in sym:
            eth_pos = portfolios.get("ETHUSDT", {})
            if eth_pos.get("position") == "OPEN" and eth_pos.get("direction") == dir_upper:
                return False, f"Correlation Shield: ETH already {dir_upper}"

    if account["halted"]:
        if account["halt_until"] <= now:
            account["halted"] = False; account["halt_reason"] = ""
        else:
            rem = int(account["halt_until"] - now) if account["halt_until"] < float("inf") else 999999
            return False, f"HALTED: {account['halt_reason']} ({rem//60}m left)"
    if account["trades_today"] >= MAX_TRADES_DAY:
        return False, f"DAY LIMIT ({MAX_TRADES_DAY} done)"
    # Dynamic, non-rigid loss cooldown based on consecutive losses
    consec = account.get("consecutive_losses", 0)
    cooldown_secs = LOSS_COOLDOWN_1 if consec <= 1 else (LOSS_COOLDOWN_2 if consec == 2 else LOSS_COOLDOWN_3)
    if account["last_loss_time"] > 0 and (now - account["last_loss_time"]) < cooldown_secs:
        rem = int(cooldown_secs - (now - account["last_loss_time"]))
        return False, f"LOSS COOLDOWN {rem//60}m{rem%60:02d}s"

    mkt = symbol_markets.get(sym, market)
    if mkt.get("spread_bps", 0.0) > 12.0:
        return False, f"SPREAD {mkt['spread_bps']:.1f}bps"
    # APEX v7.2: News blackout gate
    blackout, blackout_reason = news_guard.is_blackout(sym)
    if blackout:
        return False, f"NEWS BLACKOUT: {blackout_reason}"

    # ── Macro Market Structure Filter (1D & 4H Trend Alignment) ──
    macro = symbol_macro_structure.get(sym)
    if macro:
        if dir_upper == "LONG" and macro.get("macro_bias") == "STRONG_BEAR":
            return False, f"MACRO SHIELD: {sym} Strong Bear Macro (1D/4H aligned down)"
        elif dir_upper == "SHORT" and macro.get("macro_bias") == "STRONG_BULL":
            return False, f"MACRO SHIELD: {sym} Strong Bull Macro (1D/4H aligned up)"

    return True, "CLEAR"

def can_trade() -> tuple[bool, str]:
    return can_trade_symbol(PRIMARY_SYMBOL, "LONG")

# ─── STRUCTURAL SL (DUAL HORIZON: SCALP vs SWING) ────────────────────────────
def find_sl(direction: str, entry: float, symbol: str = SYMBOL, trade_mode: str = "SCALP") -> tuple[float, float]:
    sym = symbol.upper()
    sym_reg = symbol_regime_info.get(sym, regime_info)
    atr = max(sym_reg.get("atr", 10.0), 1.0)
    is_swing = (trade_mode == "SWING")

    # Asset & Horizon Adaptive ATR bounds
    if "ETH" in sym:
        if is_swing:
            min_sl = max(22.0, entry * 0.010)
            max_sl = min(55.0, entry * 0.025)
        else:
            min_sl = max(8.0, entry * 0.0035)
            max_sl = min(25.0, entry * 0.010)
    elif "GOLD" in sym or "XAU" in sym or "PAXG" in sym:
        if is_swing:
            min_sl = max(8.0, entry * 0.0025)
            max_sl = min(22.0, entry * 0.006)
        else:
            min_sl = max(4.0, entry * 0.0015)
            max_sl = min(12.0, entry * 0.004)
    else:  # BTC
        if is_swing:
            min_sl = max(380.0, atr * 4.0)
            max_sl = min(850.0, atr * 8.0)
        else:
            min_sl = max(160.0, atr * 1.5)
            max_sl = min(400.0, atr * 3.5)

    candidates = []

    if is_swing:
        # Swing stops anchor to 1H and 4H structural pivots
        c_1h = list(symbol_candles_1h.get(sym, []))[-15:]
        c_4h = list(symbol_candles_4h.get(sym, []))[-8:]
        if c_1h:
            if direction == "LONG":
                candidates.append(min(float(c["low"]) for c in c_1h))
            else:
                candidates.append(max(float(c["high"]) for c in c_1h))
        if c_4h:
            if direction == "LONG":
                candidates.append(min(float(c["low"]) for c in c_4h))
            else:
                candidates.append(max(float(c["high"]) for c in c_4h))
    else:
        # Scalp stops anchor to 5m and 15m micro pivots
        c_15m = list(symbol_candles_15m.get(sym, candles_15m))[-25:]
        if len(c_15m) >= 5:
            if direction == "LONG":
                candidates.append(min(float(c["low"]) for c in c_15m[-12:]))
            else:
                candidates.append(max(float(c["high"]) for c in c_15m[-12:]))

        c_5m = list(symbol_candles_5m.get(sym, candles_5m))[-40:]
        if len(c_5m) >= 10:
            if direction == "LONG":
                candidates.append(min(float(c["low"]) for c in c_5m[-20:]))
                candidates.append(min(float(c["low"]) for c in c_5m[-10:]))
            else:
                candidates.append(max(float(c["high"]) for c in c_5m[-20:]))
                candidates.append(max(float(c["high"]) for c in c_5m[-10:]))

    if direction == "LONG":
        valid = [c for c in candidates if c < entry and (entry - c) >= min_sl]
        sl_price = max(valid) - (entry * 0.0005) if valid else entry - min_sl
        sl_dist = entry - sl_price
    else:
        valid = [c for c in candidates if c > entry and (c - entry) >= min_sl]
        sl_price = min(valid) + (entry * 0.0005) if valid else entry + min_sl
        sl_dist = sl_price - entry

    sl_dist = float(np.clip(sl_dist, min_sl, max_sl))
    sl_price = entry - sl_dist if direction == "LONG" else entry + sl_dist
    return round(sl_price, 2), round(sl_dist, 2)

# ─── SIZING ───────────────────────────────────────────────────────────────────
def size_position(sl_dist: float, prob: float, entry: float, council_score: float = 0.7, direction: str = "LONG", symbol: str = SYMBOL) -> tuple[float, float]:
    sym = symbol.upper()
    # ── PROP FIRM MODE: Size directly to Challenge Target & Drawdown Limits ──
    if PROP_FIRM_MODE and prop_firm_manager:
        consec = account.get("consecutive_losses", 0)
        lots, risk_usd = prop_firm_manager.calculate_lot_size(
            sl_distance=sl_dist,
            price=entry,
            conviction=prob,
            council_score=council_score,
            consecutive_losses=consec,
            symbol=sym,
        )
        notional = round(lots * entry, 2)
        return notional, risk_usd

    # ── STANDARD BALANCE SIZING (Personal Account / Demo) ──
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

    # APEX v7.2: Sentiment modifier
    sentiment_mult = sentiment_monitor.get_regime_modifier()
    risk_pct *= sentiment_mult

    risk_usd = account["balance"] * risk_pct
    contract_size = 100.0 if ("GOLD" in sym or "XAU" in sym or "PAXG" in sym) else 1.0
    raw_lots = risk_usd / max(sl_dist * contract_size, 0.0001)
    min_vol = 0.02 if "ETH" in sym else 0.01
    lots = max(min_vol, round(raw_lots, 2))
    notional = round(lots * entry, 2)
    actual_risk = round(lots * sl_dist * contract_size, 2)
    return notional, actual_risk

# ─── ENTRY EXECUTION ──────────────────────────────────────────────────────────
def execute_signal(signal: TradingSignal, symbol: str = SYMBOL):
    """Execute a TradingSignal from the LiveSignalEngine for a specific symbol."""
    global last_signal
    sym = symbol.upper()
    port = portfolios.get(sym, portfolio)
    mkt = symbol_markets.get(sym, market)

    if port["position"] != "FLAT": return

    direction = "LONG" if signal.direction == "BUY" else "SHORT"
    ok, reason = can_trade_symbol(sym, direction)
    if not ok: return

    prob      = signal.conviction
    council   = signal.council_score

    entry = mkt["ask"] if direction == "LONG" else mkt["bid"]
    if entry <= 0: entry = mkt.get("ltp", 0.0)
    if entry <= 0: return

    # ── DUAL-HORIZON TRADE CLASSIFICATION: SWING vs PRECISION SCALP ──
    macro = compute_macro_structure(sym)
    is_ranging = "ranging" in str(signal.regime).lower()

    # Swing criteria: Trending regime + High Conviction + Macro 1D/4H Alignment
    is_swing = False
    if not is_ranging and prob >= 0.62 and council >= 0.70:
        if direction == "LONG" and macro.get("swing_eligible_long", False):
            is_swing = True
        elif direction == "SHORT" and macro.get("swing_eligible_short", False):
            is_swing = True

    trade_mode = "SWING" if is_swing else "SCALP"

    sl_price, sl_dist = find_sl(direction, entry, symbol=sym, trade_mode=trade_mode)
    if sl_dist <= 0: return

    # ── VOLATILITY & HORIZON ADAPTIVE TARGETS (Regime-Adaptive R:R) ──
    if trade_mode == "SWING":
        tp1_rr = 2.5
        tp2_rr = 5.0 if council >= 0.80 else 4.5
        max_duration_min = 2880.0  # 48 hours for swing trades
        adverse_decay_min = 720.0  # 12 hours adverse hold
    else:
        # Precision SCALP
        max_duration_min = 360.0   # 6 hours
        adverse_decay_min = 240.0  # 4 hours
        if is_ranging:
            tp1_rr = 1.5
            tp2_rr = 2.2 if "low_vol" in str(signal.regime).lower() else 2.4
        else:
            tp1_rr = 1.8 if council < 0.75 else 2.0
            tp2_rr = 3.5 if council >= 0.85 else (3.2 if council >= 0.70 else MIN_RR)

    tp1 = round(entry + sl_dist * tp1_rr if direction == "LONG" else entry - sl_dist * tp1_rr, 2)
    tp2 = round(entry + sl_dist * tp2_rr if direction == "LONG" else entry - sl_dist * tp2_rr, 2)

    # APEX v7.2: Snap TP2 to nearest Volume Profile HVN / VPOC level
    try:
        from src.features.indicators.volume_profile import compute_volume_profile_snapshot
        c_buf = symbol_candles_5m.get(sym, candles_5m)
        vp = compute_volume_profile_snapshot(pd.DataFrame(list(c_buf)[-100:]))
        if vp["profile_ready"]:
            candidates = [vp["poc"], vp["vah"]] + vp["hvn_prices"]
            if direction == "LONG":
                valid_tps = [p for p in candidates if p > entry + sl_dist * 2.0]
                if valid_tps:
                    nearest = min(valid_tps)
                    if (nearest - entry) / sl_dist >= MIN_RR:
                        tp2 = round(nearest, 2)
            else:
                valid_tps = [p for p in candidates if p < entry - sl_dist * 2.0]
                if valid_tps:
                    nearest = max(valid_tps)
                    if (entry - nearest) / sl_dist >= MIN_RR:
                        tp2 = round(nearest, 2)
    except Exception:
        pass

    notional, risk_usd = size_position(sl_dist, prob, entry, council, direction, symbol=sym)
    if notional <= 0: return

    contract_size = 100.0 if ("GOLD" in sym or "XAU" in sym or "PAXG" in sym) else 1.0
    entry_lots = round(risk_usd / max(sl_dist * contract_size, 0.0001), 2)
    min_vol = 0.02 if "ETH" in sym else 0.01
    entry_lots = max(min_vol, entry_lots)

    now_t   = time.time()
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    action  = "BUY" if direction == "LONG" else "SELL"

    port.update({
        "symbol": sym,
        "position": "OPEN", "direction": direction,
        "trade_mode": trade_mode,
        "entry_price": entry, "entry_time": now_t,
        "sl_price": sl_price, "sl_distance": sl_dist,
        "tp1_price": tp1, "tp2_price": tp2,
        "tp1_rr": tp1_rr, "tp2_rr": tp2_rr,
        "max_duration_min": max_duration_min,
        "adverse_decay_min": adverse_decay_min,
        "risk_tier": 0,
        "entry_lots": entry_lots, "risk_usd": risk_usd,
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
    mode_tag = f"[{trade_mode}]"
    log = (f"[{now_str}] OPEN {action} {sym} {mode_tag} @${entry:,.1f} | "
           f"SL:${sl_price:,.1f}(${sl_dist:.1f}) TP2:${tp2:,.1f}({tp2_rr:.1f}R) | "
           f"Prob:{prob:.3f} {council_str} Risk:${risk_usd:.2f} ({entry_lots:.2f} lots)")
    trade_log.append(log)
    if HEADLESS: print(f"\n🔥 TRADE EVENT: {log}", flush=True)
    send_challenge_alert("OPEN", {
        "symbol": sym,
        "action": action,
        "trade_mode": trade_mode,
        "price": entry,
        "sl": sl_price,
        "tp2": tp2,
        "lots": entry_lots,
        "risk_usd": risk_usd,
        "prob": prob,
        "council": council,
    })

    trade_journal.append({
        "id": len(trade_journal)+1,
        "open_ts": datetime.datetime.now().isoformat(),
        "symbol": sym,
        "action": action,
        "trade_mode": trade_mode,
        "entry_price": entry,
        "sl_price": sl_price, "tp1_price": tp1, "tp2_price": tp2,
        "sl_dist": sl_dist, "notional": notional, "risk_usd": risk_usd,
        "lots": entry_lots,
        "leverage": round(notional/account["balance"], 2),
        "prob": prob, "council_score": council,
        "regime": signal.regime, "signal_id": signal.signal_id,
        "council_votes": signal.council_votes,
        "outcome": "OPEN", "exit_price": 0.0, "pnl_usd": 0.0, "exit_reason": ""
    })
    save_journal()
    state_manager.save_state(portfolio, account)   # APEX v7.2: crash-resume

    # ── DEMO MODE: place real order on Broker (Bybit / MT5) ──────────────────────────
    if DEMO_MODE and _active_broker is not None:
        import asyncio as _aio
        async def _place():
            b_name = BROKER_TYPE.upper()
            try:
                qty = entry_lots
                side = "Buy" if direction == "LONG" else "Sell"
                oid = await _active_broker.place_order_with_stops(
                    symbol=sym,
                    side=side,
                    qty=qty,
                    stop_loss=round(sl_price, 2),
                    take_profit=round(tp2, 2),
                    sl_distance=sl_dist,
                    tp_distance=abs(tp2 - entry),
                )
                if oid:
                    port["broker_ticket"] = str(oid)
                    trade_log.append(f"[{b_name}] {sym} Order placed: {oid}")
                    telegram(f"{b_name} ORDER {side} {qty} {sym} | SL:{sl_price:.1f} TP:{tp2:.1f} | id:{oid}")
                    save_journal()
                    state_manager.save_state(portfolio, account)
                else:
                    trade_log.append(f"[{b_name}] {sym} Fill protection: Spread elevated, order held")
            except Exception as _be:
                trade_log.append(f"[{b_name}] {sym} Order error: {_be}")
        try:
            loop = _aio.get_event_loop()
            loop.create_task(_place())
        except Exception:
            pass


# ─── EXIT ENGINE ──────────────────────────────────────────────────────────────
def check_exits():
    for s in TRADED_SYMBOLS:
        _check_exit_for_symbol(s)
    if PROP_FIRM_MODE and prop_firm_manager:
        prop_firm_manager.update_multi_floating_pnl(portfolios)

def _check_exit_for_symbol(sym: str):
    port = portfolios.get(sym)
    if not port or port["position"] != "OPEN": return

    direction = port["direction"]
    entry     = port["entry_price"]
    now_t     = time.time()
    dur_min   = (now_t - port["entry_time"]) / 60.0

    mkt = symbol_markets.get(sym, market)
    cur = mkt["bid"] if direction == "LONG" else mkt["ask"]
    if cur <= 0: cur = mkt.get("ltp", 0.0)
    if cur <= 0: return

    # Unrealized PnL
    gbps = ((cur-entry)/entry if direction=="LONG" else (entry-cur)/entry) * 10000.0
    port["unrealized_pnl"] = (gbps - ROUNDTRIP_BPS) / 10000.0 * port["notional"]

    if direction == "LONG":
        port["highest_fav"] = max(port["highest_fav"], cur)
    elif port["highest_fav"] == 0 or cur < port["highest_fav"]:
        port["highest_fav"] = cur

    def close(price: float, reason: str):
        if direction == "LONG":
            nbps = ((price-entry)/entry * 10000.0) - ROUNDTRIP_BPS
        else:
            nbps = ((entry-price)/entry * 10000.0) - ROUNDTRIP_BPS

        net_usd = (nbps/10000.0) * port["notional"]
        total   = round(port["banked_pnl"] + net_usd, 2)
        is_win  = total > 0

        account["balance"] += net_usd
        update_health()

        sym_eng = signal_engines.get(sym, signal_engine)
        if is_win:
            account["wins"]             += 1
            account["consecutive_losses"] = 0
            account["total_wins_usd"]   += total
        else:
            account["losses"]           += 1
            account["consecutive_losses"] += 1
            account["total_losses_usd"] += abs(total)
            account["last_loss_time"]    = time.time()
            if sym_eng and port["signal_id"]:
                sym_eng.resolve_signal(port["signal_id"], False, total)
            # Anti-spiral
            if account["consecutive_losses"] >= SESSION_LOSSES:
                account["halted"]     = True
                account["halt_reason"] = f"{SESSION_LOSSES} consecutive losses"
                account["halt_until"] = time.time() + 4*3600
            elif account["consecutive_losses"] >= 2:
                account["halted"]     = True
                account["halt_reason"] = "2 consecutive losses — cooling"
                account["halt_until"] = time.time() + LOSS_COOLDOWN_2

        if is_win and sym_eng and port["signal_id"]:
            sym_eng.resolve_signal(port["signal_id"], True, total)

        # Update prop firm challenge progress
        if PROP_FIRM_MODE and prop_firm_manager:
            prop_firm_manager.on_trade_closed(total, reason)

        account["total"]   += 1
        account["win_rate"] = round(account["wins"]/account["total"]*100, 1) if account["total"] > 0 else 0
        account["avg_win"]  = round(account["total_wins_usd"]/account["wins"],   2) if account["wins"]   > 0 else 0
        account["avg_loss"] = round(account["total_losses_usd"]/account["losses"],2) if account["losses"] > 0 else 0
        account["profit_factor"] = round(account["total_wins_usd"]/account["total_losses_usd"],3) if account["total_losses_usd"] > 0 else 999.0
        account["best"]  = max(account["best"], total)
        account["worst"] = min(account["worst"], total)

        s = "+" if total >= 0 else ""
        trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                         f"CLOSE {sym} @${price:,.1f} PnL:{s}${total:.2f} | {reason}")
        send_challenge_alert("TRADE CLOSED", {
            "symbol": sym,
            "action": direction,
            "price": price,
            "pnl_usd": total,
            "reason": reason,
        })

        for t in reversed(trade_journal):
            if t.get("outcome") == "OPEN" and t.get("symbol") == sym:
                t.update({"exit_ts": datetime.datetime.now().isoformat(),
                          "exit_price": price, "pnl_usd": total,
                          "outcome": "WIN" if is_win else "LOSS",
                          "exit_reason": reason, "dur_min": round(dur_min,1)})
                break
        save_journal()
        state_manager.save_state(portfolio, account)   # APEX v7.2: crash-resume

        # ── Physical Broker Execution: Close on MT5/Broker ──
        if DEMO_MODE and _active_broker is not None:
            t_id = port.get("broker_ticket")
            if t_id:
                async def _exec_broker_close(ticket_id):
                    try:
                        if hasattr(_active_broker, "close_position_by_ticket"):
                            await _active_broker.close_position_by_ticket(int(ticket_id))
                        else:
                            close_side = "Sell" if direction == "LONG" else "Buy"
                            await _active_broker.place_market_order(sym, close_side, port.get("notional", 0) / max(price, 1), reduce_only=True)
                    except Exception as _ce:
                        trade_log.append(f"[{BROKER_TYPE.upper()}] Close err: {_ce}")
                try:
                    import asyncio as _aio
                    _loop = _aio.get_event_loop()
                    if _loop.is_running():
                        _loop.create_task(_exec_broker_close(t_id))
                except Exception:
                    pass

        port.update({"position": "FLAT", "direction": None,
                     "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0, "broker_ticket": "", "broker_pnl": 0.0})

    sl = port["sl_price"]
    tp1 = port["tp1_price"]
    tp2 = port["tp2_price"]
    sym_reg = symbol_regime_info.get(sym, regime_info)
    atr = max(sym_reg.get("atr", 10.0), 1.0)
    sl_dist = max(port.get("sl_distance", 1.0), 1.0)
    fav_dist = (cur - entry) if direction == "LONG" else (entry - cur)
    r_mult = fav_dist / sl_dist
    t_id = port.get("broker_ticket")
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    # 1. SL (Delegated to broker matching engine when active; otherwise managed in memory)
    if not (DEMO_MODE and port.get("broker_ticket")):
        if direction == "LONG" and cur <= sl: close(cur, "STOP LOSS"); return
        if direction == "SHORT" and cur >= sl: close(cur, "STOP LOSS"); return

    # ── DYNAMIC TIER 1: Risk Reduction at +1.0R (Tighten SL from -1.0R to -0.25R) ──
    if port.get("risk_tier", 0) < 1 and r_mult >= 1.0:
        new_sl = entry - sl_dist * 0.25 if direction == "LONG" else entry + sl_dist * 0.25
        port["sl_price"] = round(new_sl, 1)
        port["risk_tier"] = 1
        log_msg = f"[{now_str}] {sym} TIER 1 (+1.0R): Risk reduced 75% | SL tightened to -0.25R @${new_sl:,.1f}"
        trade_log.append(log_msg)
        if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
        send_challenge_alert("RISK REDUCED 75%", {
            "symbol": sym, "action": direction, "price": cur, "new_sl": port["sl_price"],
        })
        if DEMO_MODE and _active_broker is not None and t_id:
            async def _exec_modify_t1(ticket_id, sl_v, tp_v):
                try:
                    if hasattr(_active_broker, "modify_position_stops"):
                        await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_v, new_tp=tp_v)
                except Exception:
                    pass
            try:
                import asyncio as _aio
                _aio.get_event_loop().create_task(_exec_modify_t1(t_id, port["sl_price"], port["tp2_price"]))
            except Exception:
                pass

    # ── DYNAMIC TIER 2: Break-Even Lock at +1.5R ──
    if port.get("risk_tier", 0) < 2 and r_mult >= 1.5:
        fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
        new_sl = entry + fee_buf if direction == "LONG" else entry - fee_buf
        port["sl_price"] = round(new_sl, 1)
        port["risk_tier"] = 2
        log_msg = f"[{now_str}] {sym} TIER 2 (+1.5R): SL moved to Break-Even + fees @${new_sl:,.1f} | RISK FREE"
        trade_log.append(log_msg)
        if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
        send_challenge_alert("BREAK-EVEN LOCKED", {
            "symbol": sym, "action": direction, "price": cur, "new_sl": port["sl_price"],
        })
        if DEMO_MODE and _active_broker is not None and t_id:
            async def _exec_modify_t2(ticket_id, sl_v, tp_v):
                try:
                    if hasattr(_active_broker, "modify_position_stops"):
                        await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_v, new_tp=tp_v)
                except Exception:
                    pass
            try:
                import asyncio as _aio
                _aio.get_event_loop().create_task(_exec_modify_t2(t_id, port["sl_price"], port["tp2_price"]))
            except Exception:
                pass

    # ── DYNAMIC EARLY PROFIT HARVEST: Violent OFI Reversal Protection ──
    if r_mult >= 1.0:
        try:
            m_eng = momentum_engines.get(sym, momentum_eng)
            mom_d = m_eng.get_diagnostics()
            ofi_v = float(mom_d.get("ofi", 0.0))
            if direction == "LONG" and ofi_v <= -0.75:
                close(cur, f"OFI REVERSAL HARVEST (OFI {ofi_v:+.2f} at +{r_mult:.1f}R)")
                return
            elif direction == "SHORT" and ofi_v >= 0.75:
                close(cur, f"OFI REVERSAL HARVEST (OFI {ofi_v:+.2f} at +{r_mult:.1f}R)")
                return
        except Exception:
            pass

    # ── DYNAMIC TIER 3: Partial TP at TP1 (Horizon Adaptive: 40% Swing / 50% Scalp) ──
    is_swing = (port.get("trade_mode") == "SWING")
    if port["phase"] == 1:
        hit = (direction=="LONG" and cur>=tp1) or (direction=="SHORT" and cur<=tp1)
        if hit:
            part_frac = 0.40 if is_swing else 0.50
            partial = port["unrealized_pnl"] * part_frac
            port["banked_pnl"] += partial
            account["balance"]      += partial
            update_health()
            port["notional"]   *= (1.0 - part_frac)
            fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
            port["sl_price"] = entry + fee_buf if direction=="LONG" else entry - fee_buf
            port["phase"] = 2
            port["risk_tier"] = 3
            if PROP_FIRM_MODE and prop_firm_manager:
                prop_firm_manager.on_trade_closed(partial, "TP1 PARTIAL")

            log_msg = f"[{now_str}] {sym} [{port.get('trade_mode','SCALP')}] TP1 HIT (+{port.get('tp1_rr', 2.0):.1f}R) +${partial:.2f} | {int(part_frac*100)}% locked, SL→BE"
            trade_log.append(log_msg)
            if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
            send_challenge_alert("TP1 HIT (LOCKED)", {
                "symbol": sym, "action": direction, "price": cur, "pnl_usd": partial,
                "reason": f"+{port.get('tp1_rr', 2.0):.1f}R TP1 ({port.get('trade_mode','SCALP')})",
            })

            # ── Execute Physical Partial Close on Broker (MT5) ──
            if DEMO_MODE and _active_broker is not None and t_id:
                cur_vol = port.get("entry_lots", 0.0)
                part_vol = round(cur_vol * part_frac, 2)
                min_step = 0.02 if "ETH" in sym else 0.01
                part_vol = max(min_step, part_vol)
                new_be_sl = port["sl_price"]
                new_tp2 = port["tp2_price"]
                async def _exec_broker_partial(ticket_id, vol, sl_v, tp_v):
                    try:
                        if hasattr(_active_broker, "partial_close_position") and vol >= 0.01:
                            await _active_broker.partial_close_position(int(ticket_id), volume=vol)
                        if hasattr(_active_broker, "modify_position_stops"):
                            await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_v, new_tp=tp_v)
                    except Exception as _e:
                        trade_log.append(f"[{BROKER_TYPE.upper()}] {sym} TP1 sync err: {_e}")
                try:
                    import asyncio as _aio
                    _aio.get_event_loop().create_task(_exec_broker_partial(t_id, part_vol, new_be_sl, new_tp2))
                except Exception:
                    pass

    # ── DYNAMIC TIER 4: Dynamic Runner Chandelier Trail + Full TP2 + Stagnant Runner Harvest ──
    if port["phase"] == 2:
        if not is_swing and r_mult >= 1.0 and dur_min >= 75.0:
            try:
                m_eng = momentum_engines.get(sym, momentum_eng)
                mom_d = m_eng.get_diagnostics()
                ofi_val = abs(float(mom_d.get("ofi", 0.0)))
                if ofi_val < 0.20:
                    close(cur, f"STAGNANT RUNNER HARVEST (+{r_mult:.1f}R after {dur_min:.0f}m, OFI neutral)")
                    return
            except Exception:
                pass

        if is_swing:
            trail_buf = max(atr * 3.5, 25.0 if "ETH" in sym else (10.0 if "GOLD" in sym else 400.0))
        else:
            trail_buf = max(atr * 2.0, 10.0 if "ETH" in sym else (5.0 if "GOLD" in sym else 160.0))

        if direction == "LONG":
            new_sl = cur - trail_buf
            if new_sl > port["sl_price"]:
                old_sl = port["sl_price"]
                port["sl_price"] = round(new_sl, 1)
                if (new_sl - old_sl) >= (2.0 if "ETH" in sym else 25.0) and DEMO_MODE and _active_broker is not None and t_id:
                    async def _ratchet_sl(ticket_id, sl_v, tp_v):
                        try:
                            if hasattr(_active_broker, "modify_position_stops"):
                                await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_v, new_tp=tp_v)
                        except Exception:
                            pass
                    try:
                        import asyncio as _aio
                        _aio.get_event_loop().create_task(_ratchet_sl(t_id, port["sl_price"], port["tp2_price"]))
                    except Exception:
                        pass
            if cur >= tp2: close(cur, f"FULL TP2 ({port.get('tp2_rr', 3.5):.1f}R [{port.get('trade_mode','SCALP')}])"); return
        else:
            new_sl = cur + trail_buf
            if new_sl < port["sl_price"]:
                old_sl = port["sl_price"]
                port["sl_price"] = round(new_sl, 1)
                if (old_sl - new_sl) >= (2.0 if "ETH" in sym else 25.0) and DEMO_MODE and _active_broker is not None and t_id:
                    async def _ratchet_sl(ticket_id, sl_v, tp_v):
                        try:
                            if hasattr(_active_broker, "modify_position_stops"):
                                await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_v, new_tp=tp_v)
                        except Exception:
                            pass
                    try:
                        import asyncio as _aio
                        _aio.get_event_loop().create_task(_ratchet_sl(t_id, port["sl_price"], port["tp2_price"]))
                    except Exception:
                        pass
            if cur <= tp2: close(cur, f"FULL TP2 ({port.get('tp2_rr', 3.5):.1f}R [{port.get('trade_mode','SCALP')}])"); return

    # 4. Time decay (Horizon-adaptive)
    max_dur = port.get("max_duration_min", 360.0)
    adv_dur = port.get("adverse_decay_min", 240.0)
    if dur_min > max_dur:
        close(cur, f"MAX DURATION {max_dur/60:.0f}h ({port.get('trade_mode', 'SCALP')})")
        return
    if dur_min > adv_dur and port["phase"] == 1 and port["unrealized_pnl"] < 0:
        close(cur, f"TIME DECAY {adv_dur/60:.0f}h adverse ({port.get('trade_mode', 'SCALP')})")
        return

    # 5. Model-halt exit
    sym_eng = signal_engines.get(sym, signal_engine)
    if sym_eng and sym_eng.is_halted and port["unrealized_pnl"] > 0:
        close(cur, "ENGINE HALTED — locking profits"); return

# ─── CANDLE PIPELINE ──────────────────────────────────────────────────────────
async def _candle_worker():
    """Async worker: drains _candle_queue and processes each 5m candle through the symbol's ML engine."""
    global last_signal
    while True:
        item = await _candle_queue.get()
        if isinstance(item, tuple):
            sym, c5 = item
        else:
            sym, c5 = PRIMARY_SYMBOL, item

        eng = signal_engines.get(sym, signal_engine)
        if not eng:
            _candle_queue.task_done()
            continue
        try:
            sig = await eng.process_candle(c5)

            # APEX v7.2: Feed candle to retrain scheduler (non-blocking)
            if sym == PRIMARY_SYMBOL and retrain_scheduler:
                try:
                    retrain_scheduler.on_candle(c5)
                    retrain_scheduler.check_pending_swap()   # pick up new models if ready
                except Exception:
                    pass

            # Always extract live probabilities from the ensemble — even if no signal
            try:
                feats = eng.live_store.update(c5)  # already called inside; uses cached
                if feats is not None and eng.ensemble_long and eng.ensemble_short:
                    re = eng.regime_detector.current_regime
                    pl, _, _ = eng.ensemble_long.predict(feats, current_regime=re.value)
                    ps, _, _ = eng.ensemble_short.predict(feats, current_regime=re.value)
                    if sym in symbol_live_probs:
                        symbol_live_probs[sym]["long"]    = round(pl, 4)
                        symbol_live_probs[sym]["short"]   = round(ps, 4)
                        symbol_live_probs[sym]["updated"] = True
                        symbol_live_probs[sym]["regime"]  = re.value
                    if sym == PRIMARY_SYMBOL:
                        live_probs["long"]    = round(pl, 4)
                        live_probs["short"]   = round(ps, 4)
                        live_probs["updated"] = True
                        regime_info["current"] = re.value
            except Exception:
                pass

            if sig is not None:
                last_signal = sig
                if sym == PRIMARY_SYMBOL:
                    regime_info["current"] = sig.regime
                    regime_info["atr"]     = sig.atr

                compute_macro_structure(sym)
                direction = "LONG" if sig.direction == "BUY" else "SHORT"
                ok, reason = can_trade_symbol(sym, direction)
                sym_port = portfolios.get(sym, portfolio)
                if sym_port["position"] == "FLAT" and ok:
                    if sig.council_approved and sig.direction in ("BUY", "SELL"):
                        execute_signal(sig, symbol=sym)
                elif sym_port["position"] == "FLAT" and sig.council_approved and sig.direction in ("BUY", "SELL"):
                    # P3: Approved but gated — track for dashboard
                    last_gate_rejection["reason"] = f"[{sym}] {reason}"
                    last_gate_rejection["signal"] = f"[{sym}] {direction} p={sig.conviction:.3f}"
                    last_gate_rejection["ts"] = time.time()
        except Exception:
            pass
        finally:
            _candle_queue.task_done()


def process_tick_for_symbol(sym: str, price: float, qty: float, is_buyer_maker: bool):
    global cur_min_ts, cvd_bucket

    mkt = symbol_markets.get(sym, market)
    mkt["ltp"] = price
    mkt["ticks"] += 1
    if sym == PRIMARY_SYMBOL:
        market["ltp"] = price
        market["ticks"] += 1
    if price <= 0 or qty <= 0: return

    now = time.time()

    # Build 1m candle for fallback
    live_c = symbol_live_candle.get(sym)
    if live_c is not None:
        cur_min = symbol_cur_min_ts.get(sym, 0)
        if cur_min == 0:
            symbol_cur_min_ts[sym] = int(now // 60) * 60
            live_c.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0, "timestamp": symbol_cur_min_ts[sym] * 1000})
            if sym == PRIMARY_SYMBOL:
                cur_min_ts = symbol_cur_min_ts[sym]
                live_candle.update(live_c)

        if now >= cur_min + 60:
            acc1 = symbol_acc_1m.get(sym, [])
            acc1.append(dict(live_c))
            if len(acc1) >= 5:
                c5 = {"open": acc1[0]["open"], "high": max(x["high"] for x in acc1),
                      "low": min(x["low"] for x in acc1), "close": acc1[-1]["close"],
                      "volume": sum(x["volume"] for x in acc1), "timestamp": int(now * 1000)}
                acc1.clear()
                symbol_candles_5m[sym].append(dict(c5))
                if sym == PRIMARY_SYMBOL:
                    candles_5m.append(dict(c5))
                if _candle_queue is not None:
                    try:
                        _candle_queue.put_nowait((sym, c5))
                    except asyncio.QueueFull:
                        pass
            symbol_cur_min_ts[sym] = int(now // 60) * 60
            live_c.update({"open": price, "high": price, "low": price, "close": price, "volume": 0.0, "timestamp": symbol_cur_min_ts[sym] * 1000})
            if sym == PRIMARY_SYMBOL:
                cur_min_ts = symbol_cur_min_ts[sym]
                live_candle.update(live_c)
        else:
            if live_c["high"] == 0: live_c.update({"open": price, "high": price, "low": price})
            live_c["high"]   = max(live_c["high"], price)
            live_c["low"]    = min(live_c["low"],  price)
            live_c["close"]  = price
            live_c["volume"] += qty
            if sym == PRIMARY_SYMBOL:
                live_candle.update(live_c)

    if mkt["bid"] > 0 and mkt["ask"] > 0:
        mkt["spread_bps"] = ((mkt["ask"] - mkt["bid"]) / price) * 10000.0
        if sym == PRIMARY_SYMBOL:
            market["spread_bps"] = mkt["spread_bps"]

    # CVD (primary symbol)
    if sym == PRIMARY_SYMBOL:
        cvd_bucket += -qty if is_buyer_maker else qty

    # Momentum engine (tick-level signals for quick entries)
    m_eng = momentum_engines.get(sym, momentum_eng)
    mom_sig = m_eng.add_tick(price, qty, is_buyer_maker, int(now * 1000))

    # On every tick: check exits across all open positions
    check_exits()


def process_tick(price: float, qty: float, is_buyer_maker: bool):
    process_tick_for_symbol(PRIMARY_SYMBOL, price, qty, is_buyer_maker)


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

    # 3. Try Bybit Linear
    interval_map = {'1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30', '1h': '60', '2h': '120', '4h': '240', '1d': 'D'}
    bybit_interval = interval_map.get(interval, '5')
    try:
        r = requests.get(
            f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval={bybit_interval}&limit={limit}",
            timeout=6
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

    # 4. Fallback to Kraken (US exchange, 100% unblocked on US cloud datacenters like Railway)
    try:
        pair_map = {'BTCUSDT': 'XBTUSDT', 'ETHUSDT': 'ETHUSDT', 'SOLUSDT': 'SOLUSDT'}
        pair = pair_map.get(symbol, 'XBTUSDT')
        k_map = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60, '4h': 240, '1d': 1440}
        k_intv = k_map.get(interval, 5)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get(f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval={k_intv}', headers=headers, timeout=6)
        data = r.json()
        if not data.get('error'):
            result = data.get('result', {})
            key = [k for k in result if k != 'last'][0]
            raw = result[key][-limit:]
            return [
                {"timestamp": int(x[0]) * 1000, "open": float(x[1]), "high": float(x[2]),
                 "low": float(x[3]), "close": float(x[4]), "volume": float(x[6])}
                for x in raw
            ]
    except Exception:
        pass

    return []

def prewarm():
    global signal_engine, signal_engines, _loaded_model_count

    print("=" * 72)
    print("  APEX INTELLIGENCE ENGINE v7.1 — MULTI-ASSET INITIALIZATION")
    print("=" * 72)

    total_models_loaded = 0
    regimes_dir = ["ranging_high_vol", "ranging_low_vol", "trending_low_vol", "trending_high_vol"]

    for sym in TRADED_SYMBOLS:
        sym_key = sym.lower()
        print(f"  Loading XGBoost ensemble models for {sym}...")
        models: dict[str, ApexXGBoostModel] = {}
        for regime in regimes_dir:
            for direction in ["LONG", "SHORT"]:
                for bag_n in [1, 2, 3]:
                    name = f"{sym_key}_crypto_{regime}_{direction}_bag{bag_n}"
                    meta_path = MODEL_DIR / f"{name}_metadata.json"
                    model_path = MODEL_DIR / f"{name}.json"
                    if model_path.exists() and meta_path.exists():
                        try:
                            meta = json.load(open(meta_path))
                            m = ApexXGBoostModel(name=name, feature_columns=meta["feature_columns"])
                            m.load(path=MODEL_DIR)
                            models[name] = m
                        except Exception as e:
                            print(f"    WARN: Could not load {name}: {e}")

        # Also try to load transformer model if available
        try:
            from src.models.transformer_model import ApexTransformerModel
            tf_path = MODEL_DIR / f"{sym_key}_crypto_transformer.pth"
            if tf_path.exists():
                tf_model = ApexTransformerModel.load(f"{sym_key}_crypto_transformer", str(MODEL_DIR))
                if tf_model:
                    models[f"{sym_key}_crypto_transformer"] = tf_model
                    print(f"  ✓ Transformer model loaded ({sym})")
        except Exception:
            pass

        print(f"  ✓ [{sym}] {len(models)} model files loaded")
        total_models_loaded += len(models)

        # Build LiveSignalEngine for this symbol
        engine = LiveSignalEngine(
            symbol=sym,
            market_type="crypto",
            models=models,
            conviction_threshold=MIN_PROB,
            risk_per_trade=25.0 if PROP_FIRM_MODE else (account["balance"] * BASE_RISK_PCT),
            default_rr=MIN_RR,
            max_concurrent=1,
            enable_ensemble=True,
            enable_monitoring=True,
            enable_walk_forward=True,
            enable_council=True,
            enable_statistical_alpha=False,
            retrain_interval=2000,
            consensus_threshold=0.60,
        )
        signal_engines[sym] = engine
        if sym == PRIMARY_SYMBOL:
            signal_engine = engine

        # Warm up with historical candles
        print(f"  Fetching historical candles for {sym}...")
        raw5m = fetch_klines(sym, "5m", 210)
        raw15m = fetch_klines(sym, "15m", 100)
        raw1h = fetch_klines(sym, "1h", 72)
        raw4h = fetch_klines(sym, "4h", 48)
        raw1d = fetch_klines(sym, "1d", 60)

        if not raw5m:
            print(f"  ⚠️ Warning: Could not fetch 5m candles for {sym}, starting with empty buffer")
            if sym == PRIMARY_SYMBOL:
                prewarm.replay_candles = []
        else:
            hist = pd.DataFrame(raw5m)
            df_15 = pd.DataFrame(raw15m) if raw15m else None
            df_1h = pd.DataFrame(raw1h) if raw1h else None
            df_4h = pd.DataFrame(raw4h) if raw4h else None
            engine.warm_up(hist[["timestamp","open","high","low","close","volume"]], df_15m=df_15, df_1h=df_1h, df_4h=df_4h)
            for c in raw5m:
                symbol_candles_5m[sym].append(c)
                if sym == PRIMARY_SYMBOL:
                    candles_5m.append(c)

            last_close = float(raw5m[-1]["close"])
            symbol_markets[sym]["ltp"] = last_close
            symbol_markets[sym]["bid"] = last_close
            symbol_markets[sym]["ask"] = last_close
            if sym == PRIMARY_SYMBOL:
                market["ltp"] = last_close
                market["bid"] = last_close
                market["ask"] = last_close
                prewarm.replay_candles = raw5m[-3:]

        for c in (raw15m or []):
            symbol_candles_15m[sym].append(c)
            if sym == PRIMARY_SYMBOL:
                candles_15m.append(c)
        for c in (raw1h or []):
            symbol_candles_1h[sym].append(c)
            if sym == PRIMARY_SYMBOL:
                candles_1h.append(c)
        for c in (raw4h or []):
            symbol_candles_4h[sym].append(c)
            if sym == PRIMARY_SYMBOL:
                candles_4h.append(c)
        for c in (raw1d or []):
            symbol_candles_1d[sym].append(c)
            if sym == PRIMARY_SYMBOL:
                candles_1d.append(c)

        macro = compute_macro_structure(sym)
        bias_str = macro.get("macro_bias", "NEUTRAL")
        print(f"  ✓ [{sym}] Buffers ready: 5m={len(symbol_candles_5m[sym])}, 15m={len(symbol_candles_15m[sym])}, 1h={len(symbol_candles_1h[sym])}, 4h={len(symbol_candles_4h[sym])}, 1d={len(symbol_candles_1d[sym])}")
        print(f"    Macro Bias: {bias_str} (1D:{macro['trend_1d']}, 4H:{macro['trend_4h']}, 1H:{macro['trend_1h']})")

    _loaded_model_count = total_models_loaded
    regime_info["atr"] = 200.0  # Default until first signal

    update_session()
    load_journal()
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
        symbol=PRIMARY_SYMBOL,
        retrain_every_n_candles=2000,
        swap_callback=_on_models_ready,
    )

    print(f"\n  ── Concurrent Multi-Asset State ──")
    for sym in TRADED_SYMBOLS:
        p_val = symbol_markets[sym]["ltp"]
        print(f"  {sym:<9}: ${p_val:,.2f}")
    print(f"  Account:   ${account['balance']:,.2f} | PnL: ${account['net_pnl']:+.2f} | W{account['wins']}/L{account['losses']}")
    print(f"  Ensembles: {total_models_loaded} models active across {len(TRADED_SYMBOLS)} symbols")
    print("=" * 72)
    print("  ALL SYSTEMS GO — MULTI-ASSET INTELLIGENCE STACK ACTIVE")
    print("=" * 72)

    # ── Broker init (Bybit / MT5 — DEMO_MODE only) ──────────────────────────
    global _bybit_broker, _active_broker
    if DEMO_MODE:
        if BROKER_TYPE == "mt5":
            try:
                from src.execution.mt5_broker import MT5Broker
                _active_broker = MT5Broker()
                _bybit_broker = _active_broker
                print(f"  ✓ XM / MetaTrader 5 broker ready (Server: {os.getenv('MT5_SERVER', 'Default')})")
            except Exception as e:
                print(f"  ⚠️ Could not load MT5Broker: {e} → falling back to paper")
        else:
            if BYBIT_API_KEY and BYBIT_API_SECRET:
                from src.execution.bybit_broker import BybitBroker
                _active_broker = BybitBroker(
                    api_key=BYBIT_API_KEY,
                    api_secret=BYBIT_API_SECRET,
                    use_testnet=True,
                )
                _bybit_broker = _active_broker
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

    W = 98
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    def row(c):
        vl = vlen(c)
        if vl>W: c=trunc(c,W); vl=vlen(c)
        return f"{CYN}║{R} {c}{' '*(W-vl)} {CYN}║{R}\033[K"

    TOP=f"{CYN}╔{'═'*(W+2)}╗{R}\033[K"; MID=f"{CYN}╠{'═'*(W+2)}╣{R}\033[K"
    DIV=f"{CYN}╟{'─'*(W+2)}╢{R}\033[K"; BOT=f"{CYN}╚{'═'*(W+2)}╝{R}\033[K"

    buf=["\033[H", TOP]
    buf.append(row(f"{BOLD}{WHT}⚡ APEX INTELLIGENCE ENGINE v7.2 — CONCURRENT MULTI-ASSET CORE (BTC · ETH · GOLD) ⚡{R}"))
    ws = f"{GRN}LIVE ● (3 Symbols Active){R}" if market["ws_connected"] else f"{RED}DISCONNECTED{R}"
    buf.append(row(f"{DIM}Time:{R} {now} IST  │  {CYN}{session['name']} ({session['risk_mult']:.0%}){R}  │  WS: {ws}"))
    buf.append(MID)

    # 1. ACCOUNT & PROP FIRM HEADER
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

    if PROP_FIRM_MODE and prop_firm_manager:
        pf = prop_firm_manager.get_hud_metrics()
        pf_st_c = GRN if pf["status"] in ("IN_PROGRESS", "PHASE_1_PASSED") else RED
        target_str = f"+${pf['current_profit_usd']:.2f} / +${pf['target_usd']:.0f} ({pf['progress_pct']:.1f}%)"
        daily_loss_str = f"-${abs(min(0.0, pf['daily_pnl_usd'])):.2f} / -${pf['daily_limit_usd']:.0f} ({pf['daily_loss_used_pct']:.0f}% used)"
        dd_str = f"-${pf['total_dd_usd']:.2f} / -${pf['max_dd_usd']:.0f} ({pf['total_dd_pct']:.1f}%)"
        sl_badge = f"  │  {YLW}⚡ SOFT-LANDING (Tapered Risk){R}" if pf.get("soft_landing_active") else ""
        buf.append(DIV)
        buf.append(row(f"{BOLD}{YLW}🏆 PROP FIRM CHALLENGE [{pf['challenge_name']}]:{R}  Status: {pf_st_c}{pf['status']}{R}{sl_badge}"))
        buf.append(row(f" Virtual Equity: {BOLD}${pf['virtual_balance']:,.2f}{R}  │  Target: {GRN}{target_str}{R}"))
        buf.append(row(f" Daily Loss: {daily_loss_str}  │  Max DD: {dd_str}"))
    buf.append(DIV)

    # 2. CONCURRENT MULTI-ASSET MATRIX
    buf.append(row(f"{BOLD}{CYN}CONCURRENT MULTI-ASSET MATRIX:{R}"))
    buf.append(row(f" {BOLD}{'ASSET':<8} {'PRICE':<10} │ {'POSITION':<16} │ {'ML PROB (L / S)':<15} │ {'MOMENTUM / OFI':<15} │ {'REGIME (ATR)':<18}{R}"))
    for sym in TRADED_SYMBOLS:
        m = symbol_markets.get(sym, {})
        p = portfolios.get(sym, {})
        lp = symbol_live_probs.get(sym, {})
        s_reg = symbol_regime_info.get(sym, {})

        ltp_val = m.get("ltp", 0.0)
        ltp_str = f"${ltp_val:,.1f}" if "BTC" in sym else f"${ltp_val:,.2f}"
        pos_st = p.get("position", "FLAT")
        if pos_st == "OPEN":
            d = p.get("direction", "")
            d_c = GRN if d == "LONG" else RED
            pnl = p.get("broker_pnl") if p.get("broker_ticket") else p.get("unrealized_pnl", 0.0)
            pnl_c = GRN if pnl >= 0 else RED
            mode_tag = f"[{p.get('trade_mode','SCALP')[:2]}]"
            pos_str = f"{d_c}{BOLD}{d}{R}{mode_tag}({pnl_c}${pnl:+.2f}{R})"
        else:
            pos_str = f"{GRY}FLAT{R}"

        pl = lp.get("long", 0.5)
        ps = lp.get("short", 0.5)
        pl_c = GRN if pl >= MIN_PROB else (YLW if pl >= 0.52 else GRY)
        ps_c = RED if ps >= MIN_PROB else (YLW if ps >= 0.52 else GRY)

        m_eng = momentum_engines.get(sym, momentum_eng)
        mom_d = m_eng.get_diagnostics() if m_eng else {}
        mom_sig_str = mom_d.get("signal", "FLAT")
        mom_c = GRN if mom_sig_str == "LONG" else (RED if mom_sig_str == "SHORT" else GRY)
        ofi_v = float(mom_d.get("ofi", 0.0))
        mom_display = f"{mom_c}{mom_sig_str:<5}{R} (OFI:{ofi_v:+.2f})"

        reg = s_reg.get("current", lp.get("regime", "UNKNOWN"))
        reg_short = reg.replace("TRENDING_", "TREND_").replace("RANGING_", "RANGE_").replace("_VOL", "")
        atr = s_reg.get("atr", 0.0)
        reg_display = f"{reg_short} (${atr:.1f})"

        buf.append(row(f" {BOLD}{sym:<8}{R} {ltp_str:>10} │ {pos_str:<25} │ L:{pl_c}{pl:.3f}{R} S:{ps_c}{ps:.3f}{R} │ {mom_display:<24} │ {reg_display:<18}"))

    # Higher-Timeframe Macro Market Structure Row
    macro_items = []
    for s_m in TRADED_SYMBOLS:
        ms = symbol_macro_structure.get(s_m, {})
        bias = ms.get("macro_bias", "NEUTRAL")
        b_c = GRN if "BULL" in bias else (RED if "BEAR" in bias else GRY)
        s_lbl = "GOLD" if "PAXG" in s_m else s_m.replace("USDT", "")
        a1d = "▲" if "BULL" in ms.get("trend_1d", "") else ("▼" if "BEAR" in ms.get("trend_1d", "") else "▬")
        a4h = "▲" if "BULL" in ms.get("trend_4h", "") else ("▼" if "BEAR" in ms.get("trend_4h", "") else "▬")
        bias_short = bias.replace("STRONG_", "STR_")
        macro_items.append(f"{BOLD}{s_lbl}:{R} 1D:{a1d} 4H:{a4h} ({b_c}{bias_short}{R})")
    buf.append(row(f" {BOLD}{CYN}MACRO (1D/4H):{R} {' │ '.join(macro_items)}"))
    buf.append(DIV)

    # 3. ACTIVE POSITIONS & PORTFOLIO RISK POOL SUPERVISOR
    buf.append(row(f"{BOLD}{CYN}ACTIVE POSITIONS & PORTFOLIO RISK POOL:{R}"))
    open_ports = [p for p in portfolios.values() if p.get("position") == "OPEN"]
    open_count = len(open_ports)
    open_risk = sum(p.get("risk_usd", 25.0) for p in open_ports)
    corr_str = "CLEAR"
    if any(p.get("symbol") == "BTCUSDT" and p.get("position") == "OPEN" for p in portfolios.values()):
        btc_dir = portfolios["BTCUSDT"]["direction"]
        corr_str = f"BTC {btc_dir} (ETH Shielded)"
    elif any(p.get("symbol") == "ETHUSDT" and p.get("position") == "OPEN" for p in portfolios.values()):
        eth_dir = portfolios["ETHUSDT"]["direction"]
        corr_str = f"ETH {eth_dir} (BTC Shielded)"

    risk_pool_str = f"Positions: {open_count}/2 max  │  Open Risk: ${open_risk:.2f}/$50.00  │  Shield: {corr_str}"
    buf.append(row(f" {risk_pool_str}"))

    if open_ports:
        for op in open_ports:
            sym_o = op["symbol"]
            d = op["direction"]
            d_c = GRN if d == "LONG" else RED
            pnl_val = op.get("broker_pnl") if op.get("broker_ticket") else op.get("unrealized_pnl", 0.0)
            pnl_c = GRN if pnl_val >= 0 else RED
            lots = op.get("entry_lots", 0.0)
            entry = op.get("entry_price", 0.0)
            sl_p = op.get("sl_price", 0.0)
            tp1_p = op.get("tp1_price", 0.0)
            tp2_p = op.get("tp2_price", 0.0)
            t_id = op.get("broker_ticket", "PAPER")
            ph = "FULL" if op.get("phase") == 1 else "RUNNER(50%)"
            dur = (time.time() - op.get("entry_time", time.time())) / 60.0
            mode_tag = f"[{op.get('trade_mode', 'SCALP')}]"
            buf.append(row(f"  ► {BOLD}{sym_o}{R}: {d_c}{BOLD}{d}{R} {mode_tag} {lots}L @${entry:,.1f} │ PnL: {pnl_c}{BOLD}${pnl_val:+.2f}{R} │ Phase:{ph} │ Age:{dur:.0f}m │ Ticket:#{t_id}"))
            buf.append(row(f"    SL:${sl_p:,.1f} │ TP1:${tp1_p:,.1f}({op.get('tp1_rr', 1.8):.1f}R) │ TP2:${tp2_p:,.1f}({op.get('tp2_rr', 3.5):.1f}R) │ P:{op.get('prob',0.0):.3f} Cncl:{op.get('council_score',0.0):.2f}"))
    else:
        buf.append(row(f"  {GRY}► All assets FLAT — Monitoring 5m candles (Dual Scalp/Swing) across 3 symbols for Council entry{R}"))
    buf.append(DIV)

    # 4. LAST SIGNAL & GATE REJECTION
    ls = last_signal
    if ls:
        ls_sym = getattr(ls, "symbol", PRIMARY_SYMBOL)
        ls_dir = "LONG" if ls.direction == "BUY" else "SHORT"
        ls_c = GRN if ls.direction == "BUY" else RED
        council_c = GRN if ls.council_approved else RED
        buf.append(row(f"{BOLD}{CYN}LATEST SIGNAL:{R} {BOLD}{ls_sym}{R} {ls_c}{ls_dir}{R} p={ls.conviction:.3f} │ Council: {council_c}{'APPROVED' if ls.council_approved else 'BLOCKED'}{R} score={ls.council_score:.2f} [{ls.council_summary}]"))
    gr = last_gate_rejection
    if gr.get("reason") and gr.get("signal") and (time.time() - gr.get("ts", 0)) < 600:
        age = int(time.time() - gr["ts"])
        buf.append(row(f" {YLW}⚠  Approved {gr['signal']} NOT executed — {gr['reason']} ({age}s ago){R}"))
    if ls or (gr.get("reason") and gr.get("signal")):
        buf.append(DIV)

    # 5. MARKET INTELLIGENCE
    buf.append(row(f"{BOLD}{CYN}MARKET INTEL:{R}"))
    fg_val   = sentiment_monitor.get_fear_greed()
    fg_label = sentiment_monitor.get_label()
    fg_col   = sentiment_monitor.get_color_tag()
    fr_btc   = funding_monitor.get_funding_pct("BTCUSDT")
    fr_eth   = funding_monitor.get_funding_pct("ETHUSDT")
    nxt = news_guard.get_next_event()
    if nxt:
        mins_to = (nxt["time"] - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0
        evt_str = f"{nxt['name'][:30]} in {mins_to:.0f}m"
        evt_col = RED if mins_to < 30 else (YLW if mins_to < 90 else GRY)
    else:
        evt_str, evt_col = "No events scheduled", GRY
    is_blk, _ = news_guard.is_blackout("BTCUSDT")
    news_str = f"{RED}⛔ BLACKOUT{R}" if is_blk else f"{GRN}✓ CLEAR{R}"

    buf.append(row(f" F&G: {fg_col}{fg_val} ({fg_label}){R}  │  Funding: BTC {fr_btc:+.4f}%, ETH {fr_eth:+.4f}%  │  News: {news_str}"))
    cm_dxy = cross_market_monitor.dxy_trend
    cm_dxy_c = RED if cm_dxy == "USD_SURGING" else (GRN if cm_dxy == "USD_WEAKENING" else GRY)
    cm_eth = cross_market_monitor.eth_trend
    cm_eth_c = GRN if cm_eth == "ETH_OUTPERFORMING" else (RED if cm_eth == "ETH_UNDERPERFORMING" else GRY)
    buf.append(row(f" Macro DXY: {cm_dxy_c}{cm_dxy} ({cross_market_monitor.dxy_momentum_pct:+.2f}%){R}  │  ETH/BTC: {cm_eth_c}{cm_eth} ({cross_market_monitor.eth_btc_rel_pct:+.2f}%){R}"))
    buf.append(row(f" Next Event: {evt_col}{evt_str}{R}"))
    if retrain_scheduler:
        rs = retrain_scheduler.get_status()
        rt_str = f"{YLW}RETRAINING...{R}" if rs["is_retraining"] else f"{GRY}Next in {rs['next_retrain_in']} candles{R}"
        buf.append(row(f" Retrain: {rt_str}  │  Count: {rs['retrain_count']}  │  Candles: {rs['candles_accumulated']}"))
    buf.append(DIV)

    # 6. TRADE LOG
    buf.append(row(f"{BOLD}{CYN}TRADE LOG:{R}"))
    logs = list(trade_log)[-6:]
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
    # Construct combined multiplexed stream URL across all TRADED_SYMBOLS
    streams = []
    for s in TRADED_SYMBOLS:
        sl = s.lower()
        streams.extend([f"{sl}@trade", f"{sl}@bookTicker", f"{sl}@depth20@500ms", f"{sl}@kline_5m"])
    url = f"wss://fstream.binance.com/stream?streams={'/'.join(streams)}"

    while True:
        try:
            # Resilient ping and buffer settings so rapid price spikes never drop connection
            async with websockets.connect(
                url,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=15,
                max_queue=8192,
                max_size=2**22,
            ) as ws:
                for s in TRADED_SYMBOLS:
                    symbol_markets[s]["ws_connected"] = True
                market["ws_connected"] = True
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    st, pay = data.get("stream",""), data.get("data",{})
                    if not st:
                        continue
                    # Match symbol from stream string (e.g., btcusdt@trade -> BTCUSDT)
                    raw_sym = st.split('@')[0].upper()
                    sym = raw_sym
                    for ts in TRADED_SYMBOLS:
                        if ts.upper() == raw_sym:
                            sym = ts
                            break

                    if "@trade" in st:
                        try:
                            p = float(pay["p"])
                            q = float(pay["q"])
                            m = pay["m"]
                            process_tick_for_symbol(sym, p, q, m)
                            if sym == PRIMARY_SYMBOL:
                                update_session()
                        except Exception:
                            pass
                    elif "@bookTicker" in st:
                        mkt = symbol_markets.get(sym, market)
                        mkt["bid"] = float(pay["b"]); mkt["ask"] = float(pay["a"])
                        if sym == PRIMARY_SYMBOL:
                            market["bid"] = mkt["bid"]; market["ask"] = mkt["ask"]
                    elif "@depth20" in st:
                        try:
                            bids=pay.get("b",[]); asks=pay.get("a",[])
                            tb=sum(float(b[1]) for b in bids); ta=sum(float(a[1]) for a in asks)
                            if tb+ta>0:
                                obi_val = round((tb-ta)/(tb+ta), 3)
                                symbol_markets.get(sym, market)["obi"] = obi_val
                                if sym == PRIMARY_SYMBOL:
                                    market["obi"] = obi_val
                        except Exception: pass
                    elif "@kline_5m" in st:
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
                            # Update symbol buffers
                            if sym in symbol_candles_5m:
                                symbol_candles_5m[sym].append(dict(c5))
                            if sym == PRIMARY_SYMBOL:
                                candles_5m.append(dict(c5))

                            acc15 = symbol_acc_5m_15m.get(sym)
                            if acc15 is not None:
                                acc15.append(dict(c5))
                                if len(acc15) >= 3:
                                    c15 = {"open": acc15[0]["open"], "high": max(x["high"] for x in acc15),
                                           "low": min(x["low"] for x in acc15), "close": acc15[-1]["close"],
                                           "volume": sum(x["volume"] for x in acc15), "timestamp": int(k["t"])}
                                    acc15.clear()
                                    symbol_candles_15m[sym].append(dict(c15))
                                    if sym == PRIMARY_SYMBOL:
                                        candles_15m.append(dict(c15))
                                    acc1h = symbol_acc_15m_1h.get(sym)
                                    if acc1h is not None:
                                        acc1h.append(dict(c15))
                                        if len(acc1h) >= 4:
                                            c1h = {"open": acc1h[0]["open"], "high": max(x["high"] for x in acc1h),
                                                   "low": min(x["low"] for x in acc1h), "close": acc1h[-1]["close"],
                                                   "volume": sum(x["volume"] for x in acc1h), "timestamp": int(k["t"])}
                                            acc1h.clear()
                                            symbol_candles_1h[sym].append(dict(c1h))
                                            if sym == PRIMARY_SYMBOL:
                                                candles_1h.append(dict(c1h))
                                            acc4h = symbol_acc_1h_4h.get(sym)
                                            if acc4h is not None:
                                                acc4h.append(dict(c1h))
                                                if len(acc4h) >= 4:
                                                    c4h = {"open": acc4h[0]["open"], "high": max(x["high"] for x in acc4h),
                                                           "low": min(x["low"] for x in acc4h), "close": acc4h[-1]["close"],
                                                           "volume": sum(x["volume"] for x in acc4h), "timestamp": int(k["t"])}
                                                    acc4h.clear()
                                                    symbol_candles_4h[sym].append(dict(c4h))
                                                    if sym == PRIMARY_SYMBOL:
                                                        candles_4h.append(dict(c4h))
                                                    acc1d = symbol_acc_4h_1d.get(sym)
                                                    if acc1d is not None:
                                                        acc1d.append(dict(c4h))
                                                        if len(acc1d) >= 6:
                                                            c1d = {"open": acc1d[0]["open"], "high": max(x["high"] for x in acc1d),
                                                                   "low": min(x["low"] for x in acc1d), "close": acc1d[-1]["close"],
                                                                   "volume": sum(x["volume"] for x in acc1d), "timestamp": int(k["t"])}
                                                            acc1d.clear()
                                                            symbol_candles_1d[sym].append(dict(c1d))
                                                            if sym == PRIMARY_SYMBOL:
                                                                candles_1d.append(dict(c1d))
                                                    compute_macro_structure(sym)

                            if _candle_queue is not None:
                                try:
                                    _candle_queue.put_nowait((sym, c5))
                                except asyncio.QueueFull:
                                    pass
        except Exception:
            for s in TRADED_SYMBOLS:
                symbol_markets[s]["ws_connected"] = False
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
    asyncio.create_task(_safe_fetch(cross_market_monitor.update(), "cross_market"))
    # Don't await — let them run in background while WS + dashboard start immediately
    counter = 0
    while True:
        await asyncio.sleep(30)   # 30-second cycle for agile cross-market checks
        counter += 1
        try:
            # 1. Cross-market momentum updates every 30s
            await asyncio.wait_for(cross_market_monitor.update(), timeout=8)

            # 2. News, funding & sentiment update every 300s (every 10 cycles)
            if counter % 10 == 0:
                await asyncio.gather(
                    asyncio.wait_for(news_guard.update_events(), timeout=10),
                    asyncio.wait_for(funding_monitor.update(SYMBOL), timeout=10),
                    asyncio.wait_for(sentiment_monitor.update(), timeout=10),
                    return_exceptions=True,
                )

            # 3. Push fresh sentiment and cross-market data into Council
            if signal_engine and signal_engine.council:
                try:
                    fg_val = int(sentiment_monitor.get_fear_greed())
                    fund_pct = float(funding_monitor.get_funding_pct(SYMBOL))
                    signal_engine.council.update_sentiment(fg_val, fund_pct)
                    signal_engine.council.update_cross_market(cross_market_monitor.get_state())
                except Exception:
                    pass
        except Exception:
            pass


def get_dashboard_state() -> dict:
    """Extract real-time metrics for Web Dashboard & remote monitoring."""
    dur_min = 0
    if portfolio.get("entry_time", 0) > 0:
        dur_min = round((time.time() - portfolio["entry_time"]) / 60.0, 1)

    # Next event
    mins_to = None
    try:
        nxt = news_guard.get_next_event()
        if nxt:
            mins_to = round((nxt["time"] - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0, 1)
        is_blk, _ = news_guard.is_blackout(SYMBOL)
    except Exception:
        is_blk = False

    # Format journal for web UI (reverse chronological, latest first)
    journal_ui = []
    for t in reversed(trade_journal[-30:]):
        open_time_str = str(t.get("open_ts", ""))
        if "T" in open_time_str:
            open_time_str = open_time_str.split("T")[1][:8]
        journal_ui.append({
            "id": t.get("id", 0),
            "open_ts": open_time_str,
            "action": t.get("action", ""),
            "entry_price": t.get("entry_price", 0.0),
            "exit_price": t.get("exit_price", 0.0),
            "pnl_usd": t.get("pnl_usd", 0.0),
            "exit_reason": t.get("exit_reason", ""),
        })

    # OFI diagnostic
    ofi_val = 0.0
    try:
        mom_d = momentum_eng.get_diagnostics()
        ofi_val = round(float(mom_d.get("ofi", 0.0)), 3)
    except Exception:
        pass

    fg_val = 50
    fg_lbl = "Neutral"
    try:
        fg_val = sentiment_monitor.get_fear_greed()
        fg_lbl = sentiment_monitor.get_label()
    except Exception:
        pass

    funding_pct = 0.0
    try:
        funding_pct = round(funding_monitor.get_funding_pct(SYMBOL), 4)
    except Exception:
        pass

    return {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": {
            "ltp": market.get("ltp", 0.0),
            "bid": market.get("bid", 0.0),
            "ask": market.get("ask", 0.0),
            "ws_connected": market.get("ws_connected", False),
            "ticks": market.get("ticks", 0),
        },
        "account": {
            "balance": round(account.get("balance", 0.0), 2),
            "net_pnl": round(account.get("net_pnl", 0.0), 2),
            "drawdown_pct": round(account.get("drawdown_pct", 0.0), 2),
            "wins": account.get("wins", 0),
            "losses": account.get("losses", 0),
            "win_rate": round(account.get("win_rate", 0.0), 1),
            "profit_factor": round(account.get("profit_factor", 0.0), 2),
            "trades_today": account.get("trades_today", 0),
            "max_trades_day": MAX_TRADES_DAY,
        },
        "intel": {
            "fear_greed": fg_val,
            "fear_greed_label": fg_lbl,
            "funding_pct": funding_pct,
            "news_clear": not is_blk,
            "next_event_min": mins_to,
        },
        "cross_market": cross_market_monitor.get_state(),
        "ml": {
            "regime": regime_info.get("current", "UNKNOWN"),
            "atr": round(regime_info.get("atr", 0.0), 1),
            "prob_long": round(live_probs.get("long", 0.5), 4),
            "prob_short": round(live_probs.get("short", 0.5), 4),
            "candles_evaluated": signal_engine._candles_processed if signal_engine else 0,
            "models_count": _loaded_model_count,
        },
        "momentum": {
            "ofi": ofi_val,
        },
        "position": {
            "status": portfolio.get("position", "FLAT"),
            "direction": portfolio.get("direction", ""),
            "entry_price": portfolio.get("entry_price", 0.0),
            "sl_price": portfolio.get("sl_price", 0.0),
            "tp1_price": portfolio.get("tp1_price", 0.0),
            "tp2_price": portfolio.get("tp2_price", 0.0),
            "unrealized_pnl": round(portfolio.get("broker_pnl") if portfolio.get("broker_ticket") and portfolio.get("broker_pnl") != 0.0 else portfolio.get("unrealized_pnl", 0.0), 2),
            "duration_min": dur_min,
            "broker_ticket": portfolio.get("broker_ticket", ""),
        },
        "symbols": {
            sym: {
                "ltp": symbol_markets.get(sym, {}).get("ltp", 0.0),
                "bid": symbol_markets.get(sym, {}).get("bid", 0.0),
                "ask": symbol_markets.get(sym, {}).get("ask", 0.0),
                "prob_long": symbol_live_probs.get(sym, {}).get("long", 0.5),
                "prob_short": symbol_live_probs.get(sym, {}).get("short", 0.5),
                "regime": symbol_regime_info.get(sym, {}).get("current", "UNKNOWN"),
                "atr": symbol_regime_info.get(sym, {}).get("atr", 0.0),
                "position": portfolios.get(sym, {}).get("position", "FLAT"),
                "direction": portfolios.get(sym, {}).get("direction"),
                "entry_price": portfolios.get(sym, {}).get("entry_price", 0.0),
                "unrealized_pnl": portfolios.get(sym, {}).get("broker_pnl") if portfolios.get(sym, {}).get("broker_ticket") else portfolios.get(sym, {}).get("unrealized_pnl", 0.0),
            } for sym in TRADED_SYMBOLS
        },
        "prop_firm": prop_firm_manager.get_hud_metrics() if (PROP_FIRM_MODE and prop_firm_manager) else None,
        "journal": journal_ui,
    }


def _do_close_symbol(sym: str, exit_profit: float, exit_price: float, reason: str):
    """Unified handler: updates stats, balance, prop firm challenge, and journal upon trade close."""
    sym_port = portfolios.get(sym, portfolio)
    if sym_port.get("position") != "OPEN":
        return
    cur_dir = sym_port.get("direction", "LONG")
    is_win = exit_profit > 0
    account["balance"] += exit_profit
    update_health()
    if PROP_FIRM_MODE and prop_firm_manager:
        prop_firm_manager.on_trade_closed(exit_profit, reason)
    if is_win:
        account["wins"] += 1
        account["consecutive_losses"] = 0
        account["total_wins_usd"] += exit_profit
    else:
        account["losses"] += 1
        account["consecutive_losses"] += 1
        account["total_losses_usd"] += abs(exit_profit)
        account["last_loss_time"] = time.time()
        if account["consecutive_losses"] >= 2:
            account["halted"] = True
            account["halt_reason"] = "2 consecutive losses — cooling"
            account["halt_until"] = time.time() + LOSS_COOLDOWN_2
    account["total"] += 1
    account["win_rate"] = round(account["wins"]/account["total"]*100, 1) if account["total"] > 0 else 0
    account["avg_win"]  = round(account["total_wins_usd"]/account["wins"], 2) if account["wins"] > 0 else 0
    account["avg_loss"] = round(account["total_losses_usd"]/account["losses"], 2) if account["losses"] > 0 else 0
    account["profit_factor"] = round(account["total_wins_usd"]/account["total_losses_usd"], 3) if account["total_losses_usd"] > 0 else 999.0
    account["best"]  = max(account["best"], exit_profit)
    account["worst"] = min(account["worst"], exit_profit)

    s = "+" if exit_profit >= 0 else ""
    trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                     f"CLOSE [{sym}] @${exit_price:,.2f} PnL:{s}${exit_profit:.2f} | {reason}")
    send_challenge_alert("TRADE CLOSED", {
        "symbol": sym,
        "action": cur_dir,
        "price": exit_price,
        "pnl_usd": exit_profit,
        "reason": reason,
    })
    for t in reversed(trade_journal):
        if t.get("symbol") == sym and t.get("outcome") == "OPEN":
            t.update({"exit_ts": datetime.datetime.now().isoformat(),
                      "exit_price": exit_price, "pnl_usd": exit_profit,
                      "outcome": "WIN" if is_win else "LOSS",
                      "exit_reason": reason})
            break
    save_journal()
    sym_port.update({"position": "FLAT", "direction": None,
                     "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0,
                     "broker_ticket": "", "broker_pnl": 0.0})
    if sym == PRIMARY_SYMBOL:
        portfolio.update(sym_port)
    state_manager.save_state(portfolio, account)


# ─── BROKER POSITION MONITOR & CONTROLLER (APEX v7.2) ────────────────────────
async def _broker_sync_loop():
    """
    Dedicated background supervisor: monitors MT5/Broker open positions every 2s across all traded symbols.
    Detects native broker SL/TP fills, synchronizes real PnL, and updates journal.
    """
    while True:
        await asyncio.sleep(2)
        if not DEMO_MODE or _active_broker is None:
            continue

        for sym in TRADED_SYMBOLS:
            sym_port = portfolios.get(sym)
            if not sym_port:
                continue

            ticket_str = sym_port.get("broker_ticket")
            if sym_port.get("position") == "OPEN" and ticket_str:
                try:
                    ticket = int(ticket_str)
                    if hasattr(_active_broker, "get_open_position"):
                        pos = await _active_broker.get_open_position(ticket=ticket)
                        if pos:
                            sym_port["broker_pnl"] = float(pos.get("profit", 0.0))
                            if sym == PRIMARY_SYMBOL:
                                portfolio["broker_pnl"] = sym_port["broker_pnl"]
                            if PROP_FIRM_MODE and prop_firm_manager:
                                prop_firm_manager.update_multi_floating_pnl(portfolios)
                        else:
                            deal = None
                            if hasattr(_active_broker, "get_closed_deal"):
                                deal = await _active_broker.get_closed_deal(ticket=ticket)

                            sym_mkt = symbol_markets.get(sym, market)
                            exit_price = float(deal.get("price", sym_mkt.get("ltp", 0.0))) if deal else sym_mkt.get("ltp", 0.0)
                            exit_profit = float(deal.get("profit", 0.0)) if deal else 0.0
                            comment = deal.get("comment", "") if deal else ""
                            reason = f"MT5 {comment}" if comment else "MT5 SL/TP EXIT"

                            trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                                             f"BROKER SYNC [{sym}]: MT5 closed ticket {ticket} ({reason}) PnL: ${exit_profit:+.2f}")
                            _do_close_symbol(sym, exit_profit, exit_price, reason)
                except Exception:
                    pass

            # Auto-adoption per symbol
            elif sym_port.get("position") == "FLAT" and hasattr(_active_broker, "get_open_position"):
                try:
                    active_pos = await _active_broker.get_open_position(symbol=sym)
                    if active_pos:
                        claimed = any(p.get("broker_ticket") == str(active_pos["ticket"]) for p in portfolios.values())
                        if not claimed:
                            p_dir = "LONG" if active_pos["type"] == "BUY" else "SHORT"
                            p_entry = float(active_pos["price_open"])
                            p_sl = float(active_pos["sl"])
                            p_tp = float(active_pos["tp"])
                            p_vol = float(active_pos["volume"])
                            p_ticket = str(active_pos["ticket"])
                            p_profit = float(active_pos["profit"])
                            sl_dist_val = abs(p_entry - p_sl) if p_sl > 0 else (10.0 if "ETH" in sym else (5.0 if "GOLD" in sym else 300.0))

                            sym_port.update({
                                "symbol": sym,
                                "position": "OPEN",
                                "direction": p_dir,
                                "entry_price": p_entry,
                                "entry_time": time.time(),
                                "sl_price": p_sl,
                                "sl_distance": sl_dist_val,
                                "tp1_price": p_entry + sl_dist_val if p_dir == "LONG" else p_entry - sl_dist_val,
                                "tp2_price": p_tp,
                                "notional": round(p_vol * p_entry, 2),
                                "initial_notional": round(p_vol * p_entry, 2),
                                "unrealized_pnl": p_profit,
                                "broker_ticket": p_ticket,
                                "broker_pnl": p_profit,
                                "phase": 1,
                            })
                            trade_log.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                                             f"BROKER ADOPT [{sym}]: Linked active MT5 ticket {p_ticket} ({p_dir} {p_vol} {sym} @ ${p_entry:,.2f})")
                            if sym == PRIMARY_SYMBOL:
                                portfolio.update(sym_port)
                            state_manager.save_state(portfolio, account)
                except Exception:
                    pass

            # Ghost position check per symbol
            elif sym_port.get("position") == "OPEN" and not ticket_str and DEMO_MODE and hasattr(_active_broker, "get_open_position"):
                try:
                    active_pos = await _active_broker.get_open_position(symbol=sym)
                    if not active_pos:
                        sym_port.update({
                            "position": "FLAT", "direction": None, "entry_price": 0.0,
                            "sl_price": 0.0, "tp1_price": 0.0, "tp2_price": 0.0,
                            "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0,
                            "broker_ticket": "", "broker_pnl": 0.0
                        })
                        if sym == PRIMARY_SYMBOL:
                            portfolio.update(sym_port)
                        state_manager.save_state(portfolio, account)
                except Exception:
                    pass

async def main():
    global _candle_queue
    _candle_queue = asyncio.Queue(maxsize=50)  # Must be created inside running event loop
    prewarm()
    update_health()

    # ── Replay last few historical candles to populate regime/ATR across all symbols ──
    for sym in TRADED_SYMBOLS:
        eng = signal_engines.get(sym)
        sym_buf = list(symbol_candles_5m[sym])[-3:] if len(symbol_candles_5m[sym]) >= 3 else []
        if sym_buf and eng:
            print(f"  Replaying {len(sym_buf)} historical candles for {sym} to initialize ML state...")
            for c in sym_buf:
                try:
                    sig = await eng.process_candle(c)
                    if sig is not None and sym == PRIMARY_SYMBOL:
                        global last_signal
                        last_signal = sig
                except Exception:
                    pass
            eng._suppression_counter = 0
            eng._open_position_count = 0
            eng._pending_signals.clear()
            eng._signals_generated = 0
            if hasattr(eng, "regime_detector"):
                r_val = eng.regime_detector.current_regime.value
                symbol_regime_info[sym]["current"] = r_val
                symbol_live_probs[sym]["regime"] = r_val
                if sym == PRIMARY_SYMBOL:
                    regime_info["current"] = r_val
            try:
                buf = eng.live_store.get_buffer_as_df()
                if not buf.empty:
                    feats = eng.live_store._feature_store.build_features(buf)
                    if not feats.empty:
                        atr_v = float(feats["ATR"].iloc[-1])
                        symbol_regime_info[sym]["atr"] = atr_v
                        if sym == PRIMARY_SYMBOL:
                            regime_info["atr"] = atr_v
                            regime_info["adx"] = float(feats["ADX"].iloc[-1]) if "ADX" in feats else 25.0
                            regime_info["chop"] = float(feats["CHOP"].iloc[-1]) if "CHOP" in feats else 50.0
            except Exception:
                pass
            print(f"  ✓ [{sym}] Replay complete — Regime: {symbol_regime_info[sym]['current']} | ATR: ${symbol_regime_info[sym]['atr']:,.1f}")

    # ── Broker Connection Validation ──────────────────────────────────────────
    if DEMO_MODE and _active_broker:
        acc_info = await _active_broker.validate_account()
        if acc_info.get("valid"):
            bal = acc_info.get('balance', 0.0)
            cur = acc_info.get('currency', 'USDT')
            srv = acc_info.get('server', BROKER_TYPE.upper())
            print(f"  ✓ {_active_broker.__class__.__name__} connected ({srv}) — Balance: ${bal:,.2f} {cur}")
        else:
            print(f"  ⚠️  Broker warning: {acc_info.get('error')} (Order placement disabled until verified)")

    # ── Live Web Dashboard Server (Local & Cloud) ─────────────────────────────
    web_port = int(os.getenv("PORT", "8080"))
    from src.ui.web_server import ApexWebServer
    web_server = ApexWebServer(
        state_provider=get_dashboard_state,
        host="0.0.0.0",
        port=web_port
    )
    await web_server.start()

    await asyncio.gather(
        binance_ws(),
        ui_loop(),
        _candle_worker(),
        _market_intel_loop(),   # APEX v7.2: news + funding + sentiment
        _broker_sync_loop(),    # Real-time MT5 position supervisor
    )

if __name__ == "__main__":
    asyncio.run(main())
