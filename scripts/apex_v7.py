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
from src.intelligence.liquidity_engine import LiquidityEngine
from src.intelligence.lead_lag_engine import LeadLagEngine
from src.intelligence.microstructure_engine import MicrostructureEngine

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
        mode_str = f" [{details.get('trade_mode')}]" if details.get("trade_mode") else ""
        txt = (
            f"{icon} APEX: {event_type}{mode_str}\n"
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
MIN_RR          = _rc("trade_management", "default_rr_target", 2.2)   # Realistic intraday R:R from risk.yaml
MIN_SL_USD      = 250.0     # Never tighter than $250 stop
MAX_SL_USD      = 800.0     # Never wider than $800

def compute_auction_location(symbol: str, direction: str, price: float, atr: float) -> tuple[float, float, str]:
    """
    Evaluates trade location using Auction Market Theory (proximity to 5m value equilibrium & liquidity origin).
    Rewards entries near institutional value/sweeps (-0.02 threshold) and adds smooth progressive stretch
    friction (+0.025/ATR beyond 1.2 ATR) so late extended entries require proportionally higher conviction.
    """
    sym = symbol.upper()
    if atr <= 0 or price <= 0:
        return 0.0, 0.0, "neutral"

    c5_buf = list(symbol_candles_5m.get(sym, []))
    if len(c5_buf) < 20:
        return 0.0, 0.0, "warmup"

    closes = [float(c["close"]) for c in c5_buf[-20:]]
    ema20 = closes[0]
    k = 2.0 / 21.0
    for val in closes[1:]:
        ema20 = (val - ema20) * k + ema20

    # Directional extension away from 5m value mean (positive = extended in trade direction)
    dir_u = direction.upper()
    ext_pts = (price - ema20) if dir_u in ("LONG", "BUY") else (ema20 - price)
    ext_atr = ext_pts / atr

    lev = liquidity_engine.get_levels(sym) if "liquidity_engine" in globals() else {}
    at_origin = (
        abs(ext_atr) <= 0.65
        or (dir_u in ("LONG", "BUY") and (lev.get("sweep_bullish") or lev.get("fvg_bullish")))
        or (dir_u in ("SHORT", "SELL") and (lev.get("sweep_bearish") or lev.get("fvg_bearish")))
    )

    if at_origin and ext_atr <= 0.90:
        return round(ext_atr, 2), -0.02, "ORIGIN/VALUE"
    if ext_atr > 1.2:
        penalty = min(0.06, round(0.025 * (ext_atr - 1.2), 3))
        return round(ext_atr, 2), penalty, f"STRETCHED({ext_atr:.1f}ATR)"
    return round(ext_atr, 2), 0.0, "BALANCED"

def get_calibrated_threshold(symbol: str, regime: str = "UNKNOWN", loc_delta: float = 0.0) -> float:
    """
    Asset-Calibrated Conviction Thresholds with Regime & Auction Location Sensitivity.
    """
    sym = symbol.upper()
    base = 0.59 if "ETH" in sym else 0.57
    reg_l = str(regime).lower()
    if "trending_high_vol" in reg_l:
        base -= 0.02
    elif "ranging_low_vol" in reg_l:
        base += 0.03
    elif "ranging_high_vol" in reg_l:
        base += 0.02

    s_name = session.get("name", "UNKNOWN")
    if s_name in ("LONDON", "NEW_YORK"):
        base -= 0.01
    elif s_name == "PACIFIC":
        base += 0.04

    if session.get("is_kill_zone", False):
        base -= 0.01

    base += loc_delta
    return round(float(np.clip(base, 0.53, 0.66)), 3)

# Risk
BASE_RISK_PCT   = _rc("sizing", "min_allocation_pct", 0.010)   # from risk.yaml
MAX_RISK_PCT   = _rc("sizing", "max_allocation_pct", 0.015)   # from risk.yaml
MIN_RISK_PCT    = 0.005
MAX_LEVERAGE    = 5.0
MAX_TRADES_DAY  = int(os.getenv("APEX_MAX_TRADES_DAY", "3"))  # Sniper discipline: 3 quality trades/day max
MIN_INTERVAL    = 300       # 5min (1 candle) between trades
LOSS_COOLDOWN_1 = 1800      # 30min cooling pause after 1 loss (allows market chop to settle)
LOSS_COOLDOWN_2 = 3600      # 60min cooling pause after 2 consecutive losses
LOSS_COOLDOWN_3 = 7200      # 120min anti-spiral protection after 3 losses
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
        "broker_entry_price": 0.0, "broker_offset": 0.0,
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
    cur_px = float(symbol_markets.get(sym, {}).get("ltp", 0.0))
    if cur_px <= 0 and symbol_candles_5m.get(sym):
        cur_px = float(symbol_candles_5m[sym][-1].get("close", 0.0))

    # 1. Daily (1D) Analysis
    if len(c_1d) >= 15:
        closes_1d = [float(c["close"]) for c in c_1d]
        cur_1d = cur_px if cur_px > 0 else closes_1d[-1]
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
        cur_4h = cur_px if cur_px > 0 else closes_4h[-1]
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
        cur_1h = cur_px if cur_px > 0 else closes_1h[-1]
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
_last_queued_5m_ts: dict[str, int] = {}
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

# Institutional Intelligence Subsystems
liquidity_engine     = LiquidityEngine()
lead_lag_engine      = LeadLagEngine()
microstructure_engines: dict[str, MicrostructureEngine] = {
    sym: MicrostructureEngine(sym) for sym in TRADED_SYMBOLS
}

retrain_schedulers: dict[str, RetrainScheduler] = {} # Multi-asset independent retrainers
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
# P3: per-symbol gate rejections (avoids one symbol's skip hiding another's)
last_gate_rejection = {"reason": "", "signal": "", "ts": 0.0}
symbol_gate_rejections: dict[str, dict] = {}
# Institutional Fair-Value Pullback Retest Queue {symbol: {signal, direction, retest_px, inv_px, expires_ts}}
pending_retest_zones: dict[str, dict] = {}

def _is_grade_a_plus(prob: float, council: float) -> bool:
    """Unified Grade A+ institutional threshold across worker, gates, and execution."""
    return (prob >= 0.65 and council >= 0.68) or prob >= 0.70

def _record_gate_rejection(sym: str, signal_desc: str, reason: str):
    """Store per-symbol gate rejection without letting another asset overwrite it."""
    now_t = time.time()
    symbol_gate_rejections[sym] = {"symbol": sym, "signal": signal_desc, "reason": reason, "ts": now_t}
    last_gate_rejection["reason"] = reason
    last_gate_rejection["signal"] = signal_desc
    last_gate_rejection["ts"] = now_t

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
        # Reclassify $0 / sub-$0.50 flat sync closes as BE so they never poison streak or W/L stats
        for t in trade_journal:
            if t.get("outcome") in ("WIN", "LOSS") and abs(float(t.get("pnl_usd", 0.0))) < 0.50:
                t["outcome"] = "BE"
        closed = [t for t in trade_journal if t.get("outcome") in ["WIN", "LOSS"]]
        if not closed:
            account["consecutive_losses"] = 0
            if "consecutive losses" in account.get("halt_reason", "").lower():
                account["halted"] = False; account["halt_reason"] = ""; account["halt_until"] = 0.0
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
        if streak < 2 and "consecutive losses" in account.get("halt_reason", "").lower():
            account["halted"] = False; account["halt_reason"] = ""; account["halt_until"] = 0.0

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
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    utc_h = now_dt.hour
    weekday = now_dt.weekday()  # 0=Monday ... 6=Sunday
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    if today != account["today_date"]:
        account["trades_today"] = 0; account["today_date"] = today
        if "MAX TRADES" in account.get("halt_reason","") or "DAY" in account.get("halt_reason",""):
            account["halted"] = False; account["halt_reason"] = ""; account["halt_until"] = 0.0

    # Sunday Weekly Open Liquidity Expansion (Sunday 22:00 UTC to Monday 04:00 UTC)
    is_sunday_open = (weekday == 6 and utc_h >= 22) or (weekday == 0 and utc_h < 4)
    session["sunday_weekly_open"] = is_sunday_open

    # ICT Institutional Kill Zones (London Open 07-09, NY Open 12-15, London Close 15:30-16:30, Asian 00-02 UTC)
    kz_name = ""
    if 7 <= utc_h < 9:
        kz_name = "LONDON_OPEN_KZ"
    elif 12 <= utc_h < 15:
        kz_name = "NY_OPEN_KZ"
    elif (utc_h == 15 and now_dt.minute >= 30) or (utc_h == 16 and now_dt.minute <= 30):
        kz_name = "LONDON_CLOSE_KZ"
    elif 0 <= utc_h < 2:
        kz_name = "ASIAN_KZ"

    session["kill_zone"] = kz_name
    session["is_kill_zone"] = bool(kz_name)

    session["name"] = ("LONDON" if 7<=utc_h<13 else "NEW_YORK" if 13<=utc_h<21
                        else "ASIAN" if 0<=utc_h<7 else "PACIFIC")
    if is_sunday_open:
        session["risk_mult"] = 1.05  # Sunday open weekly liquidity expansion
    else:
        session["risk_mult"] = 1.0 if session["name"] in ("LONDON","NEW_YORK") else (0.75 if session["name"]=="ASIAN" else 0.6)

def is_rollover_or_maintenance() -> tuple[bool, str]:
    """
    Detects daily broker server rollover spikes and maintenance windows.
    Protects against artificial spread widening (e.g. 21:50-22:15 UTC and 23:55-00:15 UTC).
    Configurable via APEX_WEEKEND_SHIELD (default true).
    """
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    # 1. Daily Broker Rollover windows:
    # A. 21:50 to 22:15 UTC (GMT+2 / GMT+3 broker midnight rollover: XM, Eightcap, IC Markets, ThinkMarkets)
    if (now_utc.hour == 21 and now_utc.minute >= 50) or (now_utc.hour == 22 and now_utc.minute <= 15):
        return True, "BROKER SERVER ROLLOVER WINDOW (21:50-22:15 UTC) — SPREAD SHIELD"
    # B. 23:55 to 00:15 UTC (UTC midnight day rollover)
    if (now_utc.hour == 23 and now_utc.minute >= 55) or (now_utc.hour == 0 and now_utc.minute <= 15):
        return True, "UTC MIDNIGHT ROLLOVER WINDOW (23:55-00:15 UTC) — SPREAD SHIELD"
    # 2. Weekend Maintenance window on XM: Saturday 06:00 to 12:00 UTC
    weekend_shield_enabled = os.getenv("APEX_WEEKEND_SHIELD", "true").lower() == "true"
    if weekend_shield_enabled and now_utc.weekday() == 5 and (6 <= now_utc.hour < 12):
        return True, "WEEKEND SERVER MAINTENANCE WINDOW (06:00-12:00 UTC) — TRADING PAUSED"
    # 3. Abnormal Broker Spread (> 12.0 bps for BTC)
    current_spread = market.get("spread_bps", 0.0)
    if current_spread > 12.0:
        return True, f"ABNORMAL SPREAD ({current_spread:.1f} bps > 12.0 bps)"
    return False, "CLEAR"

def can_trade_symbol(symbol: str = SYMBOL, direction: str = "LONG", is_high_prob: bool = False) -> tuple[bool, str]:
    now = time.time()
    sym = symbol.upper()
    dir_upper = direction.upper()

    # Weekend & Daily Rollover Spread Shield (XM MT5 Protection)
    shielded, shield_reason = is_rollover_or_maintenance()
    if shielded:
        return False, f"SPREAD SHIELD: {shield_reason}"

    # ── Modular Prop Firm Gate (SharkFunded, FTMO, FundedNext, etc.) ──
    if PROP_FIRM_MODE and prop_firm_manager:
        cfg_max_c = int(prop_firm_manager.config.get("max_concurrent_trades", 1))
        est_risk = float(prop_firm_manager.virtual_balance * (float(prop_firm_manager.config.get("base_risk_pct", 0.6)) / 100.0))
        pf_ok, pf_reason = prop_firm_manager.can_open_multi_asset_trade(
            symbol=sym,
            direction=dir_upper,
            active_positions=portfolios,
            estimated_risk_usd=est_risk,
            max_concurrent=cfg_max_c,
            is_grade_a_plus=is_high_prob,
        )
        if not pf_ok:
            return False, f"PROP_FIRM: {pf_reason}"
    else:
        # ── Standard Account Multi-Asset Risk Gate (Personal Account / Demo) ──
        # 1. Per-symbol isolation
        if portfolios.get(sym, {}).get("position") == "OPEN":
            return False, f"{sym} already has an OPEN position"

        # 2. Maximum concurrent trades across portfolio
        max_c = int(_rc("sizing", "max_concurrent_trades", 2))
        active_count = sum(1 for p in portfolios.values() if p.get("position") == "OPEN")
        if active_count >= max_c:
            return False, f"Max concurrent trades reached ({active_count}/{max_c})"

    # ── Universal Portfolio Directional Coherence & Correlation Cluster Guard ──
    # Runs unconditionally in both Prop Firm and Personal mode.
    for p_sym, pos in portfolios.items():
        if pos.get("position") == "OPEN":
            p_dir = str(pos.get("direction", "")).upper()
            is_gold_p = ("GOLD" in p_sym.upper() or "PAXG" in p_sym.upper())
            is_crypto_p = ("BTC" in p_sym.upper() or "ETH" in p_sym.upper())
            is_curr_gold = ("GOLD" in sym or "PAXG" in sym)
            is_curr_crypto = ("BTC" in sym or "ETH" in sym)

            # Cross-asset conflict: Gold vs Crypto opposing directions cancel each other out
            if (is_gold_p and is_curr_crypto) or (is_crypto_p and is_curr_gold):
                if p_dir != dir_upper:
                    return False, f"Coherence Lock: Cannot {dir_upper} {sym} while {p_sym} is {p_dir} (Macro Conflict eating profits)"

            # Crypto Correlation Cluster Lock: BTC & ETH are 0.92 correlated — NEVER hold BOTH open simultaneously!
            if is_crypto_p and is_curr_crypto and sym != p_sym.upper():
                return False, f"Crypto Cluster Lock: Cannot open {sym} while correlated {p_sym} is already OPEN (prevents 2x leveraged crypto exposure)"

    if account["halted"]:
        if account["halt_until"] <= now:
            account["halted"] = False; account["halt_reason"] = ""
        else:
            rem = int(account["halt_until"] - now) if account["halt_until"] < float("inf") else 999999
            return False, f"HALTED: {account['halt_reason']} ({rem//60}m left)"
    if account["trades_today"] >= MAX_TRADES_DAY:
        if not is_high_prob:
            return False, f"DYNAMIC TRADE CAP: Trade #{account['trades_today']+1} requires High-Probability Grade A+ setup"
        hard_cap = 8 if session.get("name") in ("LONDON", "NEW_YORK") else 6
        if account["trades_today"] >= hard_cap:
            return False, f"HARD DAILY TRADES LIMIT ({hard_cap}/{hard_cap} trades reached)"
    # Dynamic, non-rigid loss cooldown based on consecutive losses
    consec = account.get("consecutive_losses", 0)
    cooldown_secs = LOSS_COOLDOWN_1 if consec <= 1 else (LOSS_COOLDOWN_2 if consec == 2 else LOSS_COOLDOWN_3)
    if account["last_loss_time"] > 0 and (now - account["last_loss_time"]) < cooldown_secs:
        if is_high_prob:
            logger.info("loss_cooldown_bypassed_by_grade_a_plus", symbol=sym, direction=dir_upper)
        else:
            rem = int(cooldown_secs - (now - account["last_loss_time"]))
            return False, f"LOSS COOLDOWN {rem//60}m{rem%60:02d}s"

    mkt = symbol_markets.get(sym, market)
    if mkt.get("spread_bps", 0.0) > 12.0:
        return False, f"SPREAD {mkt['spread_bps']:.1f}bps"
    # APEX v7.2: News blackout gate
    blackout, blackout_reason = news_guard.is_blackout(sym)
    if blackout:
        return False, f"NEWS BLACKOUT: {blackout_reason}"

    # ── Metals CFD Weekend Closure Guard (MT5 Gold closes Friday 21:00 UTC to Sunday 22:00 UTC) ──
    if ("GOLD" in sym or "PAXG" in sym) and BROKER_TYPE.upper() == "MT5":
        now_dt = datetime.datetime.now(datetime.timezone.utc)
        weekday = now_dt.weekday()  # 4=Friday, 5=Saturday, 6=Sunday
        hour = now_dt.hour
        is_metals_closed = (weekday == 5) or (weekday == 4 and hour >= 21) or (weekday == 6 and hour < 22)
        if is_metals_closed:
            return False, "METALS MARKET CLOSED (MT5 Gold reopens Sunday 22:00 UTC)"

    # ── Gold Institutional Structure Veto (FVG & C1/C2 Demand/Supply Gate, adapted from xau_c) ──
    if ("GOLD" in sym or "PAXG" in sym):
        try:
            from src.intelligence.gold_structure_engine import gold_structure_engine
            cur_price = mkt.get("ltp", 0.0) or (mkt.get("bid", 0.0) if dir_upper == "SHORT" else mkt.get("ask", 0.0))
            if cur_price > 0:
                c_15m = list(symbol_candles_15m.get(sym, []))
                c_1h = list(symbol_candles_1h.get(sym, []))
                if c_15m or c_1h:
                    g_eval = gold_structure_engine.evaluate_gold_structure(cur_price, c_15m, c_1h)
                    if dir_upper == "SHORT" and g_eval.get("veto_short"):
                        return False, g_eval.get("veto_reason", "Gold FVG Demand Zone: Shorts Prohibited")
                    elif dir_upper == "LONG" and g_eval.get("veto_long"):
                        return False, g_eval.get("veto_reason", "Gold FVG Supply Zone: Longs Prohibited")
        except Exception as e:
            logger.debug("gold_structure_check_error", error=str(e))

    # ── Dynamic Multi-Timeframe Alignment (No rigid 1H EMA lag stranglehold) ──
    # 1) Never fight an aligned 4H trend unless Grade A+ AND a confirmed Liquidity Sweep Turn fired.
    # 2) If 4H aligns with the signal (or is NEUTRAL), Grade A+ signals bypass lagging 1H EMA flips;
    #    standard signals are permitted through 1H lag on Sweep Turn, Flow Turn, or aligned 4H momentum.
    macro = symbol_macro_structure.get(sym)
    if macro:
        trend_4h = str(macro.get("trend_4h", "NEUTRAL")).upper()
        trend_1h = str(macro.get("trend_1h", "NEUTRAL")).upper()

        lev = liquidity_engine.get_levels(sym) if 'liquidity_engine' in globals() else {}
        cur_obi = float(mkt.get("obi", 0.0))
        cur_ofi = float(momentum_engines[sym].get_diagnostics().get("ofi", 0.0)) if sym in momentum_engines else 0.0

        has_sweep_turn = (
            (dir_upper == "SHORT" and (lev.get("sweep_bearish") or lev.get("breakout_bearish")))
            or (dir_upper == "LONG" and (lev.get("sweep_bullish") or lev.get("breakout_bullish")))
        )
        has_flow_turn = (
            (dir_upper == "SHORT" and (cur_obi <= -0.15 or cur_ofi <= -0.25))
            or (dir_upper == "LONG" and (cur_obi >= 0.15 or cur_ofi >= 0.25))
        )
        if dir_upper == "SHORT":
            four_h_aligned = ("BEAR" in trend_4h)
            four_h_opposing = ("BULL" in trend_4h)
            one_h_opposing = ("BULL" in trend_1h)
            allow_counter_1h = is_high_prob or four_h_aligned or has_sweep_turn or has_flow_turn
            if four_h_opposing and not (is_high_prob and (has_sweep_turn or has_flow_turn)):
                return False, f"MACRO SHIELD: {sym} 4H is Bullish ({trend_4h}) — Shorts require Grade A+ sweep/flow turn"
            if one_h_opposing and not allow_counter_1h:
                return False, f"MTF SHIELD: {sym} 1H is Bullish ({trend_1h}) without Sweep/OFI turn confirmation — Shorts paused"
        elif dir_upper == "LONG":
            four_h_aligned = ("BULL" in trend_4h)
            four_h_opposing = ("BEAR" in trend_4h)
            one_h_opposing = ("BEAR" in trend_1h)
            allow_counter_1h = is_high_prob or four_h_aligned or has_sweep_turn or has_flow_turn
            if four_h_opposing and not (is_high_prob and (has_sweep_turn or has_flow_turn)):
                return False, f"MACRO SHIELD: {sym} 4H is Bearish ({trend_4h}) — Longs require Grade A+ sweep/flow turn"
            if one_h_opposing and not allow_counter_1h:
                return False, f"MTF SHIELD: {sym} 1H is Bearish ({trend_1h}) without Sweep/OFI turn confirmation — Longs paused"

    return True, "CLEAR"

def can_trade(is_high_prob: bool = False) -> tuple[bool, str]:
    """Check portfolio-wide account gates (is trading globally permitted?)."""
    now = time.time()
    shielded, shield_reason = is_rollover_or_maintenance()
    if shielded:
        return False, f"SPREAD SHIELD: {shield_reason}"

    if PROP_FIRM_MODE and prop_firm_manager:
        pf_ok, pf_reason = prop_firm_manager.can_trade(is_grade_a_plus=is_high_prob)
        if not pf_ok:
            return False, f"PROP_FIRM: {pf_reason}"

    if account["halted"]:
        if account["halt_until"] <= now:
            account["halted"] = False; account["halt_reason"] = ""
        else:
            rem = int(account["halt_until"] - now) if account["halt_until"] < float("inf") else 999999
            return False, f"HALTED: {account['halt_reason']} ({rem//60}m left)"

    if account["trades_today"] >= MAX_TRADES_DAY:
        if not is_high_prob:
            return False, f"DYNAMIC TRADE CAP: Trade #{account['trades_today']+1} requires High-Probability Grade A+ setup"
        hard_cap = 8 if session.get("name") in ("LONDON", "NEW_YORK") else 6
        if account["trades_today"] >= hard_cap:
            return False, f"HARD DAILY TRADES LIMIT ({hard_cap}/{hard_cap} trades reached)"

    consec = account.get("consecutive_losses", 0)
    cooldown_secs = LOSS_COOLDOWN_1 if consec <= 1 else (LOSS_COOLDOWN_2 if consec == 2 else LOSS_COOLDOWN_3)
    if account["last_loss_time"] > 0 and (now - account["last_loss_time"]) < cooldown_secs:
        if is_high_prob:
            logger.info("loss_cooldown_bypassed_by_grade_a_plus", mode="account_global")
        else:
            rem = int(cooldown_secs - (now - account["last_loss_time"]))
            return False, f"LOSS COOLDOWN {rem//60}m{rem%60:02d}s"

    return True, "CLEAR"

# ─── STRUCTURAL SL (DUAL HORIZON: SCALP vs SWING) ────────────────────────────
def find_sl(direction: str, entry: float, symbol: str = SYMBOL, trade_mode: str = "SCALP") -> tuple[float, float]:
    sym = symbol.upper()
    sym_reg = symbol_regime_info.get(sym, regime_info)
    atr = max(sym_reg.get("atr", 10.0), 1.0)
    is_swing = (trade_mode == "SWING")

    # Asset & Horizon Adaptive ATR bounds (prevents 4-5 ATR stops that make TP unreachable)
    if "ETH" in sym:
        if is_swing:
            min_sl = max(18.0, atr * 2.5)
            max_sl = min(40.0, atr * 4.2)
        else:
            min_sl = max(7.0, atr * 1.3)
            max_sl = min(18.0, atr * 2.1)
    elif "GOLD" in sym or "XAU" in sym or "PAXG" in sym:
        if is_swing:
            min_sl = max(8.0, atr * 2.5)
            max_sl = min(18.0, atr * 4.2)
        else:
            min_sl = max(4.5, atr * 1.3)
            max_sl = min(11.0, atr * 2.1)
    else:  # BTC
        if is_swing:
            min_sl = max(320.0, atr * 2.5)
            max_sl = min(600.0, atr * 4.2)
        else:
            min_sl = max(120.0, atr * 1.2)
            max_sl = min(260.0, atr * 2.0)

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
def size_position(sl_dist: float, prob: float, entry: float, council_score: float = 0.7, direction: str = "LONG", symbol: str = SYMBOL, trade_mode: str = "SCALP", potential_rr: float = 2.5) -> tuple[float, float, float]:
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
            trade_mode=trade_mode,
            potential_rr=potential_rr,
        )
        notional = round(lots * entry, 2)
        return lots, risk_usd, notional

    # ── STANDARD BALANCE SIZING (Personal Account / Demo) ──
    is_high_potential = (trade_mode.upper() == "SWING" or potential_rr >= 3.0 or (prob >= STRONG_PROB and council_score >= 0.75))
    if is_high_potential:
        risk_pct = MAX_RISK_PCT
    elif prob >= MIN_PROB + 0.04 or council_score >= 0.70:
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
    return lots, actual_risk, notional

# ─── INSTITUTIONAL RETEST LIMIT QUEUE & ENTRY EXECUTION ──────────────────────
def check_pending_retests(sym: str, price: float):
    """Evaluate queued fair-value pullback limit zones on every incoming tick."""
    z = pending_retest_zones.get(sym)
    if not z or price <= 0:
        return

    now = time.time()
    direction = z["direction"]
    if now > z["expires_ts"]:
        pending_retest_zones.pop(sym, None)
        trade_log.append(f"[{sym}] RETEST EXPIRED: Missed pullback zone ${z['retest_px']:,.2f} — stood aside safely")
        return

    # Invalidation fence protects against fakeouts before fill
    if (direction == "LONG" and price <= z["inv_px"]) or (direction == "SHORT" and price >= z["inv_px"]):
        pending_retest_zones.pop(sym, None)
        trade_log.append(f"[{sym}] RETEST INVALIDATED: Structure broke ${z['inv_px']:,.2f} before entry ($0 loss)")
        return

    touched = (direction == "LONG" and price <= z["retest_px"]) or (direction == "SHORT" and price >= z["retest_px"])
    if touched:
        pending_retest_zones.pop(sym, None)
        sig = z["signal"]
        is_hp = _is_grade_a_plus(sig.conviction, sig.council_score)
        ok, _ = can_trade_symbol(sym, direction, is_high_prob=is_hp)
        if ok and portfolios.get(sym, portfolio)["position"] == "FLAT":
            trade_log.append(f"[{sym}] RETEST FILLED @ ${price:,.2f} (Zone: ${z['retest_px']:,.2f})")
            execute_signal(sig, symbol=sym, bypass_retest=True)

def execute_signal(signal: TradingSignal, symbol: str = SYMBOL, bypass_retest: bool = False):
    """Execute a TradingSignal from the LiveSignalEngine for a specific symbol."""
    global last_signal
    sym = symbol.upper()
    port = portfolios.get(sym, portfolio)
    mkt = symbol_markets.get(sym, market)

    if port["position"] != "FLAT": return

    # ── DOUBLE-LOCK BROKER PROTECTION: Never stack or open duplicate position if broker already holds one ──
    if DEMO_MODE and _active_broker is not None:
        try:
            if hasattr(_active_broker, "_get_open_position_sync"):
                existing = _active_broker._get_open_position_sync(symbol=sym)
                if existing:
                    # Self-heal local portfolio to match active broker position
                    port["position"] = "OPEN"
                    port["direction"] = existing.get("type", "SHORT")
                    port["broker_ticket"] = str(existing.get("ticket", ""))
                    if port.get("entry_price", 0.0) <= 0:
                        port["entry_price"] = float(existing.get("price_open", 0.0))
                        port["sl_price"] = float(existing.get("sl", 0.0))
                    log_msg = f"[{sym}] DUPLICATE PREVENTED: Broker already holds active position (Ticket #{existing.get('ticket')})"
                    trade_log.append(log_msg)
                    if HEADLESS: print(f"  🛑 {log_msg}", flush=True)
                    return
        except Exception:
            pass

    direction = "LONG" if signal.direction == "BUY" else "SHORT"
    prob      = signal.conviction
    council   = signal.council_score
    is_high_prob = _is_grade_a_plus(prob, council)
    sig_desc = f"[{sym}] {direction} p={prob:.3f}"

    ok, reason = can_trade_symbol(sym, direction, is_high_prob=is_high_prob)
    if not ok:
        _record_gate_rejection(sym, sig_desc, f"[{sym}] {reason}")
        return

    # ── PROP FIRM DYNAMIC QUALITY GATE ──
    trades_done = account.get("trades_today", 0)
    if PROP_FIRM_MODE and trades_done >= MAX_TRADES_DAY and not is_high_prob:
        log_msg = f"[{sym}] SNIPER QUALITY GATE: Signal ({prob:.3f} / {council:.2f}) below Grade A+ threshold for Trade #{trades_done+1}"
        trade_log.append(log_msg)
        _record_gate_rejection(sym, sig_desc, log_msg)
        if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
        return

    entry = mkt["ask"] if direction == "LONG" else mkt["bid"]
    if entry <= 0: entry = mkt.get("ltp", 0.0)
    if entry <= 0: return

    # ── NO-CHASE GATE ──
    # Prohibit entering only if market price slipped > 0.50 ATR from trigger (avoids spread/tick noise false-skips)
    sym_atr = symbol_regime_info.get(sym, {}).get("atr", 0.0)
    sig_price = getattr(signal, "entry_price", 0.0) or getattr(signal, "price", 0.0)
    if sym_atr > 0 and sig_price > 0:
        chase_dist = (entry - sig_price) if direction == "LONG" else (sig_price - entry)
        max_chase = 0.50 * sym_atr
        if chase_dist > max_chase:
            log_msg = (
                f"[{sym}] NO-CHASE GATE: Market price ${entry:,.2f} chased +${chase_dist:.2f} "
                f"(> 0.50 ATR=${max_chase:.2f}) from trigger ${sig_price:,.2f} — Entry cancelled"
            )
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return

    # ── 5m V-REVERSAL, EXHAUSTION & EXTREME OBI COUNTER-FLOW GUARD ──
    c5_list = list(symbol_candles_5m.get(sym, []))
    if len(c5_list) >= 20:
        last_c = c5_list[-1]
        prev_c = c5_list[-2]
        lc_o = float(last_c["open"])
        lc_c = float(last_c["close"])
        pc_h = float(prev_c["high"])
        pc_l = float(prev_c["low"])
        is_last_green = lc_c > lc_o
        is_last_red = lc_c < lc_o

        # 1. Major opposing 5m V-reversal engulfing check (0.60 ATR for standard, 0.85 ATR for Grade A+)
        rev_mult = 0.85 if is_high_prob else 0.60
        if direction == "SHORT" and is_last_green and (lc_c - lc_o) > rev_mult * max(sym_atr, 1.0) and lc_c > pc_h:
            log_msg = f"[{sym}] ANTI-REVERSAL GUARD: Cancelled SHORT into strong impulsive green 5m reversal (O:{lc_o:,.2f} -> C:{lc_c:,.2f})"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return
        if direction == "LONG" and is_last_red and (lc_o - lc_c) > rev_mult * max(sym_atr, 1.0) and lc_c < pc_l:
            log_msg = f"[{sym}] ANTI-REVERSAL GUARD: Cancelled LONG into strong impulsive red 5m selloff (O:{lc_o:,.2f} -> C:{lc_c:,.2f})"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return

        # 2. Extreme Rubber-Band Exhaustion check (> 2.8 ATR stretch away from 5m EMA20)
        closes_20 = [float(x["close"]) for x in c5_list[-20:]]
        ema20 = closes_20[0]
        k = 2.0 / 21.0
        for val in closes_20[1:]:
            ema20 = (val - ema20) * k + ema20
        if sym_atr > 0:
            if direction == "SHORT" and (ema20 - entry) > (2.8 * sym_atr):
                log_msg = f"[{sym}] EXHAUSTION GUARD: Cancelled SHORT {ema20 - entry:.1f} pts (> 2.8 ATR) oversold below 5m EMA20"
                trade_log.append(log_msg)
                _record_gate_rejection(sym, sig_desc, log_msg)
                if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
                return
            if direction == "LONG" and (entry - ema20) > (2.8 * sym_atr):
                log_msg = f"[{sym}] EXHAUSTION GUARD: Cancelled LONG {entry - ema20:.1f} pts (> 2.8 ATR) overbought above 5m EMA20"
                trade_log.append(log_msg)
                _record_gate_rejection(sym, sig_desc, log_msg)
                if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
                return

    # 3. Extreme L2 OBI Counter-Flow Guard (only blocks extreme opposing walls > 0.55 on non-A+ setups)
    cur_obi = float(mkt.get("obi", 0.0))
    if not is_high_prob:
        if direction == "SHORT" and cur_obi > 0.55:
            log_msg = f"[{sym}] OBI COUNTER-FLOW GUARD: Cancelled SHORT into extreme buy wall (OBI: {cur_obi:+.3f})"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return
        if direction == "LONG" and cur_obi < -0.55:
            log_msg = f"[{sym}] OBI COUNTER-FLOW GUARD: Cancelled LONG into extreme sell wall (OBI: {cur_obi:+.3f})"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return

    # ── ANTI-TRAP GUARD: Dual Advisor Veto ──
    v_dict = getattr(signal, "council_votes", None) or getattr(signal, "metadata", {}).get("council_votes", {})
    if v_dict:
        mom_vote = v_dict.get("Momentum", {}).get("vote")
        vol_vote = v_dict.get("Volume", {}).get("vote")
        if mom_vote == "REJECT" and vol_vote == "REJECT":
            log_msg = f"[{sym}] ANTI-TRAP VETO: Momentum & Volume both voted REJECT opposing {direction}"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return

    # ── INSTITUTIONAL FAIR-VALUE RETEST QUEUE ──
    # Grade A+ and Breakout signals execute immediately at market unless extreme chase (>1.50 ATR);
    # standard lower-conviction signals wait for pullback only when stretched (>1.35 ATR or >1.25 ATR candle).
    ext_atr, _, _ = compute_auction_location(sym, direction, entry, max(sym_atr, 1e-6))
    lev_snap = liquidity_engine.get_levels(sym) if "liquidity_engine" in globals() else {}
    has_sweep_or_brk = (
        (direction == "LONG" and (lev_snap.get("sweep_bullish") or lev_snap.get("breakout_bullish")))
        or (direction == "SHORT" and (lev_snap.get("sweep_bearish") or lev_snap.get("breakout_bearish")))
    )
    if not bypass_retest and not is_high_prob and not has_sweep_or_brk and sym_atr > 0 and len(c5_list) >= 20:
        lc = c5_list[-1]
        lc_o, lc_c = float(lc["open"]), float(lc["close"])
        bar_move = (lc_c - lc_o) if direction == "LONG" else (lc_o - lc_c)
        if bar_move > (1.25 * sym_atr) or ext_atr > 1.35:
            mid_bar = (lc_o + lc_c) / 2.0
            if direction == "LONG":
                retest_px = round(min(mid_bar, entry - 0.25 * sym_atr), 2)
                inv_px = round(float(lc["low"]) - 0.35 * sym_atr, 2)
            else:
                retest_px = round(max(mid_bar, entry + 0.25 * sym_atr), 2)
                inv_px = round(float(lc["high"]) + 0.35 * sym_atr, 2)

            pending_retest_zones[sym] = {
                "signal": signal,
                "direction": direction,
                "retest_px": retest_px,
                "inv_px": inv_px,
                "expires_ts": time.time() + 600.0,
            }
            log_msg = f"[{sym}] RETEST LIMIT QUEUED ({direction}): Wait pullback to ${retest_px:,.2f} (from ${entry:,.2f}, inv ${inv_px:,.2f})"
            trade_log.append(log_msg)
            _record_gate_rejection(sym, sig_desc, log_msg)
            if HEADLESS: print(f"  ⏳ {log_msg}", flush=True)
            return

    # ── DUAL-HORIZON TRADE CLASSIFICATION: SWING vs PRECISION SCALP ──
    macro = compute_macro_structure(sym)
    is_ranging = "ranging" in str(signal.regime).lower()

    # Swing only when high-conviction AND near equilibrium (never swing-tag late extended moves)
    is_swing = False
    if not is_ranging and prob >= 0.65 and council >= 0.72 and ext_atr <= 1.2:
        if direction == "LONG" and macro.get("swing_eligible_long", False):
            is_swing = True
        elif direction == "SHORT" and macro.get("swing_eligible_short", False):
            is_swing = True

    trade_mode = "SWING" if is_swing else "SCALP"

    sl_price, sl_dist = find_sl(direction, entry, symbol=sym, trade_mode=trade_mode)
    if sl_dist <= 0: return

    # ── SPREAD-TO-RISK GATE (Adapted from xau_c / breakout_admission.py) ──
    bid = mkt.get("bid", 0.0)
    ask = mkt.get("ask", 0.0)
    if bid > 0 and ask > 0 and sl_dist > 0:
        spread_usd = abs(ask - bid)
        spread_share = spread_usd / sl_dist
        if spread_share > 0.10:
            log_msg = (
                f"[{sym}] SPREAD-TO-RISK GATE: Spread (${spread_usd:.2f}) consumes {spread_share*100:.1f}% "
                f"of stop distance (${sl_dist:.2f}), exceeding 10% cap — Entry cancelled"
            )
            trade_log.append(log_msg)
            if HEADLESS: print(f"  🛡️ {log_msg}", flush=True)
            return

    # ── MARKET-ANCHORED TARGET ARCHITECTURE (Realistic ATR & Liquidity Targets) ──
    eff_atr = sym_atr if sym_atr > 0 else sl_dist
    if trade_mode == "SWING":
        tp1_dist = min(sl_dist * 1.6, eff_atr * 2.8)
        tp2_dist = min(sl_dist * 3.2, eff_atr * 5.5)
        tp1_dist = max(tp1_dist, sl_dist * 1.3)
        tp2_dist = max(tp2_dist, sl_dist * 2.4)
        max_duration_min = 2880.0  # 48h
        adverse_decay_min = 480.0  # 8h adverse hold
    else:
        max_duration_min = 360.0   # 6h
        adverse_decay_min = 240.0  # 4h
        base_tp1_r = 1.25 if is_ranging else 1.4
        base_tp2_r = 2.2 if is_ranging else 2.5
        tp1_dist = max(sl_dist * 1.15, min(sl_dist * base_tp1_r, eff_atr * 1.8))
        tp2_dist = max(sl_dist * 1.9, min(sl_dist * base_tp2_r, eff_atr * 3.5))

    tp1 = round(entry + tp1_dist if direction == "LONG" else entry - tp1_dist, 2)
    tp2 = round(entry + tp2_dist if direction == "LONG" else entry - tp2_dist, 2)

    # Snap TP2 to nearest genuine Volume Profile / Liquidity Pool within achievable reach
    try:
        from src.features.indicators.volume_profile import compute_volume_profile_snapshot
        c_buf = symbol_candles_5m.get(sym, candles_5m)
        vp = compute_volume_profile_snapshot(pd.DataFrame(list(c_buf)[-100:]))
        candidates = []
        if vp["profile_ready"]:
            candidates.extend([vp["poc"], vp["vah"]] + vp["hvn_prices"])
        for lvl_k in ("pdh", "pdl", "ash", "asl"):
            lv = float(lev_snap.get(lvl_k, 0.0))
            if lv > 0:
                candidates.append(lv)

        min_reach = sl_dist * 1.85
        max_reach = eff_atr * (5.5 if trade_mode == "SWING" else 3.6)
        if direction == "LONG":
            valid_tps = [p for p in candidates if entry + min_reach <= p <= entry + max_reach]
            if valid_tps:
                tp2 = round(min(valid_tps), 2)
        else:
            valid_tps = [p for p in candidates if entry - max_reach <= p <= entry - min_reach]
            if valid_tps:
                tp2 = round(max(valid_tps), 2)
    except Exception:
        pass

    tp1_rr = round(abs(tp1 - entry) / max(sl_dist, 1e-6), 2)
    tp2_rr = round(abs(tp2 - entry) / max(sl_dist, 1e-6), 2)

    entry_lots, risk_usd, notional = size_position(
        sl_dist, prob, entry, council, direction,
        symbol=sym, trade_mode=trade_mode, potential_rr=tp2_rr
    )
    if entry_lots <= 0 or notional <= 0: return

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
    symbol_gate_rejections[sym] = {"symbol": sym, "reason": "", "signal": "", "ts": 0.0}
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
                    try:
                        if hasattr(_active_broker, "get_open_position"):
                            pos_info = await _active_broker.get_open_position(ticket=int(oid))
                            if pos_info and float(pos_info.get("price_open", 0.0)) > 0:
                                b_entry = float(pos_info["price_open"])
                                port["broker_entry_price"] = b_entry
                                port["broker_offset"] = round(b_entry - entry, 4)
                    except Exception:
                        pass
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
        t_id = port.get("broker_ticket")
        broker_pnl_val = float(port.get("broker_pnl", 0.0))

        # 1. PRIORITY: If an active MT5 broker ticket exists, ground truth is the broker profit!
        if t_id and (broker_pnl_val != 0.0 or DEMO_MODE):
            if broker_pnl_val != 0.0:
                net_usd = broker_pnl_val
            else:
                raw_diff = (price - entry) if direction == "LONG" else (entry - price)
                net_usd = (raw_diff / entry) * port.get("notional", 1000.0)
        else:
            # Paper mode: calculate net after slippage/fees
            if direction == "LONG":
                nbps = ((price-entry)/entry * 10000.0) - ROUNDTRIP_BPS
            else:
                nbps = ((entry-price)/entry * 10000.0) - ROUNDTRIP_BPS

            net_usd = (nbps/10000.0) * port["notional"]

        total   = round(port["banked_pnl"] + net_usd, 2)
        is_be   = abs(total) < 0.50
        is_win  = total >= 0.50

        account["balance"] += net_usd
        update_health()

        sym_eng = signal_engines.get(sym, signal_engine)
        if is_be:
            account["trades_today"] = max(0, account.get("trades_today", 0) - 1)
            outcome_str = "BE"
        elif is_win:
            account["wins"]             += 1
            account["consecutive_losses"] = 0
            account["total_wins_usd"]   += total
            outcome_str = "WIN"
        else:
            account["losses"]           += 1
            account["consecutive_losses"] += 1
            account["total_losses_usd"] += abs(total)
            account["last_loss_time"]    = time.time()
            outcome_str = "LOSS"
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

        if not is_be:
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
                          "outcome": outcome_str,
                          "exit_reason": reason, "dur_min": round(dur_min,1)})
                break
        save_journal()

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
                     "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0,
                     "broker_ticket": "", "broker_pnl": 0.0,
                     "broker_entry_price": 0.0, "broker_offset": 0.0})
        if sym == PRIMARY_SYMBOL:
            portfolio.update(port)
        state_manager.save_state(portfolio, account)   # APEX v7.2: crash-resume

    def _schedule_broker_sync(ticket_id, binance_sl: float, binance_tp: float | None = None, part_vol: float = 0.0, err_label: str = ""):
        """Push trailing/BE stop updates (and optional partial close) offset to broker's actual fill quote."""
        if not (DEMO_MODE and _active_broker is not None and ticket_id):
            return
        offset = float(port.get("broker_offset", 0.0))
        sl_adj = round(binance_sl + offset, 2)
        tp_adj = round(binance_tp + offset, 2) if binance_tp is not None else None

        async def _run_sync():
            try:
                if part_vol >= 0.01 and hasattr(_active_broker, "partial_close_position"):
                    await _active_broker.partial_close_position(int(ticket_id), volume=part_vol)
                if hasattr(_active_broker, "modify_position_stops"):
                    if tp_adj is not None:
                        await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_adj, new_tp=tp_adj)
                    else:
                        await _active_broker.modify_position_stops(int(ticket_id), new_sl=sl_adj)
            except Exception as _be_err:
                if err_label:
                    trade_log.append(f"[{BROKER_TYPE.upper()}] {sym} {err_label}: {_be_err}")

        try:
            import asyncio as _aio
            _aio.get_event_loop().create_task(_run_sync())
        except Exception:
            pass

    sl = port["sl_price"]
    tp1 = port["tp1_price"]
    tp2 = port["tp2_price"]
    sym_reg = symbol_regime_info.get(sym, regime_info)
    atr = max(sym_reg.get("atr", 10.0), 1.0)
    sl_dist = max(port.get("sl_distance", 1.0), 1.0)
    fav_dist = (cur - entry) if direction == "LONG" else (entry - cur)
    r_mult = fav_dist / sl_dist
    t_id = port.get("broker_ticket")
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")

    # Anti-Tick Scalping: calculate seconds elapsed in trade to respect prop firm minimum duration rules
    min_dur_sec = prop_firm_manager.get_min_trade_duration_sec() if (PROP_FIRM_MODE and prop_firm_manager) else 45
    entry_ts = float(port.get("entry_time", 0.0))
    time_in_trade_sec = (time.time() - entry_ts) if entry_ts > 0 else (dur_min * 60.0)
    is_past_min_hold = (time_in_trade_sec >= min_dur_sec)

    # ── FRIDAY WEEKEND AUTO-FLATTEN (Commodities & Non-Crypto Protection) ──
    now_dt = datetime.datetime.now(datetime.timezone.utc)
    if ("GOLD" in sym or "PAXG" in sym) and now_dt.weekday() == 4 and (now_dt.hour > 20 or (now_dt.hour == 20 and now_dt.minute >= 45)):
        close(cur, "FRIDAY WEEKEND AUTO-FLATTEN (Metals market closes 22:00 UTC / Gap shield)")
        return

    # ── PRE-NEWS EVENT RISK SHIELD (High-Impact FOMC / CPI / NFP Protection) ──
    if news_guard:
        try:
            nxt_ev = news_guard.get_next_event()
            if nxt_ev and nxt_ev.get("mins_until", 999) <= 3.0:
                ev_name = nxt_ev.get("event", "HIGH-IMPACT NEWS")
                m_left = nxt_ev.get("mins_until", 0.0)
                news_allowed = prop_firm_manager.is_news_trading_allowed() if (PROP_FIRM_MODE and prop_firm_manager) else True
                if not news_allowed or r_mult >= 0.5:
                    if is_past_min_hold:
                        close(cur, f"PRE-NEWS RISK SHIELD ({ev_name} in {m_left:.1f}m)")
                        return
                elif r_mult >= 0.1 and port.get("risk_tier", 0) < 2:
                    fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
                    new_sl = entry + fee_buf if direction == "LONG" else entry - fee_buf
                    port["sl_price"] = round(new_sl, 1)
                    port["risk_tier"] = 2
                    _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"])
        except Exception:
            pass

    # 1. SL (Delegated to broker matching engine when active; otherwise managed in memory)
    if not (DEMO_MODE and port.get("broker_ticket")):
        if direction == "LONG" and cur <= sl: close(cur, "STOP LOSS"); return
        if direction == "SHORT" and cur >= sl: close(cur, "STOP LOSS"); return

    # ── DYNAMIC TIER 1: Risk Reduction at +0.90R (Tighten SL to -0.35R with breathing room for pullbacks) ──
    if port.get("risk_tier", 0) < 1 and r_mult >= 0.90:
        new_sl = entry - sl_dist * 0.35 if direction == "LONG" else entry + sl_dist * 0.35
        port["sl_price"] = round(new_sl, 1)
        port["risk_tier"] = 1
        log_msg = f"[{now_str}] {sym} TIER 1 (+0.90R): Risk reduced 65% | SL tightened to -0.35R @${new_sl:,.1f}"
        trade_log.append(log_msg)
        if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
        send_challenge_alert("RISK REDUCED 65%", {
            "symbol": sym, "action": direction, "price": cur, "new_sl": port["sl_price"],
        })
        _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"])

    # ── DYNAMIC TIER 2: Break-Even Lock at +1.15R ──
    if port.get("risk_tier", 0) < 2 and r_mult >= 1.15:
        fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
        new_sl = entry + fee_buf if direction == "LONG" else entry - fee_buf
        port["sl_price"] = round(new_sl, 1)
        port["risk_tier"] = 2
        log_msg = f"[{now_str}] {sym} TIER 2 (+1.15R): SL moved to Break-Even + fees @${new_sl:,.1f} | RISK FREE"
        trade_log.append(log_msg)
        if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
        send_challenge_alert("BREAK-EVEN LOCKED", {
            "symbol": sym, "action": direction, "price": cur, "new_sl": port["sl_price"],
        })
        _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"])

    # ── DYNAMIC EARLY PROFIT HARVEST: Runner-Preserving OFI Reversal Protection ──
    if r_mult >= 0.85 and is_past_min_hold:
        try:
            m_eng = momentum_engines.get(sym, momentum_eng)
            mom_d = m_eng.get_diagnostics()
            ofi_v = float(mom_d.get("ofi", 0.0))
            is_opposing = (direction == "LONG" and ofi_v <= -0.75) or (direction == "SHORT" and ofi_v >= 0.75)
            if is_opposing:
                if port.get("phase", 1) == 1:
                    # Bank 40% partial profit, ratchet SL to Break-Even, and KEEP 60% RUNNER OPEN!
                    part_frac = 0.40
                    partial = port["unrealized_pnl"] * part_frac
                    port["banked_pnl"] += partial
                    account["balance"] += partial
                    update_health()
                    port["notional"] *= (1.0 - part_frac)
                    fee_buf = entry * (ROUNDTRIP_BPS * 1.5 / 10000.0)
                    port["sl_price"] = entry + fee_buf if direction == "LONG" else entry - fee_buf
                    port["phase"] = 2
                    port["risk_tier"] = 3
                    if PROP_FIRM_MODE and prop_firm_manager:
                        prop_firm_manager.on_trade_closed(partial, f"OFI HARVEST PARTIAL (+{r_mult:.1f}R)")

                    log_msg = f"[{now_str}] {sym} OFI REVERSAL (+{r_mult:.1f}R): Banked 40% (+${partial:.2f}), SL→BE, RUNNER KEPT OPEN FOR TREND"
                    trade_log.append(log_msg)
                    if HEADLESS: print(f"  ⚡ {log_msg}", flush=True)
                    send_challenge_alert("OFI HARVEST (RUNNER OPEN)", {
                        "symbol": sym, "action": direction, "price": cur, "pnl_usd": partial,
                        "reason": f"Banked 40% at +{r_mult:.1f}R, 60% runner trailing to TP2",
                    })

                    # Execute physical partial close on MT5 broker
                    if DEMO_MODE and _active_broker is not None and t_id:
                        cur_vol = port.get("entry_lots", 0.0)
                        part_vol = round(cur_vol * part_frac, 2)
                        min_step = 0.02 if "ETH" in sym else 0.01
                        part_vol = max(min_step, part_vol)
                        _schedule_broker_sync(t_id, port["sl_price"], part_vol=part_vol)
                elif port.get("phase", 1) == 2 and r_mult >= 2.5:
                    # In Phase 2, only close runner if it is already past +2.5R
                    close(cur, f"OFI RUNNER HARVEST (OFI {ofi_v:+.2f} at +{r_mult:.1f}R)")
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
                _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"], part_vol=part_vol, err_label="TP1 sync err")

    # ── DYNAMIC TIER 4: Dynamic Runner Chandelier Trail + Full TP2 Target Exit ──
    if port["phase"] == 2:
        if is_swing:
            trail_buf = max(atr * 3.5, 25.0 if "ETH" in sym else (10.0 if "GOLD" in sym else 400.0))
        else:
            trail_buf = max(atr * 2.0, 10.0 if "ETH" in sym else (5.0 if "GOLD" in sym else 160.0))

        ratchet_step = 2.0 if "ETH" in sym else 25.0
        if direction == "LONG":
            new_sl = cur - trail_buf
            if new_sl > port["sl_price"]:
                old_sl = port["sl_price"]
                port["sl_price"] = round(new_sl, 1)
                if (new_sl - old_sl) >= ratchet_step:
                    _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"])
            if cur >= tp2: close(cur, f"FULL TP2 ({port.get('tp2_rr', 3.5):.1f}R [{port.get('trade_mode','SCALP')}])"); return
        else:
            new_sl = cur + trail_buf
            if new_sl < port["sl_price"]:
                old_sl = port["sl_price"]
                port["sl_price"] = round(new_sl, 1)
                if (old_sl - new_sl) >= ratchet_step:
                    _schedule_broker_sync(t_id, port["sl_price"], port["tp2_price"])
            if cur <= tp2: close(cur, f"FULL TP2 ({port.get('tp2_rr', 3.5):.1f}R [{port.get('trade_mode','SCALP')}])"); return

    # 4. Time decay & Structural 1H Reversal Invalidation
    max_dur = port.get("max_duration_min", 360.0)
    adv_dur = port.get("adverse_decay_min", 240.0)
    if dur_min > max_dur:
        close(cur, f"MAX DURATION {max_dur/60:.0f}h ({port.get('trade_mode', 'SCALP')})")
        return
    if dur_min > adv_dur and port["phase"] == 1 and port["unrealized_pnl"] < 0:
        close(cur, f"TIME DECAY {adv_dur/60:.0f}h adverse ({port.get('trade_mode', 'SCALP')})")
        return

    # 4b. Early Structural Invalidation (cuts losses at -0.75R instead of waiting for -1.0R hard SL when 1H flips against us)
    if dur_min >= 45.0 and port.get("phase", 1) == 1 and r_mult <= -0.75:
        try:
            macro_t = symbol_macro_structure.get(sym, {})
            t_1h = str(macro_t.get("trend_1h", "NEUTRAL")).upper()
            m_eng = momentum_engines.get(sym, momentum_eng)
            ofi_v = float(m_eng.get_diagnostics().get("ofi", 0.0)) if m_eng else 0.0
            if (direction == "SHORT" and "BULL" in t_1h and ofi_v > 0.25) or (direction == "LONG" and "BEAR" in t_1h and ofi_v < -0.25):
                close(cur, f"STRUCTURAL INVALIDATION (1H={t_1h}, OFI={ofi_v:+.2f} at {r_mult:.2f}R)")
                return
        except Exception:
            pass

    # 5. Model-halt exit
    sym_eng = signal_engines.get(sym, signal_engine)
    if sym_eng and sym_eng.is_halted and port["unrealized_pnl"] > 0:
        close(cur, "ENGINE HALTED — locking profits"); return

def _sync_live_probs(sym: str, eng: LiveSignalEngine) -> None:
    """Sync cached L/S ensemble probabilities from LiveSignalEngine without re-appending live_store."""
    try:
        if not eng:
            return
        pl = float(getattr(eng, "_last_prob_long", 0.5))
        ps = float(getattr(eng, "_last_prob_short", 0.5))
        r_val = str(getattr(eng, "_last_regime", "") or (eng.regime_detector.current_regime.value if hasattr(eng, "regime_detector") else "UNKNOWN"))
        if sym in symbol_live_probs:
            symbol_live_probs[sym]["long"]    = round(pl, 4)
            symbol_live_probs[sym]["short"]   = round(ps, 4)
            symbol_live_probs[sym]["updated"] = True
            symbol_live_probs[sym]["regime"]  = r_val
        if sym == PRIMARY_SYMBOL:
            live_probs["long"]    = round(pl, 4)
            live_probs["short"]   = round(ps, 4)
            live_probs["updated"] = True
            regime_info["current"] = r_val
    except Exception:
        pass

# ─── CANDLE PIPELINE ──────────────────────────────────────────────────────────
async def _candle_worker():
    """Async worker: drains _candle_queue and processes each 5m candle through the symbol's ML engine."""
    global last_signal
    while True:
        try:
            item = await _candle_queue.get()
            try:
                if isinstance(item, tuple):
                    sym, c5 = item
                else:
                    sym, c5 = PRIMARY_SYMBOL, item

                eng = signal_engines.get(sym, signal_engine)
                if not eng:
                    continue

                sig = await eng.process_candle(c5)

                # APEX v7.2: Feed candle to multi-asset retrain scheduler (non-blocking)
                r_sched = retrain_schedulers.get(sym, retrain_scheduler if sym == PRIMARY_SYMBOL else None)
                if r_sched:
                    try:
                        r_sched.on_candle(c5)
                        r_sched.check_pending_swap()   # pick up new models if ready
                    except Exception:
                        pass

                # Update Institutional Liquidity Levels (PDH/PDL, Asian Range, FVG, Sweeps)
                try:
                    c1d = symbol_candles_1d.get(sym, [])
                    c1h = symbol_candles_1h.get(sym, [])
                    c5m = symbol_candles_5m.get(sym, [])
                    liquidity_engine.update_levels(sym, candles_1d=c1d, candles_1h=c1h, candles_5m=c5m)
                except Exception:
                    pass

                # Update Cross-Asset Lead-Lag & Spillover Diagnostics
                try:
                    btc_ofi = 0.0
                    if "BTCUSDT" in momentum_engines:
                        btc_ofi = float(momentum_engines["BTCUSDT"].get_diagnostics().get("ofi", 0.0))
                    lead_lag_engine.compute_spillover(dxy_trend=cross_market_monitor.dxy_trend, btc_ofi=btc_ofi)
                except Exception:
                    pass

                # Always sync live probabilities published by eng.process_candle() (never re-call live_store.update)
                _sync_live_probs(sym, eng)

                # Strictly enforce ML + 10-Advisor Council Authority: Never fabricate blind bypass signals when sig is None
                # Institutional liquidity sweeps/breakouts boost conviction cleanly via evaluate_signal_synergy below.

                if sig is not None:
                    last_signal = sig
                    if sym == PRIMARY_SYMBOL:
                        regime_info["current"] = sig.regime
                        regime_info["atr"]     = sig.atr

                    compute_macro_structure(sym)
                    direction = "LONG" if sig.direction == "BUY" else "SHORT"

                    # ── Apply Institutional Market Intelligence Synergy ──
                    liq_delta, liq_reason = liquidity_engine.evaluate_signal_synergy(sym, direction)
                    ll_delta, ll_reason = lead_lag_engine.evaluate_synergy(sym, direction)
                    
                    # Microstructure CVD & L2 Order Book Imbalance (OBI)
                    micro_delta, micro_reason = (0.0, "")
                    if sym in microstructure_engines:
                        sym_mkt = symbol_markets.get(sym, market)
                        sym_obi = sym_mkt.get("obi", 0.0)
                        # Phase 3: Institutional Delta Exchange L2 Book Imbalance for Gold
                        if any(g in sym for g in ("GOLD", "PAXG")):
                            try:
                                from src.data.providers.delta_feed import delta_gold_feed
                                d_tel = delta_gold_feed.get_telemetry()
                                if d_tel.get("fresh"):
                                    sym_obi = d_tel["book_imbalance_5"]
                                    sym_mkt["obi"] = sym_obi
                                    sym_mkt["delta_aggressor"] = d_tel["aggressor_ratio"]
                            except Exception:
                                pass
                        micro_delta, micro_reason = microstructure_engines[sym].evaluate(direction, obi=sym_obi)

                    # Extreme Funding Squeeze & Liquidation Cascade Alpha
                    fund_pct = funding_monitor.get_funding_pct(sym)
                    fund_delta, fund_reason = (0.0, "")
                    if direction == "LONG" and fund_pct <= -0.025:
                        fund_delta = +0.04
                        fund_reason = f"Short Squeeze Setup (Funding {fund_pct:+.4f}%)"
                    elif direction == "SHORT" and fund_pct >= 0.040:
                        fund_delta = +0.04
                        fund_reason = f"Long Liquidation Cascade (Funding {fund_pct:+.4f}%)"
                    elif direction == "LONG" and fund_pct >= 0.060:
                        fund_delta = -0.03
                        fund_reason = f"Warning: Overcrowded Longs (Funding {fund_pct:+.4f}%)"
                    elif direction == "SHORT" and fund_pct <= -0.045:
                        fund_delta = -0.03
                        fund_reason = f"Warning: Overcrowded Shorts (Funding {fund_pct:+.4f}%)"

                    total_intel_delta = liq_delta + ll_delta + micro_delta + fund_delta
                    calibrated_conviction = round(float(np.clip(sig.conviction + total_intel_delta, 0.0, 1.0)), 4)
                    ext_atr, loc_delta, loc_note = compute_auction_location(sym, direction, sig.price, sig.atr)
                    required_threshold = get_calibrated_threshold(sym, sig.regime, loc_delta=loc_delta)

                    # Update signal conviction with market intelligence calibration
                    sig.conviction = calibrated_conviction

                    # Check if post-intelligence conviction satisfies auction-calibrated threshold
                    if sig.conviction < required_threshold:
                        intel_notes = [n for n in (liq_reason, ll_reason, micro_reason, fund_reason) if n and n != "NEUTRAL"]
                        intel_str = " | ".join(intel_notes) if intel_notes else "Low Intel Delta"
                        _record_gate_rejection(
                            sym,
                            f"[{sym}] {direction} raw={sig.conviction - total_intel_delta:.3f} adj={sig.conviction:.3f}",
                            f"[{sym}] Intel {sig.conviction:.3f} < {required_threshold:.3f} [{loc_note}] ({intel_str})",
                        )
                        continue

                    is_high_prob = _is_grade_a_plus(sig.conviction, sig.council_score)
                    ok, reason = can_trade_symbol(sym, direction, is_high_prob=is_high_prob)
                    sym_port = portfolios.get(sym, portfolio)
                    if sym_port["position"] == "FLAT" and ok:
                        if sig.council_approved and sig.direction in ("BUY", "SELL"):
                            execute_signal(sig, symbol=sym)
                    elif sym_port["position"] == "FLAT" and sig.council_approved and sig.direction in ("BUY", "SELL"):
                        _record_gate_rejection(sym, f"[{sym}] {direction} p={sig.conviction:.3f}", f"[{sym}] {reason}")
            except Exception:
                pass
            finally:
                try:
                    _candle_queue.task_done()
                except ValueError:
                    pass
        except Exception:
            await asyncio.sleep(0.5)


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

    # Feed Cross-Asset Lead-Lag Engine & Microstructure CVD Engine
    lead_lag_engine.update_price(sym, price, now)
    if sym in microstructure_engines:
        microstructure_engines[sym].add_tick(price, qty, is_buyer_maker)

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
            # 1m rollover updates live_candle telemetry; closed 5m bars arrive authoritatively via @kline_5m
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

    # Evaluate queued institutional retest limits & active exits on every tick
    check_pending_retests(sym, price)
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

        # Build LiveSignalEngine for this symbol (pre-filtered at 0.48 to let directional candidates flow into Market Intelligence)
        engine = LiveSignalEngine(
            symbol=sym,
            market_type="crypto",
            models=models,
            conviction_threshold=0.48,
            risk_per_trade=25.0 if PROP_FIRM_MODE else (account["balance"] * BASE_RISK_PCT),
            default_rr=MIN_RR,
            max_concurrent=1,
            enable_ensemble=True,
            enable_monitoring=True,
            enable_walk_forward=True,
            enable_council=True,
            enable_statistical_alpha=False,
            retrain_interval=2000,
            consensus_threshold=0.55,
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
            # Drop the trailing unclosed REST candle so live @kline_5m/15m/1h closes never collide or duplicate
            closed_5m  = raw5m[:-1]  if len(raw5m)  > 1 else raw5m
            closed_15m = raw15m[:-1] if raw15m and len(raw15m) > 1 else (raw15m or [])
            closed_1h  = raw1h[:-1]  if raw1h  and len(raw1h)  > 1 else (raw1h or [])

            hist = pd.DataFrame(closed_5m)
            df_15 = pd.DataFrame(closed_15m) if closed_15m else None
            df_1h = pd.DataFrame(closed_1h) if closed_1h else None
            df_4h = pd.DataFrame(raw4h) if raw4h else None
            engine.warm_up(hist[["timestamp","open","high","low","close","volume"]], df_15m=df_15, df_1h=df_1h, df_4h=df_4h)
            for c in closed_5m:
                symbol_candles_5m[sym].append(c)

            last_close = float(raw5m[-1]["close"])
            symbol_markets[sym]["ltp"] = last_close
            symbol_markets[sym]["bid"] = last_close
            symbol_markets[sym]["ask"] = last_close
            if sym == PRIMARY_SYMBOL:
                market["ltp"] = last_close
                market["bid"] = last_close
                market["ask"] = last_close
                prewarm.replay_candles = closed_5m[-3:]

        for c in (raw15m[:-1] if raw15m and len(raw15m) > 1 else (raw15m or [])):
            symbol_candles_15m[sym].append(c)
        for c in (raw1h[:-1] if raw1h and len(raw1h) > 1 else (raw1h or [])):
            symbol_candles_1h[sym].append(c)
        for c in (raw4h or []):
            symbol_candles_4h[sym].append(c)
        for c in (raw1d or []):
            symbol_candles_1d[sym].append(c)

        macro = compute_macro_structure(sym)
        bias_str = macro.get("macro_bias", "NEUTRAL")
        print(f"  ✓ [{sym}] Buffers ready: 5m={len(symbol_candles_5m[sym])}, 15m={len(symbol_candles_15m[sym])}, 1h={len(symbol_candles_1h[sym])}, 4h={len(symbol_candles_4h[sym])}, 1d={len(symbol_candles_1d[sym])}")
        print(f"    Macro Bias: {bias_str} (1D:{macro['trend_1d']}, 4H:{macro['trend_4h']}, 1H:{macro['trend_1h']})")

    _loaded_model_count = total_models_loaded
    regime_info["atr"] = 200.0  # Default until first signal

    update_session()
    state_manager.load_state(portfolio, account)
    load_journal()

    # Reconcile active session's trades directly from trade journal to prevent restart amnesia
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    session_start_iso = os.getenv("APEX_SESSION_START_ISO", account.get("session_start_iso", ""))
    if not session_start_iso or not session_start_iso.startswith(today_str):
        session_start_iso = f"{today_str}T22:00:00" if today_str == "2026-11-09" else f"{today_str}T00:00:00"
    account["session_start_iso"] = session_start_iso

    session_trades = [
        t for t in trade_journal
        if str(t.get("open_ts", "")).startswith(today_str)
        and str(t.get("open_ts", "")) >= session_start_iso
        and t.get("outcome") not in ("CANCELLED", "BE")
        and (t.get("outcome") == "OPEN" or abs(float(t.get("pnl_usd", 0.0))) >= 0.50)
    ]
    account["trades_today"] = len(session_trades)
    account["today_date"] = today_str
    if PROP_FIRM_MODE and prop_firm_manager:
        prop_firm_manager.daily_trades = len(session_trades)
        prop_firm_manager.save_state()
    state_manager.save_state(portfolio, account)

    # Reset stale consecutive loss streak if last loss was more than cooldown period ago (> 2 hours)
    if account.get("last_loss_time", 0) > 0 and (time.time() - account["last_loss_time"]) > LOSS_COOLDOWN_3:
        account["consecutive_losses"] = 0

    print(f"  ✓ Session Reconciled: {account['trades_today']} trades recorded in active session (since {session_start_iso})")

    update_health()

    # Prewarm Institutional Liquidity & Lead-Lag Engines with historical data
    for sym in TRADED_SYMBOLS:
        c1d = symbol_candles_1d.get(sym, [])
        c1h = symbol_candles_1h.get(sym, [])
        c5m = symbol_candles_5m.get(sym, [])
        if c1d or c1h or c5m:
            try:
                liquidity_engine.update_levels(sym, candles_1d=c1d, candles_1h=c1h, candles_5m=c5m)
                if c5m:
                    for c in c5m[-12:]:
                        lead_lag_engine.update_price(sym, float(c["close"]), float(c.get("timestamp", 0)) / 1000.0)
            except Exception:
                pass

    # APEX v7.2: Independent Multi-Asset Walk-Forward Retrain Schedulers
    global retrain_scheduler, retrain_schedulers
    retrain_schedulers = {}

    def make_swap_callback(s: str):
        def _on_models_ready(new_models):
            sym_eng = signal_engines.get(s)
            if sym_eng and new_models:
                try:
                    sym_eng.hot_swap_models(new_models)
                    telegram(f"🔄 APEX [{s}]: {len(new_models)} models retrained + hot-swapped")
                    print(f"\n  ✓ HOT-SWAP [{s}]: {len(new_models)} new models loaded into ensemble")
                except Exception as e:
                    print(f"  WARN: hot_swap error [{s}]: {e}")
        return _on_models_ready

    for sym in TRADED_SYMBOLS:
        stg_dir = Path(f"data/models/staging/{sym.lower()}")
        stg_dir.mkdir(parents=True, exist_ok=True)
        sched = RetrainScheduler(
            symbol=sym,
            retrain_every_n_candles=2000,
            staging_dir=stg_dir,
            swap_callback=make_swap_callback(sym),
        )
        retrain_schedulers[sym] = sched

    retrain_scheduler = retrain_schedulers.get(PRIMARY_SYMBOL)

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
    kz_lbl = f" {MAG}[{session['kill_zone'].replace('_KZ', '')}]{R}" if session.get("is_kill_zone") else ""
    if session.get("sunday_weekly_open"):
        sess_str = f"{CYN}{session['name']}{R}{kz_lbl} (SUN OPEN {session['risk_mult']:.0%})"
    else:
        sess_str = f"{CYN}{session['name']}{R}{kz_lbl} ({session['risk_mult']:.0%})"
    buf.append(row(f"{DIM}Time:{R} {now} IST  │  {sess_str}  │  WS: {ws}"))
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

        reg = s_reg.get("current", lp.get("regime", "UNKNOWN"))
        reg_short = reg.replace("TRENDING_", "TREND_").replace("RANGING_", "RANGE_").replace("_VOL", "")
        atr = s_reg.get("atr", 0.0)
        reg_display = f"{reg_short} (${atr:.1f})"

        calib_thr = get_calibrated_threshold(sym, reg)
        pl = lp.get("long", 0.5)
        ps = lp.get("short", 0.5)
        pl_c = GRN if pl >= calib_thr else (YLW if pl >= 0.52 else GRY)
        ps_c = RED if ps >= calib_thr else (YLW if ps >= 0.52 else GRY)

        m_eng = momentum_engines.get(sym, momentum_eng)
        mom_d = m_eng.get_diagnostics() if m_eng else {}
        mom_sig_str = mom_d.get("signal", "FLAT")
        mom_c = GRN if mom_sig_str == "LONG" else (RED if mom_sig_str == "SHORT" else GRY)
        ofi_v = float(mom_d.get("ofi", 0.0))
        label_str = "OFI"
        if any(g in sym for g in ("GOLD", "PAXG")):
            try:
                from src.data.providers.delta_feed import delta_gold_feed
                d_tel = delta_gold_feed.get_telemetry()
                if d_tel.get("fresh") and abs(ofi_v) < 0.01:
                    ofi_v = float(d_tel.get("book_imbalance_5", 0.0))
                    label_str = "L2_OBI"
            except Exception:
                pass
        mom_display = f"{mom_c}{mom_sig_str:<5}{R} ({label_str}:{ofi_v:+.2f})"

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
    active_rej = [
        gr for gr in symbol_gate_rejections.values()
        if gr.get("reason") and gr.get("signal") and (time.time() - gr.get("ts", 0)) < 600
    ]
    active_rej.sort(key=lambda x: x.get("ts", 0), reverse=True)
    if active_rej:
        for gr in active_rej[:3]:
            age = int(time.time() - gr["ts"])
            buf.append(row(f" {YLW}⚠  Approved {gr['signal']} NOT executed — {gr['reason']} ({age}s ago){R}"))
    elif last_gate_rejection.get("reason") and last_gate_rejection.get("signal") and (time.time() - last_gate_rejection.get("ts", 0)) < 600:
        age = int(time.time() - last_gate_rejection["ts"])
        buf.append(row(f" {YLW}⚠  Approved {last_gate_rejection['signal']} NOT executed — {last_gate_rejection['reason']} ({age}s ago){R}"))
    if ls or active_rej or (last_gate_rejection.get("reason") and last_gate_rejection.get("signal")):
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

    # Key Liquidity Levels & Sweeps (ICT Turtle Soup & Key Levels)
    btc_lev = liquidity_engine.get_levels("BTCUSDT")
    pdh_v = btc_lev.get("pdh", 0.0)
    pdl_v = btc_lev.get("pdl", 0.0)
    ash_v = btc_lev.get("ash", 0.0)
    asl_v = btc_lev.get("asl", 0.0)
    if btc_lev.get("sweep_bullish"):
        sw_tag = f"{GRN}SWEEP_BULL ({btc_lev.get('sweep_level')}){R}"
    elif btc_lev.get("sweep_bearish"):
        sw_tag = f"{RED}SWEEP_BEAR ({btc_lev.get('sweep_level')}){R}"
    elif btc_lev.get("fvg_bullish"):
        sw_tag = f"{CYN}FVG_BULL{R}"
    elif btc_lev.get("fvg_bearish"):
        sw_tag = f"{YLW}FVG_BEAR{R}"
    else:
        sw_tag = f"{GRY}NONE{R}"

    ll_diag = getattr(lead_lag_engine, "_diagnostics", {})
    spill_tag = ll_diag.get("eth_spillover_bias", "NEUTRAL")
    spill_c = GRN if "BULL" in spill_tag else (RED if "BEAR" in spill_tag else GRY)
    buf.append(row(f" BTC Key Lvls: PDH:${pdh_v:,.0f} PDL:${pdl_v:,.0f} │ Asian:${asl_v:,.0f}-${ash_v:,.0f} │ Sweep:{sw_tag} │ Spill:{spill_c}{spill_tag}{R}"))
    buf.append(row(f" Next Event: {evt_col}{evt_str}{R}"))

    # Multi-Asset Retrain Schedulers
    rt_parts = []
    for s_name in TRADED_SYMBOLS:
        s_tag = "BTC" if "BTC" in s_name else ("ETH" if "ETH" in s_name else "GOLD")
        sch = retrain_schedulers.get(s_name)
        if sch:
            st = sch.get_status()
            nxt_c = f"{YLW}RETRAIN{R}" if st.get("is_retraining") else f"{st.get('next_retrain_in', 0)}c"
            rt_parts.append(f"{s_tag}:{nxt_c}(#{st.get('candles_accumulated', 0)})")
    rt_display = "  │  ".join(rt_parts) if rt_parts else "BTC: 2000c"
    buf.append(row(f" Retrain Sched: {rt_display}"))
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


def _ingest_closed_5m(sym: str, c5: dict) -> bool:
    """Thread-safe/idempotent ingestion of a closed 5m candle into symbol buffer and ML queue."""
    ts = int(c5.get("timestamp", 0))
    if ts <= 0:
        return False
    buf5 = symbol_candles_5m.get(sym)
    if buf5 is not None:
        if buf5 and buf5[-1].get("timestamp") == ts:
            buf5[-1] = dict(c5)
        elif not buf5 or ts > int(buf5[-1].get("timestamp", 0)):
            buf5.append(dict(c5))
    if _last_queued_5m_ts.get(sym, 0) >= ts:
        return False
    _last_queued_5m_ts[sym] = ts
    if _candle_queue is not None:
        try:
            _candle_queue.put_nowait((sym, dict(c5)))
            return True
        except asyncio.QueueFull:
            pass
    return False

async def _rest_kline_sync_loop():
    """
    Deterministic 5m/15m/1h REST candle watchdog: runs one immediate catch-up pass 3s after boot
    and then wakes 2.5s after every 5m close so ML probabilities, ATR, and signals never freeze.
    """
    first_pass = True
    while True:
        try:
            if first_pass:
                first_pass = False
                await asyncio.sleep(3.0)
            else:
                now_s = time.time()
                next_close_s = ((int(now_s) // 300) + 1) * 300 + 2.5
                await asyncio.sleep(max(1.0, next_close_s - now_s))

            last_closed_5m_open_ms = (int(time.time()) // 300 - 1) * 300 * 1000
            for sym in TRADED_SYMBOLS:
                try:
                    rows5 = await asyncio.to_thread(fetch_klines, sym, "5m", 6)
                    for c in rows5:
                        ts = int(c.get("timestamp", 0))
                        if 0 < ts <= last_closed_5m_open_ms:
                            _ingest_closed_5m(sym, c)
                except Exception:
                    pass

                # Refresh 15m and 1h closed bars reliably on boundaries when WS @kline_15m/1h stalls
                now_sec = int(time.time())
                if (now_sec // 300) % 3 == 0:
                    try:
                        last_15_ms = (now_sec // 900 - 1) * 900 * 1000
                        rows15 = await asyncio.to_thread(fetch_klines, sym, "15m", 4)
                        buf15 = symbol_candles_15m.get(sym)
                        if buf15 is not None:
                            for c15 in rows15:
                                ts15 = int(c15.get("timestamp", 0))
                                if 0 < ts15 <= last_15_ms and (not buf15 or ts15 > int(buf15[-1].get("timestamp", 0))):
                                    buf15.append(dict(c15))
                    except Exception:
                        pass

                if (now_sec // 300) % 12 == 0:
                    try:
                        last_1h_ms = (now_sec // 3600 - 1) * 3600 * 1000
                        rows1h = await asyncio.to_thread(fetch_klines, sym, "1h", 4)
                        buf1h = symbol_candles_1h.get(sym)
                        if buf1h is not None:
                            for c1h in rows1h:
                                ts1h = int(c1h.get("timestamp", 0))
                                if 0 < ts1h <= last_1h_ms and (not buf1h or ts1h > int(buf1h[-1].get("timestamp", 0))):
                                    buf1h.append(dict(c1h))
                        compute_macro_structure(sym)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5)

# ─── WEBSOCKET ────────────────────────────────────────────────────────────────
async def binance_ws():
    # Construct combined multiplexed stream URL across all TRADED_SYMBOLS
    streams = []
    for s in TRADED_SYMBOLS:
        sl = s.lower()
        streams.extend([f"{sl}@trade", f"{sl}@bookTicker", f"{sl}@depth20@500ms", f"{sl}@kline_5m", f"{sl}@kline_15m", f"{sl}@kline_1h"])
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
                            _ingest_closed_5m(sym, c5)
                    elif "@kline_15m" in st:
                        k = pay.get("k", {})
                        if k.get("x"):
                            c15 = {
                                "open": float(k["o"]), "high": float(k["h"]),
                                "low": float(k["l"]), "close": float(k["c"]),
                                "volume": float(k["v"]), "timestamp": int(k["t"]),
                            }
                            buf15 = symbol_candles_15m.get(sym)
                            if buf15 is not None and (not buf15 or buf15[-1].get("timestamp") != c15["timestamp"]):
                                buf15.append(dict(c15))
                    elif "@kline_1h" in st:
                        k = pay.get("k", {})
                        if k.get("x"):
                            c1h = {
                                "open": float(k["o"]), "high": float(k["h"]),
                                "low": float(k["l"]), "close": float(k["c"]),
                                "volume": float(k["v"]), "timestamp": int(k["t"]),
                            }
                            buf1h = symbol_candles_1h.get(sym)
                            if buf1h is not None and (not buf1h or buf1h[-1].get("timestamp") != c1h["timestamp"]):
                                buf1h.append(dict(c1h))
                                acc4h = symbol_acc_1h_4h.get(sym)
                                if acc4h is not None:
                                    acc4h.append(dict(c1h))
                                    if len(acc4h) >= 4:
                                        c4h = {"open": acc4h[0]["open"], "high": max(x["high"] for x in acc4h),
                                               "low": min(x["low"] for x in acc4h), "close": acc4h[-1]["close"],
                                               "volume": sum(x["volume"] for x in acc4h), "timestamp": int(k["t"])}
                                        acc4h.clear()
                                        symbol_candles_4h[sym].append(dict(c4h))
                                        acc1d = symbol_acc_4h_1d.get(sym)
                                        if acc1d is not None:
                                            acc1d.append(dict(c4h))
                                            if len(acc1d) >= 6:
                                                c1d = {"open": acc1d[0]["open"], "high": max(x["high"] for x in acc1d),
                                                       "low": min(x["low"] for x in acc1d), "close": acc1d[-1]["close"],
                                                       "volume": sum(x["volume"] for x in acc1d), "timestamp": int(k["t"])}
                                                acc1d.clear()
                                                symbol_candles_1d[sym].append(dict(c1d))
                            compute_macro_structure(sym)
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

    for s in TRADED_SYMBOLS:
        asyncio.create_task(_safe_fetch(funding_monitor.update(s), f"funding_{s}"))
    asyncio.create_task(_safe_fetch(news_guard.update_events(), "news"))
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
                fund_coros = [asyncio.wait_for(funding_monitor.update(s), timeout=10) for s in TRADED_SYMBOLS]
                await asyncio.gather(
                    asyncio.wait_for(news_guard.update_events(), timeout=10),
                    asyncio.wait_for(sentiment_monitor.update(), timeout=10),
                    *fund_coros,
                    return_exceptions=True,
                )

            # 3. Push fresh sentiment and cross-market data into Council across all symbols
            try:
                fg_val = int(sentiment_monitor.get_fear_greed())
                cm_state = cross_market_monitor.get_state()
                for s, eng in signal_engines.items():
                    if eng and eng.council:
                        try:
                            f_pct = float(funding_monitor.get_funding_pct(s))
                            eng.council.update_sentiment(fg_val, f_pct)
                            eng.council.update_cross_market(cm_state)
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception:
            pass


async def handle_web_action(action: str, payload: dict) -> dict:
    """Dispatches interactive actions from Web Dashboard REST APIs."""
    global PROP_FIRM_MODE, MAX_TRADES_DAY, prop_firm_manager
    if action == "close_all":
        closed_syms = []
        for sym in TRADED_SYMBOLS:
            sym_port = portfolios.get(sym)
            if sym_port and sym_port.get("position") == "OPEN":
                ticket = sym_port.get("broker_ticket")
                if ticket and _active_broker and hasattr(_active_broker, "close_position_by_ticket"):
                    try:
                        await _active_broker.close_position_by_ticket(int(ticket))
                    except Exception as e:
                        print(f"  [WebAction] Close ticket {ticket} failed: {e}")
                sym_mkt = symbol_markets.get(sym, market)
                exit_price = sym_mkt["bid"] if sym_port.get("direction") == "LONG" else sym_mkt["ask"]
                exit_profit = float(sym_port.get("broker_pnl", 0.0) or sym_port.get("unrealized_pnl", 0.0))
                _do_close_symbol(sym, exit_profit, exit_price, "WEB DASHBOARD EMERGENCY FLATTEN ALL")
                closed_syms.append(sym)
        return {"success": True, "message": f"Flattened: {', '.join(closed_syms) if closed_syms else 'None open'}"}

    elif action == "close_symbol":
        sym = payload.get("symbol", "").upper()
        sym_port = portfolios.get(sym)
        if sym_port and sym_port.get("position") == "OPEN":
            ticket = sym_port.get("broker_ticket")
            if ticket and _active_broker and hasattr(_active_broker, "close_position_by_ticket"):
                try:
                    await _active_broker.close_position_by_ticket(int(ticket))
                except Exception as e:
                    print(f"  [WebAction] Close ticket {ticket} failed: {e}")
            sym_mkt = symbol_markets.get(sym, market)
            exit_price = sym_mkt["bid"] if sym_port.get("direction") == "LONG" else sym_mkt["ask"]
            exit_profit = float(sym_port.get("broker_pnl", 0.0) or sym_port.get("unrealized_pnl", 0.0))
            _do_close_symbol(sym, exit_profit, exit_price, f"WEB DASHBOARD CLOSE [{sym}]")
            return {"success": True, "message": f"{sym} closed"}
        return {"success": False, "message": f"No open position for {sym}"}

    elif action == "toggle_pause":
        account["halted"] = not account.get("halted", False)
        account["halt_reason"] = "PAUSED VIA WEB DASHBOARD" if account["halted"] else ""
        return {"success": True, "halted": account["halted"]}

    elif action == "prop_firm_settings":
        enabled = payload.get("enabled")
        profile = payload.get("profile")
        reset = payload.get("reset", False)
        if enabled is not None:
            PROP_FIRM_MODE = bool(enabled)
        if profile and prop_firm_manager:
            prop_firm_manager.switch_profile(profile, reset=reset)
        elif reset and prop_firm_manager:
            prop_firm_manager.reset_challenge()
        return {
            "success": True,
            "enabled": PROP_FIRM_MODE,
            "profile": prop_firm_manager.profile_key if prop_firm_manager else "DISABLED",
            "metrics": prop_firm_manager.get_hud_metrics() if (PROP_FIRM_MODE and prop_firm_manager) else None
        }

    elif action == "prop_firm_custom":
        custom_cfg = payload.get("config", {})
        reset = payload.get("reset", False)
        enabled = payload.get("enabled")
        if enabled is not None:
            PROP_FIRM_MODE = bool(enabled)
        if prop_firm_manager and custom_cfg:
            prop_firm_manager.update_custom_config(custom_cfg, reset=reset)
        elif reset and prop_firm_manager:
            prop_firm_manager.reset_challenge()
        return {
            "success": True,
            "enabled": PROP_FIRM_MODE,
            "profile": prop_firm_manager.profile_key if prop_firm_manager else "DISABLED",
            "metrics": prop_firm_manager.get_hud_metrics() if (PROP_FIRM_MODE and prop_firm_manager) else None
        }

    elif action == "risk_settings":
        mt = payload.get("max_trades_day")
        if mt is not None:
            MAX_TRADES_DAY = int(mt)
        return {"success": True, "max_trades_day": MAX_TRADES_DAY}

    return {"error": f"Unknown action {action}"}


def get_dashboard_state() -> dict:
    """Extract real-time metrics for Web Dashboard & remote monitoring."""
    def _safe_price(sym: str, field: str) -> float:
        v = float(symbol_markets.get(sym, {}).get(field, 0.0))
        if v > 0: return v
        c_list = list(symbol_candles_5m.get(sym, []))
        if c_list: return float(c_list[-1].get("close", 0.0))
        return 0.0

    dur_min = 0
    if portfolio.get("entry_time", 0) > 0:
        dur_min = round((time.time() - portfolio["entry_time"]) / 60.0, 1)

    # Next event
    mins_to = None
    try:
        nxt = news_guard.get_next_event()
        if nxt:
            mins_to = round((nxt["time"] - datetime.datetime.now(datetime.timezone.utc)).total_seconds() / 60.0, 1)
        is_blk, _ = news_guard.is_blackout(PRIMARY_SYMBOL)
    except Exception:
        is_blk = False

    # Format journal for web UI (reverse chronological, latest first)
    journal_ui = []
    for t in reversed(trade_journal[-30:]):
        open_time_str = str(t.get("open_ts", ""))
        if "T" in open_time_str:
            open_time_str = open_time_str.split("T")[1][:8]
        is_open = (t.get("outcome") == "OPEN")
        sym_t = t.get("symbol", PRIMARY_SYMBOL)
        cur_px = _safe_price(sym_t, "ltp")
        if is_open:
            port_t = portfolios.get(sym_t, {})
            u_pnl = port_t.get("broker_pnl", 0.0) if port_t.get("broker_ticket") else port_t.get("unrealized_pnl", 0.0)
            if u_pnl == 0.0 and port_t.get("entry_price", 0.0) > 0 and cur_px > 0:
                p_dir = port_t.get("direction", "SHORT")
                diff = (cur_px - port_t["entry_price"]) if p_dir == "LONG" else (port_t["entry_price"] - cur_px)
                u_pnl = (diff / port_t["entry_price"]) * port_t.get("notional", 2500.0)
            exit_px_disp = cur_px
            pnl_disp = round(float(u_pnl), 2)
            reason_disp = "LIVE POSITION (Trailing)"
        else:
            exit_px_disp = t.get("exit_price", 0.0)
            pnl_disp = t.get("pnl_usd", 0.0)
            reason_disp = t.get("exit_reason", "")

        journal_ui.append({
            "id": t.get("id", 0),
            "symbol": sym_t,
            "open_ts": open_time_str,
            "action": t.get("action", ""),
            "entry_price": t.get("entry_price", 0.0),
            "exit_price": exit_px_disp,
            "pnl_usd": pnl_disp,
            "exit_reason": reason_disp,
            "is_open": is_open,
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
        funding_pct = round(funding_monitor.get_funding_pct(PRIMARY_SYMBOL), 4)
    except Exception:
        pass

    # Multi-asset liquidity levels
    liq_map = {}
    try:
        for sym in TRADED_SYMBOLS:
            lev = liquidity_engine.get_levels(sym)
            if not lev or lev.get("pdh", 0.0) == 0.0:
                c1d = list(symbol_candles_1d.get(sym, []))
                c1h = list(symbol_candles_1h.get(sym, []))
                c5m = list(symbol_candles_5m.get(sym, []))
                if c1d or c1h or c5m:
                    lev = liquidity_engine.update_levels(sym, candles_1d=c1d, candles_1h=c1h, candles_5m=c5m)
            liq_map[sym] = lev or {}
    except Exception:
        pass

    # Walk-forward retrain schedulers status
    retrain_map = {}
    try:
        for sym, sch in retrain_schedulers.items():
            retrain_map[sym] = sch.get_status()
    except Exception:
        pass

    lead_lag_diag = {}
    try:
        lead_lag_diag = getattr(lead_lag_engine, "_diagnostics", {})
    except Exception:
        pass

    # Build symbols payload with MTF trends and sparkline closes
    symbols_payload = {}
    for sym in TRADED_SYMBOLS:
        c5 = list(symbol_candles_5m.get(sym, []))
        ms = symbol_macro_structure.get(sym, {})
        trend_5m = "BULL" if (len(c5) >= 2 and float(c5[-1].get("close", 0)) >= float(c5[-2].get("close", 0))) else "BEAR"
        cur_ltp_val = _safe_price(sym, "ltp")
        if len(c5) >= 15 and cur_ltp_val > 0:
            spark = [round(float(c.get("close", 0.0)), 2) for c in c5[-15:]] + [round(cur_ltp_val, 2)]
        else:
            spark = [round(float(c.get("close", 0.0)), 2) for c in c5[-16:]]
        if not spark and cur_ltp_val > 0:
            spark = [round(cur_ltp_val, 2)]

        # Resolve active position status and self-heal from trade_journal / broker
        sym_port_obj = portfolios.get(sym, {})
        pos_status = sym_port_obj.get("position", "FLAT")
        if pos_status == "FLAT":
            for tj in reversed(trade_journal[-5:]):
                if tj.get("symbol") == sym and tj.get("outcome") == "OPEN":
                    pos_status = "OPEN"
                    sym_port_obj["position"] = "OPEN"
                    sym_port_obj["direction"] = "LONG" if tj.get("action") == "BUY" else "SHORT"
                    if sym_port_obj.get("entry_price", 0.0) <= 0:
                        sym_port_obj["entry_price"] = tj.get("entry_price", 0.0)
                        sym_port_obj["sl_price"] = tj.get("sl_price", 0.0)
                        sym_port_obj["tp1_price"] = tj.get("tp1_price", 0.0)
                        sym_port_obj["notional"] = tj.get("notional", 2500.0)
                    break

        cur_ltp = _safe_price(sym, "ltp")
        u_pnl = sym_port_obj.get("broker_pnl", 0.0) if sym_port_obj.get("broker_ticket") else sym_port_obj.get("unrealized_pnl", 0.0)
        if u_pnl == 0.0 and pos_status == "OPEN" and sym_port_obj.get("entry_price", 0.0) > 0 and cur_ltp > 0:
            p_dir = sym_port_obj.get("direction", "SHORT")
            diff = (cur_ltp - sym_port_obj["entry_price"]) if p_dir == "LONG" else (sym_port_obj["entry_price"] - cur_ltp)
            u_pnl = round((diff / sym_port_obj["entry_price"]) * sym_port_obj.get("notional", 2700.0), 2)

        delta_info = {}
        if any(g in sym for g in ("GOLD", "PAXG")):
            try:
                from src.data.providers.delta_feed import delta_gold_feed
                from src.intelligence.gold_structure_engine import gold_structure_engine
                d_tel = delta_gold_feed.get_telemetry()
                c15 = list(symbol_candles_15m.get(sym, []))
                c1h = list(symbol_candles_1h.get(sym, []))
                g_eval = gold_structure_engine.evaluate_gold_structure(cur_ltp, c15, c1h)
                delta_info = {
                    "delta_connected": d_tel.get("connected", False),
                    "delta_obi_5": d_tel.get("book_imbalance_5", 0.0),
                    "delta_aggressor": d_tel.get("aggressor_ratio", 0.0),
                    "fvg_demand": g_eval.get("in_bullish_fvg", False),
                    "fvg_supply": g_eval.get("in_bearish_fvg", False),
                    "c1_level": g_eval.get("c1_level", 0.0),
                    "c2_level": g_eval.get("c2_level", 0.0),
                    "structural_bias": g_eval.get("structural_bias", "NEUTRAL"),
                }
            except Exception:
                pass

        symbols_payload[sym] = {
            "ltp": _safe_price(sym, "ltp"),
            "bid": _safe_price(sym, "bid"),
            "ask": _safe_price(sym, "ask"),
            "prob_long": symbol_live_probs.get(sym, {}).get("long", 0.5),
            "prob_short": symbol_live_probs.get(sym, {}).get("short", 0.5),
            "regime": symbol_regime_info.get(sym, {}).get("current", "UNKNOWN"),
            "atr": symbol_regime_info.get(sym, {}).get("atr", 0.0),
            "position": pos_status,
            "direction": sym_port_obj.get("direction"),
            "entry_price": sym_port_obj.get("entry_price", 0.0),
            "sl_price": sym_port_obj.get("sl_price", 0.0),
            "tp1_price": sym_port_obj.get("tp1_price", 0.0),
            "unrealized_pnl": u_pnl,
            "sparkline": spark,
            "mtf": {
                "1d": "BULL" if "BULL" in ms.get("trend_1d", "") else ("BEAR" if "BEAR" in ms.get("trend_1d", "") else "NEUT"),
                "4h": "BULL" if "BULL" in ms.get("trend_4h", "") else ("BEAR" if "BEAR" in ms.get("trend_4h", "") else "NEUT"),
                "1h": "BULL" if "BULL" in ms.get("trend_1h", "") else ("BEAR" if "BEAR" in ms.get("trend_1h", "") else "NEUT"),
                "5m": trend_5m,
                "bias": ms.get("macro_bias", "NEUTRAL"),
            },
            "microstructure": delta_info if any(g in sym for g in ("GOLD", "PAXG")) else {
                "obi": symbol_markets.get(sym, {}).get("obi", 0.0)
            }
        }

    pf_metrics = prop_firm_manager.get_hud_metrics() if (PROP_FIRM_MODE and prop_firm_manager) else None
    if pf_metrics:
        init_b = float(pf_metrics.get("initial_balance", 5000.0))
        eq_pts = [init_b]
        cur_b = init_b
        for t in trade_journal[-30:]:
            cur_b += float(t.get("pnl_usd", 0.0))
            eq_pts.append(round(cur_b, 2))
        if round(cur_b, 2) != round(pf_metrics.get("virtual_balance", cur_b), 2):
            eq_pts.append(round(float(pf_metrics.get("virtual_balance", cur_b)), 2))
        pf_metrics["equity_curve"] = eq_pts

    return {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "halted": account.get("halted", False),
            "halt_reason": account.get("halt_reason", ""),
            "session_name": session.get("name", "LONDON"),
            "kill_zone": session.get("kill_zone", ""),
            "prop_firm_mode": PROP_FIRM_MODE,
            "market": {
                "ltp": _safe_price(PRIMARY_SYMBOL, "ltp") or market.get("ltp", 0.0),
                "bid": _safe_price(PRIMARY_SYMBOL, "bid") or market.get("bid", 0.0),
                "ask": _safe_price(PRIMARY_SYMBOL, "ask") or market.get("ask", 0.0),
                "ws_connected": market.get("ws_connected", False),
                "ticks": market.get("ticks", 0),
            },
            "account": {
                "balance": round(account.get("balance", 0.0), 2),
                "net_pnl": round(account.get("net_pnl", 0.0), 2),
                "drawdown_pct": round(account.get("drawdown_pct", 0.0), 2),
                "wins": account.get("wins", 0),
                "losses": account.get("losses", 0),
                "total": account.get("total", 0),
                "win_rate": round(account.get("win_rate", 0.0), 1),
                "profit_factor": round(account.get("profit_factor", 0.0), 2),
                "trades_today": account.get("trades_today", 0),
                "max_trades_day": MAX_TRADES_DAY,
            },
            "intel": {
                "fear_greed": fg_val,
                "fear_greed_label": fg_lbl,
                "funding_rate": funding_pct,
                "news_blackout": is_blk,
                "next_event_min": mins_to,
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
                "status": (lambda p: p.get("position", "FLAT"))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "symbol": (lambda p: p.get("symbol", PRIMARY_SYMBOL))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "direction": (lambda p: p.get("direction", ""))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "entry_price": (lambda p: p.get("entry_price", 0.0))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "sl_price": (lambda p: p.get("sl_price", 0.0))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "tp1_price": (lambda p: p.get("tp1_price", 0.0))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "tp2_price": (lambda p: p.get("tp2_price", 0.0))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "unrealized_pnl": round((lambda p: p.get("broker_pnl") if p.get("broker_ticket") and p.get("broker_pnl") != 0.0 else p.get("unrealized_pnl", 0.0))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)), 2),
                "duration_min": (lambda p: round((time.time() - p["entry_time"]) / 60.0, 1) if p.get("entry_time", 0) > 0 else 0)(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
                "broker_ticket": (lambda p: p.get("broker_ticket", ""))(next((p for p in portfolios.values() if p.get("position") == "OPEN"), portfolio)),
            },
            "symbols": symbols_payload,
            "liquidity": liq_map,
            "retrain_status": retrain_map,
            "prop_firm": pf_metrics,
            "journal": journal_ui,
        }


def _do_close_symbol(sym: str, exit_profit: float, exit_price: float, reason: str):
    """Unified handler: updates stats, balance, prop firm challenge, and journal upon trade close."""
    sym_port = portfolios.get(sym, portfolio)
    if sym_port.get("position") != "OPEN":
        return
    cur_dir = sym_port.get("direction", "LONG")
    is_be = abs(exit_profit) < 0.50
    is_win = exit_profit >= 0.50
    account["balance"] += exit_profit
    update_health()
    if PROP_FIRM_MODE and prop_firm_manager:
        prop_firm_manager.on_trade_closed(exit_profit, reason)
    if is_be:
        account["trades_today"] = max(0, account.get("trades_today", 0) - 1)
        outcome_str = "BE"
    elif is_win:
        account["wins"] += 1
        account["consecutive_losses"] = 0
        account["total_wins_usd"] += exit_profit
        outcome_str = "WIN"
    else:
        account["losses"] += 1
        account["consecutive_losses"] += 1
        account["total_losses_usd"] += abs(exit_profit)
        account["last_loss_time"] = time.time()
        outcome_str = "LOSS"
        if account["consecutive_losses"] >= 2:
            account["halted"] = True
            account["halt_reason"] = "2 consecutive losses — cooling"
            account["halt_until"] = time.time() + LOSS_COOLDOWN_2
    if not is_be:
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
                      "outcome": outcome_str,
                      "exit_reason": reason})
            break
    save_journal()
    sym_port.update({"position": "FLAT", "direction": None,
                     "unrealized_pnl": 0.0, "banked_pnl": 0.0, "phase": 0,
                     "broker_ticket": "", "broker_pnl": 0.0,
                     "broker_entry_price": 0.0, "broker_offset": 0.0})
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
                                if not deal:
                                    # Wait briefly for MT5 deal history to commit right after SL/TP execution
                                    await asyncio.sleep(1.0)
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

                            p_time = float(active_pos.get("time", time.time()))
                            sym_port.update({
                                "symbol": sym,
                                "position": "OPEN",
                                "direction": p_dir,
                                "entry_price": p_entry,
                                "entry_time": p_time,
                                "entry_lots": p_vol,
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
                            if PROP_FIRM_MODE and prop_firm_manager:
                                prop_firm_manager.update_multi_floating_pnl(portfolios)
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

# ─── 2-WAY INTERACTIVE TELEGRAM COMMAND CENTER (APEX v7.2) ───────────────────
async def _telegram_command_loop():
    """
    2-Way Telegram Interactive Command Center.
    Polls getUpdates from Telegram Bot API with long-polling.
    Authenticates every message against TELEGRAM_CHAT_ID for rock-solid security.
    Supported commands:
      /menu, /help, /start- Command menu & interactive quick-action buttons
      /status             - Real-time engine health, DD, win rate, prop firm metrics
      /report             - Full performance audit & deal journal
      /positions          - Detailed view of open positions across BTC, ETH, and GOLD
      /closeall, /flatten - Emergency close all open positions across all assets
      /close <symbol>     - Close specific symbol (e.g. /close btc)
      /pause              - Pause new order execution
      /resume             - Resume trading engine
      /propfirm [profile] - List or hot-switch prop firm challenge profile
      /reset              - Reset current prop firm challenge balance & stats
      /mode <prop|live>   - Toggle Prop Firm challenge rules vs Personal trading
    """
    global PROP_FIRM_MODE, prop_firm_manager
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    offset = 0
    poll_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    send_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    import aiohttp

    # Register native Telegram command popup menu
    try:
        set_commands_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setMyCommands"
        commands_payload = {
            "commands": [
                {"command": "menu", "description": "Interactive command buttons"},
                {"command": "status", "description": "Live health, balance, PnL & target"},
                {"command": "report", "description": "Performance audit & deal history"},
                {"command": "positions", "description": "View open trades across BTC/ETH/Gold"},
                {"command": "pause", "description": "Pause new order execution"},
                {"command": "resume", "description": "Resume trading engine"},
                {"command": "flatten", "description": "Emergency close all positions"},
                {"command": "help", "description": "Full cheatsheet & commands"}
            ]
        }
        async with aiohttp.ClientSession() as init_s:
            await init_s.post(set_commands_url, json=commands_payload, timeout=5)
    except Exception:
        pass

    inline_menu_markup = {
        "inline_keyboard": [
            [
                {"text": "📊 Live Status", "callback_data": "/status"},
                {"text": "📈 Full Report", "callback_data": "/report"}
            ],
            [
                {"text": "💼 Open Positions", "callback_data": "/positions"},
                {"text": "🏆 Prop Challenge", "callback_data": "/propfirm"}
            ],
            [
                {"text": "⏸️ Pause Bot", "callback_data": "/pause"},
                {"text": "▶️ Resume Bot", "callback_data": "/resume"}
            ],
            [
                {"text": "🛑 Flatten All", "callback_data": "/flatten"}
            ]
        ]
    }

    async with aiohttp.ClientSession() as session_http:
        while True:
            try:
                params = {"offset": offset, "timeout": 10}
                async with session_http.get(poll_url, params=params, timeout=15) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        for update in data.get("result", []):
                            offset = max(offset, update["update_id"] + 1)
                            
                            cb = update.get("callback_query")
                            if cb:
                                cb_id = cb.get("id")
                                msg = cb.get("message", {})
                                chat_id = str(msg.get("chat", {}).get("id", ""))
                                text = cb.get("data", "").strip()
                                try:
                                    ans_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                                    await session_http.post(ans_url, json={"callback_query_id": cb_id}, timeout=3)
                                except Exception:
                                    pass
                            else:
                                msg = update.get("message", {})
                                chat_id = str(msg.get("chat", {}).get("id", ""))
                                text = msg.get("text", "").strip()

                            # Security: only process commands from authorized chat ID
                            if chat_id != str(TELEGRAM_CHAT_ID):
                                continue

                            if not text:
                                continue

                            parts = text.split()
                            cmd = parts[0].lower()
                            arg = parts[1] if len(parts) > 1 else ""

                            reply_txt = ""
                            reply_markup = None

                            if cmd in ("/menu", "/help", "/start"):
                                reply_txt = (
                                    "⚡ *APEX Interactive Command Center*\n"
                                    "Select an action from the buttons below or use the command menu:"
                                )
                                reply_markup = inline_menu_markup

                            elif cmd == "/report":
                                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                total_t = account.get("total", 0)
                                wins = account.get("wins", 0)
                                losses = account.get("losses", 0)
                                wr = account.get("win_rate", 0.0)
                                pf_val = account.get("profit_factor", 0.0)
                                net_pnl = account.get("net_pnl", 0.0)
                                bal = account.get("balance", 0.0)
                                dd = account.get("drawdown_pct", 0.0)
                                best_t = account.get("best", 0.0)
                                worst_t = account.get("worst", 0.0)
                                avg_w = account.get("avg_win", 0.0)
                                avg_l = account.get("avg_loss", 0.0)

                                pf_info = "Disabled (Personal Mode)"
                                if PROP_FIRM_MODE and prop_firm_manager:
                                    m = prop_firm_manager.get_hud_metrics()
                                    pf_info = (
                                        f"• Profile: *{m['name']}* [{m['status']}]\n"
                                        f"• Target: `${m['cumulative_profit']:+,.2f}` / `${m['profit_target_usd']:,.2f}` ({m['progress_pct']}%)\n"
                                        f"• Daily Loss: `-${m['daily_loss_used_usd']:,.2f}` / `-${m['max_daily_loss_usd']:,.2f}` ({m['daily_loss_used_pct']}%)\n"
                                        f"• Max DD: `-${m['max_dd_used_usd']:,.2f}` / `-${m['max_loss_usd']:,.2f}`"
                                    )

                                rec_lines = []
                                for t in reversed(trade_journal[-5:]):
                                    t_pnl = t.get("pnl_usd", 0.0)
                                    s_sign = "+" if t_pnl >= 0 else ""
                                    rec_lines.append(f"• `{t.get('symbol', 'BTC')}` {t.get('action', '')} PnL: `{s_sign}${t_pnl:,.2f}` ({t.get('exit_reason', '')[:20]})")
                                rec_block = "\n".join(rec_lines) if rec_lines else "No trades recorded yet."

                                reply_txt = (
                                    f"📈 *APEX PERFORMANCE AUDIT REPORT*\n"
                                    f"🕒 `{now_str} IST`\n\n"
                                    f"💰 *Account Overview:*\n"
                                    f"• Balance: `${bal:,.2f}`\n"
                                    f"• Net PnL: `${net_pnl:+,.2f}`\n"
                                    f"• Peak Drawdown: `{dd:.1f}%`\n"
                                    f"• Win Rate: `{wr}%` ({wins}W / {losses}L | Total: {total_t})\n"
                                    f"• Profit Factor: `{pf_val:.2f}`\n"
                                    f"• Avg Win: `${avg_w:.2f}` | Avg Loss: `${avg_l:.2f}`\n"
                                    f"• Best: `${best_t:+.2f}` | Worst: `${worst_t:+.2f}`\n\n"
                                    f"🏆 *Prop Firm Metrics:*\n{pf_info}\n\n"
                                    f"📝 *Recent Closed Deals:*\n{rec_block}"
                                )

                            elif cmd == "/status":
                                pf_status = "DISABLED (Personal Mode)"
                                if PROP_FIRM_MODE and prop_firm_manager:
                                    m = prop_firm_manager.get_hud_metrics()
                                    pf_status = (
                                        f"*{m['name']}* ({m['status']})\n"
                                        f"Virtual Equity: `${m['virtual_equity']:,.2f}`\n"
                                        f"Target: `${m['cumulative_profit']:+,.2f}` / `${m['profit_target_usd']:,.2f}` ({m['progress_pct']}%)\n"
                                        f"Daily Loss: `-${m['daily_loss_used_usd']:,.2f}` / `-${m['max_daily_loss_usd']:,.2f}` ({m['daily_loss_used_pct']}%)\n"
                                        f"Max Drawdown: `-${m['max_dd_used_usd']:,.2f}` / `-${m['max_loss_usd']:,.2f}`"
                                    )
                                open_pos_count = sum(1 for p in portfolios.values() if p.get("position") == "OPEN")
                                state_icon = "⏸️ PAUSED" if account.get("halted") else "🟢 ACTIVE"
                                reply_txt = (
                                    f"⚡ *APEX Engine Status* [{state_icon}]\n"
                                    f"Balance: `${account.get('balance', 0.0):,.2f}` | PnL: `${account.get('net_pnl', 0.0):+,.2f}`\n"
                                    f"Win Rate: `{account.get('win_rate', 0.0)}%` ({account.get('wins', 0)}W/{account.get('losses', 0)}L)\n"
                                    f"Active Positions: `{open_pos_count}/{MAX_CONCURRENT_TRADES}`\n\n"
                                    f"🏆 *Prop Firm Challenge:*\n{pf_status}"
                                )

                            elif cmd == "/positions":
                                lines = ["💼 *APEX Open Positions:*"]
                                has_open = False
                                for sym in TRADED_SYMBOLS:
                                    p = portfolios.get(sym, {})
                                    if p.get("position") == "OPEN":
                                        has_open = True
                                        pnl = p.get("broker_pnl") if p.get("broker_ticket") else p.get("unrealized_pnl", 0.0)
                                        pnl_sign = "+" if pnl >= 0 else ""
                                        lines.append(
                                            f"• *{sym}* {p.get('direction')} @ `${p.get('entry_price', 0.0):,.2f}`\n"
                                            f"  PnL: `{pnl_sign}${pnl:,.2f}` | SL: `${p.get('sl_price', 0.0):,.2f}` | TP: `${p.get('tp2_price', 0.0):,.2f}`"
                                        )
                                if not has_open:
                                    lines.append("No active positions (All assets FLAT).")
                                reply_txt = "\n".join(lines)

                            elif cmd in ("/closeall", "/flatten"):
                                closed_list = []
                                for sym in TRADED_SYMBOLS:
                                    sym_port = portfolios.get(sym)
                                    if sym_port and sym_port.get("position") == "OPEN":
                                        ticket = sym_port.get("broker_ticket")
                                        if ticket and _active_broker and hasattr(_active_broker, "close_position_by_ticket"):
                                            try:
                                                await _active_broker.close_position_by_ticket(int(ticket))
                                            except Exception:
                                                pass
                                        sym_mkt = symbol_markets.get(sym, market)
                                        cur_p = sym_mkt["bid"] if sym_port.get("direction") == "LONG" else sym_mkt["ask"]
                                        pnl = float(sym_port.get("broker_pnl", 0.0) or sym_port.get("unrealized_pnl", 0.0))
                                        _do_close_symbol(sym, pnl, cur_p, "TELEGRAM /closeall COMMAND")
                                        closed_list.append(f"{sym} (${pnl:+,.2f})")
                                if closed_list:
                                    reply_txt = f"🛑 *Closed All Open Positions:*\n" + "\n".join(closed_list)
                                else:
                                    reply_txt = "ℹ️ No open positions to close."

                            elif cmd == "/close":
                                target_sym = None
                                arg_u = arg.upper()
                                for s in TRADED_SYMBOLS:
                                    if arg_u in s:
                                        target_sym = s
                                        break
                                if not target_sym:
                                    reply_txt = f"⚠️ Unknown or unspecified symbol. Options: {', '.join(TRADED_SYMBOLS)}"
                                else:
                                    sym_port = portfolios.get(target_sym)
                                    if sym_port and sym_port.get("position") == "OPEN":
                                        ticket = sym_port.get("broker_ticket")
                                        if ticket and _active_broker and hasattr(_active_broker, "close_position_by_ticket"):
                                            try:
                                                await _active_broker.close_position_by_ticket(int(ticket))
                                            except Exception:
                                                pass
                                        sym_mkt = symbol_markets.get(target_sym, market)
                                        cur_p = sym_mkt["bid"] if sym_port.get("direction") == "LONG" else sym_mkt["ask"]
                                        pnl = float(sym_port.get("broker_pnl", 0.0) or sym_port.get("unrealized_pnl", 0.0))
                                        _do_close_symbol(target_sym, pnl, cur_p, f"TELEGRAM /close {target_sym} COMMAND")
                                        reply_txt = f"✅ Closed *{target_sym}* at `${cur_p:,.2f}` | PnL: `${pnl:+,.2f}`"
                                    else:
                                        reply_txt = f"ℹ️ *{target_sym}* is currently FLAT."

                            elif cmd == "/pause":
                                account["halted"] = True
                                account["halt_reason"] = "PAUSED VIA TELEGRAM COMMAND"
                                reply_txt = "⏸️ *APEX Trading Engine PAUSED*. Existing positions remain managed, new signals will not enter."

                            elif cmd == "/resume":
                                account["halted"] = False
                                account["halt_reason"] = ""
                                reply_txt = "▶️ *APEX Trading Engine RESUMED*. Live ML evaluation active."

                            elif cmd in ("/propfirm", "/profile"):
                                from src.core.prop_firm_manager import PROFILES
                                if not arg:
                                    curr = prop_firm_manager.profile_key if prop_firm_manager else 'None'
                                    prof_list = "\n".join([f"• `{k}` ({v['name']})" for k, v in PROFILES.items()])
                                    reply_txt = (
                                        f"🏆 Current Profile: *{curr}*\n\n"
                                        f"To switch profile, type:\n`/propfirm <profile_key>`\n\n"
                                        f"Available Profiles:\n{prof_list}"
                                    )
                                else:
                                    match_key = None
                                    for k in PROFILES:
                                        if arg.lower() == k.lower():
                                            match_key = k
                                            break
                                    if match_key and prop_firm_manager:
                                        prop_firm_manager.switch_profile(match_key, reset=False)
                                        reply_txt = f"✅ Prop firm profile switched to *{PROFILES[match_key]['name']}*!"
                                    else:
                                        reply_txt = f"❌ Invalid profile `{arg}`. Type `/propfirm` to see all 11 profiles."

                            elif cmd == "/reset":
                                if prop_firm_manager:
                                    prop_firm_manager.reset_challenge()
                                    reply_txt = "🔄 Prop Firm Challenge balance & statistics cleanly reset to starting baseline."
                                else:
                                    reply_txt = "⚠️ Prop Firm Manager not active."

                            elif cmd == "/mode":
                                if "prop" in arg.lower():
                                    PROP_FIRM_MODE = True
                                    reply_txt = "🏆 Switched to *Prop Firm Challenge Mode* (Strict rule enforcement active)."
                                elif "person" in arg.lower() or "live" in arg.lower() or "off" in arg.lower():
                                    PROP_FIRM_MODE = False
                                    reply_txt = "🚀 Switched to *Personal Account Mode* (Challenge constraints disabled)."
                                else:
                                    reply_txt = f"Current mode: *{'Prop Firm' if PROP_FIRM_MODE else 'Personal'}*. Usage: `/mode prop` or `/mode personal`."

                            if reply_txt:
                                payload = {"chat_id": chat_id, "text": reply_txt, "parse_mode": "Markdown"}
                                if reply_markup:
                                    payload["reply_markup"] = reply_markup
                                await session_http.post(
                                    send_url,
                                    json=payload,
                                    timeout=5
                                )
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(2)
            await asyncio.sleep(1)


async def main():
    if os.getenv("RAILWAY_PROJECT_ID") or os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_SERVICE_ID"):
        print("🛑 [RAILWAY DETECTED] Cloud instance disabled in favor of local terminal execution.")
        print("🛑 Exiting to eliminate duplicate Telegram messages and MT5 desync.")
        return

    global _candle_queue, last_signal
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
                    await eng.process_candle(c)
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
            _sync_live_probs(sym, eng)
            if symbol_candles_5m.get(sym):
                _last_queued_5m_ts[sym] = int(symbol_candles_5m[sym][-1].get("timestamp", 0))
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

    last_signal = None  # Never display historical startup replay bar as a live pending signal

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
        action_handler=handle_web_action,
        host="0.0.0.0",
        port=web_port
    )
    await web_server.start()

    # Phase 3: Launch Delta Exchange Gold Microstructure Feed in background
    from src.data.providers.delta_feed import delta_gold_feed

    await asyncio.gather(
        binance_ws(),
        _rest_kline_sync_loop(),
        delta_gold_feed.run(),
        ui_loop(),
        _candle_worker(),
        _market_intel_loop(),   # APEX v7.2: news + funding + sentiment
        _broker_sync_loop(),    # Real-time MT5 position supervisor
        _telegram_command_loop(), # APEX v7.2: 2-way Telegram command center
    )

if __name__ == "__main__":
    asyncio.run(main())
