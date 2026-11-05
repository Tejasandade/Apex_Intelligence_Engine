"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  APEX INTELLIGENCE ENGINE v8.0 — MULTI-SYMBOL EXECUTION SHELL               ║
║  Scalp + Swing   ·   Multi-Symbol   ·   Paper Trading Only                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Runs simultaneously:
  • Scalp engine  — 5m candle ML signals for each scalp symbol (BTC, ETH, SOL)
  • Swing engine  — Daily/4H trend-following for each swing symbol
  • MultiSymbolManager — capital isolation, correlation filter, exposure cap

Paper-trade only. No live Bybit wiring. Run after proving P&L on v7.
"""

import asyncio
import json
import sys
import os
import time
import datetime
from collections import deque
from pathlib import Path

# ─── Environment ─────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np
import pandas as pd
import requests
import websockets

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if os.name == "nt":
    os.system("")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ─── Core imports ─────────────────────────────────────────────────────────────
from src.models.xgboost_model import ApexXGBoostModel
from src.engine.signal_engine import LiveSignalEngine, TradingSignal
from src.engine.multi_symbol_manager import MultiSymbolManager, SymbolState
from src.engine.swing_trade_engine import SwingTradeEngine, SwingSignal
from src.data.providers.daily_feed import DailyFeedProvider
from src.data.providers.news_guard import NewsGuard
from src.data.providers.funding_rate import FundingRateMonitor
from src.data.providers.sentiment import SentimentMonitor
from src.core.state_manager import StateManager

# ─── ANSI ─────────────────────────────────────────────────────────────────────
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
BLU = "\033[94m"; MAG = "\033[95m"; CYN = "\033[96m"
WHT = "\033[97m"; GRY = "\033[90m"

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TOTAL_CAPITAL      = float(os.getenv("APEX_INITIAL_CAPITAL", "1000.0"))
JOURNAL_FILE       = "trade_journal_v8_multi.json"
MODEL_DIR          = Path("data/models")

# Scalp symbols — must have trained models in data/models/
SCALP_SYMBOLS: list[str] = ["BTCUSDT"]   # Add "ETHUSDT", "SOLUSDT" after training

# Swing symbols — only need daily + 4H data, no ML model required
SWING_SYMBOLS: list[str] = ["BTCUSDT"]   # Swing is capital-isolated from scalp pool

# Risk limits loaded from risk.yaml (with inline defaults)
def _load_risk_yaml() -> dict:
    try:
        import yaml
        p = Path(__file__).parent.parent / "configs" / "risk.yaml"
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}

_risk = _load_risk_yaml()
def _rc(section: str, key: str, default):
    return _risk.get(section, {}).get(key, default)

MIN_PROB       = _rc("conviction", "min_signal_threshold", 0.58)
BASE_RISK_PCT  = _rc("sizing", "min_allocation_pct", 0.010)
MAX_RISK_PCT   = _rc("sizing", "max_allocation_pct", 0.015)
MIN_RR         = _rc("trade_management", "default_rr_target", 3.0)
MAX_TRADES_DAY = _rc("sizing", "max_concurrent_trades", 3)
DAILY_LOSS_LIM = _rc("drawdown", "daily_limit_pct", 0.05)

# Telegram (optional)
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TAKER_FEE_BPS = 5.0
SLIPPAGE_BPS  = 2.0
ROUNDTRIP_BPS = (TAKER_FEE_BPS + SLIPPAGE_BPS) * 2

# ─── GLOBALS ──────────────────────────────────────────────────────────────────
# Scalp capital pool (SCALP_SYMBOLS share this, isolated per symbol via MSM)
SCALP_CAPITAL  = TOTAL_CAPITAL * 0.70   # 70% of total goes to scalp
SWING_CAPITAL  = TOTAL_CAPITAL * 0.30   # 30% to swing

trade_journal_scalp: list[dict] = []
trade_journal_swing: list[dict] = []

# Market intel providers
news_guard       = NewsGuard()
funding_monitor  = FundingRateMonitor()
sentiment_monitor = SentimentMonitor()
state_manager    = StateManager()

# Per-symbol live market data
live_market: dict[str, dict] = {}  # symbol → {ltp, bid, ask, spread_bps, ws_connected}

# Multi-symbol scalp manager
msm: MultiSymbolManager | None = None

# Swing engines — one per swing symbol
swing_engines: dict[str, SwingTradeEngine] = {}

# Swing open positions
swing_positions: dict[str, dict] = {}  # symbol → position dict or {}


# ─── UTILS ────────────────────────────────────────────────────────────────────
def telegram(msg: str) -> None:
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
    except Exception:
        pass


def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


# ─── SCALP HELPERS ───────────────────────────────────────────────────────────
def size_scalp_position(
    capital: float, sl_dist: float, entry: float, prob: float,
    council: float, direction: str = "LONG"
) -> tuple[float, float, float]:
    """
    Returns (notional_usd, risk_usd, leverage).
    Uses risk.yaml sizing with funding/sentiment modifiers.
    """
    if sl_dist <= 0 or entry <= 0:
        return 0.0, 0.0, 1.0

    # Base risk % from probability band
    if prob >= MIN_PROB + 0.07:
        risk_pct = MAX_RISK_PCT
    elif prob >= MIN_PROB + 0.04:
        risk_pct = BASE_RISK_PCT
    else:
        risk_pct = BASE_RISK_PCT * 0.6

    # Council bonus
    if council >= 0.85:
        risk_pct = min(risk_pct * 1.15, MAX_RISK_PCT)
    elif council < 0.65:
        risk_pct *= 0.80

    # Funding multiplier
    fund_mult = funding_monitor.size_multiplier(symbol="BTCUSDT", direction=direction)
    # Sentiment multiplier
    sent_mult = sentiment_monitor.get_regime_modifier()

    risk_pct = risk_pct * fund_mult * sent_mult
    risk_pct = max(0.003, min(risk_pct, MAX_RISK_PCT))

    risk_usd  = capital * risk_pct
    notional  = risk_usd / (sl_dist / entry)
    leverage  = notional / capital
    leverage  = round(min(leverage, 5.0), 1)
    notional  = capital * leverage

    fee = notional * (ROUNDTRIP_BPS / 10_000)
    if fee > risk_usd * 0.25:
        return 0.0, 0.0, 1.0

    return round(notional, 2), round(risk_usd, 2), leverage


def save_scalp_journal() -> None:
    with open(JOURNAL_FILE.replace(".json", "_scalp.json"), "w") as f:
        json.dump(trade_journal_scalp, f, indent=2)


def save_swing_journal() -> None:
    with open(JOURNAL_FILE.replace(".json", "_swing.json"), "w") as f:
        json.dump(trade_journal_swing, f, indent=2)


# ─── SCALP SIGNAL EXECUTION (per symbol) ─────────────────────────────────────
async def execute_scalp_signal(
    symbol: str,
    signal: TradingSignal,
    sym_state: SymbolState,
    ltp: float,
) -> None:
    """Paper-execute a scalp signal for one symbol."""
    portfolio = sym_state.portfolio
    account   = sym_state.account_slice

    # Already in a position?
    if portfolio["position"] != "FLAT":
        return

    # News blackout?
    if news_guard.is_blackout():
        return

    direction   = signal.direction      # "LONG" or "SHORT"
    prob        = signal.probability
    council     = signal.council_score if hasattr(signal, "council_score") else 0.60
    entry       = ltp
    sl_dist_raw = getattr(signal, "sl_distance", 0.0)
    sl_dist     = sl_dist_raw if sl_dist_raw > 0 else entry * 0.005

    notional, risk_usd, leverage = size_scalp_position(
        account["balance"], sl_dist, entry, prob, council, direction
    )
    if notional <= 0:
        return

    # Structural stops
    if direction == "LONG":
        sl_price = round(entry - sl_dist, 2)
        rr = 3.5 if council >= 0.85 else (3.0 if council >= 0.70 else MIN_RR)
        tp1 = round(entry + sl_dist * (rr / 2), 2)
        tp2 = round(entry + sl_dist * rr, 2)
    else:
        sl_price = round(entry + sl_dist, 2)
        rr = 3.5 if council >= 0.85 else (3.0 if council >= 0.70 else MIN_RR)
        tp1 = round(entry - sl_dist * (rr / 2), 2)
        tp2 = round(entry - sl_dist * rr, 2)

    portfolio.update({
        "position":    "OPEN",
        "direction":   direction,
        "entry_price": entry,
        "entry_time":  time.time(),
        "sl_price":    sl_price,
        "sl_distance": sl_dist,
        "tp1_price":   tp1,
        "tp2_price":   tp2,
        "notional":    notional,
        "initial_notional": notional,
        "unrealized_pnl": 0.0,
        "banked_pnl":  0.0,
        "phase":       1,
        "signal_id":   f"{symbol}_{int(time.time())}",
        "prob":        prob,
        "council_score": council,
        "regime_entry": getattr(signal, "regime", "unknown"),
        "source":      "SCALP",
        "highest_fav": entry,
    })

    fee = notional * (TAKER_FEE_BPS / 10_000)
    account["balance"] -= fee
    account["trades_today"] = account.get("trades_today", 0) + 1

    msg = (
        f"[{now_ts()}] 📊 PAPER {direction} {symbol}\n"
        f"Entry: {entry:.2f}  SL: {sl_price:.2f}  TP2: {tp2:.2f}\n"
        f"Size: ${notional:.0f}  Risk: ${risk_usd:.0f}  P={prob:.2f}  C={council:.2f}"
    )
    print(f"\n{GRN if direction=='LONG' else RED}{msg}{R}")
    telegram(f"🤖 APEX v8 | PAPER | {msg}")

    journal_entry = {
        "symbol": symbol, "direction": direction, "entry_price": entry,
        "sl_price": sl_price, "tp1_price": tp1, "tp2_price": tp2,
        "notional": notional, "risk_usd": risk_usd, "leverage": leverage,
        "prob": prob, "council": council,
        "open_ts": now_ts(), "close_ts": None,
        "outcome": "OPEN", "pnl_usd": 0.0, "type": "SCALP"
    }
    trade_journal_scalp.append(journal_entry)
    save_scalp_journal()


async def manage_scalp_position(
    symbol: str, sym_state: SymbolState, ltp: float
) -> None:
    """Check SL/TP1/TP2 for open scalp position."""
    portfolio = sym_state.portfolio
    account   = sym_state.account_slice

    if portfolio["position"] != "OPEN":
        return

    direction = portfolio["direction"]
    entry     = portfolio["entry_price"]
    sl        = portfolio["sl_price"]
    tp1       = portfolio["tp1_price"]
    tp2       = portfolio["tp2_price"]
    phase     = portfolio["phase"]
    notional  = portfolio["notional"]
    sl_dist   = portfolio["sl_distance"]

    # Update unrealized pnl
    if direction == "LONG":
        unrealized = (ltp - entry) / entry * notional
        portfolio["highest_fav"] = max(portfolio["highest_fav"], ltp)
    else:
        unrealized = (entry - ltp) / entry * notional
        portfolio["highest_fav"] = min(portfolio["highest_fav"], ltp)
    portfolio["unrealized_pnl"] = round(unrealized, 2)

    hit_sl = (direction == "LONG" and ltp <= sl) or (direction == "SHORT" and ltp >= sl)
    hit_tp1 = phase == 1 and (
        (direction == "LONG" and ltp >= tp1) or
        (direction == "SHORT" and ltp <= tp1)
    )
    hit_tp2 = phase == 2 and (
        (direction == "LONG" and ltp >= tp2) or
        (direction == "SHORT" and ltp <= tp2)
    )

    def close_position(exit_price: float, outcome: str, reason: str) -> None:
        pnl = portfolio["banked_pnl"]
        if direction == "LONG":
            pnl += (exit_price - entry) / entry * notional
        else:
            pnl += (entry - exit_price) / entry * notional
        fee = notional * (TAKER_FEE_BPS / 10_000)
        pnl -= fee

        account["balance"] += pnl
        account["peak"]     = max(account["peak"], account["balance"])

        portfolio["position"] = "FLAT"
        portfolio["unrealized_pnl"] = 0.0

        # Update journal
        for t in trade_journal_scalp:
            if t.get("signal_id") == portfolio["signal_id"] or (
                t.get("symbol") == symbol and t.get("outcome") == "OPEN"
            ):
                t.update({
                    "close_ts": now_ts(),
                    "exit_price": exit_price,
                    "outcome": outcome,
                    "pnl_usd": round(pnl, 2),
                    "dur_min": round((time.time() - portfolio["entry_time"]) / 60, 1),
                })
                break
        save_scalp_journal()

        icon = GRN if outcome == "WIN" else RED
        print(f"\n{icon}[{now_ts()}] {outcome} {symbol} @ {exit_price:.2f} | PnL: ${pnl:.2f}{R}")
        telegram(f"🤖 APEX v8 | {outcome} {symbol} | PnL: ${pnl:.2f}")

    if hit_sl:
        close_position(sl, "LOSS", "SL hit")
    elif hit_tp1:
        # Bank 50% at TP1, trail SL to breakeven
        half_pnl = (tp1 - entry) / entry * (notional * 0.5) if direction == "LONG" \
            else (entry - tp1) / entry * (notional * 0.5)
        half_pnl -= notional * 0.5 * (TAKER_FEE_BPS / 10_000)
        portfolio["banked_pnl"] += half_pnl
        portfolio["notional"]   *= 0.5
        portfolio["phase"]       = 2
        portfolio["sl_price"]    = entry  # move stop to breakeven
        print(f"\n{YLW}[{now_ts()}] TP1 {symbol} +${half_pnl:.2f} banked | runner to TP2{R}")
    elif hit_tp2:
        close_position(tp2, "WIN", "TP2 hit")


# ─── SWING ENGINE MANAGEMENT ──────────────────────────────────────────────────
async def run_swing_check(symbol: str, daily_feed: DailyFeedProvider) -> None:
    """Evaluate daily/4H data and manage open swing position."""
    engine = swing_engines.get(symbol)
    if engine is None:
        return

    # Refresh + update feeds
    try:
        await daily_feed.refresh(symbol)
        daily_df = daily_feed.get_daily_candles(symbol)
        h4_df    = daily_feed.get_4h_candles(symbol)
        if not daily_df.empty and not h4_df.empty:
            engine.update_feeds(daily_df, h4_df)
    except Exception as e:
        print(f"[SWING] Feed error {symbol}: {e}")
        return

    pos = swing_positions.get(symbol, {})
    ltp = live_market.get(symbol, {}).get("ltp", 0.0)

    if ltp <= 0:
        return

    # Manage existing swing position
    if pos.get("open"):
        direction = pos["direction"]
        entry     = pos["entry_price"]
        sl        = pos["sl_price"]
        tp1       = pos["tp1_price"]
        tp2       = pos["tp2_price"]
        notional  = pos["notional"]

        hit_sl  = (direction == "LONG" and ltp <= sl) or (direction == "SHORT" and ltp >= sl)
        hit_tp1 = not pos.get("tp1_banked") and (
            (direction == "LONG" and ltp >= tp1) or (direction == "SHORT" and ltp <= tp1)
        )
        hit_tp2 = pos.get("tp1_banked") and (
            (direction == "LONG" and ltp >= tp2) or (direction == "SHORT" and ltp <= tp2)
        )
        # 7-day time stop
        age_days = (time.time() - pos["entry_time"]) / 86400.0

        if hit_sl or age_days >= 7:
            reason = "SL" if hit_sl else "7-day time stop"
            pnl = (ltp - entry) / entry * notional if direction == "LONG" \
                else (entry - ltp) / entry * notional
            pnl -= notional * (ROUNDTRIP_BPS / 10_000)
            outcome = "WIN" if pnl > 0 else "LOSS"
            pos["open"] = False
            swing_positions[symbol] = {}
            print(f"\n{GRN if outcome=='WIN' else RED}[SWING {now_ts()}] {outcome} {symbol} {reason} | PnL ${pnl:.2f}{R}")
            telegram(f"🤖 APEX v8 SWING | {outcome} {symbol} {reason} | PnL: ${pnl:.2f}")

            jentry = {k: v for k, v in pos.items() if k != "open"}
            jentry.update({"close_ts": now_ts(), "pnl_usd": round(pnl, 2),
                           "outcome": outcome, "close_reason": reason})
            trade_journal_swing.append(jentry)
            save_swing_journal()
            return

        if hit_tp1:
            # Bank 50%, trail to breakeven
            pos["tp1_banked"] = True
            pos["sl_price"]   = entry
            tp1_pnl = (tp1 - entry) / entry * (notional * 0.5) if direction == "LONG" \
                else (entry - tp1) / entry * (notional * 0.5)
            pos["banked_pnl"] = tp1_pnl
            print(f"\n{YLW}[SWING {now_ts()}] TP1 {symbol} +${tp1_pnl:.2f} | runner to TP2{R}")

        if hit_tp2:
            pnl = pos.get("banked_pnl", 0.0)
            pnl += (tp2 - entry) / entry * (notional * 0.5) if direction == "LONG" \
                else (entry - tp2) / entry * (notional * 0.5)
            pnl -= notional * (ROUNDTRIP_BPS / 10_000)
            pos["open"] = False
            swing_positions[symbol] = {}
            print(f"\n{GRN}[SWING {now_ts()}] WIN TP2 {symbol} | PnL ${pnl:.2f}{R}")
            telegram(f"🤖 APEX v8 SWING | WIN {symbol} TP2 | PnL: ${pnl:.2f}")

            jentry = {k: v for k, v in pos.items() if k != "open"}
            jentry.update({"close_ts": now_ts(), "pnl_usd": round(pnl, 2), "outcome": "WIN"})
            trade_journal_swing.append(jentry)
            save_swing_journal()
        return

    # Evaluate for new swing entry
    if news_guard.is_blackout():
        return

    swing_sig: SwingSignal | None = engine.evaluate()
    if swing_sig is None:
        return

    direction = swing_sig.direction
    entry     = ltp
    sl        = swing_sig.sl_price
    sl_dist   = abs(entry - sl)
    if sl_dist <= 0:
        return

    swing_cap = SWING_CAPITAL / max(len(SWING_SYMBOLS), 1)
    risk_usd  = swing_cap * 0.015   # 1.5% per swing trade
    notional  = risk_usd / (sl_dist / entry)
    notional  = min(notional, swing_cap * 0.5)   # max 50% of swing pool per trade

    swing_positions[symbol] = {
        "open": True,
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry,
        "entry_time": time.time(),
        "sl_price": sl,
        "tp1_price": swing_sig.tp1_price,
        "tp2_price": swing_sig.tp2_price,
        "notional": round(notional, 2),
        "risk_usd": round(risk_usd, 2),
        "conviction": swing_sig.conviction,
        "reason": swing_sig.reason,
        "type": "SWING",
        "open_ts": now_ts(),
        "tp1_banked": False,
        "banked_pnl": 0.0,
    }

    msg = (
        f"[SWING {now_ts()}] 📈 {direction} {symbol}\n"
        f"Entry: {entry:.2f}  SL: {sl:.2f}  TP2: {swing_sig.tp2_price:.2f}\n"
        f"Size: ${notional:.0f}  Conviction: {swing_sig.conviction:.2f}"
    )
    print(f"\n{GRN if direction=='LONG' else RED}{msg}{R}")
    telegram(f"🤖 APEX v8 SWING | PAPER | {msg}")


# ─── WEBSOCKET (MULTI-SYMBOL) ─────────────────────────────────────────────────
async def _ws_symbol(symbol: str, candle_queue: asyncio.Queue) -> None:
    """Single symbol WebSocket — streams aggTrade + 5m kline."""
    sym_lower = symbol.lower()
    uri = (
        f"wss://fstream.binance.com/stream?streams="
        f"{sym_lower}@aggTrade/{sym_lower}@kline_5m"
    )
    while True:
        try:
            async with websockets.connect(uri, ping_interval=20, ping_timeout=15) as ws:
                live_market.setdefault(symbol, {
                    "ltp": 0.0, "bid": 0.0, "ask": 0.0,
                    "spread_bps": 0.0, "ws_connected": False, "ticks": 0
                })
                live_market[symbol]["ws_connected"] = True
                async for raw in ws:
                    data = json.loads(raw)
                    stream = data.get("stream", "")
                    payload = data.get("data", {})

                    if "aggTrade" in stream:
                        price = float(payload.get("p", 0))
                        live_market[symbol]["ltp"] = price
                        live_market[symbol]["ticks"] = live_market[symbol].get("ticks", 0) + 1

                    elif "kline_5m" in stream:
                        k = payload.get("k", {})
                        if k.get("x"):   # closed candle
                            candle = {
                                "symbol": symbol,
                                "open":   float(k["o"]),
                                "high":   float(k["h"]),
                                "low":    float(k["l"]),
                                "close":  float(k["c"]),
                                "volume": float(k["v"]),
                                "timestamp": int(k["t"]),
                            }
                            await candle_queue.put(candle)
        except Exception as e:
            live_market.setdefault(symbol, {})["ws_connected"] = False
            print(f"[WS] {symbol} error: {e} — reconnecting in 5s")
            await asyncio.sleep(5)


# ─── CANDLE WORKER ────────────────────────────────────────────────────────────
async def _candle_worker(candle_queue: asyncio.Queue) -> None:
    """Processes closed 5m candles → scalp signals → position management."""
    while True:
        candle: dict = await candle_queue.get()
        symbol = candle["symbol"]

        # Get this symbol's state from MSM
        if msm is None:
            continue
        sym_state = msm.get_symbol_state(symbol)
        if sym_state is None:
            continue

        engine = sym_state.signal_engine
        if engine is None:
            continue

        ltp = live_market.get(symbol, {}).get("ltp", candle["close"])

        # Feed candle to signal engine
        try:
            df_row = pd.Series(candle)
            sig: TradingSignal | None = await engine.process_candle(candle)
        except Exception as e:
            print(f"[CANDLE] {symbol} engine error: {e}")
            sig = None

        # Manage existing position first
        await manage_scalp_position(symbol, sym_state, ltp)

        # Execute new signal if flat
        if sig is not None and sym_state.portfolio["position"] == "FLAT":
            can_open, _ = msm.can_open_position(symbol, sig.direction)
            if can_open:
                await execute_scalp_signal(symbol, sig, sym_state, ltp)


# ─── MARKET INTEL LOOP ───────────────────────────────────────────────────────
async def _market_intel_loop() -> None:
    """Every 5 min: refresh news, funding, sentiment; push to MSM signal engines."""
    # Initial fetch
    try:
        await asyncio.gather(
            news_guard.update_events(),
            *[funding_monitor.update(sym) for sym in SCALP_SYMBOLS],
            sentiment_monitor.update(),
        )
    except Exception:
        pass

    while True:
        await asyncio.sleep(300)
        try:
            await asyncio.gather(
                news_guard.update_events(),
                *[funding_monitor.update(sym) for sym in SCALP_SYMBOLS],
                sentiment_monitor.update(),
            )
            # Push to council SentimentAdvisor for each symbol
            if msm:
                fg_val   = int(sentiment_monitor.get_fear_greed())
                fund_pct = float(funding_monitor.get_funding_pct("BTCUSDT"))
                for sym_state in msm.all_states().values():
                    if sym_state.signal_engine and sym_state.signal_engine.council:
                        try:
                            sym_state.signal_engine.council.update_sentiment(fg_val, fund_pct)
                        except Exception:
                            pass
        except Exception:
            pass


# ─── SWING DAILY LOOP ─────────────────────────────────────────────────────────
async def _swing_daily_loop(daily_feed: DailyFeedProvider) -> None:
    """Check swing signals once at startup, then every 4 hours."""
    for symbol in SWING_SYMBOLS:
        await run_swing_check(symbol, daily_feed)

    while True:
        await asyncio.sleep(14400)   # 4 hours
        for symbol in SWING_SYMBOLS:
            try:
                await run_swing_check(symbol, daily_feed)
            except Exception as e:
                print(f"[SWING] Error {symbol}: {e}")


# ─── DASHBOARD ────────────────────────────────────────────────────────────────
def draw_dashboard() -> None:
    clear()
    now_str = now_ts()
    total_scalp_bal = sum(
        (msm.get_symbol_state(s).account_slice.get("balance", 0) if msm and msm.get_symbol_state(s) else 0)
        for s in SCALP_SYMBOLS
    )
    total_swing_pnl = sum(
        t.get("pnl_usd", 0) for t in trade_journal_swing if t.get("outcome") in ("WIN", "LOSS")
    )

    print(f"{BOLD}{CYN}╔══════════════════════════════════════════════════════════════╗{R}")
    print(f"{BOLD}{CYN}║  APEX v8 MULTI-SYMBOL │ PAPER TRADING │ {now_str}  ║{R}")
    print(f"{BOLD}{CYN}╚══════════════════════════════════════════════════════════════╝{R}")
    print()

    # Scalp section
    print(f"{BOLD}{WHT}  ─── SCALP SYMBOLS ───────────────────────────────────────────{R}")
    for sym in SCALP_SYMBOLS:
        st = msm.get_symbol_state(sym) if msm else None
        ltp = live_market.get(sym, {}).get("ltp", 0.0)
        bal = st.account_slice.get("balance", 0) if st else 0
        pos = st.portfolio.get("position", "FLAT") if st else "FLAT"
        direction = st.portfolio.get("direction", "") if st else ""
        upnl = st.portfolio.get("unrealized_pnl", 0.0) if st else 0.0

        pos_color = GRN if pos == "OPEN" and direction == "LONG" else (RED if pos == "OPEN" else GRY)
        upnl_color = GRN if upnl >= 0 else RED

        print(
            f"  {BOLD}{sym}{R}  "
            f"Price:{WHT}{ltp:.2f}{R}  "
            f"Bal:{YLW}${bal:.2f}{R}  "
            f"Pos:{pos_color}{pos}{' ' + direction if pos=='OPEN' else ''}{R}  "
            f"uPnL:{upnl_color}${upnl:.2f}{R}"
        )

    print()
    print(f"{BOLD}{WHT}  ─── SWING POSITIONS ─────────────────────────────────────────{R}")
    any_swing = False
    for sym in SWING_SYMBOLS:
        pos = swing_positions.get(sym, {})
        if pos.get("open"):
            any_swing = True
            ltp = live_market.get(sym, {}).get("ltp", 0.0)
            direction = pos["direction"]
            entry = pos["entry_price"]
            upnl = (ltp - entry) / entry * pos["notional"] if direction == "LONG" \
                else (entry - ltp) / entry * pos["notional"]
            days = (time.time() - pos["entry_time"]) / 86400.0
            color = GRN if upnl >= 0 else RED
            print(
                f"  {BOLD}{sym}{R}  {direction}  "
                f"Entry:{entry:.2f}  LTP:{ltp:.2f}  "
                f"uPnL:{color}${upnl:.2f}{R}  "
                f"Day:{days:.1f}/{7}"
            )
    if not any_swing:
        print(f"  {GRY}No open swing positions{R}")

    print()
    print(f"{BOLD}{WHT}  ─── MARKET INTEL ────────────────────────────────────────────{R}")
    fg = sentiment_monitor.get_fear_greed()
    fg_label = sentiment_monitor.get_label()
    fg_color = RED if fg < 25 else (GRN if fg > 75 else YLW)
    blackout = news_guard.is_blackout()
    fund_pct = funding_monitor.get_funding_pct("BTCUSDT")

    print(
        f"  F&G:{fg_color}{fg} {fg_label}{R}  "
        f"Funding:{YLW}{fund_pct:+.4f}%{R}  "
        f"News:{'🔴 BLACKOUT' if blackout else f'{GRN}✓ CLEAR{R}'}"
    )

    print()
    print(f"{BOLD}{WHT}  ─── SESSION TOTALS ──────────────────────────────────────────{R}")
    scalp_wins  = sum(1 for t in trade_journal_scalp if t.get("outcome") == "WIN")
    scalp_loss  = sum(1 for t in trade_journal_scalp if t.get("outcome") == "LOSS")
    scalp_pnl   = sum(t.get("pnl_usd", 0) for t in trade_journal_scalp if t.get("outcome") in ("WIN","LOSS"))
    swing_wins  = sum(1 for t in trade_journal_swing if t.get("outcome") == "WIN")
    swing_loss  = sum(1 for t in trade_journal_swing if t.get("outcome") == "LOSS")

    print(f"  Scalp: {GRN}{scalp_wins}W{R}/{RED}{scalp_loss}L{R}  PnL:{GRN if scalp_pnl>=0 else RED}${scalp_pnl:.2f}{R}")
    print(f"  Swing: {GRN}{swing_wins}W{R}/{RED}{swing_loss}L{R}  PnL:{GRN if total_swing_pnl>=0 else RED}${total_swing_pnl:.2f}{R}")
    print()


async def _dashboard_loop() -> None:
    """Redraw dashboard every 2 seconds."""
    while True:
        try:
            draw_dashboard()
        except Exception:
            pass
        await asyncio.sleep(2)


# ─── PREWARM ──────────────────────────────────────────────────────────────────
def prewarm() -> None:
    global msm, swing_engines

    print("=" * 72)
    print("  APEX INTELLIGENCE ENGINE v8.0 — MULTI-SYMBOL INITIALIZATION")
    print("=" * 72)

    # 1. Build MultiSymbolManager
    print(f"  Building MultiSymbolManager for: {SCALP_SYMBOLS}")
    msm = MultiSymbolManager(
        symbols=SCALP_SYMBOLS,
        total_capital=SCALP_CAPITAL,
    )
    print(f"  ✓ MultiSymbolManager ready ({len(SCALP_SYMBOLS)} symbols)")

    # 1b. Build + attach a LiveSignalEngine for each scalp symbol
    for symbol in SCALP_SYMBOLS:
        sym_lower = symbol.lower()
        models = {}
        regimes_dir = ["ranging_high_vol", "ranging_low_vol", "trending_low_vol"]
        for regime in regimes_dir:
            for direction in ["LONG", "SHORT"]:
                for bag_n in [1, 2, 3]:
                    name = f"{sym_lower}_crypto_{regime}_{direction}_bag{bag_n}"
                    meta_path = MODEL_DIR / f"{name}_metadata.json"
                    model_path = MODEL_DIR / f"{name}.json"
                    if model_path.exists() and meta_path.exists():
                        try:
                            import json as _json
                            meta = _json.load(open(meta_path))
                            m = ApexXGBoostModel(name=name, feature_columns=meta["feature_columns"])
                            m.load(path=MODEL_DIR)
                            models[name] = m
                        except Exception as e:
                            print(f"    WARN {symbol}: {e}")

        engine = LiveSignalEngine(
            symbol=symbol,
            market_type="crypto",
            models=models,
            conviction_threshold=MIN_PROB,
            risk_per_trade=SCALP_CAPITAL / max(len(SCALP_SYMBOLS), 1) * BASE_RISK_PCT,
            default_rr=MIN_RR,
            max_concurrent=1,
            enable_ensemble=True,
            enable_monitoring=True,
            enable_council=True,
            consensus_threshold=0.60,
        )

        # Warm up with historical 5m candles
        try:
            import requests as _req
            res = _req.get(
                f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=5m&limit=210",
                timeout=15
            )
            raw5m = res.json()
            import pandas as _pd
            hist = _pd.DataFrame(raw5m, columns=[
                "timestamp","open","high","low","close","volume",
                "close_time","quote_vol","trades","taker_buy","tb_quote","ignore"
            ])
            for col in ["open","high","low","close","volume"]:
                hist[col] = hist[col].astype(float)
            hist["timestamp"] = hist["timestamp"].astype(int)
            engine.warm_up(hist[["timestamp","open","high","low","close","volume"]])

            # Pre-fill candle buffer
            sym_state = msm.get_symbol_state(symbol)
            for row in raw5m:
                c = {"open": float(row[1]), "high": float(row[2]), "low": float(row[3]),
                     "close": float(row[4]), "volume": float(row[5]), "timestamp": int(row[0])}
                sym_state.candles_5m.append(c)
        except Exception as e:
            print(f"    WARN warmup {symbol}: {e}")

        msm.attach_signal_engine(symbol, engine)
        print(f"  ✓ Signal engine: {symbol} ({len(models)} models)")

    # 2. Build SwingTradeEngine for each swing symbol
    for symbol in SWING_SYMBOLS:
        swing_cap_per_sym = SWING_CAPITAL / max(len(SWING_SYMBOLS), 1)
        swing_engines[symbol] = SwingTradeEngine(symbol=symbol, capital=swing_cap_per_sym)
        print(f"  ✓ SwingTradeEngine ready: {symbol} (${swing_cap_per_sym:.0f})")

    # 3. Load journals if they exist
    scalp_j = JOURNAL_FILE.replace(".json", "_scalp.json")
    swing_j = JOURNAL_FILE.replace(".json", "_swing.json")
    if os.path.exists(scalp_j):
        global trade_journal_scalp
        trade_journal_scalp = json.load(open(scalp_j))
        print(f"  ✓ Scalp journal loaded ({len(trade_journal_scalp)} entries)")
    if os.path.exists(swing_j):
        global trade_journal_swing
        trade_journal_swing = json.load(open(swing_j))
        print(f"  ✓ Swing journal loaded ({len(trade_journal_swing)} entries)")

    print(f"\n  Scalp pool:  ${SCALP_CAPITAL:.2f}")
    print(f"  Swing pool:  ${SWING_CAPITAL:.2f}")
    print(f"  Total:       ${TOTAL_CAPITAL:.2f}")
    print("\n  ► PAPER TRADE MODE — no live execution\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────
async def main() -> None:
    prewarm()

    candle_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    daily_feed = DailyFeedProvider()

    # WS stream per scalp symbol
    ws_tasks = [_ws_symbol(sym, candle_queue) for sym in SCALP_SYMBOLS]

    await asyncio.gather(
        *ws_tasks,
        _candle_worker(candle_queue),
        _market_intel_loop(),
        _swing_daily_loop(daily_feed),
        _dashboard_loop(),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YLW}  APEX v8 shutdown requested.{R}")
        save_scalp_journal()
        save_swing_journal()
