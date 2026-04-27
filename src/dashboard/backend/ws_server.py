import asyncio
import os
import math
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from collections import deque

console_queues = []
recent_logs = deque(maxlen=50)

def ws_log_sink(message):
    msg_str = str(message)
    recent_logs.append(msg_str)
    try:
        loop = asyncio.get_running_loop()
        for q in console_queues:
            loop.call_soon_threadsafe(q.put_nowait, msg_str)
    except RuntimeError:
        pass

# Add sink to pipe backend thinking
logger.add(
    ws_log_sink,
    format="{message}",
    filter=lambda r: "⚡" in r["message"] or "💹" in r["message"] or "Routing" in r["message"] or "Executed" in r["message"] or "Evaluating" in r["message"] or "Signal" in r["message"],
    enqueue=True
)

from src.agents.executor import TradeExecutor
from src.agents.risk_manager import RiskManager, MarketSessionManager
from src.brokers.angel_one_adapter import AngelOneAdapter
from src.brokers.oanda_adapter import OandaAdapter
from src.dashboard.backend.broadcaster import WebSocketBroadcaster
from src.dashboard.backend.schemas import (
    ActiveTrade,
    CapitalUpdate,
    DashboardMarket,
    DashboardSession,
    DashboardSnapshot,
    ExecutionPlan,
    ManualExecutionRequest,
    ProfessionalTraderUpdate,
    SignalAnalysis,
    SmartOrderCard,
    TradeTicket,
)
from src.dashboard.config import TRADING_PROFILES
from src.data.db import REDIS_CANDLE_KEY, WARMUP_CANDLES_REQUIRED, db_manager
from src.data.db_sanitizer import purge_stale_market_data
from src.data.news_fetcher import CryptoNewsFetcher
from src.models.dataset_builder import DataFetcher
from src.models.quant_model import ApexXGBoostModel


DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_STYLE = "momentum_burst"

def _get_provider_name(symbol: str) -> str:
    symbol_upper = symbol.upper()
    if "NIFTY" in symbol_upper:
        return "angel_one"
    elif "EURUSD" in symbol_upper:
        return "oanda_v20"
    return "binance_futures"

DEFAULT_STYLE = "Intraday"
INFERENCE_LOOP_INTERVAL_SECONDS = 0.5


def _default_warmup_status() -> dict:
    return {
        "symbol": DEFAULT_SYMBOL,
        "ready": False,
        "candle_count": 0,
        "required_candles": WARMUP_CANDLES_REQUIRED,
    }


async def _get_redis_live_price() -> float:
    """Zero-latency bridge — reads the best-bid/ask mid-price directly from the
    order book Redis key for tick-for-tick price parity with the live feed.
    Falls back to the latest candle close if the order book key is unavailable."""
    try:
        if not db_manager.redis_pool:
            return 0.0
        import json as _json

        # Primary: live order book mid-price (matches what the AI model sees)
        book_payload = await db_manager.redis_pool.get(f"book:{DEFAULT_SYMBOL.upper()}")
        if book_payload:
            book = _json.loads(book_payload)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if bids and asks:
                try:
                    mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                    if mid > 0:
                        return round(mid, 2)
                except (ValueError, TypeError, IndexError):
                    pass

        # Fallback: last candle close from Redis candle key
        payload = await db_manager.redis_pool.get(REDIS_CANDLE_KEY)
        if not payload:
            return 0.0
        candles = _json.loads(payload)
        if not isinstance(candles, list) or not candles:
            return 0.0
        latest = candles[-1]
        try:
            price = float(latest.get("close", 0.0))
        except (ValueError, TypeError):
            return 0.0
        return price if price > 0.0 else 0.0
    except Exception as exc:
        logger.warning("Redis live-price read failed: {}", exc)
        return 0.0



@dataclass
class DashboardRuntime:
    risk_manager: RiskManager = field(
        default_factory=lambda: RiskManager(
            capital_pools={
                "CRYPTO": float(os.getenv("APEX_CRYPTO_CAPITAL",  "10000")),
                "INDIA":  float(os.getenv("APEX_INDIA_CAPITAL",  "100000")),
                "FOREX":  float(os.getenv("APEX_FOREX_CAPITAL",  "10000")),
            }
        )
    )
    # Broker adapters keyed by market type — enables hot-swap without rebuilding executor
    broker_adapters: dict = field(default_factory=lambda: {
        "nse":   AngelOneAdapter(paper_balance=float(os.getenv("APEX_INDIA_CAPITAL", "100000"))),
        "forex": OandaAdapter(paper_balance=float(os.getenv("APEX_FOREX_CAPITAL", "10000"))),
    })
    executor: TradeExecutor = field(default_factory=lambda: TradeExecutor(live_trading_enabled=False))
    broadcaster: WebSocketBroadcaster = field(default_factory=WebSocketBroadcaster)
    news_fetcher: CryptoNewsFetcher = field(default_factory=CryptoNewsFetcher)
    trade_history: List[TradeTicket] = field(default_factory=list)
    live_signals: List[TradeTicket] = field(default_factory=list)
    latest_news: List[str] = field(default_factory=list)
    latest_snapshot: Optional[DashboardSnapshot] = None
    latest_signal_card: Optional[SmartOrderCard] = None
    last_signal: str = "HOLD"
    sim_pnl: float = 0.0
    cycle_count: int = 0
    synthetic_tick: int = 0
    latest_features: dict = field(default_factory=dict)
    latest_probability: float = 0.5
    latest_sentiment: float = 0.5
    last_probability_zone: str = "NEUTRAL"
    latest_warmup_status: dict = field(default_factory=_default_warmup_status)
    db_status: str = "connected"
    active_tab: str = "CRYPTO"
    trading_style: str = "Intraday"
    global_best_signal: Optional[dict] = None
    crypto_probabilities: list = field(default_factory=list)
    nse_probabilities: list = field(default_factory=list)


runtime = DashboardRuntime()

model_registry = {
    "crypto": ApexXGBoostModel(model_name="crypto_model"),
    "nse": ApexXGBoostModel(model_name="banknifty_model")
}


def _sanitize_features(features: Optional[dict]) -> dict:
    """
    Merge live features onto the synthetic baseline.
    For NSE index instruments (zero volume), 'VWAP' holds the session TWAP
    computed by train_nse_v3. If still zero/NaN, fall back gracefully to
    the close price so the UI never renders NaN.
    """
    merged = {**_build_synthetic_features(), **(features or {})}

    # VWAP / TWAP guard: never let this be 0 or NaN — fall back to close price
    vwap_val = merged.get("VWAP", 0.0)
    if not vwap_val or not math.isfinite(float(vwap_val)) or float(vwap_val) == 0.0:
        merged["VWAP"] = merged.get("close", 0.0)

    return merged


def _safe_predict(features: dict) -> float:
    try:
        model_key = "nse" if "NIFTY" in DEFAULT_SYMBOL.upper() else "crypto"
        model = model_registry[model_key]
        
        probability = model.predict(
            pd.DataFrame(
                [{"symbol": DEFAULT_SYMBOL.lower(), "timestamp": datetime.utcnow(), **features}]
            )
        )
        if not isinstance(probability, (float, int)) or not math.isfinite(float(probability)):
            logger.warning("Model returned a non-finite probability. Falling back to neutral.")
            return 0.5
        return float(probability)
    except Exception as exc:
        logger.error(f"Safe predict fallback triggered: {exc}")
        return 0.5


async def _build_order_details(
    signal: str,
    features: dict,
    style_key: str,
    broker_adapter,
    probability: float,
) -> ExecutionPlan:
    style = TRADING_PROFILES["styles"].get(
        style_key,
        TRADING_PROFILES["styles"][DEFAULT_STYLE],
    )
    entry = float(features.get("close", 0.0))
    atr = float(features.get("ATR", entry * 0.005 if entry else 1.0))

    tp_pct = style.get("take_profit_pct", 1.5) / 100.0
    sl_pct = style.get("trailing_stop_pct", 0.75) / 100.0

    if signal == "BUY":
        stop_loss = round(entry * (1.0 - sl_pct), 2)
        take_profit = round(entry * (1.0 + tp_pct), 2)
    else:
        stop_loss = round(entry * (1.0 + sl_pct), 2)
        take_profit = round(entry * (1.0 - tp_pct), 2)

    risk_per_unit = abs(entry - stop_loss)
    reward_per_unit = abs(entry - take_profit)
    risk_reward = round(reward_per_unit / risk_per_unit, 2) if risk_per_unit > 0 else 0.0

    high_confidence_confluence = (
        (signal == "BUY" and features.get("fvg_signal", 0.0) > 0 and features.get("liquidity_sweep_signal", 0.0) > 0)
        or (signal == "SELL" and features.get("fvg_signal", 0.0) < 0 and features.get("liquidity_sweep_signal", 0.0) < 0)
    )
    try:
        allocation = await runtime.risk_manager.calculate_position_size(
            symbol=DEFAULT_SYMBOL,
            action=signal,
            confidence=probability if signal == "BUY" else 1.0 - probability,
            reward_risk_ratio=max(risk_reward, 1.0),
            kelly_fraction=style["kelly_fraction"],
            high_confidence_confluence=high_confidence_confluence,
            broker_adapter=broker_adapter,
        )
    except Exception as exc:
        logger.error(f"Error calculating position size: {exc}")
        allocation = 0.0

    current_capital = await broker_adapter.get_account_balance() if broker_adapter else runtime.risk_manager.total_capital
    position_qty = round(allocation / entry, 4) if entry > 0 and allocation > 0 else 0.0
    leverage = round(allocation / current_capital, 1) if current_capital > 0 else 1.0

    return ExecutionPlan(
        entry_price=round(entry, 2),
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        position_qty=position_qty,
        allocation=allocation,
        leverage=leverage,
        est_pnl=round(position_qty * reward_per_unit, 2),
        max_loss=round(position_qty * risk_per_unit, 2),
        sl_distance_pct=round((risk_per_unit / entry) * 100, 3) if entry > 0 else 0.0,
        tp_distance_pct=round((reward_per_unit / entry) * 100, 3) if entry > 0 else 0.0,
        style=style_key,
        timeframe=style["timeframe"],
    )


def _get_style_thresholds(style_key: str) -> tuple[float, float]:
    style = TRADING_PROFILES["styles"].get(
        style_key,
        TRADING_PROFILES["styles"][DEFAULT_STYLE],
    )
    return float(style["buy_threshold"]), float(style["sell_threshold"])


def _probability_zone(probability: float, style_key: str = DEFAULT_STYLE) -> str:
    buy_threshold, sell_threshold = _get_style_thresholds(style_key)
    if probability >= buy_threshold:
        return "BUY"
    if probability <= sell_threshold:
        return "SELL"
    return "NEUTRAL"


def _crossed_execution_threshold(
    previous_probability_zone: str,
    probability: float,
    style_key: str = DEFAULT_STYLE,
) -> tuple[str, bool]:
    current_zone = _probability_zone(probability, style_key)
    crossed = current_zone in {"BUY", "SELL"} and current_zone != previous_probability_zone
    return current_zone, crossed


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def _build_structural_edge(features: dict) -> tuple[str, float, list[str], bool]:
    structure_signals = []
    structural_score = 0.0

    fvg_signal = float(features.get("fvg_signal", 0.0))
    sweep_signal = float(features.get("liquidity_sweep_signal", 0.0))
    structure_break_signal = float(features.get("structure_break_signal", 0.0))
    reclaim_strength = float(features.get("liquidity_reclaim_strength", 0.0))
    break_strength = float(features.get("structure_break_strength", 0.0))
    structural_confluence = bool(features.get("structural_confluence", 0.0))

    if fvg_signal > 0:
        structure_signals.append("Bullish FVG")
        structural_score += 0.2 + min(float(features.get("fvg_gap_pct", 0.0)) * 10, 0.15)
    elif fvg_signal < 0:
        structure_signals.append("Bearish FVG")
        structural_score += 0.2 + min(float(features.get("fvg_gap_pct", 0.0)) * 10, 0.15)

    if sweep_signal > 0:
        structure_signals.append("Bullish Liquidity Sweep")
        structural_score += 0.25 + min(reclaim_strength * 0.08, 0.15)
    elif sweep_signal < 0:
        structure_signals.append("Bearish Liquidity Sweep")
        structural_score += 0.25 + min(reclaim_strength * 0.08, 0.15)

    if structure_break_signal == 2:
        structure_signals.append("Bullish BOS")
        structural_score += 0.25 + min(break_strength * 0.08, 0.15)
    elif structure_break_signal == 1:
        structure_signals.append("Bullish CHoCH")
        structural_score += 0.22 + min(break_strength * 0.08, 0.12)
    elif structure_break_signal == -2:
        structure_signals.append("Bearish BOS")
        structural_score += 0.25 + min(break_strength * 0.08, 0.15)
    elif structure_break_signal == -1:
        structure_signals.append("Bearish CHoCH")
        structural_score += 0.22 + min(break_strength * 0.08, 0.12)

    if structural_confluence:
        structure_signals.append("High-Confidence Confluence")
        structural_score += 0.15

    if not structure_signals:
        return "No major institutional structure edge detected.", 0.0, [], False

    return " + ".join(structure_signals) + " detected", round(_clamp(structural_score), 4), structure_signals, structural_confluence


def _build_signal_analysis(features: dict, bullish_probability: float, sentiment: float) -> SignalAnalysis:
    close_price = float(features.get("close") or 0.0)
    vwap = float(features.get("VWAP") or close_price)
    rsi = float(features.get("RSI") or 50.0)
    macd_hist = float(features.get("MACD_hist") or 0.0)

    structural_edge, structural_score, _, _ = _build_structural_edge(features)
    technical_score = _clamp(
        0.5
        + ((close_price - vwap) / max(abs(vwap), 1.0)) * 4
        + ((rsi - 50.0) / 100.0)
        + math.tanh(macd_hist) * 0.2
        + structural_score * 0.25
    )
    sentiment_score = _clamp(sentiment)

    if close_price >= vwap and rsi >= 50:
        technical_summary = (
            f"Price is holding above VWAP with RSI at {rsi:.1f}, keeping technical momentum constructive."
        )
    elif close_price < vwap and rsi < 50:
        technical_summary = (
            f"Price is trading below VWAP while RSI sits at {rsi:.1f}, signaling weaker trend participation."
        )
    else:
        technical_summary = (
            f"Technical posture is mixed: RSI is {rsi:.1f} and price is testing its VWAP equilibrium."
        )

    if structural_score > 0:
        technical_summary = f"{technical_summary} {structural_edge}."

    if sentiment_score >= 0.6:
        sentiment_summary = "Sentiment flow is supportive and improves conviction for momentum continuation."
    elif sentiment_score <= 0.4:
        sentiment_summary = "Sentiment remains defensive, so any entry should respect tighter downside protection."
    else:
        sentiment_summary = "Sentiment is neutral, so execution should lean more heavily on price structure than headlines."

    return SignalAnalysis(
        technical_score=round(technical_score, 4),
        sentiment_score=round(sentiment_score, 4),
        technical_summary=technical_summary,
        sentiment_summary=sentiment_summary,
    )


def _build_synthetic_features(symbol: str = "BTCUSDT") -> dict:
    """
    Baseline synthetic feature vector used as fallback when live data is
    unavailable. Keys match MODEL_FEATURE_COLUMNS exactly (30 features).
    BOS signal values are clamped to {-1.0, 0.0, 1.0} matching training labels.
    """
    runtime.synthetic_tick += 1
    phase = runtime.synthetic_tick / 5
    
    symbol_upper = symbol.upper()
    if "NIFTY" in symbol_upper:
        base_price = 48500
        volatility = 120
    elif "EURUSD" in symbol_upper:
        base_price = 1.0850
        volatility = 0.0020
    else:
        base_price = 68450
        volatility = 210
        
    close_price = base_price + math.sin(phase) * volatility + math.cos(phase / 2) * (volatility * 0.4)
    atr = (volatility * 0.45) + abs(math.sin(phase / 3)) * (volatility * 0.08)
    vwap = close_price - math.sin(phase / 2) * (volatility * 0.1)
    rsi = 50 + math.sin(phase) * 11
    macd_hist = math.sin(phase / 1.8) * 0.35

    return {
        # ── OHLCV ──────────────────────────────────────────────────────────
        "open":   round(close_price - 32, 4),
        "high":   round(close_price + 58, 4),
        "low":    round(close_price - 66, 4),
        "close":  round(close_price, 4),
        "volume": round(1542 + abs(math.cos(phase)) * 325, 4),
        # ── Core indicators ────────────────────────────────────────────────
        "RSI":         round(rsi, 4),
        "EMA_14":      round(close_price - 18, 4),
        "EMA_50":      round(close_price - 42, 4),
        "MACD":        round(macd_hist * 2, 6),
        "MACD_signal": round(macd_hist, 6),
        "MACD_hist":   round(macd_hist, 6),
        "VWAP":        round(vwap, 4),   # live feed provides real VWAP/TWAP
        "ATR":         round(atr, 4),
        "CVD":         round(math.sin(phase) * 240, 4),
        # ── LOB / execution ────────────────────────────────────────────────
        "spread":              0.6,
        "best_bid":            round(close_price - 0.2, 4),
        "best_bid_qty":        12.4,
        "best_ask":            round(close_price + 0.2, 4),
        "best_ask_qty":        11.9,
        "recent_long_liq_vol": round(abs(math.sin(phase)) * 25, 4),
        "recent_short_liq_vol":round(abs(math.cos(phase)) * 21, 4),
        "liq_imbalance":       round(math.sin(phase) * 8, 4),
        # ── SMC / structure ────────────────────────────────────────────────
        "fvg_signal":                1.0 if math.sin(phase) > 0.55 else -1.0 if math.sin(phase) < -0.55 else 0.0,
        "fvg_gap_pct":               round(abs(math.sin(phase)) * 0.002, 6),
        "liquidity_sweep_signal":    1.0 if math.cos(phase) > 0.6  else -1.0 if math.cos(phase) < -0.6  else 0.0,
        "liquidity_reclaim_strength":round(abs(math.cos(phase)) * 1.0, 6),   # clamped to [0, 1]
        "structure_break_signal":    1.0 if math.sin(phase / 1.7) > 0.65 else -1.0 if math.sin(phase / 1.7) < -0.65 else 0.0,  # was ±2.0 — fixed
        "structure_break_strength":  round(abs(math.sin(phase / 1.7)) * 1.0, 6),
        "structural_confluence":     1.0 if abs(math.sin(phase)) > 0.7 and abs(math.cos(phase)) > 0.7 else 0.0,
        # ── Macro ──────────────────────────────────────────────────────────
        "macro_sentiment_score": 0.5,
    }


async def _load_market_features(symbol: str) -> dict:
    fetcher = DataFetcher(db_manager)
    # Route to NSE session-aware pipeline for any NIFTY variant (BANKNIFTY, NIFTY, etc.)
    market_type = "nse" if "NIFTY" in symbol.upper() else "crypto"
    dataframe = await fetcher.build_dataset(symbol.lower(), market_type=market_type)
    if dataframe.empty:
        warmup_status = await fetcher.fetch_warmup_status(symbol.lower())
        if not warmup_status.get("ready"):
            logger.info(
                "QuantModel warm-up pending for {} | fresh_1m_candles={}/{}",
                symbol.upper(),
                warmup_status.get("candle_count", 0),
                warmup_status.get("required_candles", 14),
            )
            return {}

        logger.warning("No warm market features available for dashboard feed.")
        return {}

    row = dataframe.iloc[0]
    features = {
        key: round(float(value), 6)
        for key, value in row.items()
        if key not in {"timestamp", "symbol"}
    }

    # NSE index instruments report zero volume — VWAP is stored as session TWAP.
    # Guard against zero/NaN so the frontend always has a valid reference price.
    vwap_val = features.get("VWAP", 0.0)
    if not vwap_val or not math.isfinite(float(vwap_val)) or float(vwap_val) == 0.0:
        features["VWAP"] = features.get("close", 0.0)

    return features


def _build_trade_ticket(
    order_id: str,
    signal: str,
    probability: float,
    sentiment: float,
    features: dict,
    execution_plan: ExecutionPlan,
    trigger_source: str,
) -> TradeTicket:
    now = datetime.now()
    return TradeTicket(
        order_id=order_id,
        symbol=DEFAULT_SYMBOL,
        signal=signal,
        status="EXECUTED" if trigger_source == "manual_override" else "STAGED",
        probability=round(probability, 4),
        provider=_get_provider_name(DEFAULT_SYMBOL),
        trigger_source=trigger_source,
        time=now.strftime("%I:%M %p"),
        date=now.strftime("%b %d, %Y"),
        sentiment=round(sentiment, 4),
        features=features,
        execution_plan=execution_plan,
    )


def _update_trade_ticket_status(order_id: str, status: str, trigger_source: Optional[str] = None):
    for trade in runtime.trade_history:
        if trade.order_id != order_id:
            continue
        trade.status = status
        if trigger_source:
            trade.trigger_source = trigger_source
        return trade
    return None


def _build_active_trades(current_price: float) -> list[ActiveTrade]:
    runtime.executor.sync_live_prices(DEFAULT_SYMBOL, current_price)
    active_trades = []
    for trade in runtime.executor.list_active_trades():
        active_trades.append(
            ActiveTrade(
                order_id=trade["order_id"],
                symbol=trade["symbol"],
                side=trade["side"],
                status=trade.get("status", "OPEN"),
                quantity=float(trade.get("quantity", 0.0)),
                entry_price=float(trade.get("entry_price", 0.0)),
                current_price=float(trade.get("current_price", 0.0)),
                pnl_pct=float(trade.get("pnl_pct", 0.0)),
                trailing_stop_level=float(trade.get("trailing_stop_level", 0.0)),
                callback_rate=float(trade.get("callback_rate", 0.0)),
                broker_status=trade.get("broker_status", "DRY_RUN"),
                exit_reason=trade.get("exit_reason"),
                opened_at=trade.get("opened_at", ""),
            )
        )
    return active_trades


def _build_snapshot(
    signal: str,
    probability: float,
    sentiment: float,
    features: dict,
    execution_plan: Optional[ExecutionPlan],
    warmup_status: Optional[dict] = None,
) -> DashboardSnapshot:
    analysis = _build_signal_analysis(features, probability, sentiment)
    structural_edge, structural_score, structure_signals, structural_confluence = _build_structural_edge(features)
    requires_manual_approval = runtime.risk_manager.requires_manual_approval and signal != "HOLD"
    squaring_off_active = any(
        trade.get("status") == "SQUARING OFF"
        for trade in runtime.executor.list_active_trades()
    )
    warmup_state = warmup_status or runtime.latest_warmup_status or _default_warmup_status()
    warmup_count = int(warmup_state.get("candle_count", 0))
    warmup_required = int(warmup_state.get("required_candles", WARMUP_CANDLES_REQUIRED))
    warmup_ready = bool(warmup_state.get("ready", False))
    warmup_message = None if warmup_ready else f"Warming Up: [{warmup_count}]/{warmup_required} Minutes"

    if squaring_off_active:
        status = "SQUARING OFF"
    elif not warmup_ready:
        status = "warming_up"
    elif signal == "HOLD":
        status = "monitoring"
    elif requires_manual_approval:
        status = "awaiting_manual_approval"
    else:
        status = "autonomous_routing"

    smart_order_card = SmartOrderCard(
        order_id=runtime.trade_history[0].order_id if runtime.trade_history else None,
        symbol=DEFAULT_SYMBOL,
        provider=_get_provider_name(DEFAULT_SYMBOL),
        signal=signal,
        bullish_probability=round(probability, 4),
        bearish_probability=round(1 - probability, 4),
        confidence_percent=round(max(probability, 1 - probability) * 100, 2),
        signal_analysis=analysis,
        structural_edge=structural_edge,
        structural_score=structural_score,
        structure_signals=structure_signals,
        structural_confluence=structural_confluence,
        execution_plan=execution_plan,
        requires_manual_approval=requires_manual_approval,
        execution_mode=runtime.risk_manager.execution_mode,
        status=status,
        warmup_count=warmup_count,
        warmup_required=warmup_required,
        warmup_message=warmup_message,
        execute_label=(
            "Autonomous Exit Running"
            if squaring_off_active
            else "Warm-up In Progress"
            if not warmup_ready
            else "Execute Now"
        ),
    )

    runtime.latest_signal_card = smart_order_card

    snapshot = DashboardSnapshot(
        generated_at=datetime.utcnow(),
        session=DashboardSession(
            cycle_count=runtime.cycle_count,
            sim_pnl=round(runtime.sim_pnl, 2),
            total_capital=runtime.risk_manager.total_capital,
            trade_allocation=runtime.risk_manager.trade_allocation,
            professional_trader_enabled=runtime.risk_manager.professional_trader_enabled,
            autonomous_mode=runtime.risk_manager.autonomous_mode,
            execution_mode=runtime.risk_manager.execution_mode,
            db_status=runtime.db_status,
            active_tab=runtime.active_tab,
            trading_style=runtime.trading_style,
            capital_pools=runtime.risk_manager.capital_pools,
        ),
        market=DashboardMarket(
            symbol=DEFAULT_SYMBOL,
            provider=_get_provider_name(DEFAULT_SYMBOL),
            close_price=float(features.get("close") or 0.0),
            rsi=float(features.get("RSI") or 50.0),
            vwap=float(features.get("VWAP") or features.get("close") or 0.0),
            atr=float(features.get("ATR") or 0.0),
            sentiment=round(sentiment, 4),
            probability=round(probability, 4),
            signal=signal,
            structural_features={
                "fvg_signal": float(features.get("fvg_signal") or 0.0),
                "fvg_gap_pct": float(features.get("fvg_gap_pct") or 0.0),
                "liquidity_sweep_signal": float(features.get("liquidity_sweep_signal") or 0.0),
                "liquidity_reclaim_strength": float(features.get("liquidity_reclaim_strength") or 0.0),
                "structure_break_signal": float(features.get("structure_break_signal") or 0.0),
                "structure_break_strength": float(features.get("structure_break_strength") or 0.0),
                "structural_confluence": float(features.get("structural_confluence") or 0.0),
            },
        ),
        active_trades=_build_active_trades(float(features.get("close") or 0.0)),
        smart_order_card=smart_order_card,
        trades=runtime.trade_history[:20],
        live_signals=runtime.live_signals[:20],
        news=runtime.latest_news[:12],
        global_best_signal=runtime.global_best_signal,
    )
    runtime.latest_snapshot = snapshot
    return snapshot


async def _refresh_snapshot(event_type: str, warmup_status: Optional[dict] = None):
    features = _sanitize_features(runtime.latest_features)
    execution_plan = None
    if runtime.last_signal != "HOLD":
        execution_plan = await _build_order_details(
            signal=runtime.last_signal,
            features=features,
            style_key=DEFAULT_STYLE,
            broker_adapter=runtime.executor.broker,
            probability=runtime.latest_probability,
        )

    snapshot = _build_snapshot(
        signal=runtime.last_signal,
        probability=runtime.latest_probability,
        sentiment=runtime.latest_sentiment,
        features=features,
        execution_plan=execution_plan,
        warmup_status=warmup_status or runtime.latest_warmup_status,
    )
    await runtime.broadcaster.broadcast(event_type, snapshot)
    return snapshot


async def _publish_snapshot(
    event_type: str,
    signal: str,
    probability: float,
    sentiment: float,
    features: dict,
    execution_plan: Optional[ExecutionPlan],
    warmup_status: Optional[dict] = None,
):
    snapshot = _build_snapshot(
        signal,
        probability,
        sentiment,
        features,
        execution_plan,
        warmup_status=warmup_status or runtime.latest_warmup_status,
    )
    await runtime.broadcaster.broadcast(event_type, snapshot)


async def _execute_signal_order(
    signal: str,
    execution_plan: ExecutionPlan,
    order_id: str,
    trigger_source: str,
):
    if execution_plan.position_qty <= 0:
        logger.warning("Skipping execution because computed position size is zero.")
        return None

    # Use the zero-latency Redis price bridge to ensure we capture the exact full price string
    live_entry_price = await _get_redis_live_price()
    if live_entry_price <= 0:
        live_entry_price = execution_plan.entry_price

    logger.info(
        "Routing {} order to TradeExecutor | order_id={} | qty={} | live_entry={:.4f} | mode={}",
        signal,
        order_id,
        execution_plan.position_qty,
        live_entry_price,
        runtime.risk_manager.execution_mode,
    )
    return await runtime.executor.execute_market_order(
        symbol=DEFAULT_SYMBOL,
        side=signal,
        quantity=execution_plan.position_qty,
        order_id=order_id,
        entry_price=live_entry_price,
        trailing_stop_level=execution_plan.stop_loss,
        callback_rate=1.0,
    )


async def inference_loop():
    await asyncio.sleep(2)

    while True:
        try:
            runtime.cycle_count += 1
            raw_features = await _load_market_features(DEFAULT_SYMBOL)
            if not raw_features:
                warmup_status = await db_manager.get_warmup_status(DEFAULT_SYMBOL)
                runtime.latest_warmup_status = warmup_status
                runtime.latest_features = {}
                runtime.latest_probability = 0.5
                runtime.latest_sentiment = 0.5
                runtime.last_signal = "HOLD"
                runtime.last_probability_zone = "NEUTRAL"
                await _publish_snapshot(
                    "signal.update",
                    "HOLD",
                    0.5,
                    0.5,
                    _build_synthetic_features(DEFAULT_SYMBOL),
                    None,
                    warmup_status=warmup_status,
                )
                await asyncio.sleep(INFERENCE_LOOP_INTERVAL_SECONDS)
                continue

            features = _sanitize_features(raw_features)
            probability = _safe_predict(features)
            runtime.latest_warmup_status = await db_manager.get_warmup_status(DEFAULT_SYMBOL)
            probability_zone, crossed_execution_threshold = _crossed_execution_threshold(
                runtime.last_probability_zone,
                probability,
                DEFAULT_STYLE,
            )
            try:
                sentiment = await db_manager.fetch_latest_sentiment()
            except Exception as exc:
                logger.warning(f"Sentiment fetch degraded. Using neutral sentiment: {exc}")
                sentiment = 0.5
            decision = runtime.risk_manager.evaluate_trade_signal(
                DEFAULT_SYMBOL,
                probability,
                signal_price=float(features.get("signal_price", features.get("close", 0.0))),
                market_price=float(features.get("market_price", features.get("close", 0.0))),
                active_trades=runtime.executor.active_trades,  # One-signal guard
            )
            signal = decision["action"]
            runtime.latest_features = features
            runtime.latest_probability = probability
            runtime.latest_sentiment = sentiment

            exit_candidates = await runtime.executor.check_exit_conditions(
                symbol=DEFAULT_SYMBOL,
                current_probability=probability,
                structure_break_signal=float(features.get("structure_break_signal", 0.0)),
                current_price=float(features.get("close", 0.0)),
            )
            if exit_candidates:
                for exit_candidate in exit_candidates:
                    _update_trade_ticket_status(
                        exit_candidate["order_id"],
                        "SQUARING OFF",
                        trigger_source="autonomous_exit",
                    )

                await _publish_snapshot(
                    "trade.square_off_started",
                    "HOLD",
                    probability,
                    sentiment,
                    features,
                    None,
                )

                for exit_candidate in exit_candidates:
                    close_result = await runtime.executor.square_off_trade(
                        exit_candidate["order_id"],
                        current_price=float(features.get("close", 0.0)),
                        exit_reason=exit_candidate["exit_reason"],
                    )
                    if close_result is not None:
                        _update_trade_ticket_status(
                            exit_candidate["order_id"],
                            "SQUARED_OFF",
                            trigger_source="autonomous_exit",
                        )
                        # Arm the 5-minute revenge-trading cooldown
                        runtime.risk_manager.record_trade_closed(DEFAULT_SYMBOL)
                    else:
                        _update_trade_ticket_status(
                            exit_candidate["order_id"],
                            "EXIT_FAILED",
                            trigger_source="autonomous_exit",
                        )

                await _publish_snapshot(
                    "trade.closed",
                    "HOLD",
                    probability,
                    sentiment,
                    features,
                    None,
                )
                runtime.last_signal = "HOLD"
                continue

            execution_plan = None
            if signal != "HOLD":
                execution_plan = await _build_order_details(
                    signal=signal,
                    features=features,
                    style_key=DEFAULT_STYLE,
                    broker_adapter=runtime.executor.broker,
                    probability=probability,
                )
                
                # Dynamic ATR-based sim_pnl
                edge = abs(probability - 0.5) * 2.0
                atr_pct = (execution_plan.sl_distance_pct + execution_plan.tp_distance_pct) / 2.0
                margin_return = edge * atr_pct * execution_plan.risk_reward * 2.0
                runtime.sim_pnl += margin_return

            live_price = float(features.get("close", 0.0))
            logger.info(
                "⚡ INFERENCE | price=${:.2f} | prob={:.4f} | signal={} | mode={}",
                live_price,
                probability,
                signal,
                runtime.risk_manager.execution_mode,
            )

            # In autonomous mode: fire on every BUY/SELL cycle (not just on first change)
            # In manual mode: fire only on signal transition to avoid flooding the trade log
            is_autonomous = runtime.risk_manager.autonomous_mode
            signal_changed = signal != runtime.last_signal
            has_open_position = len(runtime.executor.list_active_trades()) > 0
            should_route_signal = (
                signal != "HOLD"
                and execution_plan is not None
                and (signal_changed or is_autonomous)
                and not has_open_position  # one trade at a time — prevent position stacking
            )

            # Decoupled signal ticket generation
            if probability > 0.80 or probability < 0.20:
                signal_dir = "BUY" if probability >= 0.5 else "SELL"
                # Only add if it's a new signal to avoid spamming 2 per second
                if not runtime.live_signals or runtime.live_signals[0].signal != signal_dir:
                    ep = execution_plan
                    if ep is None:
                        ep = await _build_order_details(
                            signal=signal_dir,
                            features=features,
                            style_key=DEFAULT_STYLE,
                            broker_adapter=runtime.executor.broker,
                            probability=probability,
                        )
                    ticket_id = str(uuid.uuid4())[:8]
                    ticket = _build_trade_ticket(
                        order_id=ticket_id,
                        signal=signal_dir,
                        probability=probability,
                        sentiment=sentiment,
                        features=features,
                        execution_plan=ep,
                        trigger_source="signal_engine"
                    )
                    runtime.live_signals.insert(0, ticket)
                    runtime.live_signals = runtime.live_signals[:20]

            if is_autonomous and should_route_signal:
                order_id = str(uuid.uuid4())[:8]
                executed = False
                if not decision["requires_manual_approval"]:
                    logger.info(
                    "Autonomous execution | signal={} | probability={:.4f} | price=${:.2f} | zone={}",
                        signal,
                        probability,
                        live_price,
                        probability_zone,
                    )
                    execution_result = await _execute_signal_order(
                        signal=signal,
                        execution_plan=execution_plan,
                        order_id=order_id,
                        trigger_source="autonomous_engine",
                    )
                    executed = execution_result is not None

                trade_ticket = _build_trade_ticket(
                    order_id=order_id,
                    signal=signal,
                    probability=probability,
                    sentiment=sentiment,
                    features=features,
                    execution_plan=execution_plan,
                    trigger_source=(
                        "autonomous_engine"
                        if not decision["requires_manual_approval"]
                        else "signal_engine"
                    ),
                )
                trade_ticket.status = "EXECUTED" if executed else trade_ticket.status
                runtime.trade_history.insert(0, trade_ticket)
                runtime.trade_history = runtime.trade_history[:50]
                await _publish_snapshot(
                    "trade.executed" if executed else "trade.staged",
                    signal,
                    probability,
                    sentiment,
                    features,
                    execution_plan,
                )
            else:
                await _publish_snapshot(
                    "signal.update",
                    signal,
                    probability,
                    sentiment,
                    features,
                    execution_plan,
                )

            runtime.last_signal = signal
            runtime.last_probability_zone = probability_zone
        except Exception as exc:
            logger.exception(f"Dashboard inference loop error: {exc}")
            fallback_features = _sanitize_features(runtime.latest_features)
            runtime.latest_features = fallback_features
            runtime.latest_probability = 0.5
            runtime.latest_sentiment = 0.5
            runtime.last_signal = "HOLD"
            runtime.last_probability_zone = "NEUTRAL"
            await _publish_snapshot(
                "signal.update",
                "HOLD",
                0.5,
                0.5,
                fallback_features,
                None,
            )

        await asyncio.sleep(INFERENCE_LOOP_INTERVAL_SECONDS)


async def database_heartbeat_loop():
    await asyncio.sleep(5)
    while True:
        try:
            is_healthy = await db_manager.ping()
            new_status = "connected" if is_healthy else "disconnected"
            if new_status != runtime.db_status:
                logger.warning(f"Database status changed: {runtime.db_status} -> {new_status}")
                runtime.db_status = new_status
                if runtime.latest_snapshot:
                    runtime.latest_snapshot.session.db_status = new_status
                    await runtime.broadcaster.broadcast("session.db_status", runtime.latest_snapshot)
        except Exception as exc:
            logger.error(f"Heartbeat loop error: {exc}")
        await asyncio.sleep(60)


async def alpha_ranker_loop():
    await asyncio.sleep(5)
    while True:
        try:
            fetcher = DataFetcher(db_manager)
            
            # Crypto
            crypto_df = await fetcher.build_dataset("btcusdt", market_type="crypto")
            if not crypto_df.empty:
                crypto_features = {k: float(v) for k, v in crypto_df.iloc[0].items() if k not in {"timestamp", "symbol"}}
                prob_crypto = model_registry["crypto"].predict(pd.DataFrame([{"symbol": "btcusdt", "timestamp": datetime.utcnow(), **crypto_features}]))
                if isinstance(prob_crypto, (float, int)) and math.isfinite(float(prob_crypto)):
                    runtime.crypto_probabilities.append(float(prob_crypto))
                    if len(runtime.crypto_probabilities) > 10:
                        runtime.crypto_probabilities.pop(0)

            # NSE
            nse_df = await fetcher.build_dataset("banknifty", market_type="nse")
            if not nse_df.empty:
                nse_features = {k: float(v) for k, v in nse_df.iloc[0].items() if k not in {"timestamp", "symbol"}}
                prob_nse = model_registry["nse"].predict(pd.DataFrame([{"symbol": "banknifty", "timestamp": datetime.utcnow(), **nse_features}]))
                if isinstance(prob_nse, (float, int)) and math.isfinite(float(prob_nse)):
                    runtime.nse_probabilities.append(float(prob_nse))
                    if len(runtime.nse_probabilities) > 10:
                        runtime.nse_probabilities.pop(0)

            # Compare
            crypto_avg = sum(runtime.crypto_probabilities) / len(runtime.crypto_probabilities) if runtime.crypto_probabilities else 0.5
            nse_avg = sum(runtime.nse_probabilities) / len(runtime.nse_probabilities) if runtime.nse_probabilities else 0.5

            crypto_conviction = abs(crypto_avg - 0.5)
            nse_conviction = abs(nse_avg - 0.5)

            if crypto_conviction >= nse_conviction:
                best_market = "CRYPTO"
                best_prob = crypto_avg
            else:
                best_market = "INDIA (NSE)"
                best_prob = nse_avg
                
            direction = "BULLISH" if best_prob >= 0.5 else "BEARISH"

            runtime.global_best_signal = {
                "market": best_market,
                "probability": round(best_prob, 4),
                "direction": direction,
                "conviction": round(max(crypto_conviction, nse_conviction) * 200, 2)
            }
            
            if runtime.latest_snapshot:
                runtime.latest_snapshot.global_best_signal = runtime.global_best_signal

        except Exception as exc:
            logger.error(f"Alpha Ranker loop error: {exc}")
            
        await asyncio.sleep(5)



_SYNTHETIC_NEWS = [
    "📊 Apex Engine: RSI monitoring active across all market pods",
    "⚡ VWAP equilibrium zones refreshed — session TWAP aligned",
    "🔍 Liquidity sweep detection armed: watching institutional pivots",
    "📈 ATR volatility band calibrated to current session range",
    "🛡️ Risk manager: cooldown protocols active across all pods",
    "🌏 NSE session: BankNifty OI data feeding structural analysis",
    "₿ Crypto pod: BTC order-book depth analysis running",
    "🔄 FVG (Fair Value Gap) scanner: monitoring premium/discount arrays",
    "📉 Bearish momentum watch: tracking divergence signals",
    "💹 Apex Council: multi-agent consensus layer active",
    "🎯 Execution engine: manual approval mode engaged",
    "📡 WebSocket feeds: all data streams healthy",
]


async def news_loop():
    # Seed the sentinel wire immediately with synthetic intelligence
    runtime.latest_news = _SYNTHETIC_NEWS
    while True:
        try:
            headlines = await runtime.news_fetcher.fetch_news("BTC,ETH,MACRO")
            if headlines:
                runtime.latest_news = headlines[:15]
                logger.info("Sentiment Wire refreshed: {} headlines fetched", len(headlines))
            else:
                # Keep synthetic wire active when no API key or empty response
                if not runtime.latest_news:
                    runtime.latest_news = _SYNTHETIC_NEWS
            
            if runtime.latest_snapshot is not None:
                runtime.latest_snapshot.news = runtime.latest_news[:12]
                await runtime.broadcaster.broadcast("news.update", runtime.latest_snapshot)
        except Exception as exc:
            logger.warning(f"Dashboard news loop degraded: {exc}")
            if not runtime.latest_news:
                runtime.latest_news = _SYNTHETIC_NEWS

        await asyncio.sleep(300)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Starting Apex Command Center backend on port 8080...")
    try:
        await db_manager.connect()
        runtime.latest_warmup_status = await db_manager.get_warmup_status(DEFAULT_SYMBOL)
        purge_summary = await purge_stale_market_data(db_manager.pg_pool)
        logger.warning("Startup nuclear reset completed before ws_server activation: {}", purge_summary)
    except Exception as exc:
        logger.warning(f"Database startup degraded. Dashboard will run in fallback mode: {exc}")

    if runtime.executor.live_trading_enabled:
        await runtime.executor.connect()

    inference_task = asyncio.create_task(inference_loop())
    news_task = asyncio.create_task(news_loop())
    heartbeat_task = asyncio.create_task(database_heartbeat_loop())
    alpha_ranker_task = asyncio.create_task(alpha_ranker_loop())

    try:
        yield
    finally:
        inference_task.cancel()
        news_task.cancel()
        heartbeat_task.cancel()
        alpha_ranker_task.cancel()
        await asyncio.gather(inference_task, news_task, heartbeat_task, alpha_ranker_task, return_exceptions=True)
        await runtime.executor.disconnect()
        await db_manager.disconnect()
        logger.info("Apex Command Center backend stopped.")


app = FastAPI(
    title="Apex Command Center API",
    version="4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "apex-command-center"}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await runtime.broadcaster.connect(websocket, runtime.latest_snapshot)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await runtime.broadcaster.disconnect(websocket)


@app.websocket("/ws/console")
async def ws_console(websocket: WebSocket):
    await websocket.accept()
    # Send all recent logs immediately
    for log in list(recent_logs):
        await websocket.send_text(log)
    
    q = asyncio.Queue()
    console_queues.append(q)
    try:
        while True:
            msg = await q.get()
            await websocket.send_text(msg)
    except WebSocketDisconnect:
        if q in console_queues:
            console_queues.remove(q)


@app.get("/api/dashboard/snapshot")
async def get_dashboard_snapshot():
    if runtime.latest_snapshot is None:
        runtime.latest_warmup_status = await db_manager.get_warmup_status(DEFAULT_SYMBOL)
        snapshot = _build_snapshot(
            signal="HOLD",
            probability=0.5,
            sentiment=0.5,
            features=_build_synthetic_features(),
            execution_plan=None,
            warmup_status=runtime.latest_warmup_status,
        )
        return snapshot
    return runtime.latest_snapshot


@app.get("/api/capital")
async def get_capital():
    return {"total_capital": runtime.risk_manager.total_capital}


@app.put("/api/capital")
async def update_capital(update: CapitalUpdate):
    runtime.risk_manager.update_capital(update.total_capital)
    if update.trade_allocation is not None:
        runtime.risk_manager.trade_allocation = update.trade_allocation
    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("session.capital")
    return {"status": "ok", "total_capital": runtime.risk_manager.total_capital}


@app.get("/api/risk/professional-trader")
async def get_professional_trader_mode():
    return {
        "enabled": runtime.risk_manager.professional_trader_enabled,
        "autonomous_mode": runtime.risk_manager.autonomous_mode,
        "execution_mode": runtime.risk_manager.execution_mode,
    }


@app.put("/api/risk/professional-trader")
async def update_professional_trader_mode(update: ProfessionalTraderUpdate):
    execution_mode = runtime.risk_manager.set_professional_trader_mode(update.enabled)

    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("risk.mode")

    return {
        "status": "ok",
        "enabled": update.enabled,
        "autonomous_mode": runtime.risk_manager.autonomous_mode,
        "execution_mode": execution_mode,
    }


@app.post("/api/capital/pools")
async def update_capital_pools(pools: dict):
    for pool, val in pools.items():
        if pool in runtime.risk_manager.capital_pools:
            runtime.risk_manager.capital_pools[pool] = float(val)
    
    runtime.risk_manager.total_capital = runtime.risk_manager.capital_pools.get(runtime.active_tab, runtime.risk_manager.total_capital)
    await _refresh_snapshot("capital.updated")
    return {"status": "success", "capital_pools": runtime.risk_manager.capital_pools}

@app.post("/api/session/style/{style_name}")
async def set_trading_style(style_name: str):
    if style_name not in TRADING_PROFILES["styles"]:
        raise HTTPException(status_code=400, detail="Invalid trading style")
    
    runtime.trading_style = style_name
    global DEFAULT_STYLE
    DEFAULT_STYLE = style_name
    
    # Override multipliers dynamically based on the selection
    if style_name == "Scalping":
        TRADING_PROFILES["styles"]["Scalping"]["take_profit_pct"] = 0.5
        TRADING_PROFILES["styles"]["Scalping"]["trailing_stop_pct"] = 0.25
    elif style_name == "Intraday":
        TRADING_PROFILES["styles"]["Intraday"]["take_profit_pct"] = 1.5
        TRADING_PROFILES["styles"]["Intraday"]["trailing_stop_pct"] = 0.75
    elif style_name == "Swing":
        TRADING_PROFILES["styles"]["Swing"]["take_profit_pct"] = 5.0
        TRADING_PROFILES["styles"]["Swing"]["trailing_stop_pct"] = 2.0
        
    await _refresh_snapshot("style.updated")
    return {"status": "success", "style": style_name}

@app.post("/api/orders/execute")
async def execute_now(request: ManualExecutionRequest):
    card = runtime.latest_signal_card
    if card is None or card.execution_plan is None or card.signal == "HOLD":
        raise HTTPException(status_code=409, detail="No active signal available for manual execution.")

    order_id = request.order_id or str(uuid.uuid4())[:8]
    execution_result = await _execute_signal_order(
        signal=card.signal,
        execution_plan=card.execution_plan,
        order_id=order_id,
        trigger_source=request.reason,
    )
    if execution_result is None:
        raise HTTPException(status_code=409, detail="Order execution was rejected because size is zero.")

    trade_ticket = _build_trade_ticket(
        order_id=order_id,
        signal=card.signal,
        probability=card.bullish_probability,
        sentiment=runtime.latest_snapshot.market.sentiment if runtime.latest_snapshot else 0.5,
        features=runtime.latest_features or _build_synthetic_features(),
        execution_plan=card.execution_plan,
        trigger_source=request.reason,
    )
    trade_ticket.status = "EXECUTED"
    runtime.trade_history.insert(0, trade_ticket)
    runtime.trade_history = runtime.trade_history[:50]

    snapshot = _build_snapshot(
        signal=card.signal,
        probability=card.bullish_probability,
        sentiment=runtime.latest_snapshot.market.sentiment if runtime.latest_snapshot else 0.5,
        features=trade_ticket.features,
        execution_plan=card.execution_plan,
    )
    await runtime.broadcaster.broadcast("trade.executed", snapshot)
    return {"status": "ok", "order_id": order_id}


@app.get("/api/orders/active")
async def get_active_trades():
    current_price = 0.0
    if runtime.latest_snapshot is not None:
        current_price = runtime.latest_snapshot.market.close_price
    elif runtime.latest_features:
        current_price = float(runtime.latest_features.get("close", 0.0))

    runtime.executor.sync_live_prices(DEFAULT_SYMBOL, current_price)
    return {"active_trades": runtime.executor.list_active_trades()}


@app.post("/api/orders/close/{order_id}")
async def close_trade(order_id: str):
    current_price = 0.0
    if runtime.latest_snapshot is not None:
        current_price = runtime.latest_snapshot.market.close_price
    elif runtime.latest_features:
        current_price = float(runtime.latest_features.get("close", 0.0))

    _update_trade_ticket_status(order_id, "SQUARING OFF", trigger_source="manual_close")
    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("trade.square_off_started")

    close_result = await runtime.executor.close_trade(order_id, current_price=current_price)
    if close_result is None:
        raise HTTPException(status_code=404, detail="Active trade not found.")

    _update_trade_ticket_status(order_id, "SQUARED_OFF", trigger_source="manual_close")
    # Arm the 5-minute revenge-trading cooldown
    runtime.risk_manager.record_trade_closed(DEFAULT_SYMBOL)
    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("trade.closed")

    return {"status": "ok", "order_id": order_id, "closed_trade": close_result["closed_trade"]}


@app.post("/api/orders/clear-stale")
async def clear_stale_trades():
    await runtime.executor.clear_all_stale_trades()
    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("trades.cleared")
    return {"status": "ok"}


@app.get("/api/trades")
async def get_trades():
    return {"trades": runtime.trade_history[:50]}


@app.get("/inference/details/{order_id}")
async def get_trade_details(order_id: str):
    for trade in runtime.trade_history:
        if trade.order_id == order_id:
            return trade
    raise HTTPException(status_code=404, detail="Trade not found")


@app.get("/api/profiles")
async def get_profiles():
    return TRADING_PROFILES


@app.get("/api/news")
async def get_news():
    return {"news": runtime.latest_news[:15]}


@app.post("/api/market/tab/{tab_name}")
async def switch_market_tab(tab_name: str):
    global DEFAULT_SYMBOL
    runtime.active_tab = tab_name.upper()
    if runtime.active_tab == "INDIA":
        DEFAULT_SYMBOL = "BANKNIFTY"
    elif runtime.active_tab == "FOREX":
        DEFAULT_SYMBOL = "EURUSD"
    else:
        DEFAULT_SYMBOL = "BTCUSDT"
        
    if runtime.latest_snapshot is not None:
        await _refresh_snapshot("tab.switched")
    return {"status": "ok", "active_tab": runtime.active_tab, "symbol": DEFAULT_SYMBOL}


@app.get("/api/market/tick")
async def get_market_tick():
    """Zero-latency price endpoint: reads directly from Redis candle key.
    Frontend polls this every 100 ms for tick-for-tick price parity with TradingView."""
    redis_price = await _get_redis_live_price()
    # Fall back gracefully to latest snapshot price when Redis is cold / warming up
    if redis_price <= 0.0 and runtime.latest_snapshot is not None:
        redis_price = runtime.latest_snapshot.market.close_price
    return {
        "price": redis_price,
        "symbol": DEFAULT_SYMBOL,
        "ts": datetime.utcnow().isoformat(),
    }
