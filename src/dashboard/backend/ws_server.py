"""
Apex Intelligence Engine v3.0 — WebSocket Broadcast Server
High-frequency data broadcaster for the React Command Center.
Port: 8080
"""
import sys
import os
import asyncio
import json
import uuid
from datetime import datetime
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.data.db import db_manager
from src.models.dataset_builder import DataFetcher
from src.models.quant_model import ApexXGBoostModel
from src.dashboard.config import TRADING_PROFILES
from src.data.news_fetcher import CryptoNewsFetcher

# ─── Global State ───────────────────────────────────────────────────
quant_model = ApexXGBoostModel()
news_fetcher = CryptoNewsFetcher()
trade_history: List[dict] = []
sim_pnl: float = 0.0
total_capital: float = 10000.0
latest_news: List[str] = []


class CapitalUpdate(BaseModel):
    total_capital: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Apex WebSocket Broadcast Server on port 8080...")
    await db_manager.connect()
    asyncio.create_task(inference_loop())
    asyncio.create_task(news_loop())
    yield
    logger.info("Shutting down WebSocket server...")
    await db_manager.disconnect()


app = FastAPI(
    title="Apex Command Center API",
    version="3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Connected WebSocket Clients ────────────────────────────────────
connected_clients: List[WebSocket] = []


async def broadcast(payload: dict):
    """Send data to all connected WebSocket clients."""
    dead = []
    message = json.dumps(payload)
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


# ─── WebSocket: Live Data Stream ────────────────────────────────────
@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    logger.info(f"Client connected. Total clients: {len(connected_clients)}")
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
            except asyncio.TimeoutError:
                pass
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


        logger.info(f"Client disconnected. Total clients: {len(connected_clients)}")


# ─── Order Detail Calculator ─────────────────────────────────────────
def _build_order_details(signal: str, features: dict, style_key: str, capital: float) -> dict:
    """
    Compute full broker-style order details from live features and style profile.
    Uses ATR-based stop loss and take profit for a realistic R:R.
    """
    style = TRADING_PROFILES["styles"].get(style_key, TRADING_PROFILES["styles"]["Intraday"])
    entry = features.get("close", 0.0)
    atr = features.get("ATR", entry * 0.005)

    if signal == "BUY":
        stop_loss = round(entry - (atr * 1.5), 2)
        take_profit = round(entry + (atr * 3.0), 2)  # 1:2 R:R
    else:  # SELL / SHORT
        stop_loss = round(entry + (atr * 1.5), 2)
        take_profit = round(entry - (atr * 3.0), 2)

    risk_per_unit = abs(entry - stop_loss)
    reward_per_unit = abs(entry - take_profit)
    risk_reward = round(reward_per_unit / risk_per_unit, 2) if risk_per_unit > 0 else 0.0

    # Size position to risk exactly 1% of capital
    risk_capital = capital * 0.01
    position_qty = round(risk_capital / risk_per_unit, 4) if risk_per_unit > 0 else 0.0
    allocation = round(position_qty * entry, 2)
    
    # Calculate implied leverage
    leverage = round(allocation / capital, 1) if capital > 0 else 1.0

    est_pnl = round(position_qty * reward_per_unit, 2)
    max_loss = round(position_qty * risk_per_unit, 2)

    return {
        "entry_price": round(entry, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": risk_reward,
        "position_qty": position_qty,
        "allocation": allocation,
        "leverage": leverage,
        "est_pnl": est_pnl,
        "max_loss": max_loss,
        "sl_distance_pct": round((risk_per_unit / entry) * 100, 3) if entry > 0 else 0,
        "tp_distance_pct": round((reward_per_unit / entry) * 100, 3) if entry > 0 else 0,
        "style": style_key,
        "timeframe": style["timeframe"],
    }


# ─── Background: Inference Broadcast Loop ───────────────────────────
async def inference_loop():
    """
    Runs inference every 10 seconds.
    Trade Cards are ONLY created when the signal CHANGES (e.g. HOLD→SELL),
    eliminating duplicate card spam.
    """
    global sim_pnl
    await asyncio.sleep(3)

    last_signal = "HOLD"  # Tracks the previous cycle's signal

    while True:
        try:
            fetcher = DataFetcher(db_manager)
            df = await fetcher.build_dataset("btcusdt")

            if df.empty:
                await asyncio.sleep(10)
                continue

            prob = quant_model.predict(df)
            sentiment = await db_manager.fetch_latest_sentiment()

            style = TRADING_PROFILES["styles"]["Intraday"]
            if prob > style["buy_threshold"]:
                signal = "BUY"
            elif prob < style["sell_threshold"]:
                signal = "SELL"
            else:
                signal = "HOLD"

            # Accumulate simulated PnL
            if signal == "BUY":
                sim_pnl += (prob - 0.5) * 100
            elif signal == "SELL":
                sim_pnl += (0.5 - prob) * 100

            # Extract live feature weights
            row = df.iloc[0]
            features = {k: round(float(v), 6) for k, v in row.items()
                        if k not in ['timestamp', 'symbol']}

            # ── Only emit a trade card when the signal direction changes ──
            order_id = None
            if signal != "HOLD" and signal != last_signal:
                order_id = str(uuid.uuid4())[:8]
                order_details = _build_order_details(signal, features, "Intraday", total_capital)
                trade_entry = {
                    "order_id": order_id,
                    "time": datetime.now().strftime("%I:%M %p"),
                    "date": datetime.now().strftime("%b %d, %Y"),
                    "signal": signal,
                    "probability": round(prob, 4),
                    "symbol": "BTCUSDT",
                    "features": features,
                    "sentiment": round(sentiment, 4) if sentiment else 0.5,
                    **order_details,
                }
                trade_history.append(trade_entry)
                if len(trade_history) > 50:
                    trade_history.pop(0)
                logger.info(f"New trade card: {signal} @ {features.get('close', 0):.2f} | P={prob:.4f} | TP={order_details['take_profit']} | SL={order_details['stop_loss']}")

            last_signal = signal

            payload = {
                "type": "inference",
                "timestamp": datetime.now().isoformat(),
                "symbol": "BTCUSDT",
                "probability": round(prob, 4),
                "signal": signal,
                "sentiment": round(sentiment, 4) if sentiment else 0.5,
                "sim_pnl": round(sim_pnl, 2),
                "total_capital": total_capital,
                "features": features,
                "order_id": order_id,
                "close_price": features.get("close", 0),
                "rsi": features.get("RSI", 50),
                "vwap": features.get("VWAP", 0),
                "atr": features.get("ATR", 0),
            }

            await broadcast(payload)

        except Exception as e:
            logger.error(f"Inference loop error: {e}")

        await asyncio.sleep(10)


async def news_loop():
    """Fetches latest crypto headlines every 5 minutes."""
    global latest_news
    await asyncio.sleep(2)

    while True:
        try:
            headlines = await news_fetcher.fetch_news("BTC,ETH,MACRO")
            if headlines:
                latest_news = headlines[:15]
        except Exception as e:
            logger.error(f"News loop error: {e}")

        await asyncio.sleep(300)


# ─── REST Endpoints ─────────────────────────────────────────────────

@app.put("/api/capital")
async def update_capital(update: CapitalUpdate):
    """Sync total_capital from the frontend."""
    global total_capital
    total_capital = update.total_capital
    logger.info(f"Capital updated to ${total_capital:.2f}")
    return {"status": "ok", "total_capital": total_capital}


@app.get("/api/capital")
async def get_capital():
    return {"total_capital": total_capital}


@app.get("/inference/details/{order_id}")
async def get_trade_details(order_id: str):
    """Returns the full trade details for a specific order."""
    for trade in trade_history:
        if trade["order_id"] == order_id:
            return trade
    raise HTTPException(status_code=404, detail="Trade not found")


@app.get("/api/trades")
async def get_trades():
    """Returns the last 50 trades."""
    return {"trades": trade_history[-50:]}


@app.get("/api/profiles")
async def get_profiles():
    """Returns available trading profiles."""
    return TRADING_PROFILES


@app.get("/api/news")
async def get_news():
    """Returns the latest fetched crypto headlines."""
    return {"news": latest_news}
