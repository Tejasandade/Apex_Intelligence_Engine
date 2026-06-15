"""
Apex Intelligence Engine V5 — Dashboard WebSocket Server
==========================================================
Broadcasts live engine events to the web dashboard via WebSocket.

The LiveRunner pushes events here, and the dashboard connects
to receive them in real-time.

Events broadcast:
    status   — System status (balance, P&L, regime, etc.)
    signal   — New signal generated (with council decision)
    council  — Council voting details
    ensemble — Ensemble prediction breakdown
    trade_open  — New position opened
    trade_close — Position closed (with P&L)
    tick     — Price tick update (unrealized P&L)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from src.core.logging import get_logger

logger = get_logger("apex.dashboard.server")

# Try importing websockets, gracefully handle if not installed
try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not installed — dashboard server disabled")


class DashboardServer:
    """
    WebSocket server that broadcasts live engine events to the dashboard.

    Usage:
        server = DashboardServer(port=8765)
        await server.start()
        server.broadcast_signal({...})  # Called from LiveRunner
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set[websockets.WebSocketServerProtocol] = set()
        self._server = None
        self._running = False
        self._message_count = 0
        self._last_messages: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.warning("Cannot start dashboard server — websockets not installed")
            return

        self._running = True
        self._server = await serve(
            self._handle_client,
            self.host,
            self.port,
        )

        logger.info(
            "dashboard_server_started",
            host=self.host,
            port=self.port,
            url=f"ws://{self.host}:{self.port}",
        )

    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("dashboard_server_stopped", messages_sent=self._message_count)

    async def _handle_client(self, websocket) -> None:
        """Handle a new dashboard client connection."""
        self._clients.add(websocket)
        client_id = f"client_{len(self._clients)}"

        logger.info("dashboard_client_connected", client=id(websocket), total=len(self._clients))
        
        # Send cached history to new client
        for msg_type, message in self._last_messages.items():
            try:
                await websocket.send(message)
            except Exception:
                pass

        try:
            async for message in websocket:
                # Handle incoming messages from dashboard (commands like pause/resume, etc)
                cmd = json.loads(message).get("command")
                if cmd == "ping":
                    await websocket.send(json.dumps({"type": "pong", "timestamp": time.time()}))
        except Exception:
            pass
        finally:
            self._clients.remove(websocket)
            logger.info("dashboard_client_disconnected", total=len(self._clients))

    def _broadcast(self, msg_type: str, data: dict[str, Any]) -> None:
        """Broadcast a message to all connected dashboard clients."""
        # Handle potential numpy types from models
        def default_encoder(obj):
            if hasattr(obj, "item"):  # Handle numpy scalar types
                return obj.item()
        msg_dict = {"type": msg_type, "data": data, "ts": time.time()}
        message = json.dumps(msg_dict, default=default_encoder)
        
        # Cache for new connections
        self._last_messages[msg_type] = message

        if not self._clients:
            return

        self._message_count += 1

        # Fire and forget — don't block the trading engine
        disconnected = set()
        for client in self._clients:
            try:
                asyncio.create_task(client.send(message))
            except Exception:
                disconnected.add(client)

        self._clients -= disconnected

    # ── Broadcast Methods (called by LiveRunner) ─────────────────────────────

    def broadcast_status(
        self,
        symbol: str = "",
        balance: float = 0,
        initial_balance: float = 0,
        total_pnl: float = 0,
        total_trades: int = 0,
        winning_trades: int = 0,
        drawdown_pct: float = 0,
        regime: str = "",
        adx: float = 0,
        signals: int = 0,
        candles: int = 0,
        cooldown_remaining: int = 0,
        trailing_stops: dict = None,
        currency: str = "$",
    ) -> None:
        """Broadcast system status update."""
        if trailing_stops is None:
            trailing_stops = {}
            
        self._broadcast("status", {
            "symbol": symbol,
            "balance": balance,
            "initial_balance": initial_balance,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "drawdown_pct": drawdown_pct,
            "regime": regime,
            "adx": adx,
            "signals": signals,
            "candles": candles,
            "cooldown_remaining": cooldown_remaining,
            "trailing_stops": trailing_stops,
            "currency": currency,
        })

    def broadcast_ensemble(self, trending: float, ranging: float, agreement: float) -> None:
        """Broadcast underlying ensemble model probabilities."""
        self._broadcast("ensemble", {
            "trending": trending,
            "ranging": ranging,
            "agreement": agreement,
        })

    def broadcast_signal(
        self,
        direction: str,
        symbol: str,
        price: float,
        conviction: float,
        council_score: float,
        blocked: bool = False,
        reason: str = "",
    ) -> None:
        """Broadcast a new signal (approved or blocked)."""
        self._broadcast("signal", {
            "direction": direction,
            "symbol": symbol,
            "price": price,
            "conviction": conviction,
            "council_score": council_score,
            "blocked": blocked,
            "reason": reason,
        })

    def broadcast_council(
        self,
        votes: dict[str, dict],
        consensus: float,
        approved: bool,
        summary: str = "",
    ) -> None:
        """Broadcast council voting details."""
        self._broadcast("council", {
            "votes": votes,
            "consensus": consensus,
            "approved": approved,
            "summary": summary,
        })

    def broadcast_ensemble(
        self,
        trending: float,
        ranging: float,
        agreement: float,
    ) -> None:
        """Broadcast ensemble prediction breakdown."""
        self._broadcast("ensemble", {
            "trending": trending,
            "ranging": ranging,
            "agreement": agreement,
        })

    def broadcast_trade_open(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
    ) -> None:
        """Broadcast a new position opened."""
        self._broadcast("trade_open", {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "current_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

    def broadcast_trade_close(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        duration_seconds: float,
    ) -> None:
        """Broadcast a closed position."""
        self._broadcast("trade_close", {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "duration_seconds": duration_seconds,
        })

    def broadcast_tick(self, symbol: str, current_price: float, unrealized_pnl: float = 0.0) -> None:
        """Broadcast real-time price tick and unrealized P&L."""
        self._broadcast("tick", {
            "symbol": symbol,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
        })

    def broadcast_trail_update(self, update: dict) -> None:
        """Broadcast trailing stop update."""
        self._broadcast("trail_update", update)

    @property
    def client_count(self) -> int:
        return len(self._clients)
