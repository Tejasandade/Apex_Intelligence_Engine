import asyncio
from typing import Optional, Set

from fastapi import WebSocket
from loguru import logger

from src.dashboard.backend.schemas import BroadcastEnvelope, DashboardSnapshot


class WebSocketBroadcaster:
    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._sequence = 0

    async def connect(self, websocket: WebSocket, snapshot: Optional[DashboardSnapshot] = None):
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

        logger.info("Dashboard client connected. Total clients: {}", len(self._connections))

        if snapshot is not None:
            sent = await self._send(websocket, "dashboard.snapshot", snapshot)
            if not sent:
                await self.disconnect(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("Dashboard client disconnected. Total clients: {}", len(self._connections))

    async def broadcast(self, event_type: str, snapshot: DashboardSnapshot):
        async with self._lock:
            connections = list(self._connections)

        if not connections:
            return

        dead_connections = []
        envelope = self._build_envelope(event_type, snapshot)
        payload = envelope.model_dump_json()

        for websocket in connections:
            try:
                await websocket.send_text(payload)
            except Exception as exc:
                logger.warning("Dropping stale dashboard client: {}", exc)
                dead_connections.append(websocket)

        if dead_connections:
            async with self._lock:
                for websocket in dead_connections:
                    self._connections.discard(websocket)

    async def _send(self, websocket: WebSocket, event_type: str, snapshot: DashboardSnapshot) -> bool:
        envelope = self._build_envelope(event_type, snapshot)
        try:
            await websocket.send_text(envelope.model_dump_json())
            return True
        except Exception as exc:
            logger.warning("Initial dashboard snapshot send failed: {}", exc)
            return False

    def _build_envelope(self, event_type: str, snapshot: DashboardSnapshot) -> BroadcastEnvelope:
        self._sequence += 1
        return BroadcastEnvelope(
            type=event_type,
            sequence=self._sequence,
            timestamp=snapshot.generated_at,
            payload=snapshot,
        )
