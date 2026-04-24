import asyncio
import json
import websockets
from loguru import logger
from typing import Callable, Coroutine, Any

class WebSocketManager:
    """
    Manages robust WebSocket connections with aggressive auto-reconnect 
    and exponential backoff protocols.
    """
    def __init__(self, url: str, on_message: Callable[[str], Coroutine[Any, Any, None]]):
        self.url = url
        self.on_message = on_message
        self._is_running = False
        self._base_backoff = 1.0  # Starts at 1 second
        self._max_backoff = 60.0  # Max wait time 60 seconds

    async def start(self):
        self._is_running = True
        backoff = self._base_backoff
        
        while self._is_running:
            try:
                logger.info(f"Connecting to WebSocket: {self.url}")
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20) as ws:
                    logger.success(f"Connected to {self.url}")
                    # Reset backoff on successful connection
                    backoff = self._base_backoff
                    
                    async for message in ws:
                        # Yield control to the message handler
                        asyncio.create_task(self.on_message(message))
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket closed (code: {e.code}, reason: {e.reason}). Reconnecting...")
            except asyncio.CancelledError:
                logger.info("WebSocket connection task cancelled.")
                self._is_running = False
                break
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            
            if self._is_running:
                logger.info(f"Reconnecting in {backoff:.2f} seconds...")
                await asyncio.sleep(backoff)
                # Exponential backoff with a cap
                backoff = min(backoff * 2, self._max_backoff)

    async def stop(self):
        logger.info(f"Stopping WebSocket Manager for {self.url}")
        self._is_running = False
