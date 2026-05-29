import asyncio
from typing import Any, Awaitable, Callable, Iterable, Optional

import websockets
from loguru import logger

from src.data.parsers import AdapterMessage, BaseMarketAdapter, BinanceMarketAdapter


BINANCE_FUTURES_STREAM_BASE_URL = "wss://fstream.binance.com/stream?streams="
DEFAULT_BINANCE_STREAM_SUFFIXES = ("bookTicker", "forceOrder", "aggTrade")


def build_binance_futures_stream_url(
    symbol: str,
    stream_suffixes: Iterable[str] = DEFAULT_BINANCE_STREAM_SUFFIXES,
) -> str:
    normalized_symbol = symbol.lower()
    multiplexed_streams = "/".join(
        f"{normalized_symbol}@{stream_suffix}"
        for stream_suffix in stream_suffixes
    )
    return f"{BINANCE_FUTURES_STREAM_BASE_URL}{multiplexed_streams}"


class ProviderWebSocketManager:
    """
    Maintains a resilient WebSocket connection for any market data provider
    that implements the adapter contract.
    """

    def __init__(
        self,
        url: str,
        adapter: Optional[BaseMarketAdapter],
        on_message: Callable[[AdapterMessage], Awaitable[Any]],
    ):
        self.url = url
        self.adapter = adapter or BinanceMarketAdapter()
        self.on_message = on_message
        self._is_running = False
        self._base_backoff = 1.0
        self._max_backoff = 60.0

    async def start(self):
        self._is_running = True
        backoff = self._base_backoff

        while self._is_running:
            try:
                logger.info(
                    f"Connecting to {self.adapter.provider_name} WebSocket: {self.url}"
                )
                async with websockets.connect(
                    self.url,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    logger.success(f"Connected to {self.adapter.provider_name}: {self.url}")
                    backoff = self._base_backoff

                    async for raw_message in websocket:
                        parsed_message = self.adapter.parse_message(raw_message)
                        if parsed_message is None:
                            continue
                        asyncio.create_task(self.on_message(parsed_message))

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(
                    f"Provider WebSocket closed (code: {exc.code}, reason: {exc.reason})."
                )
            except asyncio.CancelledError:
                logger.info("Provider WebSocket manager cancelled.")
                self._is_running = False
                break
            except Exception as exc:
                logger.error(f"Provider WebSocket connection error: {exc}")

            if self._is_running:
                logger.info(f"Reconnecting in {backoff:.2f} seconds...")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    async def stop(self):
        logger.info(f"Stopping provider WebSocket manager for {self.url}")
        self._is_running = False


WebSocketManager = ProviderWebSocketManager
