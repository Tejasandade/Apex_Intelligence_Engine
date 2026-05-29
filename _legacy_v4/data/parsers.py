import abc
import json
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel


class OrderBookUpdate(BaseModel):
    symbol: str
    event_time: int
    first_update_id: int
    final_update_id: int
    bids: List[Tuple[str, str]]
    asks: List[Tuple[str, str]]
    is_top_of_book: bool = False


class AggTradeEvent(BaseModel):
    symbol: str
    agg_trade_id: int
    price: str
    quantity: str
    first_trade_id: int
    last_trade_id: int
    trade_time: int
    is_buyer_maker: bool


class LiquidationEvent(BaseModel):
    symbol: str
    side: str
    order_type: str
    time_in_force: str
    original_quantity: str
    price: str
    average_price: str
    order_status: str
    last_fill_quantity: str
    accumulated_fill_quantity: str
    event_time: int


ParsedPayload = Union[OrderBookUpdate, AggTradeEvent, LiquidationEvent]


class AdapterMessage(BaseModel):
    provider: str
    event_type: str
    symbol: Optional[str] = None
    stream: Optional[str] = None
    payload: ParsedPayload
    raw_payload: Dict[str, Any]


class BaseMarketAdapter(abc.ABC):
    provider_name = "unknown"

    @abc.abstractmethod
    def parse_message(self, raw_message: str) -> Optional[AdapterMessage]:
        """Parse a raw provider message into a normalized adapter envelope."""


class AdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, BaseMarketAdapter] = {}

    def register(self, adapter: BaseMarketAdapter):
        self._adapters[adapter.provider_name] = adapter

    def get(self, provider_name: str) -> BaseMarketAdapter:
        try:
            return self._adapters[provider_name]
        except KeyError as exc:
            raise KeyError(f"Adapter '{provider_name}' is not registered.") from exc


class BinanceMarketAdapter(BaseMarketAdapter):
    """
    Normalizes Binance USD-M Futures stream payloads behind a provider adapter.
    The same interface can be implemented for Nifty, Forex, or broker APIs later.
    """

    provider_name = "binance_futures"

    def parse_message(self, raw_message: str) -> Optional[AdapterMessage]:
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            logger.error(f"Failed to decode provider message: {raw_message}")
            return None

        try:
            if "stream" in data and "data" in data:
                stream_name = data["stream"]
                payload = data["data"]
            else:
                stream_name = None
                payload = data

            event_type = payload.get("e")
            if event_type == "depthUpdate":
                parsed_payload = self.parse_depth_update(payload)
            elif event_type == "bookTicker":
                parsed_payload = self.parse_book_ticker(payload)
                if parsed_payload is None:
                    return None
            elif event_type == "forceOrder":
                parsed_payload = self.parse_liquidation(payload)
            elif event_type == "aggTrade":
                parsed_payload = self.parse_agg_trade(payload)
            else:
                logger.debug(f"Unhandled provider payload: {payload}")
                return None

            return AdapterMessage(
                provider=self.provider_name,
                event_type=event_type,
                symbol=getattr(parsed_payload, "symbol", None),
                stream=stream_name,
                payload=parsed_payload,
                raw_payload=payload,
            )
        except Exception as exc:
            logger.error(f"Adapter parse error: {exc} - Payload: {raw_message}")
            return None

    @staticmethod
    def parse_depth_update(payload: Dict[str, Any]) -> OrderBookUpdate:
        return OrderBookUpdate(
            symbol=payload["s"],
            event_time=payload["E"],
            first_update_id=payload["U"],
            final_update_id=payload["u"],
            bids=payload["b"],
            asks=payload["a"],
            is_top_of_book=False,
        )

    @staticmethod
    def parse_book_ticker(payload: Dict[str, Any]) -> Optional[OrderBookUpdate]:
        update_id = int(payload.get("u", 0))
        event_time = int(payload.get("E", payload.get("T", 0)))
        
        bid_price = float(payload.get("b", 0))
        ask_price = float(payload.get("a", 0))

        if ask_price > 0 and bid_price < (ask_price * 0.8):
            logger.warning(f"Sanity Check Failed: Ghost Bid detected. Bid: {bid_price}, Ask: {ask_price}")
            return None

        return OrderBookUpdate(
            symbol=payload["s"],
            event_time=event_time,
            first_update_id=update_id,
            final_update_id=update_id,
            bids=[(payload["b"], payload["B"])],
            asks=[(payload["a"], payload["A"])],
            is_top_of_book=True,
        )

    @staticmethod
    def parse_liquidation(payload: Dict[str, Any]) -> LiquidationEvent:
        order_info = payload["o"]
        return LiquidationEvent(
            symbol=order_info["s"],
            side=order_info["S"],
            order_type=order_info["o"],
            time_in_force=order_info["f"],
            original_quantity=order_info["q"],
            price=order_info["p"],
            average_price=order_info["ap"],
            order_status=order_info["X"],
            last_fill_quantity=order_info["l"],
            accumulated_fill_quantity=order_info["z"],
            event_time=order_info["T"],
        )

    @staticmethod
    def parse_agg_trade(payload: Dict[str, Any]) -> AggTradeEvent:
        return AggTradeEvent(
            symbol=payload["s"],
            agg_trade_id=payload["a"],
            price=payload["p"],
            quantity=payload["q"],
            first_trade_id=payload["f"],
            last_trade_id=payload["l"],
            trade_time=payload["T"],
            is_buyer_maker=payload["m"],
        )


class BinanceParser:
    """
    Backwards-compatible facade while the ingestion layer migrates to adapters.
    """

    _adapter = BinanceMarketAdapter()

    @staticmethod
    def parse_message(raw_message: str) -> Optional[ParsedPayload]:
        parsed_message = BinanceParser._adapter.parse_message(raw_message)
        if parsed_message is None:
            return None
        return parsed_message.payload


adapter_registry = AdapterRegistry()
adapter_registry.register(BinanceMarketAdapter())
