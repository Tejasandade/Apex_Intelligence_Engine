import json
from loguru import logger
from pydantic import BaseModel
from typing import List, Tuple

# We define Pydantic models for structured parsing (can be optimized later if needed)

class OrderBookUpdate(BaseModel):
    symbol: str
    event_time: int
    first_update_id: int
    final_update_id: int
    bids: List[Tuple[str, str]]  # [Price, Quantity]
    asks: List[Tuple[str, str]]

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

class BinanceParser:
    """
    Parses incoming messages from Binance USD-M Futures streams.
    Handles multiplexed streams: `stream` wrapper payload.
    """
    
    @staticmethod
    def parse_message(raw_message: str):
        try:
            data = json.loads(raw_message)
            
            # If using multiplexed stream, the data is inside 'data' key, and stream name is in 'stream' key
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                payload = data['data']
            else:
                stream_name = None
                payload = data
            
            # Routing logic based on event type
            event_type = payload.get('e')
            
            if event_type == 'depthUpdate':
                return BinanceParser.parse_depth_update(payload)
            elif event_type == 'forceOrder':
                return BinanceParser.parse_liquidation(payload)
            elif event_type == 'aggTrade':
                return BinanceParser.parse_agg_trade(payload)
            else:
                # Could be a subscription confirmation or other message
                logger.debug(f"Unhandled message type or non-event payload: {payload}")
                return None
                
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON: {raw_message}")
            return None
        except Exception as e:
            logger.error(f"Error parsing message: {e} - Payload: {raw_message}")
            return None

    @staticmethod
    def parse_depth_update(payload: dict) -> OrderBookUpdate:
        """Parses @depth@100ms updates."""
        return OrderBookUpdate(
            symbol=payload['s'],
            event_time=payload['E'],
            first_update_id=payload['U'],
            final_update_id=payload['u'],
            bids=payload['b'],  # List of [Price, Quantity] strings
            asks=payload['a']   # List of [Price, Quantity] strings
        )

    @staticmethod
    def parse_liquidation(payload: dict) -> LiquidationEvent:
        """Parses forceOrder (liquidation) events."""
        order_info = payload['o']
        return LiquidationEvent(
            symbol=order_info['s'],
            side=order_info['S'],
            order_type=order_info['o'],
            time_in_force=order_info['f'],
            original_quantity=order_info['q'],
            price=order_info['p'],
            average_price=order_info['ap'],
            order_status=order_info['X'],
            last_fill_quantity=order_info['l'],
            accumulated_fill_quantity=order_info['z'],
            event_time=order_info['T']
        )

    @staticmethod
    def parse_agg_trade(payload: dict) -> AggTradeEvent:
        """Parses @aggTrade events."""
        return AggTradeEvent(
            symbol=payload['s'],
            agg_trade_id=payload['a'],
            price=payload['p'],
            quantity=payload['q'],
            first_trade_id=payload['f'],
            last_trade_id=payload['l'],
            trade_time=payload['T'],
            is_buyer_maker=payload['m']
        )
