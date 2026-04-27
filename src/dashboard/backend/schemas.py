from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class CapitalUpdate(BaseModel):
    total_capital: float = Field(gt=0)
    trade_allocation: Optional[float] = None


class ProfessionalTraderUpdate(BaseModel):
    enabled: bool


class ManualExecutionRequest(BaseModel):
    order_id: Optional[str] = None
    symbol: Optional[str] = None
    reason: str = "manual_override"


class SignalAnalysis(BaseModel):
    technical_score: float
    sentiment_score: float
    technical_summary: str
    sentiment_summary: str


class ExecutionPlan(BaseModel):
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_qty: float
    allocation: float
    leverage: float
    est_pnl: float
    max_loss: float
    sl_distance_pct: float
    tp_distance_pct: float
    style: str
    timeframe: str


class SmartOrderCard(BaseModel):
    order_id: Optional[str] = None
    symbol: str
    provider: str
    signal: Literal["BUY", "SELL", "HOLD"]
    bullish_probability: float
    bearish_probability: float
    confidence_percent: float
    signal_analysis: SignalAnalysis
    structural_edge: str
    structural_score: float
    structure_signals: List[str]
    structural_confluence: bool
    execution_plan: Optional[ExecutionPlan] = None
    requires_manual_approval: bool
    execution_mode: str
    status: str
    warmup_count: int = 0
    warmup_required: int = 14
    warmup_message: Optional[str] = None
    execute_label: str = "Execute Now"


class TradeTicket(BaseModel):
    order_id: str
    symbol: str
    signal: str
    status: str
    probability: float
    provider: str
    trigger_source: str
    time: str
    date: str
    sentiment: float
    features: dict
    execution_plan: ExecutionPlan


class ActiveTrade(BaseModel):
    order_id: str
    symbol: str
    side: str
    status: str
    quantity: float
    entry_price: float
    current_price: float
    pnl_pct: float
    pnl_value: float = 0.0
    trailing_stop_level: float
    callback_rate: float
    broker_status: str
    exit_reason: Optional[str] = None
    opened_at: str


class DashboardSession(BaseModel):
    cycle_count: int
    sim_pnl: float
    total_capital: float
    trade_allocation: float = 0.0
    professional_trader_enabled: bool
    autonomous_mode: bool
    execution_mode: str
    db_status: str = "connected"
    active_tab: str = "CRYPTO"
    trading_style: str = "Intraday"
    capital_pools: dict = {}


class DashboardMarket(BaseModel):
    symbol: str
    provider: str
    close_price: float
    rsi: float
    vwap: float
    atr: float
    sentiment: float
    probability: float
    signal: str
    structural_features: dict


class DashboardSnapshot(BaseModel):
    generated_at: datetime
    session: DashboardSession
    market: DashboardMarket
    smart_order_card: SmartOrderCard
    active_trades: List[ActiveTrade]
    trades: List[TradeTicket]
    live_signals: List[TradeTicket] = []
    news: List[str]
    global_best_signal: Optional[dict] = None


class BroadcastEnvelope(BaseModel):
    type: str
    sequence: int
    timestamp: datetime
    payload: DashboardSnapshot
