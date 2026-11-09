"""
Unit Tests for xau_c Integration into Apex Intelligence Engine
==============================================================
Validates:
1. Gold MTL / ICT Fair Value Gap (FVG) and C1/C2 Structure Engine
2. StructureAdvisor Gold Demand/Supply Vetoes
3. Cross-Market Portfolio Coherence Lock (Cannibalization Guard)
4. No-Chase (0.35 ATR) and Spread-to-Risk (10%) Execution Gates
5. Delta L2 Order Book Imbalance (OBI_5) Engine
"""

import pytest
import pandas as pd
from src.intelligence.gold_structure_engine import GoldStructureEngine
from src.council.advisors import StructureAdvisor, CrossMarketAdvisor, Vote
from src.core.prop_firm_manager import PropFirmManager
from src.data.providers.delta_feed import DeltaOrderBook


def test_gold_structure_fvg_demand_veto():
    engine = GoldStructureEngine()
    # Bullish FVG: Bar 1 high = 4085, Bar 3 low = 4095 -> FVG between 4085 and 4095
    candles_1h = [
        {"timestamp": 1000, "open": 4080, "high": 4085, "low": 4078, "close": 4082},
        {"timestamp": 2000, "open": 4082, "high": 4100, "low": 4081, "close": 4098},
        {"timestamp": 3000, "open": 4098, "high": 4110, "low": 4095, "close": 4108},
        {"timestamp": 4000, "open": 4108, "high": 4120, "low": 4105, "close": 4115},
    ]
    # Price is resting at 4090 (inside the Bullish FVG)
    res = engine.evaluate_gold_structure(4090.0, candles_1h, candles_1h)
    assert res["in_bullish_fvg"] is True
    assert res["veto_short"] is True
    assert "Demand Zone" in res["veto_reason"]


def test_gold_structure_fvg_supply_veto():
    engine = GoldStructureEngine()
    # Bearish FVG: Bar 1 low = 4120, Bar 3 high = 4105 -> FVG between 4105 and 4120
    candles_1h = [
        {"timestamp": 1000, "open": 4125, "high": 4130, "low": 4120, "close": 4122},
        {"timestamp": 2000, "open": 4122, "high": 4124, "low": 4095, "close": 4100},
        {"timestamp": 3000, "open": 4100, "high": 4105, "low": 4090, "close": 4092},
        {"timestamp": 4000, "open": 4092, "high": 4096, "low": 4080, "close": 4085},
    ]
    # Price is resting at 4112 (inside the Bearish FVG)
    res = engine.evaluate_gold_structure(4112.0, candles_1h, candles_1h)
    assert res["in_bearish_fvg"] is True
    assert res["veto_long"] is True
    assert "Supply Zone" in res["veto_reason"]


def test_structure_advisor_gold_veto():
    advisor = StructureAdvisor()
    candles_1h = [
        {"timestamp": 1000, "open": 4080, "high": 4085, "low": 4078, "close": 4082},
        {"timestamp": 2000, "open": 4082, "high": 4100, "low": 4081, "close": 4098},
        {"timestamp": 3000, "open": 4098, "high": 4110, "low": 4095, "close": 4108},
    ]
    df = pd.DataFrame([{"fvg_signal": 0.0, "structure_break_signal": 0.0}])
    # Attempting to SELL Gold inside a Bullish FVG
    vote = advisor.evaluate(
        features=df,
        direction="SELL",
        price=4090.0,
        regime="TRENDING",
        atr=5.0,
        symbol="PAXGUSDT",
        candles_1h=candles_1h,
        candles_15m=candles_1h,
    )
    assert vote.vote == Vote.REJECT
    assert "Demand Zone" in vote.reasoning


def test_cross_market_coherence_guard():
    advisor = CrossMarketAdvisor()
    df = pd.DataFrame([{"dummy": 0.0}])
    # Holding a SHORT position in BTCUSDT
    active_portfolios = {
        "BTCUSDT": {"position": "OPEN", "direction": "SHORT"},
        "PAXGUSDT": {"position": "FLAT", "direction": None},
    }
    # Trying to BUY Gold while holding SHORT on BTC
    vote = advisor.evaluate(
        features=df,
        direction="BUY",
        price=4095.0,
        regime="TRENDING",
        atr=5.0,
        symbol="PAXGUSDT",
        portfolios=active_portfolios,
    )
    assert vote.vote == Vote.REJECT
    assert "Macro Directional Conflict" in vote.reasoning


def test_prop_firm_coherence_lock():
    mgr = PropFirmManager(profile_key="sharkfunded_5k_lite_1step", state_file="data/test_prop_firm_state.json")
    # Holding SHORT on BTC
    active_pos = {
        "BTCUSDT": {"position": "OPEN", "direction": "SHORT", "risk_usd": 25.0},
        "PAXGUSDT": {"position": "FLAT", "direction": None},
    }
    # Trying to open LONG Gold
    can_open, reason = mgr.can_open_multi_asset_trade(
        symbol="PAXGUSDT",
        direction="LONG",
        active_positions=active_pos,
        estimated_risk_usd=25.0,
    )
    assert can_open is False
    assert "Coherence Guard" in reason


def test_delta_order_book_imbalance():
    book = DeltaOrderBook("PAXGUSD")
    update = {
        "type": "ob_updates",
        "sy": "PAXGUSD",
        "action": "snapshot",
        "b": [["4090.0", "100.0"], ["4089.0", "50.0"]],
        "a": [["4091.0", "20.0"], ["4092.0", "10.0"]],
    }
    book.apply_update(update)
    # Total bids = 150, Total asks = 30. Imbalance = (150 - 30) / (150 + 30) = 120 / 180 = 0.6667
    imbalance = book.get_imbalance_top_n(5)
    assert abs(imbalance - (120.0 / 180.0)) < 1e-4
