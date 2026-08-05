import asyncio
import json
import websockets
import datetime
import time
import os
import sys

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import clickhouse_connect
import requests
from collections import deque
from sklearn.cluster import KMeans
import pickle

# ---------------------------------------------------------
# SYSTEM STATE & PARAMETERS
# ---------------------------------------------------------
state = {
    "symbol": "BTCUSDT",
    "ws_connected": False,
    "ltp": 0.0,
    "bid": 0.0,
    "ask": 0.0,
    "vpin": 0.0,
    "vpin_percentile": 0,
    "cofi_z": 0.0,
    "acf_lag1": 0.0,
    "ticks_today": 0,
    "buckets_completed": 0,
    "bucket_velocity_sec": 0.0,
    "current_minute_ts": 0,
    "atr_14": 0.0,
    "avg_1min_vol": 0.0,
    "obi": 0.0
}

# Macro Structure State
macro_structure = {
    "hvns": [], # High Volume Nodes
    "poc": 0.0,  # Point of Control
    "micro_hvns": [] # Intraday Session HVNs
}

# Liquidation State (Rolling 60s)
liquidation_events = deque()
liquidations = {
    "longs_usd": 0.0,
    "shorts_usd": 0.0
}

# Machine Learning State
ml_state = {
    "features": deque(maxlen=500),
    "model": None,
    "trending_cluster_id": 0,
    "regime": "GATHERING DATA"
}

def save_ml_state():
    try:
        with open("ml_state.pkl", "wb") as f:
            pickle.dump(ml_state["features"], f)
    except Exception:
        pass

def load_ml_state():
    try:
        if os.path.exists("ml_state.pkl"):
            with open("ml_state.pkl", "rb") as f:
                features = pickle.load(f)
                ml_state["features"] = features
    except Exception:
        pass

# Virtual Portfolio State
portfolio = {
    "position": "FLAT", # FLAT, LONG, SHORT
    "phase": 0, # 0 = Flat, 1 = Initial Risk, 2 = Runner
    "entry_price": 0.0,
    "account_balance": 1000.0,
    "leverage_used": 0.0,
    "notional_size_usd": 0.0,
    "realized_pnl": 0.0,
    "unrealized_pnl": 0.0,
    "trades_executed": 0,
    "post_entry_volume": 0.0,
    "post_entry_vwap_sum": 0.0,
    "vwap": 0.0
}

# Execution Physics
ENTRY_COST_BPS = 6.0 # 0.05% Taker + 0.01% Slippage
EXIT_COST_BPS = 6.0  # 0.05% Taker + 0.01% Slippage
TOTAL_ROUNDTRIP_COST_BPS = ENTRY_COST_BPS + EXIT_COST_BPS

# Dynamic Parameters (will be overridden by prewarm)
DYNAMIC_PARAMS = {
    "BUCKET_VOLUME_SIZE": 50.0,
    "STOP_LOSS_BPS": 20.0,
    "SCALE_OUT_BPS": 20.0,
    "BREAKEVEN_BPS": 2.0
}

# Microstructure Buffers
current_bucket = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0, "start_time": time.time()}
vpin_history = deque(maxlen=200)
order_flow_imbalances = deque(maxlen=300)
recent_trades = deque(maxlen=100)

# Volatility Buffers
minute_candles = deque(maxlen=60)
live_candle = {"open": 0, "high": 0, "low": 0, "close": 0, "volume": 0}

# 5-Minute Candle Buffer (for FVG & OB detection)
five_min_candles = deque(maxlen=50)
five_min_accumulator = []  # Collects 1-min candles to aggregate

# Dynamic Cooldown State
cooldown = {
    "active": False,
    "buckets_remaining": 0,
    "base_buckets": 15,
    "elevated_vpin": 98,     # Cooldown lifts if VPIN >= 98th
    "elevated_cofi": 4.0     # AND COFI Z > 4.0
}

# Session Awareness (UTC hours)
SESSION_MAP = {
    "DEAD_ZONE": (21, 0),    # UTC 21:00-00:00 | IST 2:30-5:30 AM
    "ASIAN":     (0, 7),     # UTC 00:00-07:00 | IST 5:30 AM-12:30 PM
    "LONDON":    (7, 13),    # UTC 07:00-13:00 | IST 12:30-6:30 PM
    "NEW_YORK":  (13, 21)    # UTC 13:00-21:00 | IST 6:30 PM-2:30 AM
}
session_state = {
    "current": "UNKNOWN",
    "execution_allowed": True,
    "thresholds_elevated": False
}

# Live FVG State
fvg_state = {
    "bullish_fvgs": [],   # List of {"bottom": float, "top": float, "ts": str}
    "bearish_fvgs": [],
    "nearest_bullish": None,
    "nearest_bearish": None
}

# ICT Order Block State
ob_state = {
    "bullish_obs": [],    # List of {"low": float, "high": float, "ts": str, "displacement": float}
    "bearish_obs": [],
    "active_count": 0
}

# CVD (Cumulative Volume Delta)
cvd_state = {
    "cvd": 0.0,
    "cvd_history": deque(maxlen=50),   # CVD snapshot per 5-min candle
    "price_history": deque(maxlen=50), # Price snapshot per 5-min candle
    "divergence": "NONE",              # NONE, BULLISH, BEARISH
    "5min_buy_vol": 0.0,
    "5min_sell_vol": 0.0
}

# 15-Minute Candle Buffer (Multi-Timeframe Confirmation)
fifteen_min_candles = deque(maxlen=30)
fifteen_min_accumulator = []  # Collects 5-min candles
mtf_state = {
    "ema_20": 0.0,
    "trend": "NEUTRAL",   # BULLISH, BEARISH, NEUTRAL
    "strength": 0.0       # How far price is from EMA (in bps)
}

# Liquidity Sweep Detection
sweep_state = {
    "recent_highs": deque(maxlen=20),  # Rolling 5m candle highs
    "recent_lows": deque(maxlen=20),   # Rolling 5m candle lows
    "last_sweep": "NONE",             # NONE, BULLISH_SWEEP, BEARISH_SWEEP
    "sweep_price": 0.0,
    "sweep_ts": ""
}

# Trade Journal (for Supervised ML)
trade_journal = []  # List of {features_at_entry, outcome, pnl, ...}
TRADE_JOURNAL_FILE = "trade_journal.json"

# Volume Profile Refresh Timer
vp_refresh = {
    "last_refresh_time": 0.0,
    "refresh_interval_sec": 4 * 3600  # 4 hours
}

# ---------------------------------------------------------
# VOLATILITY & PRE-WARM LOGIC
# ---------------------------------------------------------
def calculate_atr():
    if len(minute_candles) < 14:
        return 0.0
        
    trs = []
    candles = list(minute_candles)
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i-1]['close']
        
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        if tr > 1000:
            with open('atr_debug.txt', 'a') as f:
                f.write(f"BUG! tr={tr}, high={high}, low={low}, prev_close={prev_close}, i={i}, len={len(candles)}\n")
        trs.append(tr)
        
    atr = np.mean(trs[-14:]) # 14-period ATR
    return atr

def update_dynamic_parameters():
    global DYNAMIC_PARAMS
    if len(minute_candles) < 14 or state["ltp"] == 0:
        return
        
    # 1. Update Bucket Size (ADV Approximation)
    vols = [c['volume'] for c in minute_candles]
    state["avg_1min_vol"] = np.mean(vols)
    DYNAMIC_PARAMS["BUCKET_VOLUME_SIZE"] = max(10.0, state["avg_1min_vol"] / 2.0)
    
    # 2. Update Stop Loss & Scale Out (ATR Approximation)
    state["atr_14"] = calculate_atr()
    atr_bps = (state["atr_14"] / state["ltp"]) * 10000
    
    # STOP LOSS must be strictly > TOTAL_ROUNDTRIP_COST_BPS (12 bps)
    # Otherwise, it stops out instantly upon entering the trade!
    DYNAMIC_PARAMS["STOP_LOSS_BPS"] = max(30.0, atr_bps * 3.0)
    DYNAMIC_PARAMS["SCALE_OUT_BPS"] = max(40.0, atr_bps * 4.0)

def build_volume_profile():
    print("Building Macro Volume Profile (30-Day Lookback)...")
    try:
        res = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1h&limit=720")
        data = res.json()
        
        highs = [float(k[2]) for k in data]
        lows = [float(k[3]) for k in data]
        
        min_price = min(lows)
        max_price = max(highs)
        
        num_bins = 50
        bin_size = (max_price - min_price) / num_bins
        profile = np.zeros(num_bins)
        
        for kline in data:
            c_low = float(kline[3])
            c_high = float(kline[2])
            c_vol = float(kline[5])
            
            start_bin = int(max(0, min(num_bins-1, (c_low - min_price) / bin_size)))
            end_bin = int(max(0, min(num_bins-1, (c_high - min_price) / bin_size)))
            
            if start_bin == end_bin:
                profile[start_bin] += c_vol
            else:
                vol_per_bin = c_vol / (end_bin - start_bin + 1)
                for b in range(start_bin, end_bin + 1):
                    profile[b] += vol_per_bin
                    
        top_indices = np.argsort(profile)[-5:]
        hvns = []
        for idx in top_indices:
            price_level = min_price + (idx * bin_size) + (bin_size / 2.0)
            hvns.append(price_level)
            
        macro_structure["hvns"] = sorted(hvns)
        
        poc_idx = np.argmax(profile)
        macro_structure["poc"] = min_price + (poc_idx * bin_size) + (bin_size / 2.0)
        
        print(f"Macro Profile Complete. POC: {macro_structure['poc']:.1f}, HVNs: {[round(x, 1) for x in macro_structure['hvns']]}")
        
    except Exception as e:
        print(f"Volume Profile failed: {e}")

def build_session_volume_profile():
    print("Building Session Volume Profile (24-Hour Lookback)...")
    try:
        res = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=15m&limit=96")
        data = res.json()
        highs = [float(k[2]) for k in data]
        lows = [float(k[3]) for k in data]
        min_price = min(lows)
        max_price = max(highs)
        num_bins = 30
        bin_size = (max_price - min_price) / num_bins
        profile = np.zeros(num_bins)
        for kline in data:
            c_low = float(kline[3])
            c_high = float(kline[2])
            c_vol = float(kline[5])
            start_bin = int(max(0, min(num_bins-1, (c_low - min_price) / bin_size)))
            end_bin = int(max(0, min(num_bins-1, (c_high - min_price) / bin_size)))
            if start_bin == end_bin:
                profile[start_bin] += c_vol
            else:
                vol_per_bin = c_vol / (end_bin - start_bin + 1)
                for b in range(start_bin, end_bin + 1):
                    profile[b] += vol_per_bin
        top_indices = np.argsort(profile)[-3:]
        micro_hvns = []
        for idx in top_indices:
            price_level = min_price + (idx * bin_size) + (bin_size / 2.0)
            micro_hvns.append(price_level)
        macro_structure["micro_hvns"] = sorted(micro_hvns)
        print(f"Session Profile Complete. Micro-HVNs: {[round(x, 1) for x in macro_structure['micro_hvns']]}")
    except Exception as e:
        print(f"Session Volume Profile failed: {e}")


def prewarm_engine():
    print("Pre-warming Dynamic Volatility Engine from Binance REST API...")
    try:
        build_volume_profile()
        build_session_volume_profile()
        
        # 1. Pre-warm 1-minute candles (60 candles = 1 hour)
        res = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1m&limit=60")
        data = res.json()
        for kline in data:
            candle = {
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            }
            minute_candles.append(candle)
            
        state["ltp"] = float(data[-1][4])
        update_dynamic_parameters()
        state["current_minute_ts"] = int(time.time() // 60) * 60
        live_candle["open"] = live_candle["high"] = live_candle["low"] = live_candle["close"] = state["ltp"]
        live_candle["volume"] = 0.0
        print(f"  1m candles: {len(minute_candles)} loaded.")
        
        # 2. Pre-warm 5-minute candles (50 candles = ~4 hours)
        res5 = requests.get("https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=5m&limit=50")
        data5 = res5.json()
        for kline in data5:
            candle5 = {
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            }
            five_min_candles.append(candle5)
            
            # Seed CVD history from 5m buy/sell volume
            buy_vol = float(kline[9])  # Taker buy base asset volume
            total_vol = float(kline[5])
            sell_vol = total_vol - buy_vol
            cvd_snapshot = buy_vol - sell_vol
            cvd_state["cvd_history"].append(cvd_snapshot)
            cvd_state["price_history"].append(float(kline[4]))
        
        print(f"  5m candles: {len(five_min_candles)} loaded.")
        
        # 3. Run FVG and OB detection on pre-warmed data
        detect_fvgs()
        detect_order_blocks()
        detect_cvd_divergence()
        
        # 4. Pre-warm 15-minute candles from 5m data
        candles_5m = list(five_min_candles)
        for i in range(0, len(candles_5m) - 2, 3):
            group = candles_5m[i:i+3]
            if len(group) == 3:
                c15 = {
                    "open": group[0]["open"],
                    "high": max(g["high"] for g in group),
                    "low": min(g["low"] for g in group),
                    "close": group[-1]["close"],
                    "volume": sum(g["volume"] for g in group)
                }
                fifteen_min_candles.append(c15)
        update_mtf_trend()
        print(f"  15m candles: {len(fifteen_min_candles)} loaded. MTF Trend: {mtf_state['trend']}")
        
        # 5. Seed swing levels for liquidity sweep detection
        for c5 in candles_5m:
            track_swing_levels(c5)
        print(f"  Swing levels: {len(sweep_state['recent_highs'])} highs, {len(sweep_state['recent_lows'])} lows tracked")
        
        # 6. Load trade journal
        load_trade_journal()
        print(f"  Trade journal: {len(trade_journal)} historical trades loaded")
        
        # 7. Init VP refresh timer
        vp_refresh["last_refresh_time"] = time.time()
        
        bull_fvgs = len(fvg_state["bullish_fvgs"])
        bear_fvgs = len(fvg_state["bearish_fvgs"])
        bull_obs = len(ob_state["bullish_obs"])
        bear_obs = len(ob_state["bearish_obs"])
        print(f"  FVGs: {bull_fvgs} bullish, {bear_fvgs} bearish")
        print(f"  ICT Order Blocks: {bull_obs} bullish, {bear_obs} bearish")
        print(f"  CVD Divergence: {cvd_state['divergence']}")
        print(f"Pre-warm complete! Engine is combat-ready.")
    except Exception as e:
        print(f"Pre-warm failed: {e}. Will rely on live ticks.")

# ---------------------------------------------------------
# SESSION AWARENESS
# ---------------------------------------------------------
def get_current_session():
    utc_hour = datetime.datetime.utcnow().hour
    if 7 <= utc_hour < 13:
        session_state["current"] = "LONDON"
        session_state["execution_allowed"] = True
        session_state["thresholds_elevated"] = False
    elif 13 <= utc_hour < 21:
        session_state["current"] = "NEW_YORK"
        session_state["execution_allowed"] = True
        session_state["thresholds_elevated"] = False
    elif 0 <= utc_hour < 7:
        session_state["current"] = "ASIAN"
        session_state["execution_allowed"] = True
        session_state["thresholds_elevated"] = True  # Elevated thresholds
    else:  # 21-24
        session_state["current"] = "DEAD_ZONE"
        session_state["execution_allowed"] = False
        session_state["thresholds_elevated"] = True

# ---------------------------------------------------------
# DYNAMIC COOLDOWN
# ---------------------------------------------------------
def activate_cooldown():
    cooldown["active"] = True
    cooldown["buckets_remaining"] = cooldown["base_buckets"]

def tick_cooldown():
    """Called on each completed volume bucket."""
    if cooldown["active"]:
        cooldown["buckets_remaining"] -= 1
        if cooldown["buckets_remaining"] <= 0:
            cooldown["active"] = False
            cooldown["buckets_remaining"] = 0

def check_cooldown_override(vpin_pct, cofi_z):
    """Returns True if the signal is strong enough to override cooldown."""
    if not cooldown["active"]:
        return True
    if vpin_pct >= cooldown["elevated_vpin"] and abs(cofi_z) >= cooldown["elevated_cofi"]:
        cooldown["active"] = False
        cooldown["buckets_remaining"] = 0
        print("⚡ COOLDOWN OVERRIDE: Exceptional signal detected, cooldown lifted!")
        return True
    return False

# ---------------------------------------------------------
# 5-MINUTE CANDLE AGGREGATOR
# ---------------------------------------------------------
def aggregate_5min_candle(one_min_candle):
    """Collect 1-min candles and aggregate every 5 into a 5-min candle."""
    five_min_accumulator.append(one_min_candle)
    if len(five_min_accumulator) >= 5:
        o = five_min_accumulator[0]["open"]
        h = max(c["high"] for c in five_min_accumulator)
        l = min(c["low"] for c in five_min_accumulator)
        c = five_min_accumulator[-1]["close"]
        v = sum(c_["volume"] for c_ in five_min_accumulator)
        
        candle_5m = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        five_min_candles.append(candle_5m)
        five_min_accumulator.clear()
        
        # Snapshot CVD and price for divergence detection
        cvd_state["cvd_history"].append(cvd_state["cvd"])
        cvd_state["price_history"].append(c)
        cvd_state["cvd"] = 0.0  # Reset per 5-min window
        
        # Run detection on new 5-min candle
        detect_fvgs()
        detect_order_blocks()
        detect_cvd_divergence()
        
        # Tier 1: Feed into 15-min aggregator
        aggregate_15min_candle(candle_5m)
        
        # Tier 1: Liquidity Sweep Detection
        track_swing_levels(candle_5m)
        detect_liquidity_sweep(candle_5m)
        
        # Tier 1: Auto-refresh Volume Profile every 4 hours
        check_vp_refresh()

# ---------------------------------------------------------
# LIVE FVG DETECTION
# ---------------------------------------------------------
def detect_fvgs():
    """Detect Fair Value Gaps on the 5-minute chart."""
    if len(five_min_candles) < 3:
        return
    
    candles = list(five_min_candles)
    ltp = state["ltp"]
    
    # Purge mitigated FVGs (price closed inside the gap)
    fvg_state["bullish_fvgs"] = [
        f for f in fvg_state["bullish_fvgs"]
        if not (ltp > f["bottom"] and ltp < f["top"])
    ]
    fvg_state["bearish_fvgs"] = [
        f for f in fvg_state["bearish_fvgs"]
        if not (ltp < f["top"] and ltp > f["bottom"])
    ]
    
    # Check latest 3 candles for new FVG
    c0 = candles[-3]  # Candle 1
    c1 = candles[-2]  # Candle 2 (impulse)
    c2 = candles[-1]  # Candle 3
    
    # Bullish FVG: candle 3's low > candle 1's high (gap up)
    if c2["low"] > c0["high"]:
        gap_bottom = c0["high"]
        gap_top = c2["low"]
        # Wick rule: candle 2's low must not go below candle 1's low
        if c1["low"] >= c0["low"]:
            fvg = {"bottom": gap_bottom, "top": gap_top, "ts": datetime.datetime.now().strftime("%H:%M")}
            # Only add if not duplicate
            if not any(abs(f["bottom"] - gap_bottom) < 1.0 for f in fvg_state["bullish_fvgs"]):
                fvg_state["bullish_fvgs"].append(fvg)
                print(f"📊 NEW BULLISH FVG DETECTED: [{gap_bottom:,.1f} - {gap_top:,.1f}]")
    
    # Bearish FVG: candle 3's high < candle 1's low (gap down)
    if c2["high"] < c0["low"]:
        gap_top = c0["low"]
        gap_bottom = c2["high"]
        # Wick rule: candle 2's high must not go above candle 1's high
        if c1["high"] <= c0["high"]:
            fvg = {"bottom": gap_bottom, "top": gap_top, "ts": datetime.datetime.now().strftime("%H:%M")}
            if not any(abs(f["top"] - gap_top) < 1.0 for f in fvg_state["bearish_fvgs"]):
                fvg_state["bearish_fvgs"].append(fvg)
                print(f"📊 NEW BEARISH FVG DETECTED: [{gap_bottom:,.1f} - {gap_top:,.1f}]")
    
    # Find nearest FVGs to current price
    all_bull = sorted(fvg_state["bullish_fvgs"], key=lambda f: abs(ltp - (f["bottom"] + f["top"]) / 2))
    all_bear = sorted(fvg_state["bearish_fvgs"], key=lambda f: abs(ltp - (f["bottom"] + f["top"]) / 2))
    fvg_state["nearest_bullish"] = all_bull[0] if all_bull else None
    fvg_state["nearest_bearish"] = all_bear[0] if all_bear else None

def is_price_in_fvg(ltp, is_long):
    """Check if current price is inside any active FVG zone."""
    fvgs = fvg_state["bullish_fvgs"] if is_long else fvg_state["bearish_fvgs"]
    for fvg in fvgs:
        if fvg["bottom"] <= ltp <= fvg["top"]:
            return True, fvg
    return False, None

# ---------------------------------------------------------
# ICT ORDER BLOCK DETECTION
# ---------------------------------------------------------
def detect_order_blocks():
    """Detect ICT Order Blocks on the 5-minute chart."""
    if len(five_min_candles) < 15:
        return
    
    candles = list(five_min_candles)
    ltp = state["ltp"]
    
    # Purge mitigated OBs (price closed through the OB zone)
    ob_state["bullish_obs"] = [
        ob for ob in ob_state["bullish_obs"]
        if not (ltp < ob["low"])  # Only remove if price went completely below
    ]
    ob_state["bearish_obs"] = [
        ob for ob in ob_state["bearish_obs"]
        if not (ltp > ob["high"])  # Only remove if price went completely above
    ]
    
    # Calculate body sizes
    bodies = [abs(c["close"] - c["open"]) for c in candles]
    is_bullish = [c["close"] > c["open"] for c in candles]
    
    n = len(candles)
    
    # Check last candle for displacement
    if n < 12:
        return
    
    i = n - 1  # Latest candle
    avg_body = np.mean(bodies[max(0, i-10):i])
    
    if avg_body <= 0 or bodies[i] < 1.5 * avg_body:
        return  # No displacement
    
    displacement_bullish = is_bullish[i]
    
    if displacement_bullish:
        # Find last bearish candle before displacement
        for j in range(i - 1, max(0, i - 6), -1):
            if not is_bullish[j]:
                ob = {
                    "low": candles[j]["low"],
                    "high": candles[j]["high"],
                    "ts": datetime.datetime.now().strftime("%H:%M"),
                    "displacement": bodies[i] / avg_body
                }
                if not any(abs(o["low"] - ob["low"]) < 1.0 for o in ob_state["bullish_obs"]):
                    ob_state["bullish_obs"].append(ob)
                    print(f"🏗️ BULLISH ORDER BLOCK: [{ob['low']:,.1f} - {ob['high']:,.1f}] (Displacement: {ob['displacement']:.1f}x)")
                break
    else:
        # Find last bullish candle before displacement
        for j in range(i - 1, max(0, i - 6), -1):
            if is_bullish[j]:
                ob = {
                    "low": candles[j]["low"],
                    "high": candles[j]["high"],
                    "ts": datetime.datetime.now().strftime("%H:%M"),
                    "displacement": bodies[i] / avg_body
                }
                if not any(abs(o["high"] - ob["high"]) < 1.0 for o in ob_state["bearish_obs"]):
                    ob_state["bearish_obs"].append(ob)
                    print(f"🏗️ BEARISH ORDER BLOCK: [{ob['low']:,.1f} - {ob['high']:,.1f}] (Displacement: {ob['displacement']:.1f}x)")
                break
    
    ob_state["active_count"] = len(ob_state["bullish_obs"]) + len(ob_state["bearish_obs"])

def is_price_in_ict_ob(ltp, is_long):
    """Check if current price is inside any active ICT Order Block."""
    obs = ob_state["bullish_obs"] if is_long else ob_state["bearish_obs"]
    for ob in obs:
        if ob["low"] <= ltp <= ob["high"]:
            return True, ob
    return False, None

# ---------------------------------------------------------
# CVD DIVERGENCE DETECTION
# ---------------------------------------------------------
def detect_cvd_divergence():
    """Detect price vs CVD divergence across 5-min windows."""
    if len(cvd_state["cvd_history"]) < 5 or len(cvd_state["price_history"]) < 5:
        cvd_state["divergence"] = "NONE"
        return
    
    prices = list(cvd_state["price_history"])
    cvds = list(cvd_state["cvd_history"])
    
    # Compare last 3 snapshots
    p1, p2, p3 = prices[-3], prices[-2], prices[-1]
    c1, c2, c3 = cvds[-3], cvds[-2], cvds[-1]
    
    # Bullish divergence: price making lower lows but CVD making higher lows
    if p3 < p1 and c3 > c1:
        cvd_state["divergence"] = "BULLISH"
    # Bearish divergence: price making higher highs but CVD making lower highs
    elif p3 > p1 and c3 < c1:
        cvd_state["divergence"] = "BEARISH"
    else:
        cvd_state["divergence"] = "NONE"

# ---------------------------------------------------------
# MULTI-TIMEFRAME CONFIRMATION (15-MIN)
# ---------------------------------------------------------
def aggregate_15min_candle(five_min_candle):
    """Collect 5-min candles and aggregate every 3 into a 15-min candle."""
    fifteen_min_accumulator.append(five_min_candle)
    if len(fifteen_min_accumulator) >= 3:
        o = fifteen_min_accumulator[0]["open"]
        h = max(c["high"] for c in fifteen_min_accumulator)
        l = min(c["low"] for c in fifteen_min_accumulator)
        c = fifteen_min_accumulator[-1]["close"]
        v = sum(c_["volume"] for c_ in fifteen_min_accumulator)
        
        candle_15m = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        fifteen_min_candles.append(candle_15m)
        fifteen_min_accumulator.clear()
        
        update_mtf_trend()

def update_mtf_trend():
    """Calculate 20-period EMA on 15-min closes and determine trend."""
    if len(fifteen_min_candles) < 5:
        mtf_state["trend"] = "NEUTRAL"
        return
    
    closes = [c["close"] for c in fifteen_min_candles]
    
    # EMA-20 calculation
    k = 2 / (min(20, len(closes)) + 1)
    ema = closes[0]
    for close in closes[1:]:
        ema = close * k + ema * (1 - k)
    
    mtf_state["ema_20"] = ema
    current_price = closes[-1]
    
    # Strength: how far price is from EMA in bps
    strength_bps = ((current_price - ema) / ema) * 10000
    mtf_state["strength"] = strength_bps
    
    if strength_bps > 15:
        mtf_state["trend"] = "BULLISH"
    elif strength_bps < -15:
        mtf_state["trend"] = "BEARISH"
    else:
        mtf_state["trend"] = "NEUTRAL"

# ---------------------------------------------------------
# LIQUIDITY SWEEP DETECTION
# ---------------------------------------------------------
def track_swing_levels(five_min_candle):
    """Track rolling highs and lows for sweep detection."""
    sweep_state["recent_highs"].append(five_min_candle["high"])
    sweep_state["recent_lows"].append(five_min_candle["low"])

def detect_liquidity_sweep(five_min_candle):
    """Detect if latest candle swept a previous high/low and reversed."""
    if len(sweep_state["recent_highs"]) < 5:
        return
    
    highs = list(sweep_state["recent_highs"])
    lows = list(sweep_state["recent_lows"])
    candle = five_min_candle
    
    # Find the highest high in the last 10-20 candles (excluding latest)
    prev_highs = highs[:-1] if len(highs) > 1 else highs
    prev_lows = lows[:-1] if len(lows) > 1 else lows
    
    swing_high = max(prev_highs[-10:]) if len(prev_highs) >= 3 else 0
    swing_low = min(prev_lows[-10:]) if len(prev_lows) >= 3 else float('inf')
    
    # Bearish sweep: wick above swing high, close below it
    if candle["high"] > swing_high and candle["close"] < swing_high:
        sweep_state["last_sweep"] = "BEARISH_SWEEP"
        sweep_state["sweep_price"] = swing_high
        sweep_state["sweep_ts"] = datetime.datetime.now().strftime("%H:%M")
        print(f"🎯 LIQUIDITY SWEEP: Bearish sweep above {swing_high:,.1f} — stop hunt reversal!")
    
    # Bullish sweep: wick below swing low, close above it
    elif candle["low"] < swing_low and candle["close"] > swing_low:
        sweep_state["last_sweep"] = "BULLISH_SWEEP"
        sweep_state["sweep_price"] = swing_low
        sweep_state["sweep_ts"] = datetime.datetime.now().strftime("%H:%M")
        print(f"🎯 LIQUIDITY SWEEP: Bullish sweep below {swing_low:,.1f} — stop hunt reversal!")
    else:
        # Decay sweep signal after 3 candles
        if sweep_state["last_sweep"] != "NONE":
            if len(highs) > 3:
                sweep_state["last_sweep"] = "NONE"

# ---------------------------------------------------------
# DISPLACEMENT SCORING (ICT OB QUALITY)
# ---------------------------------------------------------
def score_displacement(ob, candles_list):
    """Score an order block by its displacement candle's strength.
    Returns a score from 0-3: 0=weak, 1=normal, 2=strong, 3=extreme."""
    disp = ob.get("displacement", 1.5)
    if disp >= 3.0:
        return 3  # Extreme
    elif disp >= 2.0:
        return 2  # Strong
    elif disp >= 1.5:
        return 1  # Normal
    return 0

def get_best_ob_score(is_long):
    """Get the highest displacement score among active OBs near current price."""
    obs = ob_state["bullish_obs"] if is_long else ob_state["bearish_obs"]
    ltp = state["ltp"]
    best_score = 0
    for ob in obs:
        if ob["low"] <= ltp <= ob["high"]:
            score = score_displacement(ob, [])
            best_score = max(best_score, score)
    return best_score

# ---------------------------------------------------------
# TRADE JOURNAL (FOR SUPERVISED ML)
# ---------------------------------------------------------
def save_trade_journal():
    try:
        with open(TRADE_JOURNAL_FILE, "w") as f:
            json.dump(trade_journal, f, indent=2)
    except Exception:
        pass

def load_trade_journal():
    global trade_journal
    try:
        if os.path.exists(TRADE_JOURNAL_FILE):
            with open(TRADE_JOURNAL_FILE, "r") as f:
                trade_journal = json.load(f)
    except Exception:
        trade_journal = []

def journal_entry(action, entry_price, setup_grade, confluences):
    """Record features at trade entry for later ML training."""
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "action": action,
        "entry_price": entry_price,
        "setup_grade": setup_grade,
        "risk_multiplier": portfolio.get("confidence_multiplier", 1.0),
        "session": session_state["current"],
        "regime": ml_state["regime"],
        "vpin_pct": state["vpin_percentile"],
        "cofi_z": state["cofi_z"],
        "obi": state["obi"],
        "cvd_divergence": cvd_state["divergence"],
        "mtf_trend": mtf_state["trend"],
        "mtf_strength": mtf_state["strength"],
        "atr": state["atr_14"],
        "sweep": sweep_state["last_sweep"],
        "confluences": confluences,
        "ob_displacement_score": get_best_ob_score(action == "BUY"),
        # Filled on exit:
        "exit_price": 0.0,
        "pnl": 0.0,
        "outcome": "OPEN",
        "exit_reason": ""
    }
    trade_journal.append(entry)
    return len(trade_journal) - 1  # Return index for later update

def journal_exit(pnl, exit_price, exit_reason):
    """Update the last open journal entry with exit data."""
    for entry in reversed(trade_journal):
        if entry["outcome"] == "OPEN":
            entry["exit_price"] = exit_price
            entry["pnl"] = pnl
            entry["outcome"] = "WIN" if pnl > 0 else "LOSS"
            entry["exit_reason"] = exit_reason
            break
    save_trade_journal()

# ---------------------------------------------------------
# DYNAMIC VOLUME PROFILE REFRESH
# ---------------------------------------------------------
def check_vp_refresh():
    """Refresh volume profile every 4 hours."""
    now = time.time()
    if vp_refresh["last_refresh_time"] == 0:
        vp_refresh["last_refresh_time"] = now
        return
    
    if now - vp_refresh["last_refresh_time"] >= vp_refresh["refresh_interval_sec"]:
        print("🔄 Auto-refreshing Volume Profiles...")
        build_volume_profile()
        build_session_volume_profile()
        vp_refresh["last_refresh_time"] = now

# ---------------------------------------------------------
# DATABASE & LOGGING
# ---------------------------------------------------------
def init_clickhouse():
    try:
        client = clickhouse_connect.get_client(host='localhost', port=8123)
        client.command("""
            CREATE TABLE IF NOT EXISTS crypto_paper_trades (
                timestamp DateTime64(3),
                symbol String,
                action String,
                price Float64,
                pnl Float64,
                reason String
            ) ENGINE = MergeTree()
            ORDER BY timestamp
        """)
        return client
    except Exception as e:
        return None

db_client = init_clickhouse()

def log_trade(action, price, pnl, reason):
    portfolio["trades_executed"] += 1
    
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    pnl_str = f"${pnl:+.2f}" if pnl != 0 else " $0.00"
    
    if "STOP LOSS" in reason or "TAKE PROFIT" in reason or "EXIT" in reason:
        entry_px = portfolio['entry_price']
        log_msg = f"[{now_str}] CLOSE {action:4} | Entry: {entry_px:,.1f} -> Exit: {price:,.1f} | PnL: {pnl_str} | {reason}"
    else:
        log_msg = f"[{now_str}] OPEN  {action:4} | Entry: {price:,.1f} {' ' * 19}| PnL: {pnl_str} | {reason}"
        
    recent_trades.append(log_msg)
    
    if db_client:
        try:
            now = datetime.datetime.now()
            db_client.insert('crypto_paper_trades', 
                [[now, state["symbol"], action, float(price), float(pnl), reason]], 
                column_names=['timestamp', 'symbol', 'action', 'price', 'pnl', 'reason'])
        except:
            pass

# ---------------------------------------------------------
# EXECUTION LOGIC
# ---------------------------------------------------------
def execute_signal():
    if portfolio["position"] != "FLAT":
        return
    
    # --- SESSION GATE ---
    get_current_session()
    if not session_state["execution_allowed"]:
        return  # Dead zone: no trades
    
    vpin_pct = state["vpin_percentile"]
    cofi = state["cofi_z"]
    acf = state["acf_lag1"]
    velocity = state["bucket_velocity_sec"]
    
    if velocity > 60.0:
        return # Block fake breakouts
    
    # --- DYNAMIC COOLDOWN GATE ---
    if cooldown["active"] and not check_cooldown_override(vpin_pct, cofi):
        return  # In cooldown, signal not strong enough to override
    
    # --- SETUP B: LIQUIDATION SQUEEZE (unchanged, but with session gate) ---
    longs_wiped = liquidations["longs_usd"]
    shorts_wiped = liquidations["shorts_usd"]
    
    if longs_wiped > 500000.0 or shorts_wiped > 500000.0:
        is_long = longs_wiped > shorts_wiped
        sl_pct = DYNAMIC_PARAMS["STOP_LOSS_BPS"] / 10000.0
        risk_dollars = portfolio["account_balance"] * 0.02
        notional_size = risk_dollars / sl_pct
        max_notional = portfolio["account_balance"] * 50.0
        portfolio["notional_size_usd"] = min(notional_size, max_notional)
        portfolio["leverage_used"] = portfolio["notional_size_usd"] / portfolio["account_balance"]
        
        portfolio["phase"] = 1
        portfolio["post_entry_volume"] = 0.0
        portfolio["post_entry_vwap_sum"] = 0.0
        portfolio["vwap"] = 0.0
        
        if is_long:
            portfolio["position"] = "LONG"
            portfolio["entry_price"] = state["ask"]
            print("🔥 SETUP B - SQUEEZE REVERSAL (Capitulation Caught)")
            log_trade("BUY", state["ask"], 0.0, "SETUP B - LONG")
        else:
            portfolio["position"] = "SHORT"
            portfolio["entry_price"] = state["bid"]
            print("🔥 SETUP B - SQUEEZE REVERSAL (Short Squeeze Caught)")
            log_trade("SELL", state["bid"], 0.0, "SETUP B - SHORT")
            
        liquidations["longs_usd"] = 0.0
        liquidations["shorts_usd"] = 0.0
        liquidation_events.clear()
        return
    
    # --- SETUP A: INSTITUTIONAL FLOW + ORDER BLOCK + ABSORPTION ---
    # --- PROBABILISTIC SCORING ENGINE ---
    is_long = cofi > 0
    ltp = state["ltp"]
    hvns = macro_structure["hvns"]
    
    # --- MULTI-SOURCE ORDER BLOCK CHECK (THE TRAP/SWEEP) ---
    # Calculate Dynamic Radius based on Volatility (Stop Loss)
    atr_bps = (state["atr_14"] / state["ltp"]) * 10000 if state["ltp"] > 0 else 0
    adjusted_sl_bps = max(20.0, min(60.0, atr_bps * 0.8))
    dynamic_radius = (adjusted_sl_bps / 10000.0) * 0.25
    
    def did_sweep_level(level, radius):
        radius_abs = level * radius
        zone_upper = level + radius_abs
        zone_lower = level - radius_abs
        if zone_lower <= ltp <= zone_upper:
            return True
        lookback = min(10, len(minute_candles))
        for c in list(minute_candles)[-lookback:]:
            if c["low"] <= zone_upper and c["high"] >= zone_lower:
                return True
        return False
    
    # Source 1: Volume Profile HVNs (Macro & Micro)
    in_vp_ob = False
    ob_price = 0.0
    for hvn in hvns + macro_structure["micro_hvns"]:
        if did_sweep_level(hvn, dynamic_radius):
            in_vp_ob = True
            ob_price = hvn
            break
    if not in_vp_ob:
        if did_sweep_level(macro_structure["poc"], dynamic_radius):
            in_vp_ob = True
            ob_price = macro_structure["poc"]
    
    # Source 2: ICT Order Blocks (5-min structural)
    in_ict_ob, ict_ob = is_price_in_ict_ob(ltp, is_long)
    
    # Source 3: FVG (Fair Value Gaps)
    in_fvg, fvg = is_price_in_fvg(ltp, is_long)
    
    # Source 4: Liquidity Sweep
    sweep = sweep_state["last_sweep"]
    in_sweep = (is_long and sweep == "BULLISH_SWEEP") or (not is_long and sweep == "BEARISH_SWEEP")
    
    # --- PROBABILISTIC SCORING MODEL (12-Point System) ---
    total_pts = 0
    
    # 1. Structure Points (Max 3)
    if in_vp_ob: total_pts += 3
    elif in_ict_ob: total_pts += 2
    elif in_fvg or in_sweep: total_pts += 1
    
    # 2. Order Flow / COFI Points (Max 3)
    abs_cofi = abs(cofi)
    if abs_cofi >= 3.0: total_pts += 3
    elif abs_cofi >= 2.0: total_pts += 2
    elif abs_cofi >= 1.0: total_pts += 1
    
    # 3. Toxicity / VPIN Points (Max 3)
    if vpin_pct >= 95: total_pts += 3
    elif vpin_pct >= 85: total_pts += 2
    elif vpin_pct >= 70: total_pts += 1
    
    # 4. Confluence Points
    cvd_div = cvd_state["divergence"]
    mtf_trend = mtf_state["trend"]
    
    cvd_aligned = (is_long and cvd_div == "BULLISH") or (not is_long and cvd_div == "BEARISH")
    cvd_opposing = (is_long and cvd_div == "BEARISH") or (not is_long and cvd_div == "BULLISH")
    mtf_aligned = (is_long and mtf_trend == "BULLISH") or (not is_long and mtf_trend == "BEARISH")
    mtf_opposing = (is_long and mtf_trend == "BEARISH") or (not is_long and mtf_trend == "BULLISH")
    
    if cvd_aligned: total_pts += 1
    elif cvd_opposing: total_pts -= 1
    
    if mtf_aligned: total_pts += 1
    elif mtf_opposing: total_pts -= 1
    
    # 5. Velocity Health (Anti-Knife)
    vel = state.get("bucket_velocity_sec", 999.0)
    if vel >= 30.0: total_pts += 1
    elif vel < 15.0: total_pts -= 1 # Penalize waterfall, but don't hard ban if score is huge
    
    # --- FLUID QUALITY GATE ---
    if total_pts < 4:
        return # Need at least 4/12 points to execute
        
    # --- DETERMINE SETUP GRADE & MULTIPLIER ---
    if total_pts >= 9:
        setup_grade = "S"
        risk_mult = 2.0
    elif total_pts >= 7:
        setup_grade = "A"
        risk_mult = 1.5
    elif total_pts >= 5:
        setup_grade = "B"
        risk_mult = 1.0
    else:
        setup_grade = "C"
        risk_mult = 0.5
        
    portfolio["confidence_multiplier"] = risk_mult
    
    # --- DYNAMIC POSITION SIZING & VOLATILITY STOPS ---
    atr_bps = (state["atr_14"] / state["ltp"]) * 10000 if state["ltp"] > 0 else 0
    # Dynamic SL: base is 20 bps, scales up with ATR, cap at 60 bps
    adjusted_sl_bps = max(20.0, min(60.0, atr_bps * 0.8))
    DYNAMIC_PARAMS["STOP_LOSS_BPS"] = adjusted_sl_bps
    DYNAMIC_PARAMS["SCALE_OUT_BPS"] = adjusted_sl_bps  # 1:1 risk-reward for first scale
    
    sl_pct = adjusted_sl_bps / 10000.0
    # Base risk is 2%, scaled by the confidence multiplier
    risk_dollars = portfolio["account_balance"] * 0.02 * risk_mult
    notional_size = risk_dollars / sl_pct
    max_notional = portfolio["account_balance"] * (50.0 if risk_mult > 0.5 else 20.0)
    portfolio["notional_size_usd"] = min(notional_size, max_notional)
    portfolio["leverage_used"] = portfolio["notional_size_usd"] / portfolio["account_balance"]
    
    portfolio["phase"] = 1
    portfolio["post_entry_volume"] = 0.0
    portfolio["post_entry_vwap_sum"] = 0.0
    portfolio["vwap"] = 0.0
    
    # Build reason string
    sources = []
    if in_vp_ob: sources.append(f"VP-OB:{ob_price:,.0f}")
    if in_ict_ob: sources.append(f"ICT-OB:{ict_ob['low']:,.0f}-{ict_ob['high']:,.0f}")
    if in_fvg: sources.append(f"FVG:{fvg['bottom']:,.0f}-{fvg['top']:,.0f}")
    if cvd_div != "NONE": sources.append(f"CVD:{cvd_div}")
    if in_sweep: sources.append(f"SWEEP:{sweep_state['sweep_price']:,.0f}")
    ob_disp_score = get_best_ob_score(is_long) if in_ict_ob else 0
    if ob_disp_score >= 2: sources.append(f"DISP:{ob_disp_score}")
    if mtf_trend != "NEUTRAL": sources.append(f"15m:{mtf_trend}")
    source_str = " | ".join(sources)
    
    if cofi > 0:
        portfolio["position"] = "LONG"
        portfolio["entry_price"] = state["ask"]
        portfolio["trade_open_time"] = time.time()
        reason = f"SETUP {setup_grade} - LONG [{source_str}]"
        log_trade("BUY", state["ask"], 0.0, reason)
        journal_entry("BUY", state["ask"], setup_grade, source_str)
    else:
        portfolio["position"] = "SHORT"
        portfolio["entry_price"] = state["bid"]
        portfolio["trade_open_time"] = time.time()
        reason = f"SETUP {setup_grade} - SHORT [{source_str}]"
        log_trade("SELL", state["bid"], 0.0, reason)
        journal_entry("SELL", state["bid"], setup_grade, source_str)

def check_exits():
    if portfolio["position"] == "FLAT":
        return
        
    ts_now = time.time()
    trade_duration_mins = (ts_now - portfolio.get("trade_open_time", ts_now)) / 60.0
        
    entry = portfolio["entry_price"]
    vwap = portfolio["vwap"]
    
    SL_BPS = DYNAMIC_PARAMS["STOP_LOSS_BPS"]
    SCALE_BPS = DYNAMIC_PARAMS["SCALE_OUT_BPS"]
    BE_BPS = DYNAMIC_PARAMS["BREAKEVEN_BPS"]
    
    if portfolio["position"] == "LONG":
        gross_pnl_pct = (state["bid"] - entry) / entry * 10000
        net_pnl_pct = gross_pnl_pct - TOTAL_ROUNDTRIP_COST_BPS
        
        portfolio["unrealized_pnl"] = (net_pnl_pct / 10000) * portfolio["notional_size_usd"]
        
        if portfolio["phase"] == 1:
            if net_pnl_pct <= -SL_BPS:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("SELL", state["bid"], portfolio["unrealized_pnl"], "STOP LOSS")
                journal_exit(portfolio["unrealized_pnl"], state["bid"], "STOP LOSS")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
                activate_cooldown()
            elif trade_duration_mins > 45 and portfolio["unrealized_pnl"] <= 0:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("SELL", state["bid"], portfolio["unrealized_pnl"], "TIME DECAY EXIT")
                journal_exit(portfolio["unrealized_pnl"], state["bid"], "TIME DECAY EXIT")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif net_pnl_pct >= SCALE_BPS:
                half_pnl = portfolio["unrealized_pnl"] / 2.0
                portfolio["realized_pnl"] += half_pnl
                portfolio["account_balance"] += half_pnl
                portfolio["notional_size_usd"] /= 2.0
                log_trade("SELL", state["bid"], half_pnl, "SCALE OUT (50%)")
                portfolio["phase"] = 2
                portfolio["unrealized_pnl"] /= 2.0
                
        elif portfolio["phase"] == 2:
            if net_pnl_pct <= BE_BPS:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("SELL", state["bid"], portfolio["unrealized_pnl"], "BREAKEVEN STOP")
                journal_exit(portfolio["unrealized_pnl"], state["bid"], "BREAKEVEN STOP")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif vwap > 0 and state["bid"] < vwap:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("SELL", state["bid"], portfolio["unrealized_pnl"], "VWAP EXIT")
                journal_exit(portfolio["unrealized_pnl"], state["bid"], "VWAP EXIT")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            
    elif portfolio["position"] == "SHORT":
        gross_pnl_pct = (entry - state["ask"]) / entry * 10000
        net_pnl_pct = gross_pnl_pct - TOTAL_ROUNDTRIP_COST_BPS
        
        portfolio["unrealized_pnl"] = (net_pnl_pct / 10000) * portfolio["notional_size_usd"]
        
        if portfolio["phase"] == 1:
            if net_pnl_pct <= -SL_BPS:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("BUY", state["ask"], portfolio["unrealized_pnl"], "STOP LOSS")
                journal_exit(portfolio["unrealized_pnl"], state["ask"], "STOP LOSS")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
                activate_cooldown()
            elif trade_duration_mins > 45 and portfolio["unrealized_pnl"] <= 0:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("BUY", state["ask"], portfolio["unrealized_pnl"], "TIME DECAY EXIT")
                journal_exit(portfolio["unrealized_pnl"], state["ask"], "TIME DECAY EXIT")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif net_pnl_pct >= SCALE_BPS:
                half_pnl = portfolio["unrealized_pnl"] / 2.0
                portfolio["realized_pnl"] += half_pnl
                portfolio["account_balance"] += half_pnl
                portfolio["notional_size_usd"] /= 2.0
                log_trade("BUY", state["ask"], half_pnl, "SCALE OUT (50%)")
                portfolio["phase"] = 2
                portfolio["unrealized_pnl"] /= 2.0
                
        elif portfolio["phase"] == 2:
            if net_pnl_pct <= BE_BPS:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("BUY", state["ask"], portfolio["unrealized_pnl"], "BREAKEVEN STOP")
                journal_exit(portfolio["unrealized_pnl"], state["ask"], "BREAKEVEN STOP")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif vwap > 0 and state["ask"] > vwap:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("BUY", state["ask"], portfolio["unrealized_pnl"], "VWAP EXIT")
                journal_exit(portfolio["unrealized_pnl"], state["ask"], "VWAP EXIT")
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0

# ---------------------------------------------------------
# DATA PROCESSING
# ---------------------------------------------------------
def process_trade(price, qty, is_buyer_maker):
    state["ltp"] = price
    state["ticks_today"] += 1
    
    if price <= 0:
        return
        
    # 1. Update Live Candle
    ts_now = time.time()
    if state["current_minute_ts"] == 0:
        state["current_minute_ts"] = int(ts_now // 60) * 60
        live_candle["open"] = live_candle["high"] = live_candle["low"] = live_candle["close"] = price
        live_candle["volume"] = 0.0
        
    if ts_now > state["current_minute_ts"] + 60:
        completed_candle = dict(live_candle)
        minute_candles.append(completed_candle)
        update_dynamic_parameters()
        aggregate_5min_candle(completed_candle)  # Feed into 5-min aggregator
        
        state["current_minute_ts"] = int(ts_now // 60) * 60
        live_candle["open"] = live_candle["high"] = live_candle["low"] = live_candle["close"] = price
        live_candle["volume"] = 0.0
    else:
        live_candle["high"] = max(live_candle["high"], price) if live_candle["high"] > 0 else price
        live_candle["low"] = min(live_candle["low"], price) if live_candle["low"] > 0 else price
        live_candle["close"] = price
        live_candle["volume"] += qty
        
    # 2. VWAP Tracker
    if portfolio["position"] != "FLAT":
        portfolio["post_entry_volume"] += qty
        portfolio["post_entry_vwap_sum"] += (price * qty)
        portfolio["vwap"] = portfolio["post_entry_vwap_sum"] / portfolio["post_entry_volume"]
    
    direction = "SELL" if is_buyer_maker else "BUY"
    
    # 3. CVD Tracking (cumulative per 5-min window)
    if direction == "BUY":
        cvd_state["cvd"] += qty
        cvd_state["5min_buy_vol"] += qty
    else:
        cvd_state["cvd"] -= qty
        cvd_state["5min_sell_vol"] += qty
    
    # 4. Update COFI
    signed_vol = qty if direction == "BUY" else -qty
    order_flow_imbalances.append(signed_vol)
    
    if len(order_flow_imbalances) > 10:
        cofi_arr = np.array(order_flow_imbalances)
        mean = np.mean(cofi_arr)
        std = np.std(cofi_arr) + 1e-9
        state["cofi_z"] = (signed_vol - mean) / std
        
        if len(order_flow_imbalances) > 100:
            cofi_series = np.array(order_flow_imbalances)[-100:]
            corr = np.corrcoef(cofi_series[:-1], cofi_series[1:])[0,1]
            state["acf_lag1"] = corr if not np.isnan(corr) else 0.0
    
    # 4. Update VPIN
    global current_bucket
    if direction == "BUY":
        current_bucket["buy_vol"] += qty
    else:
        current_bucket["sell_vol"] += qty
        
    current_bucket["total_vol"] += qty
    
    if current_bucket["total_vol"] >= DYNAMIC_PARAMS["BUCKET_VOLUME_SIZE"]:
        imbalance = abs(current_bucket["buy_vol"] - current_bucket["sell_vol"])
        vpin = imbalance / current_bucket["total_vol"]
        vpin_history.append(vpin)
        
        state["bucket_velocity_sec"] = time.time() - current_bucket["start_time"]
        state["buckets_completed"] += 1
        current_bucket = {"buy_vol": 0.0, "sell_vol": 0.0, "total_vol": 0.0, "start_time": time.time()}
        tick_cooldown()  # Decrement cooldown on each bucket
        
        vpin_arr = np.array(vpin_history)
        state["vpin"] = vpin_arr[-1]
        
        if len(vpin_arr) > 10:
            state["vpin_percentile"] = int(np.percentile([np.sum(vpin_arr <= x) / len(vpin_arr) * 100 for x in vpin_arr], 100) if vpin_arr[-1] == np.max(vpin_arr) else np.sum(vpin_arr <= vpin_arr[-1]) / len(vpin_arr) * 100)

        # 5. Machine Learning Regime Classification
        ml_state["features"].append([state["vpin"], abs(state["cofi_z"]), abs(state["obi"])])
        
        if state["buckets_completed"] % 5 == 0:
            save_ml_state()
            
        if len(ml_state["features"]) >= 50:
            if state["buckets_completed"] % 50 == 0:
                X = np.array(ml_state["features"])
                try:
                    model = KMeans(n_clusters=2, n_init=10, random_state=42)
                    model.fit(X)
                    ml_state["model"] = model
                    
                    centroids = model.cluster_centers_
                    score_0 = centroids[0][0] + centroids[0][1]
                    score_1 = centroids[1][0] + centroids[1][1]
                    
                    ml_state["trending_cluster_id"] = 0 if score_0 > score_1 else 1
                except Exception:
                    pass
            
            if ml_state["model"] is not None:
                try:
                    current_f = np.array([[state["vpin"], abs(state["cofi_z"]), abs(state["obi"])]])
                    pred = ml_state["model"].predict(current_f)[0]
                    if pred == ml_state["trending_cluster_id"]:
                        ml_state["regime"] = "TRENDING (HIGH VOL)"
                    else:
                        ml_state["regime"] = "CHOP (LOW VOL)"
                except Exception:
                    pass

def render_dashboard():
    os.system('cls' if os.name == 'nt' else 'clear')
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    vpin_pct = state["vpin_percentile"]
    cofi_z = state["cofi_z"]
    acf = state["acf_lag1"]
    
    vpin_bar = "████" + "░░" * int(10 - (vpin_pct/10))
    
    print("="*65)
    print(f"  CRYPTO PAPER TRADING LAB (INSTITUTIONAL PHYSICS) | {now_str}")
    print("="*65)
    ws_status = "🟢 CONNECTED" if state["ws_connected"] else "🔴 DISCONNECTED"
    print(f"Binance WS: {ws_status} | {state['symbol']} LTP: {state['ltp']:,.1f}")
    print(f"Bid: {state['bid']:,.1f} | Ask: {state['ask']:,.1f}")
    print("")
    obi = state["obi"]
    obi_status = "MASSIVE BUY WALL (Spoofing)" if obi > 0.60 else "MASSIVE SELL WALL (Spoofing)" if obi < -0.60 else "BALANCED"
    obi_color = "🟢" if obi > 0.60 else "🔴" if obi < -0.60 else "⚪"
    
    print(f"VPIN:      {state['vpin']:.3f}  [{vpin_bar:10}]  Percentile: {vpin_pct}th")
    ml_regime = ml_state["regime"]
    print(f"COFI Z:    {cofi_z:+.2f}  |  ML Regime: {ml_regime} ({len(ml_state['features'])}/500)")
    print(f"L2 OBI:    {obi:+.2f}  |  Wall: {obi_color} {obi_status}")
    
    # CVD Display
    cvd_val = cvd_state["cvd"]
    cvd_div = cvd_state["divergence"]
    cvd_div_color = "\U0001f7e2" if cvd_div == "BULLISH" else "\U0001f534" if cvd_div == "BEARISH" else "\u26aa"
    print(f"CVD:       {cvd_val:+.1f}  |  Divergence: {cvd_div_color} {cvd_div}")
    
    vel_color = "\U0001f7e2" if state["bucket_velocity_sec"] > 0 and state["bucket_velocity_sec"] < 60.0 else "\U0001f534"
    print(f"Bucket Vel: {state['bucket_velocity_sec']:>5.1f}s {vel_color}  (Needs < 60s for Entry)")
    print("\u2500"*65)
    
    # Session & Cooldown
    get_current_session()
    sess = session_state["current"]
    sess_color = "\U0001f7e2" if sess in ("LONDON", "NEW_YORK") else "\U0001f7e1" if sess == "ASIAN" else "\U0001f534"
    cd_str = f"ACTIVE ({cooldown['buckets_remaining']} buckets)" if cooldown["active"] else "CLEAR"
    cd_color = "\U0001f534" if cooldown["active"] else "\U0001f7e2"
    print(f"SESSION: {sess_color} {sess} (Sizing Modifier) | Cooldown: {cd_color} {cd_str}")
    
    # MTF Trend
    mtf_t = mtf_state["trend"]
    mtf_color = "\U0001f7e2" if mtf_t == "BULLISH" else "\U0001f534" if mtf_t == "BEARISH" else "\u26aa"
    mtf_ema = mtf_state["ema_20"]
    print(f"15m Trend: {mtf_color} {mtf_t} (EMA-20: {mtf_ema:,.1f} | Sizing Modifier)")
    
    # Liquidity Sweep
    sw = sweep_state["last_sweep"]
    sw_color = "\U0001f3af" if sw != "NONE" else "\u26aa"
    sw_display = f"{sw} @ {sweep_state['sweep_price']:,.0f}" if sw != "NONE" else "NONE"
    print(f"Liq Sweep: {sw_color} {sw_display}")
    print("\u2500"*65)
    
    print(f"MACRO STRUCTURE (30-Day Volume Profile):")
    ltp = state["ltp"]
    hvns = macro_structure["hvns"]
    support = [h for h in hvns if h < ltp]
    resistance = [h for h in hvns if h > ltp]
    nearest_sup = max(support) if support else 0.0
    nearest_res = min(resistance) if resistance else 0.0
    print(f"Point of Control (POC): {macro_structure['poc']:,.1f}")
    print(f"Nearest Resistance:     {nearest_res:,.1f}")
    print(f"Nearest Support:        {nearest_sup:,.1f}")
    
    # Order Block Status (VP + ICT combined)
    in_vp_ob = False
    atr_bps_dash = (state["atr_14"] / state["ltp"]) * 10000 if state["ltp"] > 0 else 0
    dynamic_radius_dash = (max(20.0, min(60.0, atr_bps_dash * 0.8)) / 10000.0) * 0.25
    def did_sweep_level_dash(level, radius):
        radius_abs = level * radius
        zone_upper = level + radius_abs
        zone_lower = level - radius_abs
        if zone_lower <= ltp <= zone_upper:
            return True
        lookback = min(10, len(minute_candles))
        for c in list(minute_candles)[-lookback:]:
            if c["low"] <= zone_upper and c["high"] >= zone_lower:
                return True
        return False

    for h in hvns + macro_structure["micro_hvns"] + [macro_structure["poc"]]:
        if did_sweep_level_dash(h, dynamic_radius_dash):
            in_vp_ob = True
            break
    in_ict_bull, _ = is_price_in_ict_ob(ltp, True)
    in_ict_bear, _ = is_price_in_ict_ob(ltp, False)
    
    in_fvg_bull, _ = is_price_in_fvg(ltp, True)
    in_fvg_bear, _ = is_price_in_fvg(ltp, False)
    in_sweep_act = sweep_state["last_sweep"] != "NONE"
    
    in_execution_zone = in_vp_ob or in_ict_bull or in_ict_bear or in_fvg_bull or in_fvg_bear or in_sweep_act
    
    trend_str = "\U0001f7e2 IN STRUCTURAL ZONE (Execution Allowed)" if in_execution_zone else "\u26aa NO-MAN'S LAND (Execution Banned)"
    print(f"Liquidity State:        {trend_str}")
    
    # ICT Order Blocks with displacement scores
    bull_ob_count = len(ob_state["bullish_obs"])
    bear_ob_count = len(ob_state["bearish_obs"])
    ob_strs = []
    for ob in ob_state["bullish_obs"][-2:]:
        d_score = score_displacement(ob, [])
        d_label = ["⚪", "🟡", "🟠", "🔴"][min(d_score, 3)]
        ob_strs.append(f"Bull:{ob['low']:,.0f}-{ob['high']:,.0f}{d_label}")
    for ob in ob_state["bearish_obs"][-2:]:
        d_score = score_displacement(ob, [])
        d_label = ["⚪", "🟡", "🟠", "🔴"][min(d_score, 3)]
        ob_strs.append(f"Bear:{ob['low']:,.0f}-{ob['high']:,.0f}{d_label}")
    ob_display = " | ".join(ob_strs) if ob_strs else "None"
    print(f"ICT Order Blocks:       {bull_ob_count + bear_ob_count} Active ({ob_display})")
    
    # FVGs
    bull_fvg_count = len(fvg_state["bullish_fvgs"])
    bear_fvg_count = len(fvg_state["bearish_fvgs"])
    fvg_strs = []
    if fvg_state["nearest_bullish"]:
        f = fvg_state["nearest_bullish"]
        fvg_strs.append(f"Bull:{f['bottom']:,.0f}-{f['top']:,.0f}")
    if fvg_state["nearest_bearish"]:
        f = fvg_state["nearest_bearish"]
        fvg_strs.append(f"Bear:{f['bottom']:,.0f}-{f['top']:,.0f}")
    fvg_display = " | ".join(fvg_strs) if fvg_strs else "None"
    print(f"Active FVGs:            {bull_fvg_count + bear_fvg_count} ({fvg_display})")
    print("\u2500"*65)
    
    print(f"LIQUIDATIONS (Rolling 60s):")
    longs_wiped = liquidations["longs_usd"]
    shorts_wiped = liquidations["shorts_usd"]
    print(f"Longs Wiped:  ${longs_wiped:,.0f}  {'\U0001f534 CAPITULATION DUMP' if longs_wiped > 500000 else ''}")
    print(f"Shorts Wiped: ${shorts_wiped:,.0f}  {'\U0001f7e2 SHORT SQUEEZE' if shorts_wiped > 500000 else ''}")
    print("\u2500"*65)
    
    print("DYNAMIC ENGINE (Live Calibrated):")
    print(f"ATR (14m): {state['atr_14']:.1f}  |  Avg 1m Vol: {state['avg_1min_vol']:.1f} BTC")
    print(f"Bucket Size: {DYNAMIC_PARAMS['BUCKET_VOLUME_SIZE']:.1f} BTC  |  SL: {DYNAMIC_PARAMS['STOP_LOSS_BPS']:.1f} bps  |  Scale: {DYNAMIC_PARAMS['SCALE_OUT_BPS']:.1f} bps")
    print(f"Cost/Trade:  {TOTAL_ROUNDTRIP_COST_BPS:.1f} bps  (Taker + Slippage applied to Net PnL)")
    
    # Journal stats
    wins = sum(1 for t in trade_journal if t.get("outcome") == "WIN")
    losses = sum(1 for t in trade_journal if t.get("outcome") == "LOSS")
    wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    print(f"5m: {len(five_min_candles)}  |  15m: {len(fifteen_min_candles)}  |  Journal: {len(trade_journal)} trades (WR: {wr:.0f}%)")
    print("\u2500"*65)
    
    pos_color = "🟢" if portfolio["position"] == "LONG" else "🔴" if portfolio["position"] == "SHORT" else "⚪"
    pnl_color = "🟢" if portfolio["unrealized_pnl"] > 0 else "🔴" if portfolio["unrealized_pnl"] < 0 else "⚪"
    total_color = "🟢" if portfolio["realized_pnl"] > 0 else "🔴" if portfolio["realized_pnl"] < 0 else "⚪"
    
    vwap_str = f"|  Runner VWAP: {portfolio['vwap']:,.1f}" if portfolio["phase"] == 2 else ""
    
    print(f"VIRTUAL PORTFOLIO (NET PnL):")
    print(f"Balance:  ${portfolio['account_balance']:,.2f}  |  Total PnL: {total_color} ${portfolio['realized_pnl']:.2f}  |  Trades: {portfolio['trades_executed']}")
    print(f"Position: {pos_color} {portfolio['position']:5}  |  Entry: {portfolio['entry_price']:,.1f} {vwap_str}")
    print(f"Size:     ${portfolio['notional_size_usd']:,.0f} (Lev: {portfolio['leverage_used']:.1f}x)  |  Phase: {portfolio['phase']}")
    print(f"Unreal PnL: {pnl_color} ${portfolio['unrealized_pnl']:.2f}")
    print("="*65)
    print(f"Ticks Processed: {state['ticks_today']:,}  |  Buckets: {state['buckets_completed']}")
    
    if len(recent_trades) > 0:
        print("-" * 65)
        print("RECENT TRADES:")
        for trade in recent_trades:
            print(f"  {trade}")

# ---------------------------------------------------------
# WEBSOCKET LOOP
# ---------------------------------------------------------
def clean_liquidations():
    current_time = time.time()
    while len(liquidation_events) > 0 and current_time - liquidation_events[0]["time"] > 60.0:
        event = liquidation_events.popleft()
        if event["side"] == "SELL":
            liquidations["longs_usd"] -= event["usd_value"]
        else:
            liquidations["shorts_usd"] -= event["usd_value"]
        
        liquidations["longs_usd"] = max(0.0, liquidations["longs_usd"])
        liquidations["shorts_usd"] = max(0.0, liquidations["shorts_usd"])

async def binance_ws():
    url = "wss://fstream.binance.com/stream?streams=btcusdt@trade/btcusdt@bookTicker/btcusdt@depth20@100ms/btcusdt@forceOrder"
    
    while True:
        try:
            async with websockets.connect(url) as ws:
                state["ws_connected"] = True
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    stream = data.get("stream")
                    payload = data.get("data")
                    
                    if stream == "btcusdt@trade":
                        price = float(payload["p"])
                        qty = float(payload["q"])
                        is_buyer_maker = payload["m"]
                        process_trade(price, qty, is_buyer_maker)
                        clean_liquidations()
                        execute_signal()
                        check_exits()
                        
                    elif stream == "btcusdt@bookTicker":
                        state["bid"] = float(payload["b"])
                        state["ask"] = float(payload["a"])
                        
                    elif stream == "btcusdt@depth20@100ms":
                        try:
                            bids = payload.get("b", [])
                            asks = payload.get("a", [])
                            
                            total_bid_vol = sum(float(b[1]) for b in bids)
                            total_ask_vol = sum(float(a[1]) for a in asks)
                            
                            total_vol = total_bid_vol + total_ask_vol
                            if total_vol > 0:
                                state["obi"] = (total_bid_vol - total_ask_vol) / total_vol
                        except Exception:
                            pass
                            
                    elif stream == "btcusdt@forceOrder":
                        try:
                            order = payload.get("o", {})
                            side = order.get("S")
                            qty = float(order.get("q", 0))
                            price = float(order.get("p", 0))
                            
                            usd_value = qty * price
                            current_time = time.time()
                            
                            liquidation_events.append({"side": side, "usd_value": usd_value, "time": current_time})
                            if side == "SELL":
                                liquidations["longs_usd"] += usd_value
                            else:
                                liquidations["shorts_usd"] += usd_value
                        except Exception:
                            pass

                        
        except Exception as e:
            state["ws_connected"] = False
            await asyncio.sleep(2)

async def ui_loop():
    while True:
        if state["ws_connected"] and state["ticks_today"] > 0:
            render_dashboard()
        await asyncio.sleep(1)

async def main():
    load_ml_state()
    prewarm_engine()
    await asyncio.gather(
        binance_ws(),
        ui_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
