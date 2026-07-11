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
    "poc": 0.0  # Point of Control
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

def prewarm_engine():
    print("Pre-warming Dynamic Volatility Engine from Binance REST API...")
    try:
        build_volume_profile()
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
            
        state["ltp"] = float(data[-1][4]) # Last close price
        update_dynamic_parameters()
        state["current_minute_ts"] = int(time.time() // 60) * 60
        live_candle["open"] = live_candle["high"] = live_candle["low"] = live_candle["close"] = state["ltp"]
        live_candle["volume"] = 0.0
        print(f"Pre-warm complete! Found {len(minute_candles)} candles. Starting Live Execution.")
    except Exception as e:
        print(f"Pre-warm failed: {e}. Will rely on live ticks.")

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
        
    vpin_pct = state["vpin_percentile"]
    cofi = state["cofi_z"]
    acf = state["acf_lag1"]
    velocity = state["bucket_velocity_sec"]
    
    if velocity > 60.0:
        return # Block fake breakouts
        
    longs_wiped = liquidations["longs_usd"]
    shorts_wiped = liquidations["shorts_usd"]
    
    if longs_wiped > 500000.0 or shorts_wiped > 500000.0:
        is_long = longs_wiped > shorts_wiped
        sl_pct = DYNAMIC_PARAMS["STOP_LOSS_BPS"] / 10000.0
        risk_dollars = portfolio["account_balance"] * 0.02 # 2% risk
        notional_size = risk_dollars / sl_pct
        max_notional = portfolio["account_balance"] * 50.0 # Max 50x leverage
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
    cofi_threshold = 3.0 if ml_state["regime"] == "CHOP (LOW VOL)" else 2.0
    
    if vpin_pct >= 95 and abs(cofi) > cofi_threshold:
        is_long = cofi > 0
        ltp = state["ltp"]
        hvns = macro_structure["hvns"]
        
        # Institutional Liquidity Filter (Order Blocks & Absorption)
        in_order_block = False
        ob_price = 0.0
        
        for hvn in hvns:
            if abs(ltp - hvn) / ltp <= 0.0030: # 0.30% Order Block Zone
                in_order_block = True
                ob_price = hvn
                break
                
        if not in_order_block:
            if abs(ltp - macro_structure["poc"]) / ltp <= 0.0030:
                in_order_block = True
                ob_price = macro_structure["poc"]
                
        if not in_order_block:
            return # Ban trades in no-man's land
            
        obi = state["obi"]
        if is_long and obi < 0.30:
            print(f"Liquidity Engine: Blocked LONG at {ob_price:,.1f} OB. No Limit Buy Wall Absorption (OBI: {obi:.2f}).")
            return
        if not is_long and obi > -0.30:
            print(f"Liquidity Engine: Blocked SHORT at {ob_price:,.1f} OB. No Limit Sell Wall Absorption (OBI: {obi:.2f}).")
            return
            
        sl_pct = DYNAMIC_PARAMS["STOP_LOSS_BPS"] / 10000.0
        risk_dollars = portfolio["account_balance"] * 0.02 # 2% risk
        notional_size = risk_dollars / sl_pct
        max_notional = portfolio["account_balance"] * 50.0 # Max 50x leverage
        portfolio["notional_size_usd"] = min(notional_size, max_notional)
        portfolio["leverage_used"] = portfolio["notional_size_usd"] / portfolio["account_balance"]
        
        portfolio["phase"] = 1
        portfolio["post_entry_volume"] = 0.0
        portfolio["post_entry_vwap_sum"] = 0.0
        portfolio["vwap"] = 0.0
        
        if cofi > 0:
            portfolio["position"] = "LONG"
            portfolio["entry_price"] = state["ask"]
            log_trade("BUY", state["ask"], 0.0, "SETUP A - LONG")
        else:
            portfolio["position"] = "SHORT"
            portfolio["entry_price"] = state["bid"]
            log_trade("SELL", state["bid"], 0.0, "SETUP A - SHORT")

def check_exits():
    if portfolio["position"] == "FLAT":
        return
        
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
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif vwap > 0 and state["bid"] < vwap:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("SELL", state["bid"], portfolio["unrealized_pnl"], "VWAP EXIT")
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
                portfolio["position"] = "FLAT"
                portfolio["phase"] = 0
                portfolio["unrealized_pnl"] = 0.0
            elif vwap > 0 and state["ask"] > vwap:
                portfolio["realized_pnl"] += portfolio["unrealized_pnl"]
                portfolio["account_balance"] += portfolio["unrealized_pnl"]
                log_trade("BUY", state["ask"], portfolio["unrealized_pnl"], "VWAP EXIT")
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
        minute_candles.append(dict(live_candle))
        update_dynamic_parameters()
        
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
    
    # 3. Update COFI
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
    
    vel_color = "🟢" if state["bucket_velocity_sec"] > 0 and state["bucket_velocity_sec"] < 60.0 else "🔴"
    print(f"Bucket Vel: {state['bucket_velocity_sec']:>5.1f}s {vel_color}  (Needs < 60s for Entry)")
    print("─"*65)
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
    in_ob = False
    for h in hvns + [macro_structure["poc"]]:
        if abs(ltp - h) / ltp <= 0.0030:
            in_ob = True
            break
            
    trend_str = "🟢 INSIDE ORDER BLOCK (Hunting Liquidity)" if in_ob else "⚪ NO-MAN'S LAND (Execution Banned)"
    print(f"Liquidity State:        {trend_str}")
    print("─"*65)
    print(f"LIQUIDATIONS (Rolling 60s):")
    longs_wiped = liquidations["longs_usd"]
    shorts_wiped = liquidations["shorts_usd"]
    print(f"Longs Wiped:  ${longs_wiped:,.0f}  {'🔴 CAPITULATION DUMP' if longs_wiped > 500000 else ''}")
    print(f"Shorts Wiped: ${shorts_wiped:,.0f}  {'🟢 SHORT SQUEEZE' if shorts_wiped > 500000 else ''}")
    print("─"*65)
    
    print("DYNAMIC ENGINE (Live Calibrated):")
    print(f"ATR (14m): {state['atr_14']:.1f}  |  Avg 1m Vol: {state['avg_1min_vol']:.1f} BTC")
    print(f"Bucket Size: {DYNAMIC_PARAMS['BUCKET_VOLUME_SIZE']:.1f} BTC  |  SL: {DYNAMIC_PARAMS['STOP_LOSS_BPS']:.1f} bps  |  Scale: {DYNAMIC_PARAMS['SCALE_OUT_BPS']:.1f} bps")
    print(f"Cost/Trade:  {TOTAL_ROUNDTRIP_COST_BPS:.1f} bps  (Taker + Slippage applied to Net PnL)")
    print("─"*65)
    
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
