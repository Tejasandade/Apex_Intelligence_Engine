# -*- coding: utf-8 -*-
"""
PROOF OF EDGE -- The Only Test That Matters
============================================
Does the XGBoost model predict BTC direction better than a coin flip?

No Council. No Position Manager. No Broker. No Trailing Stops.
Just: Model -> Prediction -> Did price move in that direction?
"""
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.store import FeatureStore

# 1. Load data
print("Loading data...")
df = pd.read_csv("data/historical/cleaned/BTCUSDT_1m_cleaned.csv")
if 'time' in df.columns:
    df = df.rename(columns={'time': 'timestamp'})
print(f"  Loaded {len(df):,} bars")

# 2. Build features
print("Building features...")
store = FeatureStore("crypto")
features = store.build_features(df)
print(f"  Built {features.shape[1]} features across {features.shape[0]:,} bars")

# 3. Load the TRENDING model
print("Loading XGBoost model...")
model = xgb.Booster()
model.load_model("data/models/btcusdt_crypto_trending.json")

# 4. Get predictions
print("Running predictions...")
start = 200
end = len(df) - 100

feature_names = list(features.columns)
dmatrix = xgb.DMatrix(features.iloc[start:end], feature_names=feature_names)
predictions = model.predict(dmatrix)

print(f"  Got {len(predictions):,} predictions")
print(f"  Mean probability: {predictions.mean():.4f}")
print(f"  Std:  {predictions.std():.4f}")
print(f"  Min:  {predictions.min():.4f}  Max: {predictions.max():.4f}")

# 5. Distribution
above_58 = (predictions >= 0.58).sum()
below_42 = (predictions <= 0.42).sum()
neutral = len(predictions) - above_58 - below_42
print(f"\n  Signals at 0.58 threshold:")
print(f"    BUY  (prob >= 0.58): {above_58:,} ({above_58/len(predictions):.1%})")
print(f"    SELL (prob <= 0.42): {below_42:,} ({below_42/len(predictions):.1%})")
print(f"    HOLD (in between):  {neutral:,} ({neutral/len(predictions):.1%})")

# 6. THE PROOF
print("\n" + "="*60)
print("  THE PROOF: Forward Direction Check")
print("="*60)

for threshold_name, buy_thresh, sell_thresh in [
    ("Loose (0.55)", 0.55, 0.45),
    ("Current (0.58)", 0.58, 0.42),
    ("Tight (0.62)", 0.62, 0.38),
]:
    wins_1rr = 0; losses_1rr = 0
    wins_2rr = 0; losses_2rr = 0

    for idx in range(len(predictions)):
        prob = predictions[idx]
        bar_idx = start + idx

        if prob >= buy_thresh:
            direction = "BUY"
        elif prob <= sell_thresh:
            direction = "SELL"
        else:
            continue

        close = df.iloc[bar_idx]["close"]
        atr = features.iloc[bar_idx].get("ATR", 0)
        if atr <= 0:
            continue

        stop_dist = atr * 1.5

        # 1:1 RR check
        for j in range(bar_idx + 1, min(bar_idx + 100, len(df))):
            hi = df.iloc[j]["high"]
            lo = df.iloc[j]["low"]

            if direction == "BUY":
                if lo <= close - stop_dist:
                    losses_1rr += 1; break
                if hi >= close + stop_dist:
                    wins_1rr += 1; break
            else:
                if hi >= close + stop_dist:
                    losses_1rr += 1; break
                if lo <= close - stop_dist:
                    wins_1rr += 1; break

    # 2:1 RR pass
    for idx in range(len(predictions)):
        prob = predictions[idx]
        bar_idx = start + idx

        if prob >= buy_thresh:
            direction = "BUY"
        elif prob <= sell_thresh:
            direction = "SELL"
        else:
            continue

        close = df.iloc[bar_idx]["close"]
        atr = features.iloc[bar_idx].get("ATR", 0)
        if atr <= 0:
            continue

        stop_dist = atr * 1.5
        tp_dist = stop_dist * 2.0

        for j in range(bar_idx + 1, min(bar_idx + 100, len(df))):
            hi = df.iloc[j]["high"]
            lo = df.iloc[j]["low"]

            if direction == "BUY":
                if lo <= close - stop_dist:
                    losses_2rr += 1; break
                if hi >= close + tp_dist:
                    wins_2rr += 1; break
            else:
                if hi >= close + stop_dist:
                    losses_2rr += 1; break
                if lo <= close - tp_dist:
                    wins_2rr += 1; break

    total_1rr = wins_1rr + losses_1rr
    total_2rr = wins_2rr + losses_2rr

    print(f"\n  Threshold: {threshold_name}")
    print(f"  ---------------------------------")

    if total_1rr > 0:
        wr_1rr = wins_1rr / total_1rr
        edge_1rr = (wr_1rr * 1.0) - ((1-wr_1rr) * 1.0)
        print(f"  1:1 RR -> {total_1rr:,} trades | Win Rate: {wr_1rr:.1%} | Edge/trade: {edge_1rr:.3f}R")
        if wr_1rr > 0.50:
            print(f"           [PASS] EDGE EXISTS (>50% needed at 1:1)")
        else:
            print(f"           [FAIL] NO EDGE ({wr_1rr:.1%} < 50% needed at 1:1)")
    else:
        print(f"  1:1 RR -> No trades resolved")

    if total_2rr > 0:
        wr_2rr = wins_2rr / total_2rr
        edge_2rr = (wr_2rr * 2.0) - ((1-wr_2rr) * 1.0)
        pf = (wins_2rr * 2.0) / (losses_2rr * 1.0) if losses_2rr > 0 else float('inf')
        print(f"  2:1 RR -> {total_2rr:,} trades | Win Rate: {wr_2rr:.1%} | Edge/trade: {edge_2rr:.3f}R | PF: {pf:.2f}")
        if wr_2rr > 0.334:
            print(f"           [PASS] EDGE EXISTS (>33.4% needed at 2:1)")
        else:
            print(f"           [FAIL] NO EDGE ({wr_2rr:.1%} < 33.4% needed at 2:1)")
    else:
        print(f"  2:1 RR -> No trades resolved")

# 7. Baseline
print(f"\n{'='*60}")
print("  BASELINE: Random coin flip at same trade count")
print("="*60)
print(f"  At 1:1 RR, coin flip = 50.0% win rate, 0.000R edge")
print(f"  At 2:1 RR, coin flip = ~33.3% win rate, 0.000R edge")
print(f"\n  If our model beats these baselines, it has genuine")
print(f"  predictive power. If not, no amount of architecture helps.")
print(f"{'='*60}")
