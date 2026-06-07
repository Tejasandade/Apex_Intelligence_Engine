"""
Apex Intelligence Engine V5 — Pipeline Verification
===================================================
A comprehensive end-to-end verification script that tests every layer in isolation
AND together, using real data from the cached Parquet.
"""

import sys
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import DATA_DIR
from src.features.store import FeatureStore
from src.features.live_store import LiveFeatureStore
from src.models.xgboost_model import ApexXGBoostModel
from src.models.regime import classify_regime
from src.models.ensemble import ModelEnsemble
from src.council.council import Council
from src.engine.signal_engine import LiveSignalEngine
from src.execution.paper_broker import PaperBroker
from src.execution.position_manager import PositionManager

async def main():
    print("\n" + "="*60)
    print("  APEX PIPELINE VERIFICATION (BTCUSDT)")
    print("="*60 + "\n")

    # 1. Data Layer
    print("[1] Testing Data Layer...")
    data_path = DATA_DIR / "historical" / "crypto" / "btcusdt.parquet"
    if not data_path.exists():
        print("[FAIL] Data file missing!")
        return
    df = pd.read_parquet(data_path)
    df = df.tail(250).reset_index(drop=True)
    print(f"[OK] Loaded {len(df)} candles from {data_path.name}")

    # 2. Feature Layer
    print("\n[2] Testing Feature Layer...")
    feature_store = FeatureStore("crypto")
    features = feature_store.build_features(df)
    if len(features) == len(df):
        print(f"[OK] Built {features.shape[1]} features for {features.shape[0]} rows without issues")
    else:
        print("[FAIL] Feature build failed or row mismatch")

    # 3. Model Layer
    print("\n[3] Testing Model Layer...")
    models_dir = DATA_DIR / "models"
    trending_model = ApexXGBoostModel(
        name="btcusdt_crypto_trending",
        feature_columns=feature_store.feature_columns
    )
    try:
        trending_model.load(models_dir)
        print(f"[OK] Loaded trending model: is_trained={trending_model.is_trained}")
        
        # Test prediction
        last_features = features.iloc[[-1]]
        prob = trending_model.predict(last_features)
        print(f"[OK] Prediction works: p={prob:.4f}")
    except Exception as e:
        print(f"[FAIL] Model test failed: {e}")

    # 4. Regime Layer
    print("\n[4] Testing Regime Layer...")
    try:
        last_row = features.iloc[-1]
        regime = classify_regime(
            adx=last_row.get("ADX_14", 25.0),
            chop=last_row.get("CHOP_14", 50.0),
            atr=last_row.get("ATR_14", 1.0),
            atr_median=last_row.get("ATR_14", 1.0)
        )
        print(f"[OK] Regime detected: {regime.value}")
    except Exception as e:
        print(f"[FAIL] Regime test failed: {e}")

    # 5. Ensemble Layer
    print("\n[5] Testing Ensemble Layer...")
    try:
        ensemble = ModelEnsemble(weighting="performance")
        ensemble.add_member(trending_model, regime="TRENDING")
        prob, conviction, preds = ensemble.predict(last_features, current_regime=regime.value)
        print(f"[OK] Ensemble prediction works: p={prob:.4f}, conviction={conviction:.4f}")
    except Exception as e:
        print(f"[FAIL] Ensemble test failed: {e}")

    # 6. Council Layer
    print("\n[6] Testing Council Layer...")
    try:
        council = Council(consensus_threshold=0.6)
        council.setup_default_advisors()
        decision = council.evaluate(
            features=last_features,
            direction="BUY",
            price=df['close'].iloc[-1],
            regime=regime.value,
            atr=last_row.get("ATR_14", 1.0)
        )
        print(f"[OK] Council evaluation works:")
        print(f"   Approved: {decision.approved}")
        print(f"   Consensus: {decision.consensus_score:.4f}")
        for v in decision.votes:
            print(f"     - {v.advisor_name}: {v.vote} (w={v.weight:.1f}, c={v.conviction:.2f})")
    except Exception as e:
        print(f"[FAIL] Council test failed: {e}")

    # 7. Live Feature Store
    print("\n[7] Testing Live Feature Store...")
    try:
        live_store = LiveFeatureStore("crypto")
        live_store.warm_up(df.iloc[:-5])
        
        # Feed last 5 candles one by one
        latest_feats = None
        for i in range(len(df)-5, len(df)):
            candle = df.iloc[i].to_dict()
            latest_feats = live_store.update(candle)
            
        if latest_feats is not None:
            print(f"[OK] Live store state: buffer length {live_store.buffer_length}")
            print(f"[OK] Features generated correctly")
        else:
            print("[FAIL] Live store returned None")
    except Exception as e:
        print(f"[FAIL] Live feature store failed: {e}")

    # 8. Execution Layer
    print("\n[8] Testing Execution Layer...")
    try:
        broker = PaperBroker(initial_balance=10000.0)
        await broker.connect()
        pos_mgr = PositionManager(broker=broker)
        await pos_mgr.initialize()
        
        broker.update_price("btcusdt", 60000.0)
        order = await pos_mgr.open_position(
            signal_id="test_1",
            symbol="btcusdt",
            direction="BUY",
            quantity=0.01,
            price=60000.0,
            stop_loss=59000.0,
            take_profit=62000.0,
            regime="TRENDING",
            council_score=0.8
        )
        
        if order and order.status.value == "FILLED":
            print(f"[OK] Position opened successfully. Open positions: {pos_mgr.open_position_count}")
        else:
            print("[FAIL] Position open failed")
            
        # Simulate price moving to take profit
        trade, partial = await pos_mgr.update_price("btcusdt", 62050.0)
        if trade:
            print(f"[OK] Position closed successfully. Exit reason: {trade.exit_reason}")
            print(f"   PnL: ${trade.pnl:.2f}")
        else:
            print("[FAIL] Position close failed")
    except Exception as e:
        print(f"[FAIL] Execution layer failed: {e}")

    print("\n" + "="*60)
    print("  VERIFICATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
