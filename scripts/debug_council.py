import asyncio
import pandas as pd
from src.core.config import DATA_DIR
from src.features.store import FeatureStore
from src.engine.signal_engine import LiveSignalEngine

async def debug_signals():
    symbol = "btcusdt"
    market_type = "crypto"
    
    # 1. Load Data
    data_path = DATA_DIR / "historical" / "cleaned" / f"{symbol.upper()}_1m_cleaned.csv"
    df = pd.read_csv(data_path)
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Setup engine
    engine = LiveSignalEngine(
        symbol=symbol,
        market_type=market_type,
        models=None, # Will auto-load
        conviction_threshold=0.58,
        consensus_threshold=0.50,
        enable_ensemble=True,
        enable_council=True
    )
    engine.warm_up(df.iloc[:60])
    
    # 3. Simulate
    print(f"Running debug on {len(df)} bars...")
    signals = 0
    blocked = 0
    passed = 0
    wins = 0
    losses = 0
    
    # Pre-build features for speed
    store = FeatureStore(market_type)
    features_df = store.build_features(df)
    
    for i in range(60, len(df) - 100): # Allow 100 bars for forward check
        row = df.iloc[i]
        feat_row = features_df.iloc[[i]]
        
        # We need a real model probability here if we want real results
        probability = engine.models["XGBoost"].predict(feat_row) if "XGBoost" in engine.models else 0.6
        if probability < 0.58:
            continue
            
        direction = "BUY" if probability > 0.5 else "SELL"
        close = row["close"]
        atr = feat_row.iloc[0].get("ATR", close*0.01)
        
        council_decision = engine.council.evaluate(
            features=feat_row, direction=direction, price=close,
            regime="TRENDING", atr=atr, symbol=symbol
        )
        
        if not council_decision.approved:
            blocked += 1
            continue
            
        passed += 1
        
        # Forward check 2:1 RR
        sl = close - (atr * 1.5) if direction == "BUY" else close + (atr * 1.5)
        tp = close + (atr * 3.0) if direction == "BUY" else close - (atr * 3.0)
        
        won = False
        lost = False
        for j in range(i+1, i+100):
            future_low = df.iloc[j]["low"]
            future_high = df.iloc[j]["high"]
            if direction == "BUY":
                if future_low <= sl: lost = True; break
                if future_high >= tp: won = True; break
            else:
                if future_high >= sl: lost = True; break
                if future_low <= tp: won = True; break
                
        if won: wins += 1
        elif lost: losses += 1
            
    print(f"\nStats: Passed={passed}, Blocked={blocked}")
    print(f"Wins: {wins}, Losses: {losses}")
    if wins+losses > 0:
        win_rate = wins / (wins+losses)
        pf = (wins * 2.0) / (losses * 1.0) if losses > 0 else float('inf')
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Profit Factor (2:1 RR): {pf:.2f}")

if __name__ == "__main__":
    asyncio.run(debug_signals())
