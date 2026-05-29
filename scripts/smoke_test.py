"""Smoke test — verifies that the V5 core infrastructure boots correctly."""

import sys
sys.path.insert(0, ".")

def main():
    print("Running Apex V5 Smoke Test...\n")

    # Test 1: Config loads
    from src.core.config import market_configs, risk_config, model_config, get_market, get_active_symbols
    print("✅ Config loaded")
    print(f"   Markets: {list(market_configs.keys())}")
    print(f"   Active crypto: {get_active_symbols('crypto')}")
    print(f"   Active india: {get_active_symbols('india_equity')}")

    # Test 2: Feature store initializes
    from src.features.store import FeatureStore
    crypto_store = FeatureStore("crypto")
    india_store = FeatureStore("india_equity")
    print(f"✅ FeatureStore: crypto={crypto_store.num_features} features, india={india_store.num_features} features")

    # Test 3: Indicators compute on dummy data
    import pandas as pd
    import numpy as np
    np.random.seed(42)
    n = 100
    dummy = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min", tz="UTC"),
        "open": 100 + np.random.randn(n).cumsum(),
        "high": 0.0,
        "low": 0.0,
        "close": 100 + np.random.randn(n).cumsum(),
        "volume": np.random.randint(100, 1000, n).astype(float),
    })
    dummy["high"] = dummy[["open", "close"]].max(axis=1) + abs(np.random.randn(n)) * 0.5
    dummy["low"] = dummy[["open", "close"]].min(axis=1) - abs(np.random.randn(n)) * 0.5

    features = crypto_store.build_features(dummy)
    print(f"✅ Feature build: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"   NaN count: {features.isna().sum().sum()}")
    nan_cols = features.columns[features.isna().any()].tolist()
    if nan_cols:
        print(f"   NaN columns: {nan_cols}")

    # Test 4: Triple barrier labeling
    from src.features.labeling.triple_barrier import apply_triple_barrier_labels
    labeled = apply_triple_barrier_labels(dummy, profit_target_pct=0.01, stop_loss_pct=0.006, max_holding_bars=15)
    print(f"✅ Triple barrier: {len(labeled)} labeled samples, pos_rate={labeled['target'].mean():.2%}")

    # Test 5: Regime detection
    from src.models.regime import classify_regime, MarketRegime
    r1 = classify_regime(adx=30.0, chop=45.0)
    r2 = classify_regime(adx=18.0, chop=70.0)
    print(f"✅ Regime: ADX=30,CHOP=45 -> {r1.value} | ADX=18,CHOP=70 -> {r2.value}")

    # Test 6: Cost model
    from src.backtester.cost_model import COST_MODELS
    for name, cm in COST_MODELS.items():
        print(f"   {name:15s}: round_trip={cm.round_trip_cost_pct:.4%}")
    print("✅ CostModels ready")

    # Test 7: Event bus
    from src.core.events import event_bus, MarketTick
    print("✅ EventBus ready")

    # Test 8: Data validator
    from src.data.validation.validator import DataValidator
    v = DataValidator()
    cleaned = v.validate_dataframe(dummy.copy())
    print(f"✅ DataValidator: {len(dummy)} -> {len(cleaned)} rows after validation")

    # Test 9: XGBoost model instantiation
    from src.models.xgboost_model import ApexXGBoostModel
    model = ApexXGBoostModel(
        name="test_model",
        feature_columns=crypto_store.feature_columns,
    )
    print(f"✅ XGBoost model: {model.model_name}, trained={model.is_trained}")

    # Test 10: Portfolio tracker
    from src.backtester.portfolio import PortfolioTracker
    pt = PortfolioTracker(initial_capital=10000.0)
    print(f"✅ PortfolioTracker: equity=${pt.equity:.2f}")

    print()
    print("=" * 55)
    print("  ALL SYSTEMS OPERATIONAL - Phase 0 Core Ready")
    print("=" * 55)


if __name__ == "__main__":
    main()
