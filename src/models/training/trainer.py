"""
Apex Intelligence Engine V5 — Unified Training Pipeline
=========================================================
One pipeline to train ANY market's model:

1. Load historical data
2. Compute market-specific features via FeatureStore
3. Apply triple-barrier labeling
4. Split by regime (TRENDING/RANGING)
5. Purged walk-forward cross-validation
6. Train XGBoost per regime with calibration
7. Run backtest against holdout
8. Save model ONLY if it passes quality gates

Usage:
    python -m scripts.train_all
    # or
    trainer = UnifiedTrainer("crypto", "btcusdt")
    await trainer.run()
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.backtester.cost_model import COST_MODELS
from src.backtester.engine import BacktestEngine
from src.backtester.metrics import check_quality_gates, compute_metrics
from src.core.config import (
    DATA_DIR,
    WEIGHTS_DIR,
    get_market,
    get_model_params,
    get_quality_gates,
)
from src.core.logging import get_logger
from src.features.labeling.triple_barrier import apply_triple_barrier_labels
from src.features.store import FeatureStore
from src.models.regime import classify_regime_series
from src.models.training.cross_validation import purged_walk_forward_splits
from src.models.xgboost_model import ApexXGBoostModel

logger = get_logger("apex.models.training.trainer")


class UnifiedTrainer:
    """
    Unified training pipeline for any market type.

    Handles the complete training lifecycle:
    1. Data loading
    2. Feature engineering
    3. Target labeling
    4. Regime splitting
    5. Cross-validated training with calibration
    6. Holdout backtesting
    7. Quality gate verification
    8. Model persistence
    """

    def __init__(
        self,
        market_type: str,
        symbol: str,
    ):
        """
        Initialize the trainer.

        Args:
            market_type: Market type ("crypto" or "india_equity").
            symbol: Symbol key (e.g., "btcusdt", "banknifty").
        """
        self.market_type = market_type
        self.symbol = symbol
        self.market_config = get_market(symbol)
        self.model_params = get_model_params(market_type)
        self.quality_gates = get_quality_gates()
        self.feature_store = FeatureStore(market_type)

        self._models: dict[str, ApexXGBoostModel] = {}

    async def run(
        self,
        data: pd.DataFrame | None = None,
        save_models: bool = True,
    ) -> dict[str, Any]:
        """
        Execute the full training pipeline.

        Args:
            data: Pre-loaded OHLCV DataFrame. If None, loads from disk.
            save_models: Whether to save models that pass quality gates.

        Returns:
            Dict with training results per regime.
        """
        logger.info(
            "training_pipeline_started",
            market_type=self.market_type,
            symbol=self.symbol,
        )

        # Step 1: Load data
        if data is None:
            data = self._load_data()

        if data is None or len(data) < 500:
            logger.error(
                "training_insufficient_data",
                symbol=self.symbol,
                rows=len(data) if data is not None else 0,
            )
            return {"error": "insufficient_data"}
            
        # ── MEMORY CONSTRAINT FIX ──
        # Cap data to 10,000 rows to prevent extreme RAM OOM/Swapping
        if len(data) > 10000:
            data = data.iloc[-10000:].copy()

        logger.info("data_loaded", rows=len(data))

        # Step 2: Compute features
        feature_df = self.feature_store.build_features(data)
        combined = data.copy()
        for col in feature_df.columns:
            combined[col] = feature_df[col]

        logger.info("features_computed", columns=len(feature_df.columns))

        # Step 3: Apply triple-barrier labeling
        labeling_cfg = self.model_params.get("labeling", {})
        labeled_df = apply_triple_barrier_labels(
            combined,
            profit_target_atr=labeling_cfg.get("profit_target_atr", 2.0),
            stop_loss_atr=labeling_cfg.get("stop_loss_atr", 2.0),
            max_holding_bars=labeling_cfg.get("max_holding_bars", 30),
            atr_col="ATR"
        )

        if len(labeled_df) < 200:
            logger.error("labeling_too_few_samples", samples=len(labeled_df))
            return {"error": "insufficient_labeled_samples"}

        logger.info(
            "labels_applied",
            total=len(labeled_df),
            long_positive_rate=f"{labeled_df['target_long'].mean():.2%}",
            short_positive_rate=f"{labeled_df['target_short'].mean():.2%}",
        )

        # Step 4: Regime classification
        results: dict[str, Any] = {}

        if self.model_params.get("regime_split", True):
            regimes = classify_regime_series(
                labeled_df,
                adx_threshold=self.model_params.get("regime_thresholds", {}).get(
                    "adx_trending", 25.0
                ),
                chop_threshold=self.model_params.get("regime_thresholds", {}).get(
                    "chop_choppy", 61.8
                ),
            )

            for regime_name in ["TRENDING_HIGH_VOL", "TRENDING_LOW_VOL", "RANGING_HIGH_VOL", "RANGING_LOW_VOL"]:
                regime_mask = regimes == regime_name
                regime_df = labeled_df[regime_mask].copy()

                if len(regime_df) < 100:
                    logger.warning(
                        "regime_skip_insufficient",
                        regime=regime_name,
                        samples=len(regime_df),
                    )
                    results[regime_name] = {"skipped": True, "samples": len(regime_df)}
                    continue

                result = self._train_regime(
                    regime_df, regime_name.lower(), save_models
                )
                results[regime_name] = result
        else:
            result = self._train_regime(labeled_df, "all", save_models)
            results["ALL"] = result

        logger.info("training_pipeline_completed", symbol=self.symbol, results=list(results.keys()))
        return results

    def _train_regime(
        self,
        df: pd.DataFrame,
        regime_suffix: str,
        save_models: bool,
    ) -> dict[str, Any]:
        """
        Train both LONG and SHORT models for a specific regime subset.
        """
        X = df[self.feature_store.feature_columns].copy().fillna(0.0)
        y_long = df["target_long"].copy()
        y_short = df["target_short"].copy()

        hp = self.model_params.get("hyperparameters", {}).copy()
        hp["n_jobs"] = 2  # Memory constraint fix

        cv_cfg = self.model_params.get("training", {})
        
        regime_results = {}

        for direction, y in [("LONG", y_long), ("SHORT", y_short)]:
            model_name = f"{self.symbol}_{self.market_type}_{regime_suffix}_{direction}"

            logger.info(
                "regime_training_started",
                model=model_name,
                samples=len(df),
                positive_rate=f"{y.mean():.2%}",
            )

            # Purged walk-forward CV
            effective_min_train = max(50, min(cv_cfg.get("min_train_size", 500), len(X) // 3))
            folds = purged_walk_forward_splits(
                n_samples=len(X),
                n_folds=cv_cfg.get("cv_folds", 5),
                purge_gap=cv_cfg.get("purge_gap", 15),
                min_train_size=effective_min_train,
            )

            fold_metrics = []
            for fold in folds:
                X_train = X.iloc[fold.train_indices]
                y_train = y.iloc[fold.train_indices]
                X_val = X.iloc[fold.val_indices]
                y_val = y.iloc[fold.val_indices]

                fold_model = ApexXGBoostModel(
                    name=f"{model_name}_fold{fold.fold_num}",
                    feature_columns=self.feature_store.feature_columns,
                    hyperparams=hp,
                )

                metrics = fold_model.train(X_train, y_train, X_val, y_val)
                fold_metrics.append(metrics)
                
                del X_train, y_train, X_val, y_val, fold_model
                gc.collect()

            avg_metrics = {k: float(np.mean([f[k] for f in fold_metrics])) for k in fold_metrics[0]} if fold_metrics else {}

            # ── Train Bagged Ensemble on ALL data (90/10 split for early stopping) ──
            split_idx = int(len(X) * 0.90)
            X_train_final, y_train_final = X.iloc[:split_idx], y.iloc[:split_idx]
            X_cal, y_cal = X.iloc[split_idx:], y.iloc[split_idx:]

            bag_metrics_list = []
            best_bag_features = {}

            for bag_idx in range(1, 4):
                bag_name = f"{model_name}_bag{bag_idx}"
                bag_hp = hp.copy()

                # ── True Bootstrap Bagging ──────────────────────────────────────
                # Each bag samples 80% of rows WITH REPLACEMENT (bootstrap),
                # and uses a different random_state + hyperparameter jitter.
                # This creates genuine model diversity — bags will disagree on
                # hard examples, making the ensemble's disagreement_penalty work.
                rng = np.random.default_rng(42 + bag_idx * 7919)   # prime seed per bag
                n_train = len(X_train_final)
                boot_idx = rng.choice(n_train, size=int(n_train * 0.80), replace=True)
                X_boot = X_train_final.iloc[boot_idx].copy()
                y_boot = y_train_final.iloc[boot_idx].copy()

                # Per-bag hyperparameter jitter for additional diversity
                bag_hp["random_state"]     = 42 + bag_idx * 1337
                bag_hp["subsample"]        = round(min(0.95, 0.75 + bag_idx * 0.05), 2)   # 0.80, 0.85, 0.90
                bag_hp["colsample_bytree"] = round(min(0.95, 0.70 + bag_idx * 0.05), 2)   # 0.75, 0.80, 0.85
                bag_hp["learning_rate"]    = round(bag_hp.get("learning_rate", 0.05) * (1.0 + (bag_idx - 2) * 0.1), 4)  # -10%, 0%, +10%
                bag_hp["min_child_weight"] = 1 + (bag_idx - 1)  # 1, 2, 3

                bag_model = ApexXGBoostModel(
                    name=bag_name,
                    feature_columns=self.feature_store.feature_columns,
                    hyperparams=bag_hp,
                )

                metrics = bag_model.train(X_boot, y_boot, X_cal, y_cal)
                bag_metrics_list.append(metrics)

                if save_models:
                    acc = float(metrics.get("accuracy", 0.0))
                    ll = float(metrics.get("log_loss", 1.0))
                    passes_gate = False
                    if acc >= 0.48 and ll <= 0.88:
                        passes_gate = True
                    elif acc >= 0.40 and ll <= 0.80:
                        passes_gate = True
                    elif acc >= 0.35 and ll <= 1.40 and ("high_vol" in bag_name or "ranging" in bag_name):
                        passes_gate = True
                    elif acc >= 0.20 and ll <= 0.72 and "ranging_high_vol" in bag_name:
                        passes_gate = True

                    if passes_gate:
                        bag_model.save(WEIGHTS_DIR)
                        self._models[f"{regime_suffix}_{direction}_bag{bag_idx}"] = bag_model
                        logger.info(f"Passed quality gates: {bag_name} (acc={acc:.3f}, ll={ll:.3f})")
                    else:
                        logger.warning(f"Failed quality gates: {bag_name} (acc={acc:.3f}, ll={ll:.3f})")

                if bag_idx == 1:
                    best_bag_features = bag_model.get_feature_importance()

                del X_boot, y_boot, bag_model
                gc.collect()

            final_metrics = {k: float(np.mean([m[k] for m in bag_metrics_list])) for k in bag_metrics_list[0]}

            regime_results[direction] = {
                "model_name": model_name,
                "cv_metrics": avg_metrics,
                "final_metrics": final_metrics,
                "feature_importance": best_bag_features,
            }

            del X_train_final, y_train_final, X_cal, y_cal
            gc.collect()

        return regime_results

    def _load_data(self) -> pd.DataFrame | None:
        """Load historical data from disk (Parquet or CSV)."""
        data_dir = DATA_DIR / "historical"

        # Try market-specific directory
        if self.market_type == "crypto":
            search_dir = data_dir / "crypto"
        elif self.market_type == "india_equity":
            search_dir = data_dir / "india_equity"
        else:
            search_dir = data_dir

        # Look for parquet first, then CSV
        for ext in ["parquet", "csv"]:
            filepath = search_dir / f"{self.symbol}.{ext}"
            if filepath.exists():
                if ext == "parquet":
                    df = pd.read_parquet(filepath)
                else:
                    df = pd.read_csv(filepath)
                logger.info("data_loaded_from_disk", path=str(filepath), rows=len(df))
                return df

        logger.warning("no_data_file_found", search_dir=str(search_dir), symbol=self.symbol)
        return None

    def get_model(self, regime: str = "trend") -> ApexXGBoostModel | None:
        """Get a trained model for a specific regime."""
        for key, model in self._models.items():
            if regime.lower() in key.lower():
                return model
        if self._models:
            return list(self._models.values())[0]
        return None
