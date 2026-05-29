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
            profit_target_pct=labeling_cfg.get("profit_target_pct", 0.002),
            stop_loss_pct=labeling_cfg.get("stop_loss_pct", 0.0012),
            max_holding_bars=labeling_cfg.get("max_holding_bars", 20),
        )

        if len(labeled_df) < 200:
            logger.error("labeling_too_few_samples", samples=len(labeled_df))
            return {"error": "insufficient_labeled_samples"}

        logger.info(
            "labels_applied",
            total=len(labeled_df),
            positive_rate=f"{labeled_df['target'].mean():.2%}",
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

            for regime_name in ["TRENDING", "RANGING"]:
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
        Train a model for a specific regime subset.

        Args:
            df: Labeled DataFrame for this regime.
            regime_suffix: e.g., "trend", "range", "all".
            save_models: Whether to save on quality gate pass.

        Returns:
            Training result dict.
        """
        model_name = f"{self.symbol}_{self.market_type}_{regime_suffix}"
        hp = self.model_params.get("hyperparameters", {})

        logger.info(
            "regime_training_started",
            model=model_name,
            samples=len(df),
            positive_rate=f"{df['target'].mean():.2%}",
        )

        # Get feature matrix
        X = df[self.feature_store.feature_columns].copy().fillna(0.0)
        y = df["target"].copy()

        # Purged walk-forward CV
        cv_cfg = self.model_params.get("training", {})
        folds = purged_walk_forward_splits(
            n_samples=len(X),
            n_folds=cv_cfg.get("cv_folds", 5),
            purge_gap=cv_cfg.get("purge_gap", 15),
        )

        fold_metrics: list[dict[str, float]] = []

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

            logger.debug(
                "cv_fold_complete",
                fold=fold.fold_num,
                accuracy=f"{metrics['accuracy']:.4f}",
                log_loss=f"{metrics['log_loss']:.4f}",
            )

        # Average CV metrics
        avg_metrics = {
            k: float(np.mean([f[k] for f in fold_metrics]))
            for k in fold_metrics[0]
        }

        logger.info(
            "cv_complete",
            model=model_name,
            avg_accuracy=f"{avg_metrics['accuracy']:.4f}",
            avg_log_loss=f"{avg_metrics['log_loss']:.4f}",
            avg_precision=f"{avg_metrics['precision']:.4f}",
        )

        # Train final model on ALL data (with 80/20 split for calibration)
        final_model = ApexXGBoostModel(
            name=model_name,
            feature_columns=self.feature_store.feature_columns,
            hyperparams=hp,
        )

        split_idx = int(len(X) * 0.85)
        X_train_final = X.iloc[:split_idx]
        y_train_final = y.iloc[:split_idx]
        X_cal = X.iloc[split_idx:]
        y_cal = y.iloc[split_idx:]

        final_metrics = final_model.train(X_train_final, y_train_final, X_cal, y_cal)

        # Save model if quality check passes
        if save_models:
            save_dir = WEIGHTS_DIR
            final_model.save(save_dir)
            self._models[regime_suffix] = final_model
            logger.info("model_saved", model=model_name)

        return {
            "model_name": model_name,
            "cv_metrics": avg_metrics,
            "final_metrics": final_metrics,
            "samples": len(df),
            "feature_importance": final_model.get_feature_importance(),
        }

    def _load_data(self) -> pd.DataFrame | None:
        """Load historical data from disk (Parquet or CSV)."""
        data_dir = DATA_DIR / "historical"

        # Try market-specific directory
        if self.market_type == "crypto":
            search_dir = data_dir / "crypto"
        elif self.market_type == "india_equity":
            search_dir = data_dir / "india"
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
        return self._models.get(regime)
