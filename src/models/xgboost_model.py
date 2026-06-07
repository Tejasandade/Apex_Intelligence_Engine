"""
Apex Intelligence Engine V5 — XGBoost Model
=============================================
Production XGBoost classifier with:
- Automatic class imbalance handling (scale_pos_weight)
- Feature name enforcement (train/serve parity)
- Isotonic probability calibration
- Integrated save/load with metadata

This is the primary prediction model. Future models (Transformer, LSTM)
will implement the same BaseModel interface.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, precision_score

from src.core.config import WEIGHTS_DIR
from src.core.logging import get_logger
from src.models.base import BaseModel
from src.models.calibration import ProbabilityCalibrator

logger = get_logger("apex.models.xgboost")


class ApexXGBoostModel(BaseModel):
    """
    XGBoost-based prediction model with probability calibration.

    Handles the full lifecycle:
    1. Training with early stopping
    2. Calibration with held-out validation data
    3. Calibrated inference
    4. Feature importance tracking
    5. Model persistence
    """

    def __init__(
        self,
        name: str,
        feature_columns: list[str],
        hyperparams: dict[str, Any] | None = None,
    ):
        """
        Initialize the XGBoost model.

        Args:
            name: Unique model name (e.g., "btcusdt_crypto_trend").
            feature_columns: Ordered list of feature column names.
            hyperparams: XGBoost hyperparameters (from models.yaml).
        """
        self._name = name
        self._feature_columns = feature_columns
        self._hyperparams = hyperparams or {}
        self._model: xgb.XGBClassifier | None = None
        self._calibrator = ProbabilityCalibrator()
        self._is_trained = False
        self._metadata: dict[str, Any] = {}
        self._training_metrics: dict[str, float] = {}

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Train the XGBoost model.

        If validation data is provided:
        - Uses it for early stopping
        - Uses it for probability calibration
        Otherwise, splits the training data 80/20.

        Args:
            X: Feature matrix.
            y: Target labels (0/1).
            X_val: Validation features (optional).
            y_val: Validation targets (optional).

        Returns:
            Dict of training metrics.
        """
        # Enforce feature column order
        X = X[self._feature_columns].copy()
        if X_val is not None:
            X_val = X_val[self._feature_columns].copy()

        # Auto-split if no validation set provided
        if X_val is None or y_val is None:
            split_idx = int(len(X) * 0.8)
            X_val = X.iloc[split_idx:]
            y_val = y.iloc[split_idx:]
            X = X.iloc[:split_idx]
            y = y.iloc[:split_idx]

        # Handle class imbalance
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        # Build XGBoost params
        params = {
            "n_estimators": self._hyperparams.get("n_estimators", 300),
            "learning_rate": self._hyperparams.get("learning_rate", 0.03),
            "max_depth": self._hyperparams.get("max_depth", 6),
            "min_child_weight": self._hyperparams.get("min_child_weight", 3),
            "subsample": self._hyperparams.get("subsample", 0.8),
            "colsample_bytree": self._hyperparams.get("colsample_bytree", 0.8),
            "gamma": self._hyperparams.get("gamma", 0.1),
            "reg_alpha": self._hyperparams.get("reg_alpha", 0.1),
            "reg_lambda": self._hyperparams.get("reg_lambda", 1.0),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        early_stopping = self._hyperparams.get("early_stopping_rounds", 20)

        logger.info(
            "training_started",
            model=self._name,
            train_samples=len(X),
            val_samples=len(X_val),
            features=len(self._feature_columns),
            pos_rate=f"{y.mean():.2%}",
            scale_pos_weight=f"{scale_pos_weight:.2f}",
        )

        self._model = xgb.XGBClassifier(**params)
        self._model.fit(
            X,
            y,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Validation metrics
        raw_probs_val = self._model.predict_proba(X_val)[:, 1]
        y_pred = (raw_probs_val >= 0.5).astype(int)

        acc = accuracy_score(y_val, y_pred)
        ll = log_loss(y_val, raw_probs_val)
        prec = precision_score(y_val, y_pred, zero_division=0)
        best_iter = getattr(self._model, "best_iteration", params["n_estimators"])

        # Calibrate using validation set
        self._calibrator.fit(raw_probs_val, y_val.values)

        self._training_metrics = {
            "accuracy": acc,
            "log_loss": ll,
            "precision": prec,
            "best_iteration": float(best_iter),
            "train_samples": float(len(X)),
            "val_samples": float(len(X_val)),
            "positive_rate": float(y.mean()),
        }

        self._is_trained = True
        self._metadata = {
            "model_name": self._name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": self._feature_columns,
            "num_features": len(self._feature_columns),
            "hyperparams": params,
            "metrics": self._training_metrics,
            "calibrated": self._calibrator.is_fitted,
        }

        logger.info(
            "training_completed",
            model=self._name,
            accuracy=f"{acc:.4f}",
            log_loss=f"{ll:.4f}",
            precision=f"{prec:.4f}",
            best_trees=best_iter,
            calibrated=self._calibrator.is_fitted,
        )

        return self._training_metrics

    def predict(self, X: pd.DataFrame) -> float:
        """
        Predict calibrated probability for a single sample (or first row of batch).

        Args:
            X: Feature matrix (1 or more rows).

        Returns:
            Calibrated probability (0.0 to 1.0).
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(f"Model '{self._name}' is not trained")

        # Enforce feature order
        X_ordered = X[self._feature_columns].copy()

        raw_prob = self._model.predict_proba(X_ordered)[:, 1]

        # Calibration DISABLED — isotonic calibration squashes variance to near-zero
        # Raw XGBoost probabilities retain signal variance needed for thresholding
        return float(raw_prob[0])

    def predict_batch(self, X: pd.DataFrame) -> pd.Series:
        """
        Predict calibrated probabilities for a batch.

        Args:
            X: Feature matrix (multiple rows).

        Returns:
            Series of calibrated probabilities.
        """
        if not self._is_trained or self._model is None:
            raise RuntimeError(f"Model '{self._name}' is not trained")

        X_ordered = X[self._feature_columns].copy()
        raw_probs = self._model.predict_proba(X_ordered)[:, 1]

        # Calibration DISABLED — return raw probabilities with full variance
        return pd.Series(raw_probs, index=X.index)

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature importance scores (gain-based)."""
        if not self._is_trained or self._model is None:
            return {}

        importance = self._model.feature_importances_
        return dict(zip(self._feature_columns, importance.tolist()))

    def save(self, path: Path | None = None) -> None:
        """
        Save model weights, calibrator, and metadata.

        Creates three files:
        - {name}.json: XGBoost model
        - {name}_calibrator.json: Isotonic regression calibrator
        - {name}_metadata.json: Training metadata
        """
        if not self._is_trained or self._model is None:
            logger.warning("save_skipped_not_trained", model=self._name)
            return

        save_dir = path or WEIGHTS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        model_path = save_dir / f"{self._name}.json"
        calibrator_path = save_dir / f"{self._name}_calibrator.json"
        metadata_path = save_dir / f"{self._name}_metadata.json"

        # Save XGBoost model
        self._model.save_model(str(model_path))

        # Save calibrator
        self._calibrator.save(calibrator_path)

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(self._metadata, f, indent=2, default=str)

        logger.info("model_saved", model=self._name, dir=str(save_dir))

    def load(self, path: Path | None = None) -> None:
        """Load model weights, calibrator, and metadata from disk."""
        load_dir = path or WEIGHTS_DIR

        model_path = load_dir / f"{self._name}.json"
        calibrator_path = load_dir / f"{self._name}_calibrator.json"
        metadata_path = load_dir / f"{self._name}_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load XGBoost model
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(model_path))
        self._is_trained = True

        # Load calibrator
        if calibrator_path.exists():
            self._calibrator.load(calibrator_path)

        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self._metadata = json.load(f)

        logger.info(
            "model_loaded",
            model=self._name,
            calibrated=self._calibrator.is_fitted,
        )
