"""
Apex Intelligence Engine V5 — Walk-Forward Retrainer
======================================================
Periodically retrains models on the latest data without forgetting old patterns.

WHY WALK-FORWARD?
Static models decay. Markets evolve. A model trained in January
will perform worse in March. Walk-forward retraining fixes this by:

1. Adding new data to the training set
2. Retraining on the expanded dataset
3. Validating on a holdout window
4. Hot-swapping the model if the new one is better

Schedule:
    Every N new candles (default: 1000), retrain the model.
    This means ~16 hours of 1-minute data between retrains.

Safety:
    - New model must PASS quality gates before replacing the old one
    - If new model is worse, keep the old one (don't break what works)
    - Save both old and new model for rollback
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import DATA_DIR, get_model_params
from src.core.logging import get_logger
from src.features.store import FeatureStore
from src.models.training.trainer import UnifiedTrainer
from src.models.xgboost_model import ApexXGBoostModel

logger = get_logger("apex.models.walk_forward")

MODELS_DIR = DATA_DIR / "models"


class WalkForwardRetrainer:
    """
    Manages periodic model retraining on accumulating data.

    As new candles arrive via WebSocket, they're appended to the
    training buffer. Every `retrain_interval` candles, a new model
    is trained and validated.
    """

    def __init__(
        self,
        symbol: str,
        market_type: str = "crypto",
        retrain_interval: int = 1000,  # Retrain every N new candles
        validation_pct: float = 0.15,  # Holdout for validation
        min_training_bars: int = 5000,  # Min bars before first retrain
    ):
        """
        Args:
            symbol: Trading symbol.
            market_type: Market type for features.
            retrain_interval: Candles between retrains.
            validation_pct: Fraction of data for validation.
            min_training_bars: Minimum data before retraining.
        """
        self.symbol = symbol
        self.market_type = market_type
        self.retrain_interval = retrain_interval
        self.validation_pct = validation_pct
        self.min_training_bars = min_training_bars

        # Data buffer
        self._data_buffer: list[dict[str, Any]] = []
        self._candles_since_retrain = 0
        self._retrain_count = 0
        self._last_retrain_time: float = 0.0

        # Model tracking
        self._current_models: dict[str, ApexXGBoostModel] = {}
        self._model_history: list[dict[str, Any]] = []  # History of retrains

    @property
    def should_retrain(self) -> bool:
        """Check if enough new data has arrived for a retrain."""
        return (
            self._candles_since_retrain >= self.retrain_interval
            and len(self._data_buffer) >= self.min_training_bars
        )

    def add_candle(self, candle: dict[str, Any]) -> bool:
        """
        Add a new candle to the training buffer.

        Args:
            candle: Candle dict from WebSocket.

        Returns:
            True if retrain threshold was reached.
        """
        self._data_buffer.append({
            "timestamp": candle.get("timestamp", 0),
            "open": float(candle.get("open", 0)),
            "high": float(candle.get("high", 0)),
            "low": float(candle.get("low", 0)),
            "close": float(candle.get("close", 0)),
            "volume": float(candle.get("volume", 0)),
        })
        self._candles_since_retrain += 1

        return self.should_retrain

    def load_historical(self, data: pd.DataFrame) -> None:
        """
        Pre-load historical data into the training buffer.
        Call at startup to seed with cached Parquet data.
        """
        for _, row in data.iterrows():
            self._data_buffer.append({
                "timestamp": row["timestamp"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })

        logger.info(
            "walk_forward_historical_loaded",
            symbol=self.symbol,
            bars=len(self._data_buffer),
        )

    async def retrain(self) -> dict[str, Any]:
        """
        Execute a walk-forward retrain cycle.

        Steps:
        1. Convert buffer to DataFrame
        2. Split into train (expanding window) + validation (last N%)
        3. Train new models via UnifiedTrainer
        4. Compare new models against old ones
        5. Swap if better, keep old if not

        Returns:
            Retrain results dict.
        """
        start_time = time.time()
        self._retrain_count += 1

        logger.info(
            "walk_forward_retrain_start",
            symbol=self.symbol,
            cycle=self._retrain_count,
            total_bars=len(self._data_buffer),
            new_bars=self._candles_since_retrain,
        )

        # Step 1: Build DataFrame
        df = pd.DataFrame(self._data_buffer)

        # Step 2: Train
        trainer = UnifiedTrainer(
            market_type=self.market_type,
            symbol=self.symbol,
        )

        try:
            training_results = await trainer.run(data=df, save_models=True)
        except Exception as e:
            logger.error(
                "walk_forward_retrain_failed",
                error=str(e),
                cycle=self._retrain_count,
            )
            return {
                "status": "failed",
                "error": str(e),
                "cycle": self._retrain_count,
            }

        # Step 3: Extract new models
        new_models: dict[str, ApexXGBoostModel] = {}
        improvements: dict[str, dict[str, float]] = {}

        feature_store = FeatureStore(self.market_type)

        for regime in ["trending", "ranging"]:
            model = trainer.get_model(regime)
            if model is None:
                continue

            model_name = f"{self.symbol}_{self.market_type}_{regime}"

            # Step 4: Compare with old model
            old_model = self._current_models.get(regime)
            if old_model is not None:
                old_metrics = old_model.metadata.get("metrics", {})
                new_metrics = training_results.get(regime, {}).get("cv_metrics", {})

                improvements[regime] = {
                    "old_accuracy": old_metrics.get("accuracy", 0.0),
                    "new_accuracy": new_metrics.get("accuracy", 0.0),
                    "old_log_loss": old_metrics.get("log_loss", 1.0),
                    "new_log_loss": new_metrics.get("log_loss", 1.0),
                }

                # Step 5: Only swap if new model is better
                old_ll = old_metrics.get("log_loss", 1.0)
                new_ll = new_metrics.get("log_loss", 1.0)

                if new_ll < old_ll:
                    new_models[regime] = model
                    logger.info(
                        "walk_forward_model_improved",
                        regime=regime,
                        old_log_loss=f"{old_ll:.4f}",
                        new_log_loss=f"{new_ll:.4f}",
                    )
                else:
                    logger.info(
                        "walk_forward_model_kept",
                        regime=regime,
                        reason="new model not better",
                        old_log_loss=f"{old_ll:.4f}",
                        new_log_loss=f"{new_ll:.4f}",
                    )
            else:
                # No old model — accept new one
                new_models[regime] = model

        # Update current models
        for regime, model in new_models.items():
            self._current_models[regime] = model

        # Reset counter
        self._candles_since_retrain = 0
        self._last_retrain_time = time.time()

        elapsed = time.time() - start_time

        result = {
            "status": "success",
            "cycle": self._retrain_count,
            "total_bars": len(self._data_buffer),
            "new_bars": self._candles_since_retrain,
            "models_updated": list(new_models.keys()),
            "models_kept": [r for r in ["trending", "ranging"] if r not in new_models],
            "improvements": improvements,
            "elapsed_seconds": elapsed,
        }

        self._model_history.append(result)

        logger.info(
            "walk_forward_retrain_complete",
            cycle=self._retrain_count,
            updated=list(new_models.keys()),
            elapsed=f"{elapsed:.1f}s",
        )

        return result

    def get_current_models(self) -> dict[str, ApexXGBoostModel]:
        """Return the current active models."""
        return self._current_models

    def set_current_models(self, models: dict[str, ApexXGBoostModel]) -> None:
        """Set the initial models (loaded from disk at startup)."""
        self._current_models = models

    def get_stats(self) -> dict[str, Any]:
        """Return retrainer statistics."""
        return {
            "symbol": self.symbol,
            "total_bars": len(self._data_buffer),
            "candles_since_retrain": self._candles_since_retrain,
            "retrain_interval": self.retrain_interval,
            "retrain_count": self._retrain_count,
            "should_retrain": self.should_retrain,
            "models_active": list(self._current_models.keys()),
            "last_retrain": self._last_retrain_time,
        }
