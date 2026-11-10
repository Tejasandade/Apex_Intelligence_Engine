"""
Apex Intelligence Engine — Hot-Swap Model Retraining Scheduler
===============================================================
Runs model retraining in a background thread and atomically swaps in
new models without stopping the trading bot.

Key design:
  - Background Thread (not asyncio) so retraining never blocks the event loop
  - Accumulates live candles + feature data for the retrain dataset
  - On completion, calls swap_callback(new_models) on the main thread
  - Thread-safe model swap via threading.Lock

Usage:
    scheduler = RetrainScheduler(
        symbol="BTCUSDT",
        retrain_every_n_candles=2000,
        swap_callback=lambda models: signal_engine.hot_swap_models(models),
    )

    # On each candle:
    scheduler.on_candle(candle, features)
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("apex.models.retrain_scheduler")

_STAGING_DIR = Path("data/models/staging")


class RetrainScheduler:
    """
    Background model retrainer with hot-swap capability.

    Thread model:
    - Main asyncio event loop calls on_candle() — O(1), never blocks
    - Background ThreadPoolExecutor runs retraining — heavy, isolated
    - Lock-protected swap: main loop picks up new models atomically
    """

    def __init__(
        self,
        symbol: str,
        retrain_every_n_candles: int = 2000,
        staging_dir: Path | str = _STAGING_DIR,
        swap_callback: Optional[Callable[[dict], None]] = None,
        max_buffer_size: int = 5000,
    ):
        self.symbol = symbol
        self.retrain_every_n_candles = retrain_every_n_candles
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self._swap_callback = swap_callback

        # Candle accumulator for retraining dataset
        self._buffer: deque[dict] = deque(maxlen=max_buffer_size)
        self._feature_buffer: deque[dict] = deque(maxlen=max_buffer_size)

        # State
        self._candle_count = 0
        self._last_retrain_candle = 0
        self._last_retrain_time: float = 0.0
        self._is_retraining = False
        self._retrain_count = 0
        self._last_accuracy_delta: float = 0.0

        # Thread safety
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apex_retrain")

        # Pending model swap (set by background thread, consumed by main thread)
        self._pending_models: dict | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def on_candle(self, candle: dict, features: Optional[pd.DataFrame] = None) -> bool:
        """
        Call on every closed candle. Returns True if a retrain was triggered.
        O(1) — never blocks the event loop.
        """
        self._candle_count += 1

        # Accumulate for dataset
        self._buffer.append(candle)
        if features is not None and not features.empty:
            self._feature_buffer.append(features.iloc[-1].to_dict())

        # Check if we should trigger a retrain
        candles_since = self._candle_count - self._last_retrain_candle
        if (
            candles_since >= self.retrain_every_n_candles
            and not self._is_retraining
            and len(self._buffer) >= 500   # need enough data
        ):
            self._trigger_retrain()
            return True

        return False

    def check_pending_swap(self) -> bool:
        """
        Check if retraining produced new models ready to swap.
        Call this in the main loop (e.g. after each candle).
        Returns True if models were swapped.
        """
        models = None
        with self._lock:
            if self._pending_models is not None:
                models = self._pending_models
                self._pending_models = None

        if models and self._swap_callback:
            try:
                self._swap_callback(models)
                logger.info(
                    f"[RetrainScheduler] Hot-swapped {len(models)} models "
                    f"(retrain #{self._retrain_count})"
                )
                return True
            except Exception as exc:
                logger.warning(f"[RetrainScheduler] swap_callback error: {exc}")

        return False

    @property
    def is_retraining(self) -> bool:
        return self._is_retraining

    @property
    def last_retrain_time(self) -> float:
        return self._last_retrain_time

    @property
    def next_retrain_in(self) -> int:
        """Candles until next scheduled retrain."""
        candles_since = self._candle_count - self._last_retrain_candle
        return max(0, self.retrain_every_n_candles - candles_since)

    @property
    def last_accuracy_delta(self) -> float:
        """Accuracy improvement of last retrain (+ve = new model better)."""
        return self._last_accuracy_delta

    def get_status(self) -> dict:
        return {
            "is_retraining": self._is_retraining,
            "candles_accumulated": self._candle_count,
            "next_retrain_in": self.next_retrain_in,
            "retrain_count": self._retrain_count,
            "last_retrain_time": self._last_retrain_time,
            "last_accuracy_delta": self._last_accuracy_delta,
        }

    def shutdown(self) -> None:
        """Clean shutdown — wait for any running retrain to finish."""
        self._executor.shutdown(wait=True)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _trigger_retrain(self) -> None:
        """Submit background retrain job. Non-blocking."""
        self._is_retraining = True
        self._last_retrain_candle = self._candle_count

        # Snapshot current buffer
        candle_snapshot = list(self._buffer)
        feature_snapshot = list(self._feature_buffer)

        logger.info(
            f"[RetrainScheduler] Triggering background retrain "
            f"(#{self._retrain_count + 1}) with {len(candle_snapshot)} candles"
        )

        self._executor.submit(
            self._retrain_in_background,
            candle_snapshot,
            feature_snapshot,
        )

    def _retrain_in_background(
        self,
        candles: list[dict],
        features: list[dict],
    ) -> None:
        """
        Runs in a background thread. Trains new models and sets pending swap.
        """
        try:
            start = time.time()
            new_models = self._run_training(candles, features)
            elapsed = time.time() - start

            if new_models:
                with self._lock:
                    self._pending_models = new_models
                self._retrain_count += 1
                self._last_retrain_time = time.time()
                logger.info(
                    f"[RetrainScheduler] Retrain complete in {elapsed:.1f}s — "
                    f"{len(new_models)} models staged for swap"
                )
            else:
                logger.warning("[RetrainScheduler] Retrain produced no models")

        except Exception as exc:
            logger.error(f"[RetrainScheduler] Background retrain failed: {exc}", exc_info=True)
        finally:
            self._is_retraining = False

    def _run_training(
        self,
        candles: list[dict],
        features: list[dict],
    ) -> dict | None:
        """
        Actual training logic. Returns dict of new models or None on failure.
        Attempts to use the existing trainer infrastructure.
        """
        try:
            from src.features.labeling.apex_labeler import ApexTripleBarrierLabeler
            from src.models.xgboost_model import ApexXGBoostModel
            from src.features.store import FeatureStore

            df = pd.DataFrame(candles)
            if df.empty or len(df) < 200:
                return None

            for col in ["open", "high", "low", "close", "volume"]:
                if col not in df.columns:
                    return None
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close"])

            # Build features
            store = FeatureStore("crypto")
            feat_df = store.build_features(df)
            if feat_df.empty:
                return None

            # Label
            labeler = ApexTripleBarrierLabeler(tp_multiple=3.0, sl_multiple=1.0, time_stop_bars=48)
            labeled = labeler.label_series(df)
            feat_df["label_long"]  = labeled["label_long"].values
            feat_df["label_short"] = labeled["label_short"].values

            feature_cols = store.feature_columns
            new_models = {}

            for direction in ["LONG", "SHORT"]:
                label_col = f"label_{direction.lower()}"
                valid = feat_df[feature_cols + [label_col]].dropna()
                if len(valid) < 100:
                    continue

                X = valid[feature_cols]
                y = valid[label_col].astype(int)

                # Simple 80/20 split (time-based, no shuffle)
                split = int(len(X) * 0.8)
                X_train, X_val = X.iloc[:split], X.iloc[split:]
                y_train, y_val = y.iloc[:split], y.iloc[split:]

                name = f"{self.symbol.lower()}_crypto_live_retrain_{direction.lower()}"
                model = ApexXGBoostModel(name=name, feature_columns=feature_cols)
                try:
                    model.train(X_train, y_train, X_val, y_val)
                    new_models[name] = model
                except Exception as exc:
                    logger.warning(f"[RetrainScheduler] training failed for {direction}: {exc}")

            if new_models:
                # Save to staging dir
                for name, model in new_models.items():
                    try:
                        model.save(path=self.staging_dir)
                    except Exception:
                        pass

            return new_models if new_models else None

        except Exception as exc:
            logger.error(f"[RetrainScheduler] _run_training error: {exc}", exc_info=True)
            return None

