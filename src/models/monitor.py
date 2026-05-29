"""
Apex Intelligence Engine V5 — Model Performance Monitor
==========================================================
Continuously tracks live model accuracy and detects decay.

WHY THIS MATTERS:
Markets change. A model trained on trending BTC in May may fail when
the market starts ranging in June. Without monitoring, you keep trading
a broken model until your account tells you (too late).

Detection methods:
1. BRIER SCORE: Tracks calibration quality. If Brier rises, model is miscalibrated.
2. ACCURACY DRIFT: Rolling window accuracy drops below threshold.
3. SIGNAL QUALITY: Profitable signals ratio drops.
4. REGIME SHIFT: If the market regime distribution changes significantly.

When decay is detected:
- Emit ModelDecayAlert event
- Trigger walk-forward retrain
- Optionally reduce position sizes or halt trading
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.core.events import event_bus, ModelDecayAlert
from src.core.logging import get_logger

logger = get_logger("apex.models.monitor")


@dataclass
class PredictionRecord:
    """A single prediction with its eventual outcome."""

    timestamp: float  # Unix seconds
    model_name: str
    probability: float
    direction: str  # BUY/SELL
    actual_outcome: int | None = None  # 1=profitable, 0=not, None=pending
    pnl: float | None = None  # Resolved PnL
    resolved_at: float | None = None


@dataclass
class DecayAlert:
    """Alert when model performance drops."""

    model_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str  # "warning" or "critical"
    message: str
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """
    Live model performance monitor.

    Tracks predictions vs outcomes and detects model decay through
    multiple statistical tests.
    """

    def __init__(
        self,
        model_name: str = "default",
        # Accuracy thresholds
        min_accuracy: float = 0.45,
        min_accuracy_critical: float = 0.35,
        # Brier score thresholds
        max_brier: float = 0.30,
        max_brier_critical: float = 0.40,
        # Signal quality thresholds
        min_signal_quality: float = 0.40,  # Min % of profitable signals
        # Window sizes
        accuracy_window: int = 100,  # Last N predictions for accuracy
        brier_window: int = 50,
        # Consecutive loss detection
        max_consecutive_losses: int = 8,
        # Regime tracking
        regime_window: int = 200,
    ):
        self.model_name = model_name
        self.min_accuracy = min_accuracy
        self.min_accuracy_critical = min_accuracy_critical
        self.max_brier = max_brier
        self.max_brier_critical = max_brier_critical
        self.min_signal_quality = min_signal_quality
        self.accuracy_window = accuracy_window
        self.brier_window = brier_window
        self.max_consecutive_losses = max_consecutive_losses
        self.regime_window = regime_window

        # Prediction tracking
        self._pending: dict[str, PredictionRecord] = {}  # key = unique ID
        self._resolved: deque[PredictionRecord] = deque(maxlen=1000)
        self._prediction_count = 0

        # Rolling metrics
        self._accuracy_buffer: deque[bool] = deque(maxlen=accuracy_window)
        self._brier_buffer: deque[float] = deque(maxlen=brier_window)
        self._pnl_buffer: deque[float] = deque(maxlen=accuracy_window)
        self._consecutive_losses = 0

        # Regime tracking
        self._regime_history: deque[str] = deque(maxlen=regime_window)

        # Alerts
        self._active_alerts: list[DecayAlert] = []
        self._alert_count = 0

    def record_prediction(
        self,
        prediction_id: str,
        probability: float,
        direction: str,
        model_name: str | None = None,
    ) -> None:
        """Record a new prediction (outcome pending)."""
        record = PredictionRecord(
            timestamp=time.time(),
            model_name=model_name or self.model_name,
            probability=probability,
            direction=direction,
        )
        self._pending[prediction_id] = record
        self._prediction_count += 1

    def resolve_prediction(
        self,
        prediction_id: str,
        profitable: bool,
        pnl: float = 0.0,
    ) -> list[DecayAlert]:
        """
        Resolve a pending prediction with the actual outcome.

        Args:
            prediction_id: Unique prediction identifier.
            profitable: Whether the trade was profitable.
            pnl: Actual PnL of the trade.

        Returns:
            List of any decay alerts triggered.
        """
        record = self._pending.pop(prediction_id, None)
        if record is None:
            return []

        actual = 1 if profitable else 0
        record.actual_outcome = actual
        record.pnl = pnl
        record.resolved_at = time.time()
        self._resolved.append(record)

        # Update rolling metrics
        correct = (
            (record.probability >= 0.5 and actual == 1)
            or (record.probability < 0.5 and actual == 0)
        )
        self._accuracy_buffer.append(correct)

        brier = (record.probability - actual) ** 2
        self._brier_buffer.append(brier)

        self._pnl_buffer.append(pnl)

        # Track consecutive losses
        if profitable:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Run decay detection
        alerts = self._check_decay()

        return alerts

    def record_regime(self, regime: str) -> None:
        """Track current regime for shift detection."""
        self._regime_history.append(regime)

    def _check_decay(self) -> list[DecayAlert]:
        """Run all decay detection checks."""
        alerts: list[DecayAlert] = []

        # Need minimum samples
        if len(self._accuracy_buffer) < 20:
            return alerts

        # 1. Accuracy check
        accuracy = sum(self._accuracy_buffer) / len(self._accuracy_buffer)

        if accuracy < self.min_accuracy_critical:
            alerts.append(DecayAlert(
                model_name=self.model_name,
                metric="accuracy",
                current_value=accuracy,
                threshold=self.min_accuracy_critical,
                severity="critical",
                message=f"CRITICAL: Accuracy dropped to {accuracy:.1%} "
                        f"(threshold: {self.min_accuracy_critical:.1%})",
            ))
        elif accuracy < self.min_accuracy:
            alerts.append(DecayAlert(
                model_name=self.model_name,
                metric="accuracy",
                current_value=accuracy,
                threshold=self.min_accuracy,
                severity="warning",
                message=f"WARNING: Accuracy at {accuracy:.1%} "
                        f"(threshold: {self.min_accuracy:.1%})",
            ))

        # 2. Brier score check
        if len(self._brier_buffer) >= 20:
            brier = np.mean(list(self._brier_buffer))

            if brier > self.max_brier_critical:
                alerts.append(DecayAlert(
                    model_name=self.model_name,
                    metric="brier_score",
                    current_value=brier,
                    threshold=self.max_brier_critical,
                    severity="critical",
                    message=f"CRITICAL: Brier score at {brier:.4f} "
                            f"(max: {self.max_brier_critical:.4f})",
                ))
            elif brier > self.max_brier:
                alerts.append(DecayAlert(
                    model_name=self.model_name,
                    metric="brier_score",
                    current_value=brier,
                    threshold=self.max_brier,
                    severity="warning",
                    message=f"WARNING: Brier score at {brier:.4f} "
                            f"(max: {self.max_brier:.4f})",
                ))

        # 3. Signal quality (profitable trade ratio)
        if len(self._pnl_buffer) >= 20:
            profitable_ratio = sum(1 for p in self._pnl_buffer if p > 0) / len(self._pnl_buffer)

            if profitable_ratio < self.min_signal_quality:
                alerts.append(DecayAlert(
                    model_name=self.model_name,
                    metric="signal_quality",
                    current_value=profitable_ratio,
                    threshold=self.min_signal_quality,
                    severity="warning",
                    message=f"WARNING: Only {profitable_ratio:.1%} profitable signals "
                            f"(min: {self.min_signal_quality:.1%})",
                ))

        # 4. Consecutive losses
        if self._consecutive_losses >= self.max_consecutive_losses:
            alerts.append(DecayAlert(
                model_name=self.model_name,
                metric="consecutive_losses",
                current_value=float(self._consecutive_losses),
                threshold=float(self.max_consecutive_losses),
                severity="critical",
                message=f"CRITICAL: {self._consecutive_losses} consecutive losses",
            ))

        # 5. Regime shift detection
        if len(self._regime_history) >= self.regime_window:
            alerts.extend(self._check_regime_shift())

        # Emit alerts
        for alert in alerts:
            self._active_alerts.append(alert)
            self._alert_count += 1
            logger.warning(
                "model_decay_detected",
                model=alert.model_name,
                metric=alert.metric,
                value=f"{alert.current_value:.4f}",
                threshold=f"{alert.threshold:.4f}",
                severity=alert.severity,
            )

        return alerts

    def _check_regime_shift(self) -> list[DecayAlert]:
        """Detect if the market regime distribution has shifted significantly."""
        alerts = []
        history = list(self._regime_history)
        half = len(history) // 2

        first_half = history[:half]
        second_half = history[half:]

        # Count regimes in each half
        def regime_dist(data: list[str]) -> dict[str, float]:
            counts: dict[str, int] = {}
            for r in data:
                counts[r] = counts.get(r, 0) + 1
            total = len(data)
            return {k: v / total for k, v in counts.items()}

        dist1 = regime_dist(first_half)
        dist2 = regime_dist(second_half)

        # Check if dominant regime changed
        dom1 = max(dist1, key=dist1.get) if dist1 else ""
        dom2 = max(dist2, key=dist2.get) if dist2 else ""

        if dom1 != dom2:
            alerts.append(DecayAlert(
                model_name=self.model_name,
                metric="regime_shift",
                current_value=0.0,
                threshold=0.0,
                severity="warning",
                message=f"Regime shifted from {dom1} to {dom2}. "
                        f"Consider retraining for {dom2} regime.",
            ))

        return alerts

    @property
    def current_accuracy(self) -> float:
        """Current rolling accuracy."""
        if not self._accuracy_buffer:
            return 0.5
        return sum(self._accuracy_buffer) / len(self._accuracy_buffer)

    @property
    def current_brier(self) -> float:
        """Current rolling Brier score."""
        if not self._brier_buffer:
            return 0.25
        return float(np.mean(list(self._brier_buffer)))

    @property
    def has_active_alerts(self) -> bool:
        return len(self._active_alerts) > 0

    @property
    def has_critical_alerts(self) -> bool:
        return any(a.severity == "critical" for a in self._active_alerts)

    def clear_alerts(self) -> None:
        """Clear active alerts (after retraining, for example)."""
        self._active_alerts.clear()

    def get_stats(self) -> dict[str, Any]:
        """Return comprehensive monitoring stats."""
        return {
            "model_name": self.model_name,
            "total_predictions": self._prediction_count,
            "resolved": len(self._resolved),
            "pending": len(self._pending),
            "current_accuracy": self.current_accuracy,
            "current_brier": self.current_brier,
            "consecutive_losses": self._consecutive_losses,
            "active_alerts": len(self._active_alerts),
            "critical_alerts": self.has_critical_alerts,
            "total_alerts": self._alert_count,
            "total_pnl": sum(self._pnl_buffer) if self._pnl_buffer else 0.0,
        }
