"""
Apex Intelligence Engine V5 — Domain Exception Hierarchy
=========================================================
Every module raises specific exceptions from this hierarchy.
No bare Exception() or generic ValueError — always a domain-specific error.
"""

from __future__ import annotations


class ApexError(Exception):
    """Base exception for all Apex engine errors."""

    pass


# ── Data Layer Exceptions ────────────────────────────────────────────────────
class DataError(ApexError):
    """Base exception for data-related errors."""

    pass


class DataValidationError(DataError):
    """Raised when incoming market data fails validation checks."""

    def __init__(self, symbol: str, reason: str, raw_value: object = None):
        self.symbol = symbol
        self.reason = reason
        self.raw_value = raw_value
        super().__init__(f"[{symbol}] Data validation failed: {reason}")


class InsufficientDataError(DataError):
    """Raised when there isn't enough data to compute features or run inference."""

    def __init__(self, symbol: str, required: int, available: int):
        self.symbol = symbol
        self.required = required
        self.available = available
        super().__init__(
            f"[{symbol}] Insufficient data: need {required}, have {available}"
        )


class DataProviderError(DataError):
    """Raised when a data provider (Binance, Dhan, etc.) fails."""

    def __init__(self, provider: str, reason: str):
        self.provider = provider
        self.reason = reason
        super().__init__(f"[{provider}] Data provider error: {reason}")


class StaleDataError(DataError):
    """Raised when data is too old to be trusted."""

    def __init__(self, symbol: str, age_seconds: float, max_age_seconds: float):
        self.symbol = symbol
        self.age_seconds = age_seconds
        self.max_age_seconds = max_age_seconds
        super().__init__(
            f"[{symbol}] Stale data: {age_seconds:.1f}s old "
            f"(max allowed: {max_age_seconds:.1f}s)"
        )


# ── Model Layer Exceptions ──────────────────────────────────────────────────
class ModelError(ApexError):
    """Base exception for model-related errors."""

    pass


class ModelNotTrainedError(ModelError):
    """Raised when inference is attempted on an untrained model."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' is not trained. Train before inference.")


class ModelDecayDetected(ModelError):
    """Raised when CUSUM or accuracy monitoring detects model degradation."""

    def __init__(self, model_name: str, metric: str, value: float, threshold: float):
        self.model_name = model_name
        self.metric = metric
        self.value = value
        self.threshold = threshold
        super().__init__(
            f"Model '{model_name}' decay detected: {metric}={value:.4f} "
            f"(threshold: {threshold:.4f})"
        )


class CalibrationError(ModelError):
    """Raised when probability calibration fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Calibration error: {reason}")


class QualityGateFailure(ModelError):
    """Raised when a model fails to meet the deployment quality gates."""

    def __init__(self, model_name: str, failures: dict[str, tuple[float, float]]):
        self.model_name = model_name
        self.failures = failures  # metric_name -> (actual_value, required_value)
        details = ", ".join(
            f"{k}: {v[0]:.4f} (required: {v[1]:.4f})" for k, v in failures.items()
        )
        super().__init__(f"Model '{model_name}' failed quality gates: {details}")


# ── Execution Layer Exceptions ──────────────────────────────────────────────
class ExecutionError(ApexError):
    """Base exception for execution/trading errors."""

    pass


class BrokerConnectionError(ExecutionError):
    """Raised when connection to broker API fails."""

    def __init__(self, broker: str, reason: str):
        self.broker = broker
        self.reason = reason
        super().__init__(f"[{broker}] Connection error: {reason}")


class OrderRejectedError(ExecutionError):
    """Raised when the broker rejects an order."""

    def __init__(self, broker: str, symbol: str, reason: str):
        self.broker = broker
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"[{broker}] Order rejected for {symbol}: {reason}")


class PositionReconciliationError(ExecutionError):
    """Raised when broker positions don't match local state."""

    def __init__(self, symbol: str, broker_qty: float, local_qty: float):
        self.symbol = symbol
        self.broker_qty = broker_qty
        self.local_qty = local_qty
        super().__init__(
            f"[{symbol}] Position mismatch: broker={broker_qty}, local={local_qty}"
        )


# ── Risk Layer Exceptions ───────────────────────────────────────────────────
class RiskError(ApexError):
    """Base exception for risk management errors."""

    pass


class KillSwitchTripped(RiskError):
    """Raised when the kill switch activates. This is a HARD STOP."""

    def __init__(self, reason: str, details: dict | None = None):
        self.reason = reason
        self.details = details or {}
        super().__init__(f"KILL SWITCH ACTIVATED: {reason}")


class ExposureLimitBreached(RiskError):
    """Raised when a trade would exceed exposure limits."""

    def __init__(self, symbol: str, current_exposure: float, limit: float):
        self.symbol = symbol
        self.current_exposure = current_exposure
        self.limit = limit
        super().__init__(
            f"[{symbol}] Exposure limit breached: "
            f"current={current_exposure:.2%}, limit={limit:.2%}"
        )


# ── Configuration Exceptions ────────────────────────────────────────────────
class ConfigurationError(ApexError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(f"Configuration error for '{key}': {reason}")
