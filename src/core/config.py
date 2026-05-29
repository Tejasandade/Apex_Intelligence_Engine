"""
Apex Intelligence Engine V5 — Central Configuration
=====================================================
Single source of truth for ALL configuration.
Loads .env secrets and YAML market/risk/model configs into typed dataclasses.

Usage:
    from src.core.config import settings, market_config, risk_config, model_config
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT_DIR / "configs"
DATA_DIR = ROOT_DIR / "data"
WEIGHTS_DIR = ROOT_DIR / "data" / "models"

# Load .env
load_dotenv(ROOT_DIR / ".env", override=True)


# ── Secrets (from .env) ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Secrets:
    """API keys and credentials loaded from environment variables."""

    # Binance
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_use_testnet: bool = True

    # Dhan (to be added when account is ready)
    dhan_access_token: str = ""
    dhan_client_id: str = ""

    # Angel One (legacy, kept for reference)
    angel_api_key: str = ""
    angel_totp_key: str = ""
    angel_client_id: str = ""
    angel_pin: str = ""

    # News APIs
    cryptocompare_api_key: str = ""
    news_api_key: str = ""

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    timescale_url: str = "postgres://user:password@localhost:5432/apex"

    # Telegram alerting (Phase 5)
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""


def _load_secrets() -> Secrets:
    """Load secrets from environment variables."""
    return Secrets(
        binance_api_key=os.getenv("BINANCE_API_KEY", ""),
        binance_api_secret=os.getenv("BINANCE_API_SECRET", ""),
        binance_use_testnet=os.getenv("BINANCE_USE_TESTNET", "True").lower()
        in {"true", "1", "yes"},
        dhan_access_token=os.getenv("DHAN_ACCESS_TOKEN", ""),
        dhan_client_id=os.getenv("DHAN_CLIENT_ID", ""),
        angel_api_key=os.getenv("ANGEL_API_KEY", ""),
        angel_totp_key=os.getenv("ANGEL_TOTP_KEY", ""),
        angel_client_id=os.getenv("ANGEL_CLIENT_ID", ""),
        angel_pin=os.getenv("ANGEL_PIN", ""),
        cryptocompare_api_key=os.getenv("CRYPTOCOMPARE_API_KEY", ""),
        news_api_key=os.getenv("NEWS_API_KEY", ""),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        timescale_url=os.getenv(
            "TIMESCALE_URL", "postgres://user:password@localhost:5432/apex"
        ),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    )


# ── YAML Config Loaders ─────────────────────────────────────────────────────
def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config file from the configs/ directory."""
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ── Market Configuration ────────────────────────────────────────────────────
@dataclass
class SessionConfig:
    """Trading session hours for session-bound markets."""

    timezone: str = "UTC"
    open: str = "00:00"
    close: str = "23:59"
    days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    pre_market: str = ""
    post_market: str = ""


@dataclass
class MarketSymbolConfig:
    """Configuration for a single tradeable symbol."""

    symbol_key: str  # e.g., "btcusdt", "banknifty"
    market_type: str  # e.g., "crypto", "india_equity", "forex"
    exchange: str
    display_name: str
    base_currency: str
    quote_currency: str
    tick_size: float
    lot_size: float
    min_notional: float
    session: SessionConfig
    commission_bps: float
    taker_commission_bps: float
    expected_slippage_bps: float
    max_slippage_bps: float
    data_source: str
    candle_intervals: list[str]
    warmup_candles: int
    features_version: str
    # Optional market-specific fields
    drop_open_bars: int = 0
    expiry_day: str = ""


def _parse_market_config(raw: dict[str, Any]) -> dict[str, MarketSymbolConfig]:
    """Parse the markets.yaml into typed MarketSymbolConfig objects."""
    symbols: dict[str, MarketSymbolConfig] = {}
    markets = raw.get("markets", {})

    for market_type, market_symbols in markets.items():
        if not isinstance(market_symbols, dict):
            continue
        for symbol_key, cfg in market_symbols.items():
            if not isinstance(cfg, dict):
                continue
            # Skip deferred markets
            if cfg.get("_status") == "deferred":
                continue

            # Parse session config
            session_raw = cfg.get("session", "24/7")
            if isinstance(session_raw, dict):
                session = SessionConfig(**session_raw)
            elif session_raw == "24/7":
                session = SessionConfig(
                    timezone="UTC",
                    open="00:00",
                    close="23:59",
                    days=[0, 1, 2, 3, 4, 5, 6],
                )
            elif session_raw == "24/5":
                session = SessionConfig(
                    timezone="UTC",
                    open="00:00",
                    close="23:59",
                    days=[0, 1, 2, 3, 4],
                )
            else:
                session = SessionConfig()

            symbols[symbol_key] = MarketSymbolConfig(
                symbol_key=symbol_key,
                market_type=market_type,
                exchange=cfg.get("exchange", ""),
                display_name=cfg.get("display_name", symbol_key),
                base_currency=cfg.get("base_currency", ""),
                quote_currency=cfg.get("quote_currency", ""),
                tick_size=float(cfg.get("tick_size", 0.01)),
                lot_size=float(cfg.get("lot_size", 1.0)),
                min_notional=float(cfg.get("min_notional", 0)),
                session=session,
                commission_bps=float(cfg.get("commission_bps", 2.0)),
                taker_commission_bps=float(cfg.get("taker_commission_bps", 4.0)),
                expected_slippage_bps=float(cfg.get("expected_slippage_bps", 1.0)),
                max_slippage_bps=float(cfg.get("max_slippage_bps", 3.0)),
                data_source=cfg.get("data_source", ""),
                candle_intervals=cfg.get("candle_intervals", ["1m"]),
                warmup_candles=int(cfg.get("warmup_candles", 60)),
                features_version=cfg.get("features_version", ""),
                drop_open_bars=int(cfg.get("drop_open_bars", 0)),
                expiry_day=cfg.get("expiry_day", ""),
            )

    return symbols


# ── Singleton Instances ──────────────────────────────────────────────────────
secrets: Secrets = _load_secrets()

_market_yaml = _load_yaml("markets.yaml")
_risk_yaml = _load_yaml("risk.yaml")
_model_yaml = _load_yaml("models.yaml")

# Typed market configs indexed by symbol_key
market_configs: dict[str, MarketSymbolConfig] = _parse_market_config(_market_yaml)

# Raw YAML dicts for risk and model configs (used by other modules)
risk_config: dict[str, Any] = _risk_yaml
model_config: dict[str, Any] = _model_yaml


def get_market(symbol_key: str) -> MarketSymbolConfig:
    """Get the market configuration for a symbol. Raises KeyError if not found."""
    key = symbol_key.lower()
    if key not in market_configs:
        raise KeyError(
            f"Unknown symbol '{symbol_key}'. "
            f"Available: {list(market_configs.keys())}"
        )
    return market_configs[key]


def get_active_symbols(market_type: str | None = None) -> list[str]:
    """Get all active symbol keys, optionally filtered by market type."""
    if market_type is None:
        return list(market_configs.keys())
    return [k for k, v in market_configs.items() if v.market_type == market_type]


def get_model_params(market_type: str) -> dict[str, Any]:
    """Get model hyperparameters and training config for a market type."""
    models = _model_yaml.get("models", {})
    if market_type not in models:
        raise KeyError(
            f"No model config for market type '{market_type}'. "
            f"Available: {list(models.keys())}"
        )
    return models[market_type]


def get_quality_gates() -> dict[str, Any]:
    """Get model quality gate thresholds."""
    return _model_yaml.get("quality_gates", {})
