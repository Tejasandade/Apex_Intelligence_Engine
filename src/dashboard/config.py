"""
Apex Intelligence Engine — Trading Profile Configuration Schema.
Supports multi-asset, multi-style trading with dynamic risk parameters.
"""

TRADING_PROFILES = {
    "assets": {
        "BTCUSDT": {
            "display_name": "Bitcoin (BTC/USDT)",
            "exchange": "Binance Futures",
            "asset_class": "Crypto",
            "tick_size": 0.1,
            "default_qty_precision": 3,
        },
        "NIFTY50": {
            "display_name": "Nifty 50 Index",
            "exchange": "NSE (Future)",
            "asset_class": "Index",
            "tick_size": 0.05,
            "default_qty_precision": 0,
        },
        "EURUSD": {
            "display_name": "EUR/USD",
            "exchange": "Forex",
            "asset_class": "Forex",
            "tick_size": 0.00001,
            "default_qty_precision": 2,
        },
    },
    "styles": {
        "Scalping": {
            "description": "Ultra-short-term, high-frequency entries",
            "timeframe": "1m",
            "trailing_stop_pct": 0.3,
            "take_profit_pct": 0.5,
            "max_hold_minutes": 15,
            "kelly_fraction": 0.25,
            "buy_threshold": 0.55,
            "sell_threshold": 0.45,
        },
        "Intraday": {
            "description": "Same-day positions, no overnight risk",
            "timeframe": "15m",
            "trailing_stop_pct": 1.0,
            "take_profit_pct": 2.0,
            "max_hold_minutes": 480,
            "kelly_fraction": 0.5,
            "buy_threshold": 0.55,
            "sell_threshold": 0.45,
        },
        "Swing": {
            "description": "Multi-day trend following",
            "timeframe": "4h",
            "trailing_stop_pct": 3.0,
            "take_profit_pct": 8.0,
            "max_hold_minutes": 10080,
            "kelly_fraction": 0.4,
            "buy_threshold": 0.60,
            "sell_threshold": 0.40,
        },
    },
}

# Database schema for future multi-user SaaS support
DB_SCHEMA_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(256) NOT NULL,
    full_name VARCHAR(120),
    role VARCHAR(32) NOT NULL DEFAULT 'trader',
    professional_trader_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

DB_SCHEMA_BROKER_CREDENTIALS = """
CREATE TABLE IF NOT EXISTS broker_credentials (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider_name VARCHAR(50) NOT NULL,
    broker_label VARCHAR(80) NOT NULL DEFAULT 'primary',
    account_type VARCHAR(24) NOT NULL DEFAULT 'paper',
    api_key_encrypted TEXT NOT NULL,
    api_secret_encrypted TEXT NOT NULL,
    passphrase_encrypted TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_validated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, provider_name, broker_label)
);
"""
