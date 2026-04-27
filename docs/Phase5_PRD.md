# Phase 5: Multi-Strategy Pod Architecture (Global Apex Engine)

## Objective
Evolve the Apex Intelligence Engine from a single-market (Crypto) bot into an institutional-grade Multi-Strategy engine. The system will independently monitor and trade Crypto, Indian Equity/Options (Angel One), and Forex (OANDA Demo) through isolated "Risk Pods," overseen by a Global Alpha Ranker.

## Step-by-Step Implementation Roadmap

### [Step 1] The Decoupling (Adapter Pattern)
- Abstract the current Binance-hardcoded logic into a `BaseBrokerAdapter`.
- Create a `BinanceAdapter` (Crypto), `AngelOneAdapter` (Indian Spot/Options), and `OandaAdapter` (Forex Data).
- Ensure the core `TradeExecutor` and `QuantModel` interact only with the Adapter interface, completely ignorant of the underlying broker.

### [Step 2] Multi-Pool Risk Management
- Upgrade `risk_manager.py` to support isolated capital pools (e.g., $10,000 Binance vs. ₹1,00,000 Angel One).
- Implement Market Session Logic: Automatically pause the Indian Risk Pod at 15:30 IST without disrupting the 24/7 Crypto Pod.

### [Step 3] Advanced Trade Management (Trailing & Scaling)
- Upgrade `executor.py` to support Trailing Stop Losses (e.g., move stop to breakeven after 1:1 RR).
- Implement Scale-outs (e.g., close 50% of position at Target 1). Test on Crypto first.

### [Step 4] Indian Market Data Pipeline
- Build `nse_ingestion.py` using Angel One SmartAPI to stream BankNifty/Nifty50 Options Chain data and Greeks (Delta, Theta).
- Build the localized Oracle to scrape NewsAPI (India Business) and NSE FII/DII data.

### [Step 5] Specialized AI Models
- Train and implement market-specific XGBoost models (e.g., `crypto_scalp_model`, `banknifty_options_model`).
- Update `dataset_builder.py` to route the correct feature set to the correct model.

### [Step 6] The Global Command Center (UI Overhaul)
- Upgrade the React Dashboard with Market Tabs: `[GLOBAL MASTER]`, `[CRYPTO]`, `[INDIA]`, `[FOREX]`.
- Implement the **Alpha Ranker**: A logic layer that compares signals across all active markets and routes executions based on the highest statistical edge.

### [Step 7] Architecture Finalization & Dry-Run Adapters

7.1 Multi-Pool Risk Management: Refactor risk_manager.py to use isolated capital_pools (e.g., $10,000 for Crypto, ₹1,00,000 for India) instead of a single global float.

7.2 The Angel One Dry-Run Adapter: Build src/broker/angel_one_adapter.py. It must implement BaseBrokerAdapter but strictly enforce live_trading_enabled = False (Paper Trading mode) to safely log execution intent without risking capital.

7.3 Global Master View (UI): Build the React component in App.jsx to visually render the Alpha Ranker's cross-market conviction.

7.4 The Forex Placeholder: Build a stubbed oanda_adapter.py to satisfy the adapter layer for future implementation.