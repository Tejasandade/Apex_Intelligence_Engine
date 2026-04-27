# Phase 8: The V4 Institutional Architecture (Real Money Readiness)

## Objective
Rebuild the core data, AI, and risk pipelines to institutional standards, stripping away retail assumptions and ensuring the engine can survive high-volatility events, bad exchange data, and regime shifts with real capital on the line.

## Step-by-Step Implementation Roadmap

### [Step 1] Ironclad Data Engineering (The L1 Firewall)
- **Z-Score Anomaly Rejection:** Implement a 1.5% tick-to-tick deviation threshold in `ingestion_nse.py` to silently drop exchange API glitches and flash spikes.
- **Exchange Normalization:** Explicitly handle unit conversions at the ingestion layer (e.g., Angel One paise to Rupee conversion).
- **Feature Matrix Sanitization:** Ensure `dataset_builder.py` explicitly catches `NaN` or `Infinity` values caused by zero-volume indexing (e.g., VWAP division by zero) and fails gracefully.

### [Step 2] Regime-Aware AI Architecture (The Brains)
- **Regime Detection:** Build a module to calculate Choppiness Index (CHOP) and Average Directional Index (ADX) to classify the market state (Trending, Ranging, Volatile).
- **Multi-Model Routing:** Architect the inference loop to dynamically route data to specific XGBoost weights based on the current market regime.
- **Purged Cross-Validation:** Overhaul the training pipeline to use Purged K-Fold Cross Validation to mathematically guarantee the AI is not overfitting.

### [Step 3] Institutional Risk & Capital Allocation (The Shield)
- **ATR-Based Sizing:** Move away from pure Fractional Kelly. Base the capital allocation on the dynamic distance to the logical Stop Loss (using ATR).
- **Aggression Scaling & Floors:** Ensure the AI allocates a meaningful minimum percentage of the active capital pool (e.g., 5%) while capping at `max_allowed_risk`.
- **Asymmetric Drawdown Halts:** Slash active capital allocation by 50% automatically if the system suffers 3 consecutive losses, scaling back up only after a verified win.

### [Step 4] Realistic Execution & Slippage (The Reality)
- **Slippage Simulation:** The DRY_RUN execution must penalize fills by crossing the spread and adding simulated latency (e.g., 200ms).
- **Limit vs. Market Routing:** Program the AI to submit passive limit orders for low-conviction signals and aggressive market orders for high-conviction sweeps.

### [Step 5] The Command Center (The UI)
- **True PnL Tracking:** Overhaul the dashboard to track Realized and Unrealized PnL strictly based on simulated entry/exit prices and fees.
- **Quant Analysis Cards:** Enhance the Smart Order Cards to display the AI's SHAP values (feature importance), the current Market Regime, and the ATR distance.