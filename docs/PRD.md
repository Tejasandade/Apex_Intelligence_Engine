# Apex Intelligence Engine V5.0
## Product Requirements Document (PRD)

---

### Vision Statement
Build an institutional-grade AI trading system that is smarter, more powerful, and more
capable than systems available on the market. A system that understands different markets,
different trading styles, and can guarantee your money is in safe hands — one that will
make you profitable through genuine intelligence, not simple if-else rules.

---

### Phase 0: The Proving Ground (COMPLETE)

#### Objective
Build the foundational infrastructure and prove it works end-to-end with historical data.

#### Delivered
- **Core Infrastructure**: YAML config, structured logging (structlog), event bus, exception hierarchy
- **Feature Engineering**: 23 vectorised features — technical (RSI, EMA, MACD, ATR, ADX, CHOP, BB, VWAP), structure (FVG, BOS, liquidity sweep, confluence), volume (CVD, OBV, order flow imbalance)
- **Market Profiles**: Crypto-specific and India NSE session features
- **Labeling**: Triple-barrier method (profit target, stop loss, time expiry)
- **Regime Detection**: 4 regimes (TRENDING, RANGING, VOLATILE, QUIET) with ADX/CHOP classification
- **Models**: XGBoost with auto class-imbalance handling + Isotonic probability calibration
- **Training**: Purged walk-forward CV (5-fold, temporal gap to prevent leakage)
- **Backtester**: Event-driven engine with stop/target/scale-out, risk-based sizing, 4 cost scenarios
- **Data**: Binance Futures REST provider, DataValidator, Parquet caching
- **Reports**: Text reports with full trade statistics, cost sensitivity analysis

#### Validation
- 10/10 smoke tests passing
- End-to-end pipeline: download → train → backtest → report
- 49,980 BTCUSDT candles + 49,981 ETHUSDT candles downloaded
- Out-of-sample backtest implemented (60/40 train/test split)

---

### Phase 1: The Spine (Live Data) — NEXT

#### Objective
Real-time WebSocket data streaming and live feature computation.

#### Requirements
- **Binance WebSocket**: Real-time 1-minute candle streaming for BTCUSDT, ETHUSDT
- **Live FeatureStore**: Rolling window feature computation matching backtest features exactly
- **Reconnection**: Auto-reconnect with exponential backoff on disconnects
- **Data Persistence**: Append live candles to historical Parquet for continuous training

---

### Phase 2: The Brain (Advanced Models)

#### Objective
Ensemble model architecture with adaptive regime-aware routing.

#### Requirements
- **Regime Router**: Automatically select the right model based on current market regime
- **Model Ensemble**: Combine trending/ranging models with weighted voting
- **Online Learning**: Periodic re-training on newest data without forgetting old patterns
- **Walk-Forward**: Rolling retrain schedule (e.g., retrain every 1000 new bars)

---

### Phase 3: The Council (Signal Confirmation)

#### Objective
Multi-advisor signal confirmation system to reduce false positives.

#### Requirements
- **Advisor Architecture**: Modular advisors (momentum, structure, volume, sentiment)
- **Consensus Engine**: Weighted voting with conviction scores
- **Conflict Resolution**: Handle contradicting signals gracefully
- **Explanation System**: Log WHY each advisor voted the way it did

---

### Phase 4: The Hands (Execution)

#### Objective
Real order execution with intelligent order management.

#### Requirements
- **Broker Abstraction**: Unified interface for Dhan (India) + Binance (Crypto)
- **Order Types**: Market, limit, stop-limit with bracket orders
- **Position Manager**: Track open positions, P&L, margin
- **Risk Enforcement**: Max drawdown circuit breaker, position size limits, daily loss limit
- **Paper Trading**: Full simulation mode before going live

---

### Phase 5: The Face (Dashboard UI)

#### Objective
Professional real-time monitoring dashboard.

#### Requirements
- **Live Charts**: Real-time price with entry/exit markers
- **Portfolio View**: Equity curve, open positions, trade history
- **Model View**: Feature importance, regime status, conviction scores
- **Alerts**: Telegram/Discord notifications for signals and risk events

---

### Phase 6: The Proof (Live Validation)

#### Objective
Paper trade for 30+ days to validate the system in real market conditions.

#### Requirements
- **Paper Trading**: Full simulation with real-time data and model predictions
- **Performance Tracking**: Compare paper results against backtest expectations
- **Stability**: Zero crashes, auto-recovery from errors
- **Go-Live Gate**: Paper Sharpe > 1.5, Max DD < 15%, Win Rate > 40%, 200+ trades

---

### Architecture

```
                    ┌─────────────────────────────────┐
                    │       Apex Intelligence V5       │
                    │   "The Smart Financial Machine"  │
                    └─────────────┬───────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
    ┌────▼─────┐            ┌────▼─────┐            ┌────▼─────┐
    │  Binance  │            │   Dhan   │            │  OANDA   │
    │  Crypto   │            │  India   │            │  Forex   │
    │ WebSocket │            │   API    │            │ (Future) │
    └────┬─────┘            └────┬─────┘            └────┬─────┘
         │                        │                        │
         └────────────┬───────────┘────────────────────────┘
                      │
              ┌───────▼────────┐
              │  Data Provider  │
              │   (Abstract)    │
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  FeatureStore   │──── 23 Vectorised Features
              │  (v1.0.0)      │──── Market Profiles
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │ Regime Detector │──── TRENDING / RANGING / VOLATILE / QUIET
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  XGBoost Model  │──── Calibrated Probability [0, 1]
              │  (per regime)   │──── Purged Walk-Forward CV
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  The Council    │──── Multi-Advisor Consensus
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Risk Manager   │──── Position Sizing + Circuit Breakers
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Execution      │──── Smart Order Routing
              └───────┬────────┘
                      │
              ┌───────▼────────┐
              │  Dashboard UI   │──── Real-time Monitoring
              └────────────────┘
```

---

### Tech Stack
- **Language**: Python 3.11+
- **ML**: XGBoost, scikit-learn
- **Data**: pandas, numpy, PyArrow (Parquet)
- **API**: httpx (async HTTP), websockets (streaming)
- **Logging**: structlog (structured JSON)
- **Config**: YAML + .env
- **Broker**: Binance REST/WS (crypto), Dhan API (India)
- **Future**: OANDA (forex)
