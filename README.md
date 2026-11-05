# Apex Intelligence Engine v7.2

> **Ultimate algorithmic trading bot** — multi-symbol, multi-timeframe, ML-driven with live news awareness, sentiment filtering, and auto-retraining.

---

## 🏗️ Architecture

```
Binance WebSocket (trades + depth + bookTicker)
        │
        ▼
  process_tick()  ←── CVD, OBI, spread, 1m candle builder
        │
  5m candle queue (asyncio)
        │
        ▼
  LiveSignalEngine ──── LiveFeatureStore (37 features)
        │                      │
        │               RegimeDetector (ADX + CHOP → 4 regimes)
        │                      │
        │            ModelEnsemble (18 XGBoost bags)
        │                      │
        │             PerformanceMonitor (Brier decay)
        │                      │
        │             WalkForwardRetrainer
        │                      │
        ▼
   Council (5 advisors: Momentum / Structure / Volume / Regime / Risk)
        │
        ▼
  Gate checks:  can_trade() ← NewsGuard ← FundingRateMonitor ← SentimentMonitor
        │
        ▼
  execute_signal()  → size_position() [Kelly-adjusted, sentiment-weighted]
        │                    │
        │            find_sl() [structural + ATR]
        │            VPOC-snap TP2 [Volume Profile]
        ▼
  check_exits()  [3-phase: SL → partial TP1 → trail → full TP2]
        │
        ▼
  StateManager (crash-resume)  +  Telegram alerts
```

---

## ⚡ Quick Start

### 1. Environment setup
```bash
# Clone and create venv
python -m venv venv
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

### 2. Configure `.env`
```ini
# Telegram alerts
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id

# Binance (not needed for paper trading)
BINANCE_API_KEY=...
BINANCE_API_SECRET=...

# Optional: news enrichment
CRYPTOCOMPARE_API_KEY=...
NEWS_API_KEY=...
```

### 3. Train models (first time only)
```bash
python scripts/download_6_months.py      # download BTC 1m data
python scripts/train_master.py           # train all regime XGBoost models
```

### 4. Run the bot

**Single-symbol BTC paper trader:**
```bash
python scripts/apex_v7.py
```

**With auto-restart guardian (recommended for 24/7):**
```bash
scripts\start_apex_guardian.bat
```

**Stop:**
```bash
scripts\stop_apex.bat
```

---

## 📂 Project Structure

```
Apex_Intelligence_Engine/
├── scripts/
│   ├── apex_v7.py              ← Main bot (single-symbol, paper trade)
│   ├── start_apex_guardian.bat ← Windows auto-restart wrapper
│   ├── stop_apex.bat           ← Graceful shutdown
│   ├── cleanup.py              ← Archive old backups
│   └── train_master.py         ← Train all models
│
├── src/
│   ├── core/
│   │   ├── config.py           ← Central config (loads .env + YAML)
│   │   └── state_manager.py    ← Crash-resume state persistence
│   │
│   ├── data/providers/
│   │   ├── binance_ws.py       ← WebSocket feed
│   │   ├── news_guard.py       ← News blackout filter (FOMC/CPI/NFP)
│   │   ├── funding_rate.py     ← Binance funding rate monitor
│   │   ├── sentiment.py        ← Fear & Greed index
│   │   └── daily_feed.py       ← Daily + 4H candles for swing trades
│   │
│   ├── engine/
│   │   ├── signal_engine.py    ← LiveSignalEngine (full ML pipeline)
│   │   ├── multi_symbol_manager.py ← Multi-symbol capital isolation
│   │   └── swing_trade_engine.py   ← Multi-day swing trade signals
│   │
│   ├── features/
│   │   ├── live_store.py       ← Rolling feature computation
│   │   ├── store.py            ← FeatureStore (37 crypto features)
│   │   ├── labeling/
│   │   │   └── apex_labeler.py ← Triple-barrier labels (Lopez de Prado)
│   │   └── indicators/
│   │       ├── technical.py    ← RSI, EMA, MACD, ATR, ADX, CHOP, BB
│   │       ├── structure.py    ← FVG, BOS, liquidity sweeps
│   │       └── volume_profile.py ← VPOC, VAH, VAL, HVN/LVN
│   │
│   ├── models/
│   │   ├── xgboost_model.py    ← XGBoost with Platt calibration
│   │   ├── ensemble.py         ← Regime-weighted model ensemble
│   │   ├── regime.py           ← 4-regime market classifier
│   │   ├── monitor.py          ← Brier score decay detection
│   │   └── training/
│   │       └── retrain_scheduler.py ← Background hot-swap retraining
│   │
│   ├── council/
│   │   ├── council.py          ← 5-advisor vote aggregator
│   │   └── advisors.py         ← Momentum / Structure / Volume / Regime / Risk
│   │
│   └── execution/
│       ├── bybit_broker.py     ← Bybit V5 live execution (ready, not wired)
│       └── paper_broker.py     ← Paper trade simulator
│
├── configs/
│   ├── markets.yaml            ← Symbol specs (BTC, ETH, BankNifty, Nifty)
│   ├── risk.yaml               ← Risk limits and kill-switch config
│   └── models.yaml             ← XGBoost hyperparameters
│
└── data/
    ├── models/                 ← Trained XGBoost bags (18 models)
    └── apex_state.json         ← Auto-saved crash-resume state
```

---

## 🔧 Configuration

### Risk parameters (`configs/risk.yaml`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `sizing.min_allocation_pct` | 1% | Minimum risk per trade |
| `sizing.max_allocation_pct` | 5% | Maximum risk per trade |
| `drawdown.portfolio_kill_pct` | 10% | Hard stop if portfolio drops 10% |
| `conviction.min_signal_threshold` | 0.60 | Minimum model probability to trade |

### Market Intelligence (`apex_v7.py` constants)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `MIN_PROB` | 0.58 | Minimum ensemble probability |
| `MIN_RR` | 3.0 | Minimum reward:risk ratio |
| `MIN_INTERVAL` | 1800s | Minimum time between trades |
| `LOSS_COOLDOWN` | 7200s | Cooldown after any loss |
| `MAX_TRADES_DAY` | 3 | Daily trade limit |
| `SESSION_LOSSES` | 3 | Consecutive losses before halt |

---

## 🧠 Signal Pipeline Detail

```
1. LiveFeatureStore     → 37 crypto features (MTF ADX, CVD, FVG, BOS, VWAP, etc.)
2. RegimeDetector       → TRENDING_HIGH_VOL / TRENDING_LOW_VOL / RANGING_HIGH/LOW
3. ModelEnsemble        → 18 XGBoost bags (3 regimes × 2 dirs × 3 bags)
4. PerformanceMonitor   → Brier score decay detection → auto-halt on model drift
5. Council (5 advisors) → Weighted vote: Momentum + Structure + Volume + Regime + Risk
6. NewsGuard            → Blocks 30min before / 60min after FOMC/CPI/NFP
7. FundingRateMonitor   → Reduces size in crowded funding extremes
8. SentimentMonitor     → Fear & Greed regime modifier (0.70–1.15×)
9. VPOC TP targeting    → Snaps TP2 to nearest High Volume Node
10. StateManager        → Crash-safe persistence after every trade event
```

---

## 📊 Dashboard Layout

```
╔════════════════════════════════════════════════════════════════╗
║ ⚡ APEX INTELLIGENCE ENGINE v7.2 ⚡                            ║
╠════════════════════════════════════════════════════════════════╣
║ ACCOUNT: Balance / PnL / DD / Win Rate / Profit Factor        ║
╟────────────────────────────────────────────────────────────────╢
║ ML ENGINE: Regime / ATR / Prob bars / Last signal / Momentum  ║
╟────────────────────────────────────────────────────────────────╢
║ MARKET INTEL: F&G / Funding Rate / News Gate / Next Event     ║
╟────────────────────────────────────────────────────────────────╢
║ POSITION: Direction / Entry / SL / TP1 / TP2 / Age / PnL     ║
╟────────────────────────────────────────────────────────────────╢
║ TRADE LOG: Last 7 trades with timestamps + outcome            ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🚦 Trade Entry Checklist (enforced automatically)

- [ ] Model ensemble probability ≥ 0.58
- [ ] Long vs Short probability gap ≥ 0.06
- [ ] Council score ≥ 0.60 (5-advisor weighted vote)
- [ ] Risk:Reward ≥ 3.0
- [ ] No news blackout (FOMC / CPI / NFP ±30/60min window)
- [ ] Spread ≤ 8 bps
- [ ] Not halted (drawdown / consecutive losses / daily limit)
- [ ] Minimum 30min since last trade
- [ ] Minimum 2h cooldown after any loss

---

## 🔄 Retraining

Models are automatically retrained every 2000 candles (~7 days on 5m) using:
- **Triple-barrier labels** (Lopez de Prado method) — SL/TP/time barriers aligned with real trade mechanics
- **Background thread** — retraining never pauses trading
- **Hot-swap** — new models are atomically swapped in without restart

Manual retrain:
```bash
python scripts/retrain_bootstrap.py
```

---

## 📈 Extending to Multi-Symbol

Multi-symbol support is scaffolded in `src/engine/multi_symbol_manager.py`.
To activate:

1. Edit `SYMBOLS` in `apex_v7.py`:
   ```python
   SYMBOLS = ["BTCUSDT", "ETHUSDT"]
   ```
2. Train ETH models: `python scripts/train_master.py --symbol ETHUSDT`
3. Run `apex_v8_multi.py` (coming soon)

---

## ⚠️ Disclaimer

This system is for **educational and research purposes**. Paper trade extensively before considering live deployment. Past performance in backtest does not guarantee future results. Never risk capital you cannot afford to lose.

