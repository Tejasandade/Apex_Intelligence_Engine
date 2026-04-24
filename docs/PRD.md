# Apex Market Intelligence Engine v2.5
## Product Requirements Document (PRD)

### Phase 1: The Quant Agent (Microstructure Engine)

#### 1. Objective
Establish high-frequency WebSocket data ingestion pipelines, parse Level 2/Level 3 order book data, engineer microstructure features (CVD, GEX, Liquidations), and train the quantitative ML model for real-time crypto liquidations.

#### 2. Scope
* **Target Market**: Crypto exchanges (Binance, Bybit) for Phase 1 proof-of-concept, given 24/7 liquidity and rich WebSocket APIs.
* **Core Inputs**: Raw order flow, footprint charts, Cumulative Volume Delta (CVD), resting limit block orders, and liquidation heatmaps.
* **Core Output**: A mathematical probability score [0, 1] representing directional bias (Long/Short) over a defined time horizon (e.g., 5m, 15m).

#### 3. Architecture & Tech Stack (Phase 1)
* **Language**: Python 3.11+
* **Data Ingestion**: `websockets`, `aiohttp`, `asyncio` for non-blocking high-frequency streams.
* **Data Processing/Storage**: `pandas`, `numpy`, Redis (for ultra-fast in-memory state/orderbook snapshotting), TimescaleDB/PostgreSQL (for historical tick-level persistence).
* **Machine Learning**: `xgboost`, `pytorch` (for initial modeling of tick data and sequence prediction).
* **Containerization**: Docker & Docker Compose (isolated environments for ingestion, db, and training).

#### 4. Requirements & Features
* **REQ-1 (WebSocket Pipeline)**: Establish resilient WebSocket connections to target exchanges (e.g., Binance USD-M Futures). Auto-reconnect with exponential backoff.
* **REQ-2 (Order Book Reconstruction)**: Maintain a local, real-time snapshot of the L2/L3 order book. Handle delta updates flawlessly.
* **REQ-3 (Microstructure Feature Engineering)**:
    * Calculate Cumulative Volume Delta (CVD) per timeframe.
    * Track large limit order additions/cancellations (Spoofing detection).
    * Parse and log real-time liquidation events to map liquidation cascades.
* **REQ-4 (Model Training Pipeline)**: Sandbox environment to load historical data, engineer features offline, and train the XGBoost/PyTorch model.
* **REQ-5 (Inference Engine)**: Serve the model to evaluate the real-time order book state and output the probability score via a local API.

#### 5. Autonomous Agent Workflow (GSD)
This project is built for autonomous AI agents (Aider/Cline).
* All tasks must be tracked in `progress.txt`.
* Agents pull the next uncompleted task, write code, run containerized tests, and commit.
* CodeRabbit performs automated PR reviews.

---

### Task Breakdown (Phase 1)

**Epic 1: Infrastructure & Scaffolding**
- [ ] Task 1.1: Initialize project structure, Docker configurations, and `requirements.txt`. (Completed)
- [x] Task 1.2: Set up Redis and TimescaleDB containers in `docker-compose.yml`. (Completed)
- [x] Task 1.3: Create base async database connection utilities. (Completed)

**Epic 2: High-Frequency Data Ingestion**
- [x] Task 2.1: Build robust WebSocket client manager with auto-reconnect.
- [x] Task 2.2: Implement exchange-specific parsers (Binance AggTrades, Depth Updates, Liquidations).
- [x] Task 2.3: Route parsed streams to Redis (real-time state) and TimescaleDB (historical). (Completed)

**Epic 3: Feature Engineering Engine**
- [ ] Task 3.1: Implement Local Order Book (LOB) reconstruction logic from depth updates.
- [ ] Task 3.2: Create real-time CVD calculation module.
- [ ] Task 3.3: Map liquidation events and calculate liquidation intensity metrics.

**Epic 4: The Quant Model**
- [ ] Task 4.1: Script to dump normalized feature datasets from DB.
- [ ] Task 4.2: Build PyTorch/XGBoost training pipeline.
- [ ] Task 4.3: Expose inference API (FastAPI) to output the directional probability score.
