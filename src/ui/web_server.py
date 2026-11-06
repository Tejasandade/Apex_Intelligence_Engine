"""
Apex Intelligence Engine — Live Web Dashboard
Embedded high-performance asynchronous web server using aiohttp.
Provides a real-time web UI and JSON API endpoint for remote monitoring.
"""

import os
import json
import asyncio
from typing import Callable, Any
from aiohttp import web

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Apex Intelligence Engine — Live Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #121824;
      --card-border: #1f293d;
      --text-main: #f0f6fc;
      --text-muted: #8b949e;
      --accent-green: #00f090;
      --accent-green-bg: rgba(0, 240, 144, 0.12);
      --accent-red: #ff4d4d;
      --accent-red-bg: rgba(255, 77, 77, 0.12);
      --accent-cyan: #00d8ff;
      --accent-gold: #ffb800;
      --font-mono: 'JetBrains Mono', monospace;
      --font-sans: 'Inter', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 16px;
      font-size: 14px;
      line-height: 1.5;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    /* Header */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand h1 {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 0.5px;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand h1 span { color: var(--accent-cyan); }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 12px;
      text-transform: uppercase;
      font-family: var(--font-mono);
    }
    .badge-live {
      background: rgba(0, 240, 144, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(0, 240, 144, 0.3);
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--accent-green);
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.4; transform: scale(0.9); }
      50% { opacity: 1; transform: scale(1.1); }
      100% { opacity: 0.4; transform: scale(0.9); }
    }
    .header-right {
      display: flex;
      align-items: baseline;
      gap: 16px;
    }
    .price-tag {
      font-family: var(--font-mono);
      font-size: 24px;
      font-weight: 700;
      color: var(--accent-cyan);
    }

    /* Grid Layouts */
    .grid-4 {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }
    .grid-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 768px) {
      .grid-2 { grid-template-columns: 1fr; }
    }

    /* Card Styling */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card-value {
      font-family: var(--font-mono);
      font-size: 20px;
      font-weight: 700;
    }
    .card-sub {
      font-size: 12px;
      color: var(--text-muted);
      display: flex;
      gap: 6px;
      align-items: center;
    }

    /* Probability Bars */
    .prob-row {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-top: 4px;
    }
    .prob-bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      font-family: var(--font-mono);
    }
    .prob-track {
      height: 10px;
      background: #1c2436;
      border-radius: 5px;
      overflow: hidden;
      position: relative;
    }
    .prob-fill-long {
      height: 100%;
      background: linear-gradient(90deg, #00b468, var(--accent-green));
      border-radius: 5px;
      transition: width 0.3s ease;
    }
    .prob-fill-short {
      height: 100%;
      background: linear-gradient(90deg, #d32f2f, var(--accent-red));
      border-radius: 5px;
      transition: width 0.3s ease;
    }
    .threshold-line {
      position: absolute;
      top: 0;
      bottom: 0;
      left: 60%;
      width: 2px;
      background: #ffffff;
      opacity: 0.7;
    }

    /* Active Position Banner */
    .pos-banner {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .pos-open-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pos-dir {
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 16px;
      padding: 4px 10px;
      border-radius: 6px;
    }
    .dir-long { background: var(--accent-green-bg); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .dir-short { background: var(--accent-red-bg); color: var(--accent-red); border: 1px solid var(--accent-red); }
    .pos-pnl {
      font-family: var(--font-mono);
      font-size: 22px;
      font-weight: 700;
    }

    .pos-metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
      background: #0d121c;
      padding: 12px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 13px;
    }
    .pos-metric-box span { display: block; font-size: 10px; color: var(--text-muted); text-transform: uppercase; }

    /* Tables */
    .table-container {
      overflow-x: auto;
      max-height: 340px;
      overflow-y: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      font-family: var(--font-mono);
      text-align: left;
    }
    th {
      position: sticky;
      top: 0;
      background: #161e2e;
      color: var(--text-muted);
      padding: 10px 12px;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
    }
    td {
      padding: 9px 12px;
      border-bottom: 1px solid #161e2e;
    }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    .green { color: var(--accent-green); }
    .red { color: var(--accent-red); }
    .cyan { color: var(--accent-cyan); }
    .gold { color: var(--accent-gold); }
    .muted { color: var(--text-muted); }

    footer {
      text-align: center;
      font-size: 11px;
      color: var(--text-muted);
      padding-top: 8px;
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <h1><span>⚡</span> APEX INTELLIGENCE ENGINE</h1>
        <div class="badge badge-live"><span class="pulse-dot"></span> LIVE 24/7</div>
      </div>
      <div class="header-right">
        <span class="muted" style="font-size: 12px;" id="last-update">Connecting...</span>
        <div class="price-tag" id="btc-price">$--,---.-</div>
      </div>
    </header>

    <!-- Top KPI Grid -->
    <div class="grid-4">
      <div class="card">
        <div class="card-title">Account Balance</div>
        <div class="card-value" id="account-bal">$1,000.00</div>
        <div class="card-sub">
          Net PnL: <span class="green" id="net-pnl">+$0.00</span>
          &bull; Max DD: <span id="dd-pct">0.0%</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Performance Edge</div>
        <div class="card-value green" id="win-rate">75.0%</div>
        <div class="card-sub">
          <span id="win-loss-count">3W / 1L</span> &bull; Profit Factor: <span class="gold" id="profit-factor">3.26</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Daily Activity & Risk</div>
        <div class="card-value cyan" id="trades-today">2 / 3</div>
        <div class="card-sub">
          Trades Today &bull; Gate: <span class="green">ALL CLEAR</span>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Stack Architecture</div>
        <div class="card-value" style="font-size: 16px; color: var(--accent-cyan);" id="models-active">24 Active Models</div>
        <div class="card-sub">
          Council: <span class="green">9 Advisors</span> &bull; WalkForward: <span class="green">ON</span>
        </div>
      </div>
    </div>

    <!-- Active Position Banner -->
    <div id="position-container">
      <!-- Populated dynamically -->
    </div>

    <!-- Middle Grid: ML Engine & Market Intel -->
    <div class="grid-2">
      <!-- ML Signal Engine -->
      <div class="card">
        <div class="card-title">
          <span>ML Ensemble & Conviction</span>
          <span class="badge" style="background:#1f293d; color:var(--accent-cyan);" id="regime-badge">RANGING_LOW_VOL</span>
        </div>

        <div class="prob-row">
          <div class="prob-bar-label">
            <span>LONG Model Conviction</span>
            <span id="prob-long-text">0.00%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill-long" id="prob-long-bar" style="width: 0%;"></div>
            <div class="threshold-line" title="Entry Threshold (60%)"></div>
          </div>
        </div>

        <div class="prob-row" style="margin-top: 8px;">
          <div class="prob-bar-label">
            <span>SHORT Model Conviction</span>
            <span id="prob-short-text">0.00%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill-short" id="prob-short-bar" style="width: 0%;"></div>
            <div class="threshold-line" title="Entry Threshold (60%)"></div>
          </div>
        </div>

        <div style="margin-top: 10px; font-size: 11px; display: flex; justify-content: space-between;" class="muted">
          <span>Threshold: <b>60.0%</b></span>
          <span id="candles-eval">Candles Evaluated: 0</span>
          <span id="atr-val">ATR: $--</span>
        </div>
      </div>

      <!-- Market Intel & Order Flow -->
      <div class="card">
        <div class="card-title">Market Intel & Order Flow</div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 4px;">
          <div style="background:#0d121c; padding: 10px; border-radius: 6px;">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Fear & Greed</div>
            <div style="font-size: 16px; font-weight: 700; font-family: var(--font-mono);" id="fg-val">50 (Neutral)</div>
          </div>
          <div style="background:#0d121c; padding: 10px; border-radius: 6px;">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Funding Rate</div>
            <div style="font-size: 16px; font-weight: 700; font-family: var(--font-mono);" id="funding-val">+0.0000%</div>
          </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 8px;">
          <div style="background:#0d121c; padding: 10px; border-radius: 6px;">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Order Flow Imbalance (OFI)</div>
            <div style="font-size: 14px; font-family: var(--font-mono);" id="ofi-val">+0.000</div>
          </div>
          <div style="background:#0d121c; padding: 10px; border-radius: 6px;">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Macro News Guard</div>
            <div style="font-size: 14px; font-family: var(--font-mono); color: var(--accent-green);" id="news-val">✓ CLEAR</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Trade History Table -->
    <div class="card">
      <div class="card-title">Completed Trades Journal</div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Time</th>
              <th>Action</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>Realized PnL</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody id="journal-tbody">
            <tr><td colspan="7" class="muted" style="text-align:center;">Loading trades...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <footer>
      Apex Intelligence Engine &bull; Cloud 24/7 Autonomous Algorithmic Execution System
    </footer>
  </div>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();

        // Header
        document.getElementById('btc-price').textContent = '$' + Number(data.market.ltp).toLocaleString(undefined, {minimumFractionDigits: 1, maximumFractionDigits: 1});
        document.getElementById('last-update').textContent = data.timestamp;

        // KPI
        document.getElementById('account-bal').textContent = '$' + Number(data.account.balance).toLocaleString(undefined, {minimumFractionDigits: 2});
        const pnl = Number(data.account.net_pnl);
        const pnlElem = document.getElementById('net-pnl');
        pnlElem.textContent = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);
        pnlElem.className = pnl >= 0 ? 'green' : 'red';
        document.getElementById('dd-pct').textContent = data.account.drawdown_pct.toFixed(1) + '%';
        
        document.getElementById('win-rate').textContent = data.account.win_rate.toFixed(1) + '%';
        document.getElementById('win-loss-count').textContent = `${data.account.wins}W / ${data.account.losses}L`;
        document.getElementById('profit-factor').textContent = data.account.profit_factor.toFixed(2);
        document.getElementById('trades-today').textContent = `${data.account.trades_today} / ${data.account.max_trades_day}`;
        document.getElementById('models-active').textContent = `${data.ml.models_count} Active Models`;

        // ML Bars
        const pLong = Number(data.ml.prob_long || 0.5) * 100;
        const pShort = Number(data.ml.prob_short || 0.5) * 100;
        document.getElementById('prob-long-bar').style.width = pLong + '%';
        document.getElementById('prob-long-text').textContent = pLong.toFixed(1) + '%';
        document.getElementById('prob-short-bar').style.width = pShort + '%';
        document.getElementById('prob-short-text').textContent = pShort.toFixed(1) + '%';

        document.getElementById('regime-badge').textContent = data.ml.regime;
        document.getElementById('atr-val').textContent = 'ATR: $' + data.ml.atr.toFixed(0);
        document.getElementById('candles-eval').textContent = `Candles Scanned: ${data.ml.candles_evaluated}`;

        // Intel
        document.getElementById('fg-val').textContent = `${data.intel.fear_greed} (${data.intel.fear_greed_label})`;
        const fund = Number(data.intel.funding_pct);
        document.getElementById('funding-val').textContent = (fund >= 0 ? '+' : '') + fund.toFixed(4) + '%';
        document.getElementById('ofi-val').textContent = (data.momentum.ofi >= 0 ? '+' : '') + Number(data.momentum.ofi || 0).toFixed(3);
        document.getElementById('news-val').textContent = data.intel.news_clear ? '✓ CLEAR' : '⚠ BLACKOUT';

        // Position Banner
        const posCont = document.getElementById('position-container');
        if (data.position.status === 'OPEN') {
          const isLong = data.position.direction === 'LONG';
          const posPnl = Number(data.position.unrealized_pnl);
          posCont.innerHTML = `
            <div class="pos-banner" style="border-color: ${isLong ? 'var(--accent-green)' : 'var(--accent-red)'}">
              <div class="pos-open-header">
                <div style="display: flex; align-items: center; gap: 10px;">
                  <span class="pos-dir ${isLong ? 'dir-long' : 'dir-short'}">OPEN ${data.position.direction}</span>
                  <span style="font-family: var(--font-mono); font-size: 15px;">@$${Number(data.position.entry_price).toLocaleString()}</span>
                </div>
                <div class="pos-pnl ${posPnl >= 0 ? 'green' : 'red'}">
                  ${posPnl >= 0 ? '+$' : '-$'}${Math.abs(posPnl).toFixed(2)}
                </div>
              </div>
              <div class="pos-metrics">
                <div class="pos-metric-box"><span>Current Price</span>$${Number(data.market.ltp).toLocaleString()}</div>
                <div class="pos-metric-box"><span>Stop Loss</span>$${Number(data.position.sl_price).toLocaleString()}</div>
                <div class="pos-metric-box"><span>TP1 Target</span>$${Number(data.position.tp1_price).toLocaleString()}</div>
                <div class="pos-metric-box"><span>TP2 (4.0R)</span>$${Number(data.position.tp2_price).toLocaleString()}</div>
                <div class="pos-metric-box"><span>Duration</span>${data.position.duration_min} min</div>
              </div>
            </div>
          `;
        } else {
          posCont.innerHTML = `
            <div class="pos-banner" style="align-items: center; text-align: center; color: var(--text-muted); padding: 18px;">
              <span style="font-family: var(--font-mono); font-size: 13px;">
                ● FLAT — Scanning 5m candles for next high-conviction Council setup (p ≥ 0.60)
              </span>
            </div>
          `;
        }

        // Journal Table
        const tbody = document.getElementById('journal-tbody');
        if (data.journal && data.journal.length > 0) {
          tbody.innerHTML = data.journal.map(t => {
            const p = Number(t.pnl_usd);
            const pClass = p > 0 ? 'green' : (p < 0 ? 'red' : 'muted');
            const sign = p > 0 ? '+' : '';
            return `
              <tr>
                <td>#${t.id}</td>
                <td>${t.open_ts || '--:--'}</td>
                <td style="font-weight:600; color:${t.action === 'BUY' ? 'var(--accent-green)' : 'var(--accent-red)'}">${t.action}</td>
                <td>$${Number(t.entry_price).toLocaleString()}</td>
                <td>$${Number(t.exit_price).toLocaleString()}</td>
                <td class="${pClass}">${sign}$${p.toFixed(2)}</td>
                <td><span style="font-size:11px;" class="muted">${t.exit_reason || 'MANUAL'}</span></td>
              </tr>
            `;
          }).join('');
        }
      } catch (err) {
        console.error('Dashboard fetch error:', err);
      }
    }

    // Refresh every 1.5 seconds
    setInterval(updateDashboard, 1500);
    updateDashboard();
  </script>
</body>
</html>
"""

class ApexWebServer:
    def __init__(self, state_provider: Callable[[], dict], host: str = "0.0.0.0", port: int = 8080):
        self.state_provider = state_provider
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/state", self.handle_api_state)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def handle_api_state(self, request: web.Request) -> web.Response:
        try:
            state = self.state_provider()
            return web.json_response(state)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "apex-intelligence-engine"})

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(f"  ✓ Live Web Dashboard running at http://{self.host}:{self.port}")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
