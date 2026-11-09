"""
Apex Indian Engine — Live Web Command Center
============================================
High-performance asynchronous web dashboard running on port 8081.
Provides real-time visualization of:
- NIFTY & BANKNIFTY Spot, Day Change, and VWAP
- India VIX Volatility Gauge & Regime Status
- 3-Minute Institutional Option Chain PCR & Delta-OI Walls
- Active Option Contract, PnL in ₹, and Analytical Greeks
- One-Click Square Off / Pause / Resume controls
- Trade Execution Journal with exact STT/brokerage friction accounting
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Optional
from aiohttp import web

logger = logging.getLogger("apex.indian.web")

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>APEX Indian Options Intelligence Core</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --canvas: #090b0e;
      --surface: #0f1217;
      --surface-subtle: #141820;
      --surface-elevated: #1a202c;
      --surface-hover: #1f2634;
      --border-hairline: rgba(255, 255, 255, 0.06);
      --border-subtle: rgba(255, 255, 255, 0.10);
      --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', Consolas, monospace;
      --text-primary: #f0f3f6;
      --text-secondary: #8c96a5;
      --text-muted: #525c6a;
      --green: #089981;
      --green-soft: rgba(8, 153, 129, 0.12);
      --red: #f23645;
      --red-soft: rgba(242, 54, 69, 0.12);
      --amber: #f59e0b;
      --amber-soft: rgba(245, 158, 11, 0.12);
      --blue: #2962ff;
      --blue-soft: rgba(41, 98, 255, 0.12);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--canvas);
      color: var(--text-primary);
      font-family: var(--font-ui);
      font-size: 13px;
      line-height: 1.5;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    .top-header {
      background: var(--surface);
      border-bottom: 1px solid var(--border-hairline);
      padding: 12px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-badge {
      background: linear-gradient(135deg, #ff9933, #ffffff, #138808);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      font-weight: 800;
      font-size: 18px;
      letter-spacing: 0.5px;
    }
    .tag {
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .tag-live { background: var(--green-soft); color: var(--green); border: 1px solid rgba(8,153,129,0.3); }
    .tag-paper { background: var(--blue-soft); color: var(--blue); border: 1px solid rgba(41,98,255,0.3); }
    .tag-vix { background: var(--amber-soft); color: var(--amber); border: 1px solid rgba(245,158,11,0.3); }
    
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn {
      padding: 6px 14px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      border: 1px solid var(--border-subtle);
      background: var(--surface-subtle);
      color: var(--text-primary);
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .btn:hover { background: var(--surface-hover); }
    .btn-danger { background: var(--red-soft); color: var(--red); border-color: rgba(242,54,69,0.3); }
    .btn-danger:hover { background: rgba(242,54,69,0.2); }
    
    .container {
      max-width: 1440px;
      margin: 0 auto;
      padding: 20px 24px;
      width: 100%;
      flex: 1;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    
    /* Stats Row */
    .stats-banner {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 12px;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: 8px;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .stat-label { font-size: 11px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
    .stat-val { font-size: 20px; font-weight: 700; font-family: var(--font-mono); }
    
    /* Asset Cards Grid */
    .assets-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 16px;
    }
    .asset-card {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: 10px;
      padding: 18px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .asset-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .asset-title { font-size: 16px; font-weight: 700; }
    .asset-subtitle { font-size: 11px; color: var(--text-secondary); }
    .asset-price { font-size: 24px; font-weight: 800; font-family: var(--font-mono); }
    
    .metrics-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .metrics-table td {
      padding: 6px 0;
      border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .metrics-table td:last-child {
      text-align: right;
      font-family: var(--font-mono);
      font-weight: 600;
    }
    
    /* Active Position Banner */
    .position-banner {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: 10px;
      padding: 20px;
    }
    .position-banner.active {
      border-color: rgba(41,98,255,0.3);
      box-shadow: 0 0 20px rgba(41,98,255,0.05);
    }
    .pos-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 16px;
      margin-top: 14px;
    }
    
    /* Trade Journal */
    .journal-card {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: 10px;
      padding: 20px;
    }
    .table-container {
      overflow-x: auto;
      margin-top: 12px;
    }
    table.journal-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    table.journal-table th {
      text-align: left;
      padding: 8px 12px;
      color: var(--text-secondary);
      font-size: 11px;
      text-transform: uppercase;
      border-bottom: 1px solid var(--border-subtle);
    }
    table.journal-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border-hairline);
      font-family: var(--font-mono);
    }
    .pos-win { color: var(--green); }
    .pos-loss { color: var(--red); }
  </style>
</head>
<body>

  <header class="top-header">
    <div class="brand">
      <span class="brand-badge">🇮🇳 APEX INDIAN INTELLIGENCE</span>
      <span id="mode-badge" class="tag tag-paper">PAPER MODE</span>
      <span id="status-badge" class="tag tag-live">READY</span>
      <span id="vix-badge" class="tag tag-vix">VIX ELEVATED</span>
    </div>
    <div class="header-actions">
      <span id="clock" style="font-family: var(--font-mono); font-size: 12px; color: var(--text-secondary);">--:--:-- IST</span>
      <button class="btn" onclick="triggerAction('pause')">Pause Engine</button>
      <button class="btn" onclick="triggerAction('resume')">Resume</button>
      <button class="btn btn-danger" onclick="triggerAction('square_off')">Square Off All</button>
    </div>
  </header>

  <main class="container">

    <!-- Account Stats -->
    <section class="stats-banner">
      <div class="stat-card">
        <span class="stat-label">Trading Capital</span>
        <span class="stat-val" id="val-cap">₹1,00,000.00</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Today's Realized Net PnL</span>
        <span class="stat-val" id="val-net-pnl">+₹0.00</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Daily Trades</span>
        <span class="stat-val" id="val-trades">0 / 4</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Friction & STT Deducted</span>
        <span class="stat-val" id="val-fees">₹0.00</span>
      </div>
      <div class="stat-card">
        <span class="stat-label">Daily Loss Buffer (3%)</span>
        <span class="stat-val" style="color: var(--amber);" id="val-buffer">₹3,000.00</span>
      </div>
    </section>

    <!-- Asset Cards (BankNifty, Nifty, India VIX) -->
    <section class="assets-grid">
      <!-- BankNifty -->
      <div class="asset-card">
        <div class="asset-head">
          <div>
            <div class="asset-title">BANKNIFTY</div>
            <div class="asset-subtitle">NSE Banking Index</div>
          </div>
          <span class="tag tag-live" id="bn-tag">LIVE</span>
        </div>
        <div class="asset-price" id="bn-price">₹0.00</div>
        <table class="metrics-table">
          <tr><td style="color:var(--text-secondary)">True VWAP</td><td id="bn-vwap">₹0.00</td></tr>
          <tr><td style="color:var(--text-secondary)">Put-Call Ratio (PCR)</td><td id="bn-pcr">1.00</td></tr>
          <tr><td style="color:var(--text-secondary)">Delta-OI Flow</td><td id="bn-flow">BALANCED</td></tr>
          <tr><td style="color:var(--text-secondary)">Max Put Support</td><td id="bn-put-wall">0</td></tr>
          <tr><td style="color:var(--text-secondary)">Max Call Resistance</td><td id="bn-call-wall">0</td></tr>
        </table>
      </div>

      <!-- Nifty 50 -->
      <div class="asset-card">
        <div class="asset-head">
          <div>
            <div class="asset-title">NIFTY 50</div>
            <div class="asset-subtitle">NSE Benchmark Index</div>
          </div>
          <span class="tag tag-live" id="nifty-tag">LIVE</span>
        </div>
        <div class="asset-price" id="nifty-price">₹0.00</div>
        <table class="metrics-table">
          <tr><td style="color:var(--text-secondary)">True VWAP</td><td id="nifty-vwap">₹0.00</td></tr>
          <tr><td style="color:var(--text-secondary)">Put-Call Ratio (PCR)</td><td id="nifty-pcr">1.00</td></tr>
          <tr><td style="color:var(--text-secondary)">Delta-OI Flow</td><td id="nifty-flow">BALANCED</td></tr>
          <tr><td style="color:var(--text-secondary)">Max Put Support</td><td id="nifty-put-wall">0</td></tr>
          <tr><td style="color:var(--text-secondary)">Max Call Resistance</td><td id="nifty-call-wall">0</td></tr>
        </table>
      </div>

      <!-- India VIX & Regime -->
      <div class="asset-card">
        <div class="asset-head">
          <div>
            <div class="asset-title">INDIA VIX</div>
            <div class="asset-subtitle">Volatility Regime Core</div>
          </div>
          <span class="tag tag-vix" id="vix-regime-name">ELEVATED</span>
        </div>
        <div class="asset-price" style="color: var(--amber);" id="vix-val">18.29</div>
        <table class="metrics-table">
          <tr><td style="color:var(--text-secondary)">Regime Classification</td><td id="vix-desc">ELEVATED_VOL</td></tr>
          <tr><td style="color:var(--text-secondary)">Strike Selection Rule</td><td id="vix-strike-rule">ATM / Momentum Breakout</td></tr>
          <tr><td style="color:var(--text-secondary)">Council Threshold</td><td id="vix-threshold">0.58</td></tr>
          <tr><td style="color:var(--text-secondary)">Size Multiplier</td><td id="vix-mult">90% Sizing</td></tr>
          <tr><td style="color:var(--text-secondary)">BankNifty Base SL</td><td id="vix-sl">32.0 pts</td></tr>
        </table>
      </div>
    </section>

    <!-- Active Option Position -->
    <section class="position-banner" id="pos-banner">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:14px; font-weight:700;">ACTIVE OPTION POSITION</span>
        <span class="tag tag-paper" id="pos-status">FLAT (MONITORING 5M SETUPS)</span>
      </div>
      <div class="pos-grid" id="pos-details" style="display:none;">
        <div class="stat-card">
          <span class="stat-label">Contract</span>
          <span class="stat-val" style="font-size:14px;" id="pos-symbol">--</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Direction & Lots</span>
          <span class="stat-val" style="font-size:14px;" id="pos-lots">--</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Entry Premium</span>
          <span class="stat-val" style="font-size:14px;" id="pos-entry">₹0.00</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Trailing Stop Loss</span>
          <span class="stat-val" style="font-size:14px; color:var(--red);" id="pos-sl">₹0.00</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Take Profit 1 / TP2</span>
          <span class="stat-val" style="font-size:14px; color:var(--green);" id="pos-tp">₹0.00 / ₹0.00</span>
        </div>
        <div class="stat-card">
          <span class="stat-label">Unrealized PnL</span>
          <span class="stat-val" style="font-size:16px;" id="pos-pnl">₹0.00</span>
        </div>
      </div>
    </section>

    <!-- Execution Journal -->
    <section class="journal-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:14px; font-weight:700;">TRADE EXECUTION AUDIT JOURNAL</span>
        <button class="btn" onclick="exportCsv()">Export CSV</button>
      </div>
      <div class="table-container">
        <table class="journal-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Contract</th>
              <th>Type</th>
              <th>Lots</th>
              <th>Entry Px</th>
              <th>Exit Px</th>
              <th>Dur</th>
              <th>Gross PnL</th>
              <th>Friction</th>
              <th>Net Cash</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody id="journal-body">
            <tr><td colspan="11" style="text-align:center; color:var(--text-muted); padding:16px;">No closed option trades recorded today.</td></tr>
          </tbody>
        </table>
      </div>
    </section>

  </main>

  <script>
    async function updateDashboard() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();

        // Top Banner
        document.getElementById('val-cap').innerText = '₹' + Number(data.capital_inr || 100000).toLocaleString('en-IN', {minimumFractionDigits: 2});
        const net = (data.daily_realized_pnl || 0) + (data.daily_unrealized_pnl || 0);
        const sign = net >= 0 ? '+' : '';
        const pnlEl = document.getElementById('val-net-pnl');
        pnlEl.innerText = sign + '₹' + Number(net).toLocaleString('en-IN', {minimumFractionDigits: 2});
        pnlEl.style.color = net >= 0 ? 'var(--green)' : 'var(--red)';
        
        document.getElementById('val-trades').innerText = (data.trades_today || 0) + ' / ' + (data.max_trades || 4);
        document.getElementById('val-fees').innerText = '₹' + Number(data.daily_friction_cost || 0).toFixed(2);
        
        // Spot Prices
        document.getElementById('bn-price').innerText = '₹' + Number(data.banknifty_spot || 0).toLocaleString('en-IN', {minimumFractionDigits: 2});
        document.getElementById('bn-vwap').innerText = '₹' + Number(data.banknifty_vwap || 0).toLocaleString('en-IN', {minimumFractionDigits: 2});
        document.getElementById('bn-pcr').innerText = Number(data.pcr || 1.0).toFixed(2);
        document.getElementById('bn-flow').innerText = data.oi_flow || 'BALANCED';
        document.getElementById('bn-put-wall').innerText = Number(data.put_wall || 0).toFixed(0);
        document.getElementById('bn-call-wall').innerText = Number(data.call_wall || 0).toFixed(0);

        document.getElementById('nifty-price').innerText = '₹' + Number(data.nifty_spot || 0).toLocaleString('en-IN', {minimumFractionDigits: 2});
        document.getElementById('nifty-vwap').innerText = '₹' + Number(data.nifty_vwap || 0).toLocaleString('en-IN', {minimumFractionDigits: 2});

        // VIX
        document.getElementById('vix-val').innerText = Number(data.vix || 18.0).toFixed(2);
        document.getElementById('vix-regime-name').innerText = data.vix_regime || 'ELEVATED';
        document.getElementById('vix-desc').innerText = data.vix_regime || 'ELEVATED_VOL';

        // Clock
        const now = new Date();
        document.getElementById('clock').innerText = now.toLocaleTimeString('en-US', {hour12: false}) + ' IST';

        // Position
        const p = data.open_position;
        const banner = document.getElementById('pos-banner');
        const posDetails = document.getElementById('pos-details');
        const posStatus = document.getElementById('pos-status');

        if (p) {
          banner.classList.add('active');
          posStatus.innerText = 'OPEN ' + p.direction + ' (' + p.option_type + ')';
          posStatus.className = 'tag tag-live';
          posDetails.style.display = 'grid';

          document.getElementById('pos-symbol').innerText = p.symbol;
          document.getElementById('pos-lots').innerText = p.lots + ' Lots (' + p.quantity + ' Qty)';
          document.getElementById('pos-entry').innerText = '₹' + Number(p.entry_price).toFixed(2);
          document.getElementById('pos-sl').innerText = '₹' + Number(p.sl_price).toFixed(2);
          document.getElementById('pos-tp').innerText = '₹' + Number(p.tp1_price).toFixed(2) + ' / ₹' + Number(p.tp2_price).toFixed(2);
          
          const uPnl = p.unrealized_pnl || 0;
          const uSign = uPnl >= 0 ? '+' : '';
          const uEl = document.getElementById('pos-pnl');
          uEl.innerText = uSign + '₹' + Number(uPnl).toFixed(2);
          uEl.style.color = uPnl >= 0 ? 'var(--green)' : 'var(--red)';
        } else {
          banner.classList.remove('active');
          posStatus.innerText = 'FLAT (MONITORING 5M SETUPS)';
          posStatus.className = 'tag tag-paper';
          posDetails.style.display = 'none';
        }

        // Journal
        if (data.journal && data.journal.length > 0) {
          const tbody = document.getElementById('journal-body');
          tbody.innerHTML = '';
          data.journal.forEach(t => {
            const tr = document.createElement('tr');
            const pnlClass = t.gross_pnl >= 0 ? 'pos-win' : 'pos-loss';
            const signP = t.gross_pnl >= 0 ? '+' : '';
            tr.innerHTML = `
              <td>${t.trade_id || '--'}</td>
              <td><b>${t.symbol || '--'}</b></td>
              <td>${t.option_type || '--'}</td>
              <td>${t.lots || 1} (${t.quantity || 30})</td>
              <td>₹${Number(t.entry_price || 0).toFixed(2)}</td>
              <td>₹${Number(t.exit_price || 0).toFixed(2)}</td>
              <td>${t.dur_min || 0}m</td>
              <td class="${pnlClass}">${signP}₹${Number(t.gross_pnl || 0).toFixed(2)}</td>
              <td>₹${Number(t.friction_cost || 0).toFixed(2)}</td>
              <td class="${pnlClass}"><b>${signP}₹${Number(t.net_pnl || 0).toFixed(2)}</b></td>
              <td><span style="font-size:10px; color:var(--text-muted);">${t.exit_reason || ''}</span></td>
            `;
            tbody.appendChild(tr);
          });
        }

      } catch (e) {
        console.error("Dashboard poll error:", e);
      }
    }

    async function triggerAction(action) {
      if (!confirm(`Are you sure you want to trigger: ${action.toUpperCase()}?`)) return;
      try {
        const res = await fetch('/api/action', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({action: action})
        });
        const out = await res.json();
        alert(out.message || "Action executed");
        updateDashboard();
      } catch(e) {
        alert("Failed to send action: " + e);
      }
    }

    setInterval(updateDashboard, 1000);
    updateDashboard();
  </script>
</body>
</html>
"""


class IndianWebServer:
    """
    Embedded aiohttp web server for Indian Options Dashboard on port 8081.
    """

    def __init__(
        self,
        engine_ref: Any,
        host: str = "0.0.0.0",
        port: int = 8081,
    ):
        self.engine = engine_ref
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/state", self.handle_state)
        self.app.router.add_post("/api/action", self.handle_action)

    async def handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def handle_state(self, request: web.Request) -> web.Response:
        try:
            metrics = self.engine.get_hud_metrics()
            # Attach closed trades journal from executor
            journal = []
            for t in reversed(self.engine.executor.trade_history[-25:]):
                g_pnl = t.get("gross_pnl", 0.0)
                # Friction approximation
                buy_val = t.get("entry_price", 0.0) * t.get("quantity", 30)
                sell_val = t.get("exit_price", 0.0) * t.get("quantity", 30)
                fric = self.engine.config.friction.calculate_total_friction(buy_val, sell_val)
                journal.append({
                    "trade_id": t.get("trade_id", ""),
                    "symbol": t.get("symbol", ""),
                    "option_type": t.get("option_type", ""),
                    "lots": t.get("lots", 1),
                    "quantity": t.get("quantity", 30),
                    "entry_price": t.get("entry_price", 0.0),
                    "exit_price": t.get("exit_price", 0.0),
                    "dur_min": t.get("dur_min", 0.0),
                    "gross_pnl": g_pnl,
                    "friction_cost": fric.total_friction,
                    "net_pnl": round(g_pnl - fric.total_friction, 2),
                    "exit_reason": t.get("exit_reason", ""),
                })
            metrics["journal"] = journal
            return web.json_response(metrics)
        except Exception as e:
            logger.error(f"Error serving dashboard state: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_action(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
            action = str(data.get("action", "")).lower()

            if action == "square_off":
                if not self.engine.executor.has_open_position():
                    return web.json_response({"status": "ok", "message": "No active option position to square off."})
                trade = self.engine.executor.active_trade
                cur_px = trade.highest_fav_price or trade.entry_price
                closed = self.engine.executor.close_trade(cur_px, reason="MANUAL 1-CLICK WEB DASHBOARD SQUARE OFF")
                self.engine.risk_manager.on_trade_closed(
                    gross_pnl_inr=closed["gross_pnl"],
                    buy_price=trade.entry_price,
                    sell_price=cur_px,
                    qty=trade.quantity,
                    reason="MANUAL WEB SQUARE OFF"
                )
                return web.json_response({"status": "ok", "message": f"Successfully squared off {trade.symbol}."})

            elif action == "pause":
                self.engine.is_paused = True
                return web.json_response({"status": "ok", "message": "Trading engine PAUSED."})

            elif action == "resume":
                self.engine.is_paused = False
                return web.json_response({"status": "ok", "message": "Trading engine RESUMED."})

            return web.json_response({"status": "error", "message": f"Unknown action: {action}"}, status=400)
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def start(self):
        try:
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()
            logger.info(f"🌐 Indian Options Web Dashboard running on http://{self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start Indian Web Dashboard: {e}")
