"""
Apex Intelligence Engine — Interactive Live Web Command Center
==============================================================
Embedded high-performance asynchronous web server using aiohttp.
Provides a modern institutional multi-asset trading dashboard,
dynamic prop firm configuration drawer (Phase 1, Phase 2, Funded,
1-Step, and fully editable rules), emergency position manager,
CSV audit journal exporter, and REST APIs.
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Callable, Any, Optional
from aiohttp import web
from src.core.prop_firm_manager import PROFILES, FIRM_TEMPLATES

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>APEX TRADING ENGINE — Institutional HUD</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #101623;
      --card-subtle: #141c2c;
      --card-border: #1e293b;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --green: #10b981;
      --green-bg: rgba(16, 185, 129, 0.12);
      --green-border: rgba(16, 185, 129, 0.25);
      --red: #f43f5e;
      --red-bg: rgba(244, 63, 94, 0.12);
      --red-border: rgba(244, 63, 94, 0.25);
      --blue: #38bdf8;
      --amber: #f59e0b;
      --amber-bg: rgba(245, 158, 11, 0.12);
      --amber-border: rgba(245, 158, 11, 0.25);
      --font-ui: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', 'SF Mono', 'Roboto Mono', Menlo, monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: var(--font-ui);
      padding: 16px;
      font-size: 13px;
      line-height: 1.45;
      -webkit-font-smoothing: antialiased;
    }

    .container {
      max-width: 1380px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    /* Top Navigation Bar */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-subtle);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-title {
      font-size: 17px;
      font-weight: 700;
      letter-spacing: -0.3px;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .brand-title .brand-dot {
      color: var(--blue);
    }
    .brand-sub {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
      padding-left: 8px;
      border-left: 1px solid var(--card-border);
    }
    .status-pills {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 9999px;
      font-family: var(--font-mono);
      border: 1px solid var(--border-subtle);
      background: var(--card-bg);
      color: var(--text-muted);
    }
    .pill-green {
      background: var(--green-bg);
      color: var(--green);
      border-color: var(--green-border);
    }
    .pill-green .dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--green);
      box-shadow: 0 0 8px var(--green);
    }
    .pill-amber {
      background: var(--amber-bg);
      color: var(--amber);
      border-color: var(--amber-border);
    }
    .pill-red {
      background: var(--red-bg);
      color: var(--red);
      border-color: var(--red-border);
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: all 0.15s ease;
      border: 1px solid var(--border-subtle);
      background: var(--card-bg);
      color: var(--text-main);
      font-family: var(--font-ui);
      user-select: none;
    }
    .btn:hover {
      background: var(--card-subtle);
      border-color: var(--text-muted);
    }
    .btn-amber {
      border-color: var(--amber-border);
      color: var(--amber);
      background: var(--amber-bg);
    }
    .btn-amber:hover {
      background: rgba(245, 158, 11, 0.2);
    }
    .btn-danger {
      border-color: var(--red-border);
      color: var(--red);
      background: var(--red-bg);
    }
    .btn-danger:hover {
      background: rgba(244, 63, 94, 0.2);
    }
    .btn-primary {
      background: #2563eb;
      border-color: #3b82f6;
      color: #fff;
    }
    .btn-primary:hover {
      background: #1d4ed8;
    }

    /* Cards */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Prop Firm HUD */
    .pf-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pf-summary {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .pf-name {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-main);
    }
    .pf-balance-tag {
      font-family: var(--font-mono);
      font-size: 14px;
      font-weight: 700;
      color: var(--green);
    }
    .pf-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }

    /* Progress Trackers */
    .meters-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    .meter-box {
      background: var(--card-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .meter-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
    }
    .meter-label {
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .meter-val {
      font-family: var(--font-mono);
      font-weight: 600;
    }
    .meter-bar {
      height: 6px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 3px;
      overflow: hidden;
      position: relative;
    }
    .meter-fill {
      height: 100%;
      border-radius: 3px;
      transition: width 0.3s ease;
    }
    .meter-fill-green { background: var(--green); }
    .meter-fill-amber { background: var(--amber); }
    .meter-fill-red { background: var(--red); }

    /* Interactive Prop Firm Drawer */
    #pf-drawer {
      display: none;
      background: #0d121c;
      border: 1px solid #2a374a;
      border-radius: 8px;
      padding: 16px;
      flex-direction: column;
      gap: 14px;
      margin-top: 4px;
    }
    .drawer-section-title {
      font-size: 11px;
      font-weight: 700;
      color: var(--blue);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 6px;
    }
    .tab-pills {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .tab-btn {
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 6px;
      background: var(--card-bg);
      border: 1px solid var(--border-subtle);
      color: var(--text-muted);
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: var(--font-ui);
    }
    .tab-btn:hover {
      color: var(--text-main);
      border-color: var(--blue);
    }
    .tab-btn.active {
      background: #1e293b;
      border-color: var(--blue);
      color: #fff;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
    }
    .form-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .form-group label {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .form-input {
      background: var(--card-bg);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 6px 10px;
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-main);
      outline: none;
    }
    .form-input:focus {
      border-color: var(--blue);
    }

    /* Multi-Asset Grid */
    .assets-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
    }
    .asset-card {
      background: var(--card-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .asset-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
    }
    .asset-info {
      display: flex;
      flex-direction: column;
    }
    .asset-sym {
      font-size: 15px;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: -0.2px;
    }
    .asset-subname {
      font-size: 11px;
      color: var(--text-muted);
    }
    .asset-price {
      font-family: var(--font-mono);
      font-size: 17px;
      font-weight: 700;
      color: var(--text-main);
      text-align: right;
    }
    .asset-pos-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      border-radius: 6px;
      background: var(--card-bg);
      border: 1px solid var(--border-subtle);
      font-size: 11px;
      font-weight: 600;
    }
    .asset-pos-bar.flat {
      color: var(--text-muted);
    }
    .asset-pos-bar.long {
      background: var(--green-bg);
      color: var(--green);
      border-color: var(--green-border);
    }
    .asset-pos-bar.short {
      background: var(--red-bg);
      color: var(--red);
      border-color: var(--red-border);
    }

    /* ML Dual-Split Probability Bar */
    .prob-container {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .prob-labels {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-family: var(--font-mono);
      font-weight: 600;
    }
    .prob-long-lbl { color: var(--green); }
    .prob-short-lbl { color: var(--red); }
    .prob-bar {
      height: 8px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
      overflow: hidden;
      display: flex;
    }
    .prob-bar-long {
      height: 100%;
      background: var(--green);
      transition: width 0.3s ease;
    }
    .prob-bar-short {
      height: 100%;
      background: var(--red);
      transition: width 0.3s ease;
    }
    .prob-meta {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 10px;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .asset-meta-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      color: var(--text-muted);
      padding-top: 4px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      font-family: var(--font-mono);
    }

    /* 2-Column Split: Key Levels + Macro Intel */
    .intel-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    @media (max-width: 900px) {
      .intel-grid { grid-template-columns: 1fr; }
    }

    /* Tables */
    .table-container {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th {
      text-align: left;
      padding: 8px 10px;
      color: var(--text-dim);
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border-subtle);
    }
    td {
      padding: 8px 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-family: var(--font-mono);
      font-size: 12px;
      color: var(--text-main);
    }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    .tag-sweep {
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 10px;
      font-weight: 700;
      font-family: var(--font-mono);
    }
    .tag-sweep-bull { background: var(--green-bg); color: var(--green); }
    .tag-sweep-bear { background: var(--red-bg); color: var(--red); }

    .pnl-tag {
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: var(--font-mono);
      font-size: 11px;
    }
    .pnl-pos { background: var(--green-bg); color: var(--green); }
    .pnl-neg { background: var(--red-bg); color: var(--red); }

    /* Macro Alpha Grid */
    .alpha-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
    }
    .alpha-tile {
      background: var(--card-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: 6px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 3px;
    }
    .alpha-title {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }
    .alpha-value {
      font-family: var(--font-mono);
      font-weight: 600;
      font-size: 12px;
      color: var(--text-main);
    }
  </style>
</head>
<body>
  <div class="container">

    <!-- Top Navigation Bar -->
    <header>
      <div class="brand">
        <div class="brand-title">
          <span>⚡ APEX</span>
          <span class="brand-dot">·</span>
          <span>INTELLIGENCE ENGINE</span>
        </div>
        <div class="brand-sub">Multi-Asset v7.3</div>
        <div class="status-pills">
          <div class="pill pill-green" id="ws-pill">
            <div class="dot"></div>
            <span>3 SYMBOLS ACTIVE</span>
          </div>
          <div class="pill" id="session-pill">LONDON</div>
          <div class="pill" id="kz-pill" style="display:none; color:var(--amber); border-color:var(--amber-border);">KILL ZONE</div>
          <div class="pill" id="halt-badge" style="display:none; color:var(--red); border-color:var(--red-border);">ENGINE PAUSED</div>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn btn-amber" id="btn-pause" onclick="togglePause()">⏸ Pause Bot</button>
        <button class="btn btn-danger" onclick="flattenAll()">🛑 Flatten All</button>
      </div>
    </header>

    <!-- Prop Firm Challenge Command Card -->
    <section class="card" id="prop-firm-card">
      <div class="pf-header">
        <div class="pf-summary">
          <div>
            <div class="pf-name" id="pf-title">Prop Firm Challenge</div>
            <div style="font-size: 11px; color: var(--text-muted);" id="pf-subtitle">Managed Rule Supervisor</div>
          </div>
          <div class="pf-balance-tag" id="pf-equity">$5,000.00</div>
        </div>
        <div class="pf-controls">
          <button class="btn" id="btn-toggle-drawer" onclick="toggleConfigDrawer()">⚙️ Configure Challenge ▾</button>
          <button class="btn" id="btn-mode-toggle" onclick="togglePropFirmMode()">MODE: ON</button>
          <button class="btn" onclick="resetChallengePrompt()">🔄 Reset</button>
        </div>
      </div>

      <!-- Expandable Configuration Drawer (Phase 1, Phase 2, Funded, Custom) -->
      <div id="pf-drawer">
        <div class="drawer-section-title">1. Select Prop Firm</div>
        <div class="tab-pills" id="firm-tabs">
          <button class="tab-btn active" onclick="selectFirm('sharkfunded')">SharkFunded</button>
          <button class="tab-btn" onclick="selectFirm('ftmo')">FTMO</button>
          <button class="tab-btn" onclick="selectFirm('fundednext')">FundedNext</button>
          <button class="tab-btn" onclick="selectFirm('the5ers')">The 5%ers</button>
          <button class="tab-btn" onclick="selectFirm('funding_pips')">Funding Pips</button>
          <button class="tab-btn" onclick="selectFirm('custom')">Custom Setup</button>
        </div>

        <div class="drawer-section-title" style="margin-top: 6px;">2. Challenge Size</div>
        <div class="tab-pills" id="size-pills">
          <!-- Dynamically populated based on chosen firm -->
        </div>

        <div class="drawer-section-title" style="margin-top: 6px;">3. Challenge Phase / Mode</div>
        <div class="tab-pills" id="phase-pills">
          <!-- Dynamically populated based on chosen firm -->
        </div>

        <div class="drawer-section-title" style="margin-top: 8px;">4. Fine-Tune Parameters (Editable)</div>
        <div class="form-grid">
          <div class="form-group">
            <label>Initial Balance ($)</label>
            <input type="number" class="form-input" id="cfg-balance" value="5000">
          </div>
          <div class="form-group">
            <label>Profit Target (%)</label>
            <input type="number" step="0.5" class="form-input" id="cfg-target" value="9.0">
          </div>
          <div class="form-group">
            <label>Daily Loss Limit (%)</label>
            <input type="number" step="0.25" class="form-input" id="cfg-daily-loss" value="3.0">
          </div>
          <div class="form-group">
            <label>Daily Buffer (%)</label>
            <input type="number" step="0.25" class="form-input" id="cfg-daily-buffer" value="2.25">
          </div>
          <div class="form-group">
            <label>Max Total Loss (%)</label>
            <input type="number" step="0.5" class="form-input" id="cfg-max-loss" value="6.0">
          </div>
          <div class="form-group">
            <label>Total Buffer (%)</label>
            <input type="number" step="0.5" class="form-input" id="cfg-max-buffer" value="4.8">
          </div>
          <div class="form-group">
            <label>Base Risk (%)</label>
            <input type="number" step="0.05" class="form-input" id="cfg-risk" value="0.5">
          </div>
          <div class="form-group">
            <label>Max Concurrent</label>
            <input type="number" class="form-input" id="cfg-concurrent" value="1">
          </div>
        </div>

        <div style="display:flex; justify-content: flex-end; gap: 8px; margin-top: 6px;">
          <button class="btn" onclick="toggleConfigDrawer()">Cancel</button>
          <button class="btn btn-primary" onclick="applyCustomConfig()">✓ Save & Apply Rules</button>
        </div>
      </div>

      <!-- Live Target & Drawdown Meters -->
      <div class="meters-grid">
        <div class="meter-box">
          <div class="meter-top">
            <span class="meter-label">Profit Target</span>
            <span class="meter-val" id="pf-target-text" style="color:var(--green);">+$0.00 / +$450.00 (0.0%)</span>
          </div>
          <div class="meter-bar">
            <div class="meter-fill meter-fill-green" id="pf-target-fill" style="width: 0%;"></div>
          </div>
        </div>

        <div class="meter-box">
          <div class="meter-top">
            <span class="meter-label">Daily Loss Buffer</span>
            <span class="meter-val" id="pf-daily-text" style="color:var(--amber);">-$0.00 / -$150.00 (0% used)</span>
          </div>
          <div class="meter-bar">
            <div class="meter-fill meter-fill-amber" id="pf-daily-fill" style="width: 0%;"></div>
          </div>
        </div>

        <div class="meter-box">
          <div class="meter-top">
            <span class="meter-label">Max Drawdown Stop</span>
            <span class="meter-val" id="pf-maxdd-text" style="color:var(--red);">-$0.00 / -$300.00 (0.0%)</span>
          </div>
          <div class="meter-bar">
            <div class="meter-fill meter-fill-red" id="pf-maxdd-fill" style="width: 0%;"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Multi-Asset Live Cards -->
    <section>
      <div class="card-title" style="margin-bottom: 8px;">
        <span>Multi-Asset Execution Core</span>
        <span class="pill" id="open-risk-pill">OPEN RISK: $0.00 / $50.00 (0/2 MAX)</span>
      </div>
      <div class="assets-grid" id="assets-grid">
        <!-- Rendered dynamically for BTC, ETH, and PAXG (Gold) -->
      </div>
    </section>

    <!-- 2-Column Split: Key Levels & Macro Intelligence -->
    <div class="intel-grid">

      <!-- Institutional Liquidity Levels -->
      <section class="card">
        <div class="card-header">
          <div class="card-title">Institutional Liquidity & Key Levels</div>
          <div style="font-size:11px; color:var(--text-muted);">PDH / PDL / Asian / Fix</div>
        </div>
        <div class="table-container">
          <table>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Prev Day High</th>
                <th>Prev Day Low</th>
                <th>Asian Range</th>
                <th>Sweep / Benchmark</th>
              </tr>
            </thead>
            <tbody id="liq-tbody">
              <!-- Rendered dynamically -->
            </tbody>
          </table>
        </div>
      </section>

      <!-- Macro Alpha & Retrain Status -->
      <section class="card">
        <div class="card-header">
          <div class="card-title">Macro Alpha & Intelligence</div>
          <div class="pill pill-green" id="news-pill">NEWS: CLEAR</div>
        </div>
        <div class="alpha-grid">
          <div class="alpha-tile">
            <div class="alpha-title">Fear & Greed</div>
            <div class="alpha-value" id="fg-val">50 (Neutral)</div>
          </div>
          <div class="alpha-tile">
            <div class="alpha-title">Macro DXY Trend</div>
            <div class="alpha-value" id="dxy-val">USD_NEUTRAL</div>
          </div>
          <div class="alpha-tile">
            <div class="alpha-title">ETH Spillover Bias</div>
            <div class="alpha-value" id="spill-val">NEUTRAL</div>
          </div>
          <div class="alpha-tile">
            <div class="alpha-title">Gold London Fix Bias</div>
            <div class="alpha-value" id="gold-bias-val">NEUTRAL</div>
          </div>
        </div>
        <div style="font-size: 11px; color: var(--text-muted); display:flex; justify-content:space-between; align-items:center; margin-top: 4px;">
          <span id="next-event-text">High-Impact Macro Event in 6800+ mins</span>
          <span id="retrain-text" style="font-family:var(--font-mono);">WF Retrain: Active</span>
        </div>
      </section>

    </div>

    <!-- Trade Execution Journal -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">
          <span>Trade Execution Journal</span>
          <span id="journal-stats" style="color:var(--text-muted); font-size:11px; font-weight:500;">0 Trades · 0W / 0L</span>
        </div>
        <a href="/api/export/journal" download="apex_trade_journal.csv" class="btn" style="text-decoration:none; padding: 4px 10px; font-size:11px;">📥 Export CSV</a>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Time (UTC)</th>
              <th>Action</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>PnL ($)</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody id="journal-tbody">
            <!-- Rendered dynamically -->
          </tbody>
        </table>
      </div>
    </section>

  </div>

  <script>
    let firmTemplates = {};
    let currentFirm = 'sharkfunded';
    let currentSize = 5000;
    let currentPhase = '1-step';
    let lastKnownPrices = { 'BTCUSDT': 0, 'ETHUSDT': 0, 'PAXGUSDT': 0 };
    let isHalted = false;
    let propFirmMode = true;

    // Load templates on boot
    async function loadTemplates() {
      try {
        const res = await fetch('/api/prop_firm/templates');
        firmTemplates = await res.json();
        renderFirmTabs();
        selectFirm('sharkfunded');
      } catch (e) {
        console.error('Failed to load templates:', e);
      }
    }

    function renderFirmTabs() {
      const container = document.getElementById('firm-tabs');
      if (!container || !firmTemplates) return;
      container.innerHTML = '';
      for (const [key, firm] of Object.entries(firmTemplates)) {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${key === currentFirm ? 'active' : ''}`;
        btn.textContent = firm.name;
        btn.onclick = () => selectFirm(key);
        container.appendChild(btn);
      }
    }

    function selectFirm(key) {
      currentFirm = key;
      renderFirmTabs();
      const firm = firmTemplates[key];
      if (!firm) return;

      // Render sizes
      const sizeContainer = document.getElementById('size-pills');
      sizeContainer.innerHTML = '';
      firm.sizes.forEach(sz => {
        const btn = document.createElement('button');
        btn.className = `tab-btn ${sz === currentSize ? 'active' : ''}`;
        btn.textContent = `$${sz.toLocaleString()}`;
        btn.onclick = () => selectSize(sz);
        sizeContainer.appendChild(btn);
      });
      if (!firm.sizes.includes(currentSize)) {
        currentSize = firm.default_size || firm.sizes[0];
      }

      // Render phases
      const phaseContainer = document.getElementById('phase-pills');
      phaseContainer.innerHTML = '';
      const phaseKeys = Object.keys(firm.phases);
      phaseKeys.forEach(pk => {
        const pObj = firm.phases[pk];
        const btn = document.createElement('button');
        btn.className = `tab-btn ${pk === currentPhase ? 'active' : ''}`;
        btn.textContent = pObj.name;
        btn.onclick = () => selectPhase(pk);
        phaseContainer.appendChild(btn);
      });
      if (!firm.phases[currentPhase]) {
        currentPhase = phaseKeys[0];
      }

      updateFormFields();
    }

    function selectSize(sz) {
      currentSize = sz;
      selectFirm(currentFirm);
    }

    function selectPhase(pk) {
      currentPhase = pk;
      selectFirm(currentFirm);
    }

    function updateFormFields() {
      const firm = firmTemplates[currentFirm];
      if (!firm) return;
      const phase = firm.phases[currentPhase];
      if (!phase) return;

      document.getElementById('cfg-balance').value = currentSize;
      document.getElementById('cfg-target').value = phase.profit_target_pct;
      document.getElementById('cfg-daily-loss').value = phase.max_daily_loss_pct;
      document.getElementById('cfg-daily-buffer').value = phase.daily_safety_buffer_pct;
      document.getElementById('cfg-max-loss').value = phase.max_total_loss_pct;
      document.getElementById('cfg-max-buffer').value = phase.total_safety_buffer_pct;
      document.getElementById('cfg-risk').value = phase.base_risk_pct;
      document.getElementById('cfg-concurrent').value = phase.max_concurrent_trades;
    }

    function toggleConfigDrawer() {
      const d = document.getElementById('pf-drawer');
      const btn = document.getElementById('btn-toggle-drawer');
      if (d.style.display === 'flex') {
        d.style.display = 'none';
        btn.textContent = '⚙️ Configure Challenge ▾';
      } else {
        d.style.display = 'flex';
        btn.textContent = '✕ Close Configurator ▴';
      }
    }

    async function applyCustomConfig() {
      const cfg = {
        firm: firmTemplates[currentFirm]?.name || currentFirm,
        phase: firmTemplates[currentFirm]?.phases[currentPhase]?.name || currentPhase,
        initial_balance: parseFloat(document.getElementById('cfg-balance').value),
        profit_target_pct: parseFloat(document.getElementById('cfg-target').value),
        max_daily_loss_pct: parseFloat(document.getElementById('cfg-daily-loss').value),
        daily_safety_buffer_pct: parseFloat(document.getElementById('cfg-daily-buffer').value),
        max_total_loss_pct: parseFloat(document.getElementById('cfg-max-loss').value),
        total_safety_buffer_pct: parseFloat(document.getElementById('cfg-max-buffer').value),
        base_risk_pct: parseFloat(document.getElementById('cfg-risk').value),
        max_concurrent_trades: parseInt(document.getElementById('cfg-concurrent').value, 10),
      };

      try {
        const res = await fetch('/api/settings/prop_firm_custom', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ config: cfg, enabled: true })
        });
        const data = await res.json();
        if (data.success) {
          toggleConfigDrawer();
          updateDashboard();
        } else {
          alert('Failed to apply config: ' + JSON.stringify(data));
        }
      } catch (e) {
        alert('Network error applying config: ' + e);
      }
    }

    async function togglePause() {
      try {
        const res = await fetch('/api/action/toggle_pause', { method: 'POST' });
        const data = await res.json();
        isHalted = data.halted;
        updatePauseButton();
      } catch (e) {
        alert('Error toggling pause: ' + e);
      }
    }

    function updatePauseButton() {
      const btn = document.getElementById('btn-pause');
      const badge = document.getElementById('halt-badge');
      if (isHalted) {
        btn.textContent = '▶ Resume Bot';
        btn.className = 'btn btn-primary';
        badge.style.display = 'inline-flex';
      } else {
        btn.textContent = '⏸ Pause Bot';
        btn.className = 'btn btn-amber';
        badge.style.display = 'none';
      }
    }

    async function togglePropFirmMode() {
      try {
        const res = await fetch('/api/settings/prop_firm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !propFirmMode })
        });
        const data = await res.json();
        propFirmMode = data.enabled;
        document.getElementById('btn-mode-toggle').textContent = propFirmMode ? 'MODE: ON' : 'MODE: OFF';
      } catch (e) {
        alert('Failed to toggle prop firm mode: ' + e);
      }
    }

    async function resetChallengePrompt() {
      if (!confirm('Are you sure you want to RESET the challenge to initial balance ($0 profit / $0 drawdown)?')) return;
      try {
        await fetch('/api/settings/prop_firm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reset: true })
        });
        updateDashboard();
      } catch (e) {
        alert('Failed to reset challenge: ' + e);
      }
    }

    async function flattenAll() {
      if (!confirm('EMERGENCY FLATTEN: Close ALL open positions immediately across all symbols?')) return;
      try {
        await fetch('/api/action/close_all', { method: 'POST' });
        updateDashboard();
      } catch (e) {
        alert('Failed to close positions: ' + e);
      }
    }

    async function closeSymbol(sym) {
      if (!confirm(`Close ${sym} open position at market price?`)) return;
      try {
        await fetch('/api/action/close_symbol', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: sym })
        });
        updateDashboard();
      } catch (e) {
        alert('Failed to close ' + sym + ': ' + e);
      }
    }

    // Main Update Loop
    async function updateDashboard() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();

        // 1. Session & Engine Status
        isHalted = data.halted || false;
        updatePauseButton();
        document.getElementById('session-pill').textContent = data.session_name || 'LONDON';
        const kzPill = document.getElementById('kz-pill');
        if (data.kill_zone) {
          kzPill.textContent = data.kill_zone.replace('_KZ', '') + ' KZ';
          kzPill.style.display = 'inline-flex';
        } else {
          kzPill.style.display = 'none';
        }

        // 2. Prop Firm HUD
        const pf = data.prop_firm;
        propFirmMode = data.prop_firm_mode !== false;
        document.getElementById('btn-mode-toggle').textContent = propFirmMode ? 'MODE: ON' : 'MODE: OFF';

        if (pf && propFirmMode) {
          document.getElementById('pf-title').textContent = pf.challenge_name || 'Prop Firm Challenge';
          document.getElementById('pf-subtitle').textContent = `Target: +${pf.profit_target_pct}% | Daily: -${pf.max_daily_loss_pct}% | Max DD: -${pf.max_dd_limit_pct}%`;
          document.getElementById('pf-equity').textContent = `$${(pf.virtual_balance || pf.initial_balance).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

          const targetPct = Math.min(100, Math.max(0, pf.progress_pct || 0));
          document.getElementById('pf-target-fill').style.width = `${targetPct}%`;
          document.getElementById('pf-target-text').textContent = `+$${(pf.current_profit_usd || 0).toFixed(2)} / +$${(pf.target_usd || 0).toFixed(2)} (${targetPct.toFixed(1)}%)`;

          const dailyUsedPct = Math.min(100, Math.max(0, pf.daily_loss_used_pct || 0));
          document.getElementById('pf-daily-fill').style.width = `${dailyUsedPct}%`;
          document.getElementById('pf-daily-text').textContent = `-$${Math.abs(Math.min(0, pf.daily_pnl_usd || 0)).toFixed(2)} / -$${(pf.daily_limit_usd || 0).toFixed(2)} (${dailyUsedPct.toFixed(0)}% used)`;

          const maxDdPct = Math.min(100, Math.max(0, pf.total_dd_pct || 0));
          document.getElementById('pf-maxdd-fill').style.width = `${(maxDdPct / (pf.max_dd_limit_pct || 10)) * 100}%`;
          document.getElementById('pf-maxdd-text').textContent = `-$${(pf.total_dd_usd || 0).toFixed(2)} / -$${(pf.max_dd_usd || 0).toFixed(2)} (${maxDdPct.toFixed(1)}%)`;
        } else {
          document.getElementById('pf-title').textContent = 'Personal Account Mode';
          document.getElementById('pf-subtitle').textContent = 'Prop firm risk limits disabled · Running live portfolio risk';
          document.getElementById('pf-equity').textContent = `$${(data.account?.balance || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        }

        // 3. Multi-Asset Cards
        const syms = data.symbols || {};
        const assetsContainer = document.getElementById('assets-grid');
        assetsContainer.innerHTML = '';

        let openPositionsCount = 0;
        let totalOpenRisk = 0;

        for (const [sym, info] of Object.entries(syms)) {
          // Never let LTP flicker to $0
          if (info.ltp && info.ltp > 0) {
            lastKnownPrices[sym] = info.ltp;
          }
          const displayPrice = lastKnownPrices[sym] > 0 ? lastKnownPrices[sym] : (info.ltp || 0);

          const isPosition = info.position === 'OPEN';
          if (isPosition) openPositionsCount++;

          const pClass = isPosition ? (info.direction === 'LONG' ? 'long' : 'short') : 'flat';
          const pLabel = isPosition ? `${info.direction} @ $${Number(info.entry_price || 0).toLocaleString()}` : 'FLAT';
          const pnlVal = info.unrealized_pnl || 0;
          const pnlSign = pnlVal >= 0 ? '+' : '';
          const pnlText = isPosition ? `PnL: ${pnlSign}$${pnlVal.toFixed(2)}` : 'Scanning 5m...';

          const pLong = Math.round((info.prob_long || 0.5) * 100);
          const pShort = 100 - pLong;

          let biasTag = 'NEUTRAL';
          if (pLong >= 58) biasTag = 'BULLISH BIAS';
          else if (pShort >= 58) biasTag = 'BEARISH BIAS';

          const subNames = { 'BTCUSDT': 'Bitcoin', 'ETHUSDT': 'Ethereum', 'PAXGUSDT': 'Physical Gold' };

          const card = document.createElement('div');
          card.className = 'asset-card';
          card.innerHTML = `
            <div class="asset-head">
              <div class="asset-info">
                <div class="asset-sym">${sym}</div>
                <div class="asset-subname">${subNames[sym] || 'Perpetual'}</div>
              </div>
              <div class="asset-price">$${Number(displayPrice).toLocaleString(undefined, { minimumFractionDigits: (sym === 'BTCUSDT' ? 1 : 2), maximumFractionDigits: 2 })}</div>
            </div>

            <div class="asset-pos-bar ${pClass}">
              <span>${pLabel}</span>
              <div style="display:flex; align-items:center; gap:8px;">
                <span>${pnlText}</span>
                ${isPosition ? `<button class="btn btn-danger" style="padding:2px 6px; font-size:10px;" onclick="closeSymbol('${sym}')">Close</button>` : ''}
              </div>
            </div>

            <div class="prob-container">
              <div class="prob-labels">
                <span class="prob-long-lbl">Long ${pLong}%</span>
                <span class="prob-short-lbl">Short ${pShort}%</span>
              </div>
              <div class="prob-bar">
                <div class="prob-bar-long" style="width: ${pLong}%;"></div>
                <div class="prob-bar-short" style="width: ${pShort}%;"></div>
              </div>
              <div class="prob-meta">
                <span>Ensemble Consensus</span>
                <span style="font-weight:600; color:${biasTag.includes('BULL') ? 'var(--green)' : biasTag.includes('BEAR') ? 'var(--red)' : 'var(--text-muted)'};">${biasTag}</span>
              </div>
            </div>

            <div class="asset-meta-row">
              <span>REGIME: ${info.regime || 'TRENDING'}</span>
              <span>ATR: $${(info.atr || 0).toFixed(1)}</span>
            </div>
          `;
          assetsContainer.appendChild(card);
        }

        document.getElementById('open-risk-pill').textContent = `POSITIONS: ${openPositionsCount}/2 MAX`;

        // 4. Institutional Liquidity & Key Levels Table
        const liq = data.liquidity || {};
        const liqBody = document.getElementById('liq-tbody');
        liqBody.innerHTML = '';

        for (const [sym, l] of Object.entries(liq)) {
          const pdh = l.pdh > 0 ? `$${l.pdh.toLocaleString()}` : '—';
          const pdl = l.pdl > 0 ? `$${l.pdl.toLocaleString()}` : '—';
          const asian = (l.ash > 0 && l.asl > 0) ? `$${l.asl.toLocaleString()} – $${l.ash.toLocaleString()}` : '—';

          let sweepBadge = '<span style="color:var(--text-dim);">None</span>';
          if (l.sweep_bullish) {
            sweepBadge = `<span class="tag-sweep tag-sweep-bull">SWEEP BULL (${l.sweep_level || 'PDL'})</span>`;
          } else if (l.sweep_bearish) {
            sweepBadge = `<span class="tag-sweep tag-sweep-bear">SWEEP BEAR (${l.sweep_level || 'PDH'})</span>`;
          } else if (l.lbma_pm_fix > 0) {
            sweepBadge = `<span style="color:var(--blue); font-size:11px;">Fix $${l.lbma_pm_fix.toFixed(1)}</span>`;
          }

          const row = document.createElement('tr');
          row.innerHTML = `
            <td style="font-weight:700;">${sym}</td>
            <td style="color:var(--green);">${pdh}</td>
            <td style="color:var(--red);">${pdl}</td>
            <td>${asian}</td>
            <td>${sweepBadge}</td>
          `;
          liqBody.appendChild(row);
        }

        // 5. Macro Intel & Retrain Status
        if (data.intel) {
          document.getElementById('fg-val').textContent = `${data.intel.fear_greed} (${data.intel.fear_greed_label})`;
          if (data.intel.next_event_min) {
            document.getElementById('next-event-text').textContent = `High-Impact Macro Event in ${data.intel.next_event_min} mins`;
          }
        }
        if (data.cross_market) {
          document.getElementById('dxy-val').textContent = data.cross_market.dxy_trend || 'USD_NEUTRAL';
        }
        if (data.lead_lag) {
          document.getElementById('spill-val').textContent = data.lead_lag.eth_spillover_bias || 'NEUTRAL';
          document.getElementById('gold-bias-val').textContent = data.lead_lag.gold_dxy_bias || 'NEUTRAL';
        }

        // Retrain status
        const rSched = data.retrain_status || {};
        const rParts = [];
        for (const [s, st] of Object.entries(rSched)) {
          rParts.push(`${s.replace('USDT', '')}: ${st.next_retrain_in || 2000}c`);
        }
        if (rParts.length > 0) {
          document.getElementById('retrain-text').textContent = `Retrain in ${rParts.join(' · ')}`;
        }

        // 6. Trade Journal
        const jList = data.journal || [];
        document.getElementById('journal-stats').textContent = `${data.account?.total || jList.length} Trades · ${data.account?.wins || 0}W / ${data.account?.losses || 0}L`;

        const jBody = document.getElementById('journal-tbody');
        jBody.innerHTML = '';
        if (jList.length === 0) {
          jBody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--text-dim); padding:16px;">No closed trades recorded yet.</td></tr>';
        } else {
          jList.slice(0, 15).forEach(t => {
            const pnl = Number(t.pnl_usd || 0);
            const pnlTag = pnl >= 0 ? `<span class="pnl-tag pnl-pos">+$${pnl.toFixed(2)}</span>` : `<span class="pnl-tag pnl-neg">-$${Math.abs(pnl).toFixed(2)}</span>`;
            const row = document.createElement('tr');
            row.innerHTML = `
              <td>#${t.id}</td>
              <td style="color:var(--text-muted);">${t.open_ts}</td>
              <td style="font-weight:600;">${t.action}</td>
              <td>$${Number(t.entry_price).toLocaleString()}</td>
              <td>$${Number(t.exit_price).toLocaleString()}</td>
              <td>${pnlTag}</td>
              <td style="color:var(--text-muted); font-size:11px;">${t.exit_reason || ''}</td>
            `;
            jBody.appendChild(row);
          });
        }

      } catch (e) {
        console.error('Update loop error:', e);
      }
    }

    // Startup
    loadTemplates();
    setInterval(updateDashboard, 1500);
    updateDashboard();
  </script>
</body>
</html>
"""

class ApexWebServer:
    """
    High-performance, asynchronous web server providing real-time HUD and interactive control APIs.
    """

    def __init__(
        self,
        state_provider: Callable[[], dict],
        action_handler: Optional[Callable[[str, dict], Any]] = None,
        host: str = "0.0.0.0",
        port: int = 8080
    ):
        self.state_provider = state_provider
        self.action_handler = action_handler
        self.host = host
        self.port = port
        self.app = web.Application()
        self._setup_routes()
        self._runner: web.AppRunner | None = None

    def _setup_routes(self):
        self.app.router.add_get("/", self.handle_index)
        self.app.router.add_get("/api/state", self.handle_api_state)
        self.app.router.add_get("/api/profiles", self.handle_api_profiles)
        self.app.router.add_get("/api/prop_firm/templates", self.handle_api_templates)
        self.app.router.add_post("/api/action/close_all", self.handle_close_all)
        self.app.router.add_post("/api/action/close_symbol", self.handle_close_symbol)
        self.app.router.add_post("/api/action/toggle_pause", self.handle_toggle_pause)
        self.app.router.add_post("/api/settings/prop_firm", self.handle_prop_firm_settings)
        self.app.router.add_post("/api/settings/prop_firm_custom", self.handle_prop_firm_custom)
        self.app.router.add_post("/api/settings/risk", self.handle_risk_settings)
        self.app.router.add_get("/api/export/journal", self.handle_export_journal)
        self.app.router.add_get("/health", self.handle_health)

    async def handle_index(self, request: web.Request) -> web.Response:
        return web.Response(text=DASHBOARD_HTML, content_type="text/html")

    async def handle_api_state(self, request: web.Request) -> web.Response:
        try:
            state = self.state_provider()
            return web.json_response(state)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_profiles(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(PROFILES)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_api_templates(self, request: web.Request) -> web.Response:
        try:
            return web.json_response(FIRM_TEMPLATES)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_close_all(self, request: web.Request) -> web.Response:
        if self.action_handler:
            res = await self.action_handler("close_all", {})
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_close_symbol(self, request: web.Request) -> web.Response:
        if self.action_handler:
            try:
                data = await request.json()
            except Exception:
                data = {}
            res = await self.action_handler("close_symbol", data)
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_toggle_pause(self, request: web.Request) -> web.Response:
        if self.action_handler:
            res = await self.action_handler("toggle_pause", {})
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_prop_firm_settings(self, request: web.Request) -> web.Response:
        if self.action_handler:
            try:
                data = await request.json()
            except Exception:
                data = {}
            res = await self.action_handler("prop_firm_settings", data)
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_prop_firm_custom(self, request: web.Request) -> web.Response:
        if self.action_handler:
            try:
                data = await request.json()
            except Exception:
                data = {}
            res = await self.action_handler("prop_firm_custom", data)
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_risk_settings(self, request: web.Request) -> web.Response:
        if self.action_handler:
            try:
                data = await request.json()
            except Exception:
                data = {}
            res = await self.action_handler("risk_settings", data)
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

    async def handle_export_journal(self, request: web.Request) -> web.Response:
        try:
            state = self.state_provider()
            journal = state.get("journal", [])
            lines = ["ID,Time,Action,Entry Price,Exit Price,PnL USD,Exit Reason"]
            for t in journal:
                lines.append(f"{t.get('id', '')},{t.get('open_ts', '')},{t.get('action', '')},{t.get('entry_price', 0.0)},{t.get('exit_price', 0.0)},{t.get('pnl_usd', 0.0)},\"{t.get('exit_reason', '')}\"")
            csv_content = "\n".join(lines)
            return web.Response(
                text=csv_content,
                content_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="apex_trade_journal.csv"'}
            )
        except Exception as e:
            return web.Response(text=f"Error exporting journal: {e}", status=500)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "service": "apex-intelligence-engine"})

    async def start(self):
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print(f"  ✓ Interactive Web Command Center running at http://{self.host}:{self.port}")

    async def stop(self):
        if self._runner:
            await self._runner.cleanup()
