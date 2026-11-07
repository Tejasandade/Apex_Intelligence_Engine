"""
Apex Intelligence Engine — Interactive Live Web Command Center
==============================================================
Embedded high-performance asynchronous web server using aiohttp.
Provides a real-time multi-asset web UI, interactive parameter controls,
prop firm profile switcher, emergency position manager, and JSON API.
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Callable, Any, Optional
from aiohttp import web
from src.core.prop_firm_manager import PROFILES

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ APEX INTELLIGENCE ENGINE — Command Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0a0e17;
      --card-bg: #111827;
      --card-border: #1f293d;
      --card-hover: #172236;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
      --accent-green: #10b981;
      --accent-green-bg: rgba(16, 185, 129, 0.12);
      --accent-red: #ef4444;
      --accent-red-bg: rgba(239, 68, 68, 0.12);
      --accent-cyan: #06b6d4;
      --accent-gold: #f59e0b;
      --accent-purple: #8b5cf6;
      --font-mono: 'JetBrains Mono', monospace;
      --font-sans: 'Inter', sans-serif;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: var(--font-sans);
      padding: 16px;
      font-size: 13px;
      line-height: 1.5;
    }

    .container {
      max-width: 1320px;
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
      font-weight: 800;
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
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-kz {
      background: rgba(139, 92, 246, 0.18);
      color: #c4b5fd;
      border: 1px solid rgba(139, 92, 246, 0.4);
    }
    .badge-halt {
      background: rgba(239, 68, 68, 0.18);
      color: var(--accent-red);
      border: 1px solid rgba(239, 68, 68, 0.4);
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
    .header-actions {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* Buttons */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      padding: 6px 14px;
      border-radius: 6px;
      border: 1px solid transparent;
      cursor: pointer;
      transition: all 0.15s ease;
      white-space: nowrap;
    }
    .btn:active { transform: translateY(1px); }
    .btn-pause {
      background: rgba(245, 158, 11, 0.15);
      color: var(--accent-gold);
      border-color: rgba(245, 158, 11, 0.3);
    }
    .btn-pause:hover { background: rgba(245, 158, 11, 0.25); }
    .btn-resume {
      background: rgba(16, 185, 129, 0.15);
      color: var(--accent-green);
      border-color: rgba(16, 185, 129, 0.3);
    }
    .btn-resume:hover { background: rgba(16, 185, 129, 0.25); }
    .btn-danger {
      background: rgba(239, 68, 68, 0.18);
      color: var(--accent-red);
      border-color: rgba(239, 68, 68, 0.4);
    }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.35); }
    .btn-sm {
      padding: 3px 8px;
      font-size: 11px;
    }
    .btn-secondary {
      background: #1f293d;
      color: var(--text-main);
      border-color: #374151;
    }
    .btn-secondary:hover { background: #374151; }

    /* Card Styling */
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .card-title {
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Prop Firm Control Panel */
    .prop-firm-panel {
      background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(31, 41, 61, 0.5));
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .pf-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }
    .pf-controls {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .select-custom {
      background: #1e293b;
      color: var(--text-main);
      border: 1px solid #334155;
      font-family: var(--font-mono);
      font-size: 12px;
      padding: 5px 10px;
      border-radius: 6px;
      cursor: pointer;
      outline: none;
    }
    .select-custom:focus { border-color: var(--accent-cyan); }
    .pf-bars-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
    }
    .pf-bar-item {
      background: rgba(0, 0, 0, 0.25);
      border: 1px solid rgba(255, 255, 255, 0.05);
      padding: 10px 12px;
      border-radius: 6px;
    }
    .pf-bar-labels {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-family: var(--font-mono);
      margin-bottom: 5px;
    }
    .progress-bar-bg {
      width: 100%;
      height: 8px;
      background: rgba(255, 255, 255, 0.08);
      border-radius: 4px;
      overflow: hidden;
      position: relative;
    }
    .progress-bar-fill {
      height: 100%;
      border-radius: 4px;
      transition: width 0.3s ease;
    }

    /* Multi-Asset Grid */
    .assets-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 14px;
    }
    .asset-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: border-color 0.2s ease;
    }
    .asset-card:hover { border-color: #374151; }
    .asset-card-header {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    .asset-sym {
      font-size: 16px;
      font-weight: 800;
      font-family: var(--font-mono);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .asset-price {
      font-family: var(--font-mono);
      font-size: 18px;
      font-weight: 700;
      color: var(--accent-cyan);
    }
    .asset-pos-bar {
      padding: 8px 10px;
      border-radius: 6px;
      font-family: var(--font-mono);
      font-size: 12px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .pos-flat { background: rgba(255, 255, 255, 0.03); color: var(--text-muted); }
    .pos-long { background: var(--accent-green-bg); border: 1px solid rgba(16, 185, 129, 0.3); color: var(--accent-green); }
    .pos-short { background: var(--accent-red-bg); border: 1px solid rgba(239, 68, 68, 0.3); color: var(--accent-red); }

    /* Probability Bars */
    .prob-track {
      background: rgba(255, 255, 255, 0.05);
      border-radius: 4px;
      height: 18px;
      display: flex;
      position: relative;
      overflow: hidden;
      margin: 4px 0;
    }
    .prob-fill-l { background: var(--accent-green); height: 100%; transition: width 0.3s; }
    .prob-fill-s { background: var(--accent-red); height: 100%; transition: width 0.3s; margin-left: auto; }
    .prob-label-row {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
      font-family: var(--font-mono);
    }

    /* Diagnostics Tags */
    .tags-row {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      font-size: 11px;
      font-family: var(--font-mono);
    }
    .tag {
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 2px 6px;
      border-radius: 4px;
    }

    /* Key Levels Table */
    .table-container {
      width: 100%;
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--font-mono);
      font-size: 12px;
    }
    th, td {
      padding: 8px 10px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }
    th {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 600;
    }
    tr:hover td { background: rgba(255, 255, 255, 0.02); }

    /* Utility */
    .green { color: var(--accent-green); }
    .red { color: var(--accent-red); }
    .cyan { color: var(--accent-cyan); }
    .gold { color: var(--accent-gold); }
    .muted { color: var(--text-muted); }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header>
      <div class="brand">
        <h1>⚡ <span>APEX</span> COMMAND CENTER <span style="font-size:12px; font-weight:500; color:var(--text-muted);">v7.2 Multi-Asset</span></h1>
        <span class="badge badge-live"><span class="pulse-dot"></span> LIVE ● 3 ASSETS</span>
        <span id="session-badge" class="badge badge-kz">SESSION: --</span>
        <span id="halt-badge" class="badge badge-halt" style="display:none;">⛔ HALTED</span>
      </div>
      <div class="header-actions">
        <button id="btn-toggle-pause" class="btn btn-pause" onclick="togglePause()">⏸ PAUSE BOT</button>
        <button class="btn btn-danger" onclick="closeAllPositions()">🚨 FLATTEN ALL</button>
      </div>
    </header>

    <!-- Prop Firm Challenge Management Panel -->
    <section class="prop-firm-panel">
      <div class="pf-header">
        <div style="display:flex; align-items:center; gap:8px;">
          <span style="font-size:16px;">🏆</span>
          <div>
            <div style="font-size:13px; font-weight:700;" id="pf-title">PROP FIRM CHALLENGE</div>
            <div style="font-size:11px; color:var(--text-muted);" id="pf-subtitle">Managed Rule Supervisor</div>
          </div>
        </div>
        <div class="pf-controls">
          <label style="font-size:11px; font-weight:600; color:var(--text-muted);">PROFILE:</label>
          <select id="select-profile" class="select-custom" onchange="onProfileChange(this.value)">
            <!-- Options dynamically loaded -->
          </select>
          <button class="btn btn-secondary btn-sm" onclick="togglePropFirmMode()" id="btn-toggle-pf">MODE: ON</button>
          <button class="btn btn-secondary btn-sm" onclick="resetChallenge()" title="Reset challenge metrics to 0">↺ RESET</button>
        </div>
      </div>

      <div class="pf-bars-grid">
        <!-- Target Progress -->
        <div class="pf-bar-item">
          <div class="pf-bar-labels">
            <span>TARGET PROGRESS</span>
            <span id="pf-target-text" class="green">+$0.00 / +$450.00 (0.0%)</span>
          </div>
          <div class="progress-bar-bg">
            <div id="pf-target-bar" class="progress-bar-fill" style="background:var(--accent-green); width:0%;"></div>
          </div>
        </div>

        <!-- Daily Loss Drawdown -->
        <div class="pf-bar-item">
          <div class="pf-bar-labels">
            <span>DAILY LOSS BUFFER</span>
            <span id="pf-daily-text" class="gold">-$0.00 / -$150.00 (0% used)</span>
          </div>
          <div class="progress-bar-bg">
            <div id="pf-daily-bar" class="progress-bar-fill" style="background:var(--accent-gold); width:0%;"></div>
          </div>
        </div>

        <!-- Max Overall Drawdown -->
        <div class="pf-bar-item">
          <div class="pf-bar-labels">
            <span>MAX DRAWDOWN HARD STOP</span>
            <span id="pf-maxdd-text" class="red">-$0.00 / -$300.00 (0% used)</span>
          </div>
          <div class="progress-bar-bg">
            <div id="pf-maxdd-bar" class="progress-bar-fill" style="background:var(--accent-red); width:0%;"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Concurrent Multi-Asset Cards -->
    <section>
      <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px; display:flex; justify-content:space-between;">
        <span>CONCURRENT MULTI-ASSET MATRIX</span>
        <span id="risk-pool-text" class="cyan">Open Risk: $0.00 / $50.00 (0/2 max)</span>
      </div>
      <div class="assets-grid" id="assets-container">
        <!-- Asset cards rendered dynamically -->
      </div>
    </section>

    <!-- Institutional Market Intelligence & Key Levels -->
    <section style="display:grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap:14px;">
      <!-- Key Levels Table -->
      <div class="card">
        <div class="card-title">
          <span>INSTITUTIONAL LIQUIDITY & KEY LEVELS</span>
          <span class="muted">PDH / PDL / ASIAN / SWEEPS</span>
        </div>
        <div class="table-container">
          <table id="levels-table">
            <thead>
              <tr>
                <th>SYMBOL</th>
                <th>PREV DAY HIGH</th>
                <th>PREV DAY LOW</th>
                <th>ASIAN RANGE</th>
                <th>SWEEP / FVG</th>
              </tr>
            </thead>
            <tbody id="levels-tbody">
              <!-- Populated dynamically -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- Macro Sentiment & Cross-Market -->
      <div class="card">
        <div class="card-title">
          <span>MACRO INTELLIGENCE & CROSS-ASSET ALPHA</span>
          <span id="news-status" class="green">✓ NEWS CLEAR</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
          <div class="tags-row">
            <div class="tag">F&G: <span id="fg-val" class="green">50 (Neutral)</span></div>
            <div class="tag">Macro DXY: <span id="dxy-val" class="muted">USD_NEUTRAL</span></div>
            <div class="tag">ETH Spillover: <span id="spill-val" class="muted">NEUTRAL</span></div>
            <div class="tag">Gold Bias: <span id="gold-bias-val" class="muted">NEUTRAL</span></div>
          </div>
          <div style="font-size:11px; color:var(--text-muted);" id="next-event-text">Next Macro Event: Checking...</div>
          <div style="margin-top:6px;">
            <div style="font-size:10px; color:var(--text-muted); font-weight:700; text-transform:uppercase; margin-bottom:4px;">Multi-Asset Walk-Forward Retrain Cycles</div>
            <div class="tags-row" id="retrain-tags">
              <!-- Populated dynamically -->
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- Real-Time Trade Journal Table -->
    <section class="card">
      <div class="card-title">
        <span>TRADE EXECUTION JOURNAL</span>
        <span class="muted" id="journal-stats">0 Trades | 0W / 0L</span>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>TIME (UTC)</th>
              <th>SYMBOL</th>
              <th>ACTION</th>
              <th>ENTRY</th>
              <th>EXIT</th>
              <th>PNL ($)</th>
              <th>EXIT REASON</th>
            </tr>
          </thead>
          <tbody id="journal-tbody">
            <tr><td colspan="8" class="muted" style="text-align:center;">Monitoring live market for high-conviction entries...</td></tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    let isHalted = false;
    let propFirmMode = true;
    let profilesList = {};

    async function fetchProfiles() {
      try {
        const res = await fetch('/api/profiles');
        profilesList = await res.json();
        const sel = document.getElementById('select-profile');
        sel.innerHTML = Object.keys(profilesList).map(k => {
          return `<option value="${k}">${profilesList[k].name}</option>`;
        }).join('');
      } catch (e) {
        console.error('Failed to load profiles:', e);
      }
    }

    async function updateDashboard() {
      try {
        const res = await fetch('/api/state');
        const data = await res.json();
        renderHeader(data);
        renderPropFirm(data);
        renderAssets(data);
        renderIntelligence(data);
        renderJournal(data);
      } catch (err) {
        console.error('State poll error:', err);
      }
    }

    function renderHeader(data) {
      isHalted = data.halted || false;
      const haltBadge = document.getElementById('halt-badge');
      const pauseBtn = document.getElementById('btn-toggle-pause');
      if (isHalted) {
        haltBadge.style.display = 'inline-flex';
        haltBadge.textContent = '⛔ ' + (data.halt_reason || 'HALTED');
        pauseBtn.className = 'btn btn-resume';
        pauseBtn.textContent = '▶ RESUME BOT';
      } else {
        haltBadge.style.display = 'none';
        pauseBtn.className = 'btn btn-pause';
        pauseBtn.textContent = '⏸ PAUSE BOT';
      }

      const sess = data.session_name || 'LONDON';
      const kz = data.kill_zone ? ` [${data.kill_zone.replace('_KZ', '')}]` : '';
      document.getElementById('session-badge').textContent = `SESSION: ${sess}${kz}`;
    }

    function renderPropFirm(data) {
      const pf = data.prop_firm;
      propFirmMode = data.prop_firm_mode !== false;
      document.getElementById('btn-toggle-pf').textContent = propFirmMode ? 'MODE: ON' : 'MODE: OFF';
      document.getElementById('btn-toggle-pf').style.color = propFirmMode ? 'var(--accent-green)' : 'var(--text-muted)';

      if (!pf) {
        document.getElementById('pf-title').textContent = 'PERSONAL ACCOUNT MODE';
        document.getElementById('pf-subtitle').textContent = 'Risk scaled to real account balance';
        return;
      }

      document.getElementById('pf-title').textContent = pf.challenge_name || 'PROP FIRM CHALLENGE';
      const sel = document.getElementById('select-profile');
      if (sel && sel.value !== pf.profile_key) {
        sel.value = pf.profile_key;
      }

      // Target
      const targetPct = Math.min(100, Math.max(0, pf.progress_pct || 0));
      document.getElementById('pf-target-bar').style.width = targetPct + '%';
      document.getElementById('pf-target-text').textContent = `+$${pf.current_profit_usd.toFixed(2)} / +$${pf.target_usd.toFixed(2)} (${targetPct.toFixed(1)}%)`;

      // Daily
      const dailyPct = Math.min(100, Math.max(0, pf.daily_loss_used_pct || 0));
      document.getElementById('pf-daily-bar').style.width = dailyPct + '%';
      document.getElementById('pf-daily-text').textContent = `-$${Math.abs(Math.min(0, pf.daily_pnl_usd)).toFixed(2)} / -$${pf.daily_limit_usd.toFixed(2)} (${dailyPct.toFixed(0)}% used)`;

      // Max DD
      const maxDDPct = Math.min(100, Math.max(0, (pf.total_dd_usd / pf.max_dd_usd) * 100));
      document.getElementById('pf-maxdd-bar').style.width = maxDDPct + '%';
      document.getElementById('pf-maxdd-text').textContent = `-$${pf.total_dd_usd.toFixed(2)} / -$${pf.max_dd_usd.toFixed(2)} (${pf.total_dd_pct.toFixed(1)}%)`;
    }

    function renderAssets(data) {
      const cont = document.getElementById('assets-container');
      const syms = data.symbols || {};
      let totalOpen = 0;
      let totalRisk = 0;

      let html = '';
      for (const [sym, info] of Object.entries(syms)) {
        const isPos = info.position === 'OPEN';
        if (isPos) {
          totalOpen++;
          totalRisk += 25.0; // standard parity
        }

        const isLong = info.direction === 'LONG';
        const pClass = isPos ? (isLong ? 'pos-long' : 'pos-short') : 'pos-flat';
        const pnl = Number(info.unrealized_pnl || 0);
        const pnlStr = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);

        const pL = (Number(info.prob_long || 0.5) * 100).toFixed(1);
        const pS = (Number(info.prob_short || 0.5) * 100).toFixed(1);

        html += `
          <div class="asset-card">
            <div class="asset-card-header">
              <div class="asset-sym">${sym}</div>
              <div class="asset-price">$${Number(info.ltp || 0).toLocaleString()}</div>
            </div>
            <div class="asset-pos-bar ${pClass}">
              <span>${isPos ? `${info.direction} (${pnlStr})` : 'FLAT'}</span>
              ${isPos ? `<button class="btn btn-danger btn-sm" onclick="closeSymbol('${sym}')">CLOSE</button>` : `<span class="muted">Scanning 5m...</span>`}
            </div>
            <div>
              <div class="prob-label-row">
                <span class="green">LONG ${pL}%</span>
                <span class="red">SHORT ${pS}%</span>
              </div>
              <div class="prob-track">
                <div class="prob-fill-l" style="width:${pL}%;"></div>
                <div class="prob-fill-s" style="width:${pS}%;"></div>
              </div>
            </div>
            <div class="tags-row">
              <div class="tag">REGIME: <span>${info.regime || 'UNKNOWN'}</span></div>
              <div class="tag">ATR: <span>$${Number(info.atr || 0).toFixed(1)}</span></div>
            </div>
          </div>
        `;
      }
      cont.innerHTML = html;
      document.getElementById('risk-pool-text').textContent = `Open Risk: $${totalRisk.toFixed(2)} / $50.00 (${totalOpen}/2 max)`;
    }

    function renderIntelligence(data) {
      // Key levels table
      const tbody = document.getElementById('levels-tbody');
      const liq = data.liquidity || {};
      let rows = '';
      for (const [sym, lev] of Object.entries(liq)) {
        const pdh = lev.pdh ? `$${lev.pdh.toLocaleString()}` : '$0';
        const pdl = lev.pdl ? `$${lev.pdl.toLocaleString()}` : '$0';
        const asian = (lev.ash && lev.asl) ? `$${lev.asl.toLocaleString()} - $${lev.ash.toLocaleString()}` : '$0 - $0';
        let sweepTag = '<span class="muted">NONE</span>';
        if (lev.sweep_bullish) sweepTag = `<span class="green">SWEEP_BULL (${lev.sweep_level})</span>`;
        else if (lev.sweep_bearish) sweepTag = `<span class="red">SWEEP_BEAR (${lev.sweep_level})</span>`;
        else if (lev.fvg_bullish) sweepTag = '<span class="cyan">FVG_BULL</span>';
        else if (lev.fvg_bearish) sweepTag = '<span class="gold">FVG_BEAR</span>';

        rows += `
          <tr>
            <td style="font-weight:700;">${sym}</td>
            <td class="green">${pdh}</td>
            <td class="red">${pdl}</td>
            <td>${asian}</td>
            <td>${sweepTag}</td>
          </tr>
        `;
      }
      tbody.innerHTML = rows || '<tr><td colspan="5" class="muted">Calculating key liquidity levels...</td></tr>';

      // Cross market
      if (data.cross_market) {
        document.getElementById('dxy-val').textContent = data.cross_market.dxy_trend || 'USD_NEUTRAL';
      }
      if (data.lead_lag) {
        document.getElementById('spill-val').textContent = data.lead_lag.eth_spillover_bias || 'NEUTRAL';
        document.getElementById('gold-bias-val').textContent = data.lead_lag.gold_dxy_bias || 'NEUTRAL';
      }
      if (data.intel) {
        document.getElementById('fg-val').textContent = `${data.intel.fear_greed} (${data.intel.fear_greed_label})`;
        if (data.intel.next_event_min) {
          document.getElementById('next-event-text').textContent = `High-Impact Macro Event in ${data.intel.next_event_min} mins`;
        }
      }

      // Retrain tags
      const rTags = document.getElementById('retrain-tags');
      const rSched = data.retrain_status || {};
      rTags.innerHTML = Object.entries(rSched).map(([s, stat]) => {
        const nxt = stat.is_retraining ? 'RETRAINING' : `${stat.next_retrain_in}c`;
        return `<div class="tag">${s}: <span class="cyan">${nxt}</span> (#${stat.candles_accumulated || 0})</div>`;
      }).join('');
    }

    function renderJournal(data) {
      const tbody = document.getElementById('journal-tbody');
      const list = data.journal || [];
      if (list.length === 0) {
        tbody.innerHTML = '<tr><td colspan="8" class="muted" style="text-align:center;">Monitoring live market for high-conviction entries...</td></tr>';
        return;
      }

      tbody.innerHTML = list.slice(0, 15).map(t => {
        const p = Number(t.pnl_usd || 0);
        const pCol = p > 0 ? 'green' : (p < 0 ? 'red' : 'muted');
        const sign = p > 0 ? '+' : '';
        const sym = t.symbol || 'BTCUSDT';
        return `
          <tr>
            <td>#${t.id || 1}</td>
            <td>${t.open_ts || '--:--'}</td>
            <td style="font-weight:700;">${sym}</td>
            <td class="${t.action === 'BUY' ? 'green' : 'red'}" style="font-weight:700;">${t.action}</td>
            <td>$${Number(t.entry_price || 0).toLocaleString()}</td>
            <td>$${Number(t.exit_price || 0).toLocaleString()}</td>
            <td class="${pCol}">${sign}$${p.toFixed(2)}</td>
            <td class="muted">${t.exit_reason || 'TAKE PROFIT'}</td>
          </tr>
        `;
      }).join('');
    }

    // Action API Calls
    async function togglePause() {
      try {
        const res = await fetch('/api/action/toggle_pause', { method: 'POST' });
        const d = await res.json();
        updateDashboard();
      } catch (e) {
        alert('Failed to toggle bot pause: ' + e);
      }
    }

    async function closeAllPositions() {
      if (!confirm('Are you sure you want to FLATTEN ALL open positions across all assets?')) return;
      try {
        const res = await fetch('/api/action/close_all', { method: 'POST' });
        const d = await res.json();
        alert('All positions closed: ' + d.message);
        updateDashboard();
      } catch (e) {
        alert('Error closing positions: ' + e);
      }
    }

    async function closeSymbol(sym) {
      if (!confirm(`Close ${sym} position immediately?`)) return;
      try {
        const res = await fetch('/api/action/close_symbol', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ symbol: sym })
        });
        const d = await res.json();
        updateDashboard();
      } catch (e) {
        alert('Error closing ' + sym + ': ' + e);
      }
    }

    async function onProfileChange(val) {
      try {
        await fetch('/api/settings/prop_firm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ profile: val, enabled: true })
        });
        updateDashboard();
      } catch (e) {
        alert('Failed to change profile: ' + e);
      }
    }

    async function togglePropFirmMode() {
      try {
        await fetch('/api/settings/prop_firm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled: !propFirmMode })
        });
        updateDashboard();
      } catch (e) {
        alert('Failed to toggle prop firm mode: ' + e);
      }
    }

    async function resetChallenge() {
      if (!confirm('Reset challenge balance and stats to initial baseline?')) return;
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

    // Startup
    fetchProfiles();
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
        self.app.router.add_post("/api/action/close_all", self.handle_close_all)
        self.app.router.add_post("/api/action/close_symbol", self.handle_close_symbol)
        self.app.router.add_post("/api/action/toggle_pause", self.handle_toggle_pause)
        self.app.router.add_post("/api/settings/prop_firm", self.handle_prop_firm_settings)
        self.app.router.add_post("/api/settings/risk", self.handle_risk_settings)
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

    async def handle_risk_settings(self, request: web.Request) -> web.Response:
        if self.action_handler:
            try:
                data = await request.json()
            except Exception:
                data = {}
            res = await self.action_handler("risk_settings", data)
            return web.json_response(res)
        return web.json_response({"error": "No action handler configured"}, status=501)

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
