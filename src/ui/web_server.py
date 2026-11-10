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
  <title>APEX Institutional Trading Core</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      /* Surface & Canvas Tokens (True Obsidian Dark Palette - Impeccable Standard) */
      --canvas: #090b0e;
      --surface: #0f1217;
      --surface-subtle: #141820;
      --surface-elevated: #1a202c;
      --surface-hover: #1f2634;
      
      /* Hairline Structural Borders */
      --border-hairline: rgba(255, 255, 255, 0.06);
      --border-subtle: rgba(255, 255, 255, 0.10);
      --border-focus: #2962ff;
      
      /* Typography */
      --font-ui: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      --font-mono: 'JetBrains Mono', 'SF Mono', Menlo, Consolas, monospace;
      
      --text-primary: #f0f3f6;
      --text-secondary: #8c96a5;
      --text-muted: #525c6a;
      --text-dim: #3b434e;
      
      /* Institutional TradingView-Calibrated Status Accents */
      --green: #089981;
      --green-soft: rgba(8, 153, 129, 0.12);
      --green-border: rgba(8, 153, 129, 0.25);
      
      --red: #f23645;
      --red-soft: rgba(242, 54, 69, 0.12);
      --red-border: rgba(242, 54, 69, 0.25);
      
      --amber: #f59e0b;
      --amber-soft: rgba(245, 158, 11, 0.12);
      --amber-border: rgba(245, 158, 11, 0.25);
      
      --blue: #2962ff;
      --blue-soft: rgba(41, 98, 255, 0.12);
      
      /* Concentric Geometry (make-interfaces-feel-better principle) */
      --radius-lg: 8px;
      --radius-md: 6px;
      --radius-sm: 4px;
      
      /* Layered Elevation Shadow */
      --shadow-surface: 0 1px 3px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.03);
      --shadow-dropdown: 0 12px 28px rgba(0, 0, 0, 0.65), 0 0 0 1px var(--border-hairline);
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }

    body {
      background-color: var(--canvas);
      color: var(--text-primary);
      font-family: var(--font-ui);
      font-size: 13px;
      line-height: 1.5;
      padding: 16px 20px 32px 20px;
    }

    .tabular, .num, td, input, .metric-val, .badge-mono {
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    /* SVG Icon Utility */
    .icon {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 14px;
      height: 14px;
      flex-shrink: 0;
      stroke: currentColor;
      fill: none;
      vertical-align: middle;
    }

    /* Toast Notification Center */
    #toast-container {
      position: fixed;
      top: 16px;
      right: 20px;
      z-index: 99999;
      display: flex;
      flex-direction: column;
      gap: 8px;
      pointer-events: none;
      max-width: 380px;
    }
    .toast {
      pointer-events: auto;
      background: var(--surface-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-md);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.7);
      padding: 10px 14px;
      display: flex;
      align-items: flex-start;
      gap: 10px;
      font-size: 12px;
      color: var(--text-primary);
      animation: toastSlide 0.25s cubic-bezier(0.16, 1, 0.3, 1);
      transition: opacity 0.2s ease, transform 0.2s ease;
    }
    .toast.toast-success {
      border-color: var(--green-border);
      background: #0d1a16;
    }
    .toast.toast-danger {
      border-color: var(--red-border);
      background: #1c0e12;
    }
    .toast.toast-info {
      border-color: var(--blue-soft);
      background: #0e1524;
    }
    .toast-title {
      font-weight: 600;
      margin-bottom: 2px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .toast-body {
      color: var(--text-secondary);
      font-size: 12px;
    }
    @keyframes toastSlide {
      from { opacity: 0; transform: translateX(20px); }
      to { opacity: 1; transform: translateX(0); }
    }

    /* Top Masthead */
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 4px 0 14px 0;
      border-bottom: 1px solid var(--border-hairline);
      flex-wrap: wrap;
      gap: 12px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .brand-mark {
      width: 24px;
      height: 24px;
      background: var(--surface-subtle);
      border: 1px solid var(--border-subtle);
      border-radius: var(--radius-sm);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--blue);
    }
    .brand-title {
      font-size: 14px;
      font-weight: 700;
      letter-spacing: -0.2px;
      color: var(--text-primary);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .brand-title span.sub {
      font-weight: 400;
      color: var(--text-secondary);
    }
    .brand-version {
      font-size: 11px;
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    .status-group {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 500;
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      color: var(--text-secondary);
    }
    .status-badge.live {
      color: var(--green);
      background: var(--green-soft);
      border-color: var(--green-border);
    }
    .pulse-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 6px currentColor;
    }

    .header-actions {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Buttons with tactile active scale */
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      padding: 6px 12px;
      font-size: 12px;
      font-weight: 500;
      border-radius: var(--radius-md);
      cursor: pointer;
      border: 1px solid var(--border-hairline);
      background: var(--surface);
      color: var(--text-primary);
      font-family: var(--font-ui);
      user-select: none;
      transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.08s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .btn:hover {
      background: var(--surface-hover);
      border-color: var(--border-subtle);
    }
    .btn:active {
      transform: scale(0.97);
    }
    .btn-amber {
      color: var(--amber);
      background: var(--amber-soft);
      border-color: var(--amber-border);
    }
    .btn-amber:hover {
      background: rgba(245, 158, 11, 0.2);
    }
    .btn-danger {
      color: var(--red);
      background: var(--red-soft);
      border-color: var(--red-border);
    }
    .btn-danger:hover {
      background: rgba(242, 54, 69, 0.2);
    }
    .btn-primary {
      background: var(--blue);
      border-color: #3b71fe;
      color: #fff;
    }
    .btn-primary:hover {
      background: #1b52e8;
    }

    /* Surfaces / Panels */
    .surface-panel {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-surface);
      overflow: hidden;
    }

    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 0 2px 2px 2px;
    }
    .section-label {
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--text-muted);
      display: flex;
      align-items: center;
      gap: 6px;
    }

    /* Institutional Prop Firm Cockpit Bar */
    .cockpit-bar {
      display: flex;
      flex-direction: column;
    }
    .cockpit-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 12px 16px;
      border-bottom: 1px solid var(--border-hairline);
      flex-wrap: wrap;
      gap: 12px;
      background: var(--surface-subtle);
    }
    .cockpit-info {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .cockpit-name {
      font-size: 14px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .cockpit-rules {
      font-size: 11px;
      color: var(--text-secondary);
      font-family: var(--font-mono);
    }
    .cockpit-equity {
      font-family: var(--font-mono);
      font-size: 14px;
      font-weight: 600;
      color: var(--green);
      background: var(--green-soft);
      padding: 3px 8px;
      border-radius: var(--radius-sm);
      border: 1px solid var(--green-border);
    }
    .cockpit-controls {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    /* Unified Horizontal Metric Columns */
    .cockpit-metrics {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
    }
    @media (max-width: 820px) {
      .cockpit-metrics { grid-template-columns: 1fr; }
      .metric-col { border-right: none !important; border-bottom: 1px solid var(--border-hairline); }
    }
    .metric-col {
      padding: 12px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-right: 1px solid var(--border-hairline);
    }
    .metric-col:last-child {
      border-right: none;
    }
    .metric-top {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
    }
    .metric-label {
      font-weight: 500;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }
    .metric-val {
      font-weight: 600;
    }
    .meter-track {
      height: 3px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 2px;
      overflow: hidden;
      position: relative;
    }
    .meter-fill {
      height: 100%;
      border-radius: 2px;
      transition: width 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    /* Accordion Configurator Drawer */
    #pf-drawer, #pf-chart-drawer {
      display: none;
      padding: 16px;
      background: #0b0e13;
      border-bottom: 1px solid var(--border-hairline);
      flex-direction: column;
      gap: 14px;
      animation: drawerSlide 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes drawerSlide {
      from { opacity: 0; transform: translateY(-6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .drawer-label {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.3px;
      margin-bottom: 6px;
    }
    .segment-group {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .segment-btn {
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 500;
      border-radius: var(--radius-sm);
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      color: var(--text-secondary);
      cursor: pointer;
      transition: color 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
      font-family: var(--font-ui);
    }
    .segment-btn:hover {
      color: var(--text-primary);
      border-color: var(--border-subtle);
    }
    .segment-btn.active {
      background: var(--surface-elevated);
      border-color: var(--blue);
      color: #fff;
    }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }
    .form-item {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .form-item label {
      font-size: 11px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .form-input {
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-sm);
      padding: 6px 8px;
      font-size: 12px;
      color: var(--text-primary);
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    .form-input:focus {
      border-color: var(--blue);
      box-shadow: 0 0 0 2px var(--blue-soft);
    }

    /* Multi-Asset Grid */
    .assets-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 12px;
    }
    .asset-card {
      background: var(--surface);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow-surface);
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: border-color 0.15s ease;
    }
    .asset-card:hover {
      border-color: var(--border-subtle);
    }
    .asset-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .asset-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--text-primary);
      letter-spacing: -0.2px;
    }
    .asset-desc {
      font-size: 11px;
      color: var(--text-muted);
    }
    
    /* Header Right: Price + Sparkline Container */
    .price-sparkline-group {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .asset-price {
      font-size: 16px;
      font-weight: 600;
      color: var(--text-primary);
      text-align: right;
    }
    .sparkline-svg {
      width: 88px;
      height: 26px;
      overflow: visible;
    }

    /* MTF 4-Timeframe Matrix Strip */
    .mtf-matrix {
      display: flex;
      gap: 5px;
      align-items: center;
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-sm);
      padding: 4px 8px;
    }
    .mtf-cell {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      font-size: 10px;
      font-family: var(--font-mono);
      font-weight: 600;
      padding: 1px 5px;
      border-radius: 3px;
      color: var(--text-secondary);
    }
    .mtf-cell.bull {
      background: var(--green-soft);
      color: var(--green);
    }
    .mtf-cell.bear {
      background: var(--red-soft);
      color: var(--red);
    }
    .mtf-cell.neut {
      background: rgba(255, 255, 255, 0.04);
      color: var(--text-muted);
    }

    /* Position Status Strip */
    .pos-strip {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 6px 10px;
      border-radius: var(--radius-md);
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      font-size: 11px;
      font-weight: 500;
    }
    .pos-strip.flat {
      color: var(--text-muted);
    }
    .pos-strip.long {
      background: var(--green-soft);
      color: var(--green);
      border-color: var(--green-border);
    }
    .pos-strip.short {
      background: var(--red-soft);
      color: var(--red);
      border-color: var(--red-border);
    }

    /* Quant Center-Zero Delta Gauge */
    .conviction-module {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .conviction-header {
      display: flex;
      justify-content: space-between;
      font-size: 11px;
    }
    .conviction-label {
      color: var(--text-secondary);
      font-weight: 500;
    }
    .conviction-tag {
      font-weight: 600;
      font-family: var(--font-mono);
      font-size: 11px;
    }
    .delta-gauge-container {
      position: relative;
      height: 4px;
      background: rgba(255, 255, 255, 0.06);
      border-radius: 2px;
      overflow: hidden;
    }
    .delta-center-line {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      width: 2px;
      background: rgba(255, 255, 255, 0.2);
      transform: translateX(-50%);
      z-index: 2;
    }
    .delta-bar-long {
      position: absolute;
      left: 50%;
      top: 0;
      bottom: 0;
      background: var(--green);
      border-radius: 0 2px 2px 0;
      transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .delta-bar-short {
      position: absolute;
      right: 50%;
      top: 0;
      bottom: 0;
      background: var(--red);
      border-radius: 2px 0 0 2px;
      transition: width 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .conviction-footer {
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    .asset-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 6px;
      border-top: 1px solid var(--border-hairline);
      font-size: 11px;
      color: var(--text-muted);
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

    /* Tables with Tabular Figures */
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
      padding: 9px 14px;
      color: var(--text-muted);
      font-size: 11px;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.3px;
      border-bottom: 1px solid var(--border-hairline);
      background: var(--surface-subtle);
    }
    td {
      padding: 9px 14px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      color: var(--text-primary);
      font-size: 12px;
    }
    tr:last-child td {
      border-bottom: none;
    }
    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .pill-tag {
      padding: 2px 6px;
      border-radius: var(--radius-sm);
      font-size: 11px;
      font-weight: 500;
      display: inline-block;
    }
    .pill-tag.bull {
      background: var(--green-soft);
      color: var(--green);
    }
    .pill-tag.bear {
      background: var(--red-soft);
      color: var(--red);
    }

    /* Macro Alpha Grid */
    .alpha-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 8px;
      padding: 14px 16px;
    }
    .alpha-card {
      background: var(--surface-subtle);
      border: 1px solid var(--border-hairline);
      border-radius: var(--radius-md);
      padding: 10px 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .alpha-title {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
      font-weight: 500;
    }
    .alpha-data {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-primary);
    }
    .alpha-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      border-top: 1px solid var(--border-hairline);
      font-size: 11px;
      color: var(--text-secondary);
      background: var(--surface-subtle);
    }
  </style>
</head>
<body>
  <!-- Toast Notification Center -->
  <div id="toast-container"></div>

  <div class="container">

    <!-- Top Masthead -->
    <header>
      <div class="brand">
        <div class="brand-mark">
          <svg class="icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
        </div>
        <div class="brand-title">
          <span>APEX</span>
          <span class="sub">INTELLIGENCE ENGINE</span>
        </div>
        <div class="brand-version">v7.3</div>
        <div class="status-group">
          <div class="status-badge live" id="ws-badge">
            <div class="pulse-dot"></div>
            <span>3 ASSETS LIVE</span>
          </div>
          <div class="status-badge" id="session-badge">LONDON</div>
          <div class="status-badge" id="kz-badge" style="display:none; color:var(--amber); border-color:var(--amber-border);">KILL ZONE</div>
          <div class="status-badge" id="halt-badge" style="display:none; color:var(--red); border-color:var(--red-border);">ENGINE PAUSED</div>
        </div>
      </div>
      <div class="header-actions">
        <button class="btn" id="btn-audio" onclick="toggleAudio()">
          <svg class="icon" id="audio-icon" viewBox="0 0 24 24"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          <span id="audio-text">Audio: ON</span>
        </button>
        <button class="btn btn-amber" id="btn-pause" onclick="togglePause()">
          <svg class="icon" id="pause-icon" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1"></rect><rect x="14" y="4" width="4" height="16" rx="1"></rect></svg>
          <span id="pause-text">Pause Bot</span>
        </button>
        <button class="btn btn-danger" onclick="flattenAll()">
          <svg class="icon" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"></line></svg>
          <span>Flatten All</span>
        </button>
      </div>
    </header>

    <!-- Prop Firm Cockpit Bar (Unified, NO nested cards) -->
    <section class="surface-panel cockpit-bar" id="prop-firm-card">
      <div class="cockpit-top">
        <div class="cockpit-info">
          <div>
            <div class="cockpit-name" id="pf-title">Prop Firm Challenge</div>
            <div class="cockpit-rules" id="pf-subtitle">Target: +9% | Daily: -3% | Max DD: -6%</div>
          </div>
          <div class="cockpit-equity" id="pf-equity">$5,000.00</div>
        </div>
        <div class="cockpit-controls">
          <button class="btn" id="btn-toggle-chart" onclick="toggleChartDrawer()">
            <svg class="icon" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
            <span id="chart-btn-text">Equity Trajectory ▾</span>
          </button>
          <button class="btn" id="btn-toggle-drawer" onclick="toggleConfigDrawer()">
            <svg class="icon" viewBox="0 0 24 24"><line x1="4" y1="21" x2="4" y2="14"></line><line x1="4" y1="10" x2="4" y2="3"></line><line x1="12" y1="21" x2="12" y2="12"></line><line x1="12" y1="8" x2="12" y2="3"></line><line x1="20" y1="21" x2="20" y2="16"></line><line x1="20" y1="12" x2="20" y2="3"></line><line x1="1" y1="14" x2="7" y2="14"></line><line x1="9" y1="8" x2="15" y2="8"></line><line x1="17" y1="16" x2="23" y2="16"></line></svg>
            <span id="drawer-btn-text">Configure Challenge</span>
            <svg class="icon" style="width:10px; height:10px;" viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
          <button class="btn" id="btn-mode-toggle" onclick="togglePropFirmMode()">MODE: ON</button>
          <button class="btn" onclick="resetChallengePrompt()">
            <svg class="icon" viewBox="0 0 24 24"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
            <span>Reset</span>
          </button>
        </div>
      </div>

      <!-- Expandable Equity Curve Trajectory Drawer -->
      <div id="pf-chart-drawer">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <div class="drawer-label" style="margin:0;">Virtual Equity Trajectory vs. Challenge Bounds</div>
          <div style="font-size:11px; color:var(--text-secondary); display:flex; gap:12px;" class="tabular">
            <span style="color:var(--green);">Target Ceiling: <span id="chart-tgt-lbl">$5,450.00</span></span>
            <span style="color:var(--text-muted);">Baseline: <span id="chart-base-lbl">$5,000.00</span></span>
            <span style="color:var(--red);">Max DD Stop: <span id="chart-dd-lbl">$4,700.00</span></span>
          </div>
        </div>
        <div style="position:relative; width:100%; height:130px; background:#080a0e; border:1px solid var(--border-hairline); border-radius:var(--radius-sm); overflow:hidden;">
          <svg id="equity-svg" width="100%" height="130" style="display:block;"></svg>
        </div>
      </div>

      <!-- Expandable Challenge Configurator Drawer -->
      <div id="pf-drawer">
        <div class="drawer-label">1. Select Prop Firm</div>
        <div class="segment-group" id="firm-tabs"></div>

        <div class="drawer-label" style="margin-top: 4px;">2. Challenge Size</div>
        <div class="segment-group" id="size-pills"></div>

        <div class="drawer-label" style="margin-top: 4px;">3. Challenge Phase / Tier</div>
        <div class="segment-group" id="phase-pills"></div>

        <div class="drawer-label" style="margin-top: 6px;">4. Exact Rule Parameters (Editable)</div>
        <div class="form-grid">
          <div class="form-item">
            <label>Initial Balance ($)</label>
            <input type="number" class="form-input tabular" id="cfg-balance" value="5000">
          </div>
          <div class="form-item">
            <label>Profit Target (%)</label>
            <input type="number" step="0.5" class="form-input tabular" id="cfg-target" value="9.0">
          </div>
          <div class="form-item">
            <label>Daily Loss Limit (%)</label>
            <input type="number" step="0.25" class="form-input tabular" id="cfg-daily-loss" value="3.0">
          </div>
          <div class="form-item">
            <label>Daily Buffer (%)</label>
            <input type="number" step="0.25" class="form-input tabular" id="cfg-daily-buffer" value="2.25">
          </div>
          <div class="form-item">
            <label>Max Total Loss (%)</label>
            <input type="number" step="0.5" class="form-input tabular" id="cfg-max-loss" value="6.0">
          </div>
          <div class="form-item">
            <label>Total Buffer (%)</label>
            <input type="number" step="0.5" class="form-input tabular" id="cfg-max-buffer" value="4.8">
          </div>
          <div class="form-item">
            <label>Base Risk (%)</label>
            <input type="number" step="0.05" class="form-input tabular" id="cfg-risk" value="0.5">
          </div>
          <div class="form-item">
            <label>Max Concurrent</label>
            <input type="number" class="form-input tabular" id="cfg-concurrent" value="1">
          </div>
        </div>

        <div style="display:flex; justify-content: flex-end; gap: 8px; margin-top: 4px;">
          <button class="btn" onclick="toggleConfigDrawer()">Cancel</button>
          <button class="btn btn-primary" onclick="applyCustomConfig()">Save & Apply Rules</button>
        </div>
      </div>

      <!-- Unified 3-Column Metrics -->
      <div class="cockpit-metrics">
        <div class="metric-col">
          <div class="metric-top">
            <span class="metric-label">Profit Target</span>
            <span class="metric-val tabular" id="pf-target-text" style="color:var(--green);">+$0.00 / +$450.00 (0.0%)</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill" id="pf-target-fill" style="width: 0%; background: var(--green);"></div>
          </div>
        </div>

        <div class="metric-col">
          <div class="metric-top">
            <span class="metric-label">Daily Loss Buffer</span>
            <span class="metric-val tabular" id="pf-daily-text" style="color:var(--amber);">-$0.00 / -$150.00 (0% used)</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill" id="pf-daily-fill" style="width: 0%; background: var(--amber);"></div>
          </div>
        </div>

        <div class="metric-col">
          <div class="metric-top">
            <span class="metric-label">Max Drawdown Stop</span>
            <span class="metric-val tabular" id="pf-maxdd-text" style="color:var(--red);">-$0.00 / -$300.00 (0.0%)</span>
          </div>
          <div class="meter-track">
            <div class="meter-fill" id="pf-maxdd-fill" style="width: 0%; background: var(--red);"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- Multi-Asset Execution Core -->
    <section>
      <div class="section-header" style="margin-bottom: 8px;">
        <div class="section-label">
          <span>Multi-Asset Execution Core</span>
        </div>
        <div class="status-badge tabular" id="open-risk-badge">POSITIONS: 0/2 MAX</div>
      </div>
      <div class="assets-grid" id="assets-grid">
        <!-- Rendered reactively (BTC, ETH, PAXG) without wiping innerHTML -->
      </div>
    </section>

    <!-- 2-Column Split: Key Levels + Macro Intelligence -->
    <div class="intel-grid">

      <!-- Institutional Liquidity & Key Levels -->
      <section class="surface-panel">
        <div style="padding: 12px 16px; border-bottom: 1px solid var(--border-hairline); display:flex; justify-content:space-between; align-items:center;">
          <div class="section-label">Institutional Key Levels & Liquidity</div>
          <div style="font-size:11px; color:var(--text-muted); font-family:var(--font-mono);">PDH / PDL / Asian / Fix</div>
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
              <!-- Rendered reactively -->
            </tbody>
          </table>
        </div>
      </section>

      <!-- Macro Alpha & Retrain Status -->
      <section class="surface-panel" style="display:flex; flex-direction:column; justify-content:space-between;">
        <div>
          <div style="padding: 12px 16px; border-bottom: 1px solid var(--border-hairline); display:flex; justify-content:space-between; align-items:center;">
            <div class="section-label">Macro Alpha & Intelligence</div>
            <div class="status-badge live" id="news-badge">NEWS: CLEAR</div>
          </div>
          <div class="alpha-grid">
            <div class="alpha-card">
              <div class="alpha-title">Fear & Greed</div>
              <div class="alpha-data tabular" id="fg-val">50 (Neutral)</div>
            </div>
            <div class="alpha-card">
              <div class="alpha-title">Macro DXY Trend</div>
              <div class="alpha-data tabular" id="dxy-val">USD_NEUTRAL</div>
            </div>
            <div class="alpha-card">
              <div class="alpha-title">ETH Spillover Bias</div>
              <div class="alpha-data tabular" id="spill-val">NEUTRAL</div>
            </div>
            <div class="alpha-card">
              <div class="alpha-title">Gold London Fix Bias</div>
              <div class="alpha-data tabular" id="gold-bias-val">NEUTRAL</div>
            </div>
          </div>
        </div>
        <div class="alpha-footer">
          <span id="next-event-text">High-Impact Event: None scheduled</span>
          <span id="retrain-text" class="tabular">WF Retrain: Active</span>
        </div>
      </section>

    </div>

    <!-- Trade Execution Journal -->
    <section class="surface-panel">
      <div style="padding: 12px 16px; border-bottom: 1px solid var(--border-hairline); display:flex; justify-content:space-between; align-items:center;">
        <div class="section-label">
          <span>Trade Execution Journal</span>
          <span id="journal-stats" style="color:var(--text-muted); font-size:11px; font-weight:400; margin-left:6px;" class="tabular">0 Trades · 0W / 0L</span>
        </div>
        <a href="/api/export/journal" download="apex_trade_journal.csv" class="btn" style="text-decoration:none; padding: 4px 10px; font-size:11px;">
          <svg class="icon" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
          <span>Export CSV</span>
        </a>
      </div>
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Time</th>
              <th>Symbol</th>
              <th>Action</th>
              <th>Entry Price</th>
              <th>Exit Price</th>
              <th>PnL ($)</th>
              <th>Exit Reason</th>
            </tr>
          </thead>
          <tbody id="journal-tbody">
            <!-- Rendered reactively -->
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
    let audioEnabled = true;

    // State tracking for audio alerts & toasts
    let knownPositions = {};
    let lastKnownTradeCount = null;
    let sharedAudioCtx = null;

    function getAudioContext() {
      if (!sharedAudioCtx) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) sharedAudioCtx = new AudioCtx();
      }
      if (sharedAudioCtx && sharedAudioCtx.state === 'suspended') {
        sharedAudioCtx.resume().catch(() => {});
      }
      return sharedAudioCtx;
    }

    // Auto-unlock audio on first user interaction
    ['click', 'touchstart', 'keydown'].forEach(evt => {
      document.addEventListener(evt, () => getAudioContext(), { passive: true });
    });

    function toggleAudio() {
      audioEnabled = !audioEnabled;
      const text = document.getElementById('audio-text');
      const icon = document.getElementById('audio-icon');
      if (audioEnabled) {
        text.textContent = 'Audio: ON';
        icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>';
        showToast('AUDIO ALERTS ENABLED', 'Execution chimes active', 'info');
        playChime(true); // Test chime to confirm audio works
      } else {
        text.textContent = 'Audio: OFF';
        icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line>';
      }
    }

    function playChime(isWin = true) {
      if (!audioEnabled) return;
      try {
        const ctx = getAudioContext();
        if (!ctx) return;
        if (ctx.state === 'suspended') {
          ctx.resume().then(() => _emitChime(ctx, isWin)).catch(() => {});
        } else {
          _emitChime(ctx, isWin);
        }
      } catch (e) {}
    }

    function _emitChime(ctx, isWin) {
      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.type = 'sine';
      osc2.type = 'triangle';

      if (isWin) {
        // Melodic ascending major chime: C5 -> G5
        osc1.frequency.setValueAtTime(523.25, now);
        osc1.frequency.exponentialRampToValueAtTime(783.99, now + 0.12);
        osc2.frequency.setValueAtTime(659.25, now);
        osc2.frequency.exponentialRampToValueAtTime(1046.50, now + 0.15);
      } else {
        // Muted descending tone: A4 -> F4
        osc1.frequency.setValueAtTime(440.00, now);
        osc1.frequency.exponentialRampToValueAtTime(329.63, now + 0.14);
        osc2.frequency.setValueAtTime(349.23, now);
        osc2.frequency.exponentialRampToValueAtTime(261.63, now + 0.14);
      }

      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.linearRampToValueAtTime(0.28, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.45);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.45);
      osc2.stop(now + 0.45);
    }

    function showToast(title, body, type = 'info') {
      const container = document.getElementById('toast-container');
      if (!container) return;
      const t = document.createElement('div');
      t.className = `toast toast-${type}`;
      t.innerHTML = `
        <div>
          <div class="toast-title">${title}</div>
          <div class="toast-body">${body}</div>
        </div>
      `;
      container.appendChild(t);
      setTimeout(() => {
        t.style.opacity = '0';
        t.style.transform = 'translateX(20px)';
        setTimeout(() => t.remove(), 250);
      }, 4500);
    }

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
        btn.className = `segment-btn ${key === currentFirm ? 'active' : ''}`;
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

      const sizeContainer = document.getElementById('size-pills');
      sizeContainer.innerHTML = '';
      firm.sizes.forEach(sz => {
        const btn = document.createElement('button');
        btn.className = `segment-btn ${sz === currentSize ? 'active' : ''}`;
        btn.textContent = `$${sz.toLocaleString()}`;
        btn.onclick = () => selectSize(sz);
        sizeContainer.appendChild(btn);
      });
      if (!firm.sizes.includes(currentSize)) {
        currentSize = firm.default_size || firm.sizes[0];
      }

      const phaseContainer = document.getElementById('phase-pills');
      phaseContainer.innerHTML = '';
      const phaseKeys = Object.keys(firm.phases);
      phaseKeys.forEach(pk => {
        const pObj = firm.phases[pk];
        const btn = document.createElement('button');
        btn.className = `segment-btn ${pk === currentPhase ? 'active' : ''}`;
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
      const text = document.getElementById('drawer-btn-text');
      if (d.style.display === 'flex') {
        d.style.display = 'none';
        text.textContent = 'Configure Challenge';
      } else {
        d.style.display = 'flex';
        text.textContent = 'Close Configurator';
      }
    }

    function toggleChartDrawer() {
      const d = document.getElementById('pf-chart-drawer');
      const text = document.getElementById('chart-btn-text');
      if (d.style.display === 'flex') {
        d.style.display = 'none';
        text.textContent = 'Equity Trajectory ▾';
      } else {
        d.style.display = 'flex';
        text.textContent = 'Hide Trajectory ▴';
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
          showToast('RULES UPDATED', `${cfg.firm} ${cfg.phase} applied`, 'success');
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
        showToast(isHalted ? 'BOT PAUSED' : 'BOT RESUMED', isHalted ? 'Execution halted' : 'Scanning active', isHalted ? 'danger' : 'success');
      } catch (e) {
        alert('Error toggling pause: ' + e);
      }
    }

    function updatePauseButton() {
      const btn = document.getElementById('btn-pause');
      const text = document.getElementById('pause-text');
      const icon = document.getElementById('pause-icon');
      const badge = document.getElementById('halt-badge');
      if (isHalted) {
        text.textContent = 'Resume Bot';
        btn.className = 'btn btn-primary';
        icon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"></polygon>';
        badge.style.display = 'inline-flex';
      } else {
        text.textContent = 'Pause Bot';
        btn.className = 'btn btn-amber';
        icon.innerHTML = '<rect x="6" y="4" width="4" height="16" rx="1"></rect><rect x="14" y="4" width="4" height="16" rx="1"></rect>';
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
        showToast('CHALLENGE RESET', 'Balances and loss buffers cleared', 'info');
        updateDashboard();
      } catch (e) {
        alert('Failed to reset challenge: ' + e);
      }
    }

    async function flattenAll() {
      if (!confirm('EMERGENCY FLATTEN: Close ALL open positions immediately across all symbols?')) return;
      try {
        await fetch('/api/action/close_all', { method: 'POST' });
        showToast('EMERGENCY FLATTEN', 'All open positions closed at market', 'danger');
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
        showToast('POSITION CLOSED', `Closed ${sym} market order`, 'info');
        updateDashboard();
      } catch (e) {
        alert('Failed to close ' + sym + ': ' + e);
      }
    }

    // SVG Sparkline Generator
    function renderSparkline(svgElement, dataPoints) {
      if (!svgElement || !dataPoints || dataPoints.length < 2) return;
      const w = 88;
      const h = 26;
      const min = Math.min(...dataPoints);
      const max = Math.max(...dataPoints);
      const range = max - min || 1;
      const isUp = dataPoints[dataPoints.length - 1] >= dataPoints[0];
      const strokeColor = isUp ? 'var(--green)' : 'var(--red)';

      const coords = dataPoints.map((val, idx) => {
        const x = (idx / (dataPoints.length - 1)) * w;
        const y = h - ((val - min) / range) * (h - 6) - 3;
        return [x, y];
      });

      const lineD = 'M ' + coords.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L ');
      const areaD = lineD + ` L ${w},${h} L 0,${h} Z`;

      svgElement.innerHTML = `
        <defs>
          <linearGradient id="sg-${svgElement.id}" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="${strokeColor}" stop-opacity="0.25"/>
            <stop offset="100%" stop-color="${strokeColor}" stop-opacity="0.0"/>
          </linearGradient>
        </defs>
        <path d="${areaD}" fill="url(#sg-${svgElement.id})" />
        <path d="${lineD}" fill="none" stroke="${strokeColor}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" />
      `;
    }

    // SVG Equity Curve Generator
    function renderEquityChart(svgElement, pts, initialBal, targetBal, maxLossBal) {
      if (!svgElement) return;
      const w = svgElement.clientWidth || 800;
      const h = 130;
      svgElement.setAttribute('viewBox', `0 0 ${w} ${h}`);

      const allVals = [...(pts || [initialBal]), targetBal, maxLossBal, initialBal];
      const min = Math.min(...allVals) * 0.998;
      const max = Math.max(...allVals) * 1.002;
      const range = max - min || 1;

      const getY = (val) => h - ((val - min) / range) * (h - 24) - 12;

      const yTgt = getY(targetBal);
      const yBase = getY(initialBal);
      const yLoss = getY(maxLossBal);

      const stepX = w / Math.max(1, (pts.length - 1));
      const curveCoords = pts.map((val, idx) => [idx * stepX, getY(val)]);
      const lineD = 'M ' + curveCoords.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' L ');
      const lastPt = curveCoords[curveCoords.length - 1];

      svgElement.innerHTML = `
        <!-- Reference Target Line -->
        <line x1="0" y1="${yTgt}" x2="${w}" y2="${yTgt}" stroke="var(--green)" stroke-dasharray="4,4" stroke-width="1" opacity="0.6"/>
        <text x="${w - 8}" y="${yTgt - 4}" fill="var(--green)" font-size="9" text-anchor="end" font-family="var(--font-mono)">Target Ceiling</text>

        <!-- Reference Baseline -->
        <line x1="0" y1="${yBase}" x2="${w}" y2="${yBase}" stroke="rgba(255,255,255,0.2)" stroke-dasharray="2,2" stroke-width="1"/>
        
        <!-- Reference Stop Line -->
        <line x1="0" y1="${yLoss}" x2="${w}" y2="${yLoss}" stroke="var(--red)" stroke-dasharray="4,4" stroke-width="1" opacity="0.6"/>
        <text x="${w - 8}" y="${yLoss + 10}" fill="var(--red)" font-size="9" text-anchor="end" font-family="var(--font-mono)">Max DD Stop</text>

        <!-- Equity Curve Path -->
        <path d="${lineD}" fill="none" stroke="var(--blue)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <circle cx="${lastPt[0]}" cy="${lastPt[1]}" r="3.5" fill="var(--blue)" stroke="#fff" stroke-width="1.5"/>
      `;
    }

    // Reactive Asset Card Map
    const renderedCards = {};

    function getOrCreateAssetCard(sym, container) {
      if (renderedCards[sym]) return renderedCards[sym];

      const subNames = { 'BTCUSDT': 'Bitcoin Perpetual', 'ETHUSDT': 'Ethereum Perpetual', 'PAXGUSDT': 'Physical Gold Spot/Perp' };
      const card = document.createElement('div');
      card.className = 'asset-card';
      card.id = `card-${sym}`;
      card.innerHTML = `
        <div class="asset-header">
          <div>
            <div class="asset-title">${sym}</div>
            <div class="asset-desc">${subNames[sym] || 'Perpetual'}</div>
          </div>
          <div class="price-sparkline-group">
            <svg class="sparkline-svg" id="spark-${sym}"></svg>
            <div class="asset-price tabular" id="price-${sym}">$0.00</div>
          </div>
        </div>

        <!-- 4-Timeframe Structure Matrix (1D / 4H / 1H / 5m) -->
        <div class="mtf-matrix" id="mtf-${sym}">
          <span style="font-size:10px; color:var(--text-muted); font-weight:600; margin-right:2px;">MTF:</span>
          <span class="mtf-cell neut" id="mtf-1d-${sym}">1D —</span>
          <span class="mtf-cell neut" id="mtf-4h-${sym}">4H —</span>
          <span class="mtf-cell neut" id="mtf-1h-${sym}">1H —</span>
          <span class="mtf-cell neut" id="mtf-5m-${sym}">5m —</span>
        </div>

        <div class="pos-strip flat" id="pos-strip-${sym}">
          <span id="pos-label-${sym}">FLAT</span>
          <div style="display:flex; align-items:center; gap:8px;">
            <span id="pos-pnl-${sym}" class="tabular">Scanning 5m...</span>
            <button class="btn btn-danger" id="pos-close-${sym}" style="display:none; padding:2px 6px; font-size:10px;" onclick="closeSymbol('${sym}')">Close</button>
          </div>
        </div>

        <!-- Center-Zero Delta Bias Gauge -->
        <div class="conviction-module">
          <div class="conviction-header">
            <span class="conviction-label">Model Convexity & Order Flow</span>
            <span class="conviction-tag" id="bias-tag-${sym}">NEUTRAL</span>
          </div>
          <div class="delta-gauge-container">
            <div class="delta-center-line"></div>
            <div class="delta-bar-long" id="gauge-long-${sym}" style="width: 0%;"></div>
            <div class="delta-bar-short" id="gauge-short-${sym}" style="width: 0%;"></div>
          </div>
          <div class="conviction-footer">
            <span id="prob-long-${sym}">P(L) 0.50</span>
            <span id="prob-short-${sym}">P(S) 0.50</span>
          </div>
        </div>

        <div class="asset-footer">
          <span id="regime-${sym}" class="tabular">REGIME: TRENDING</span>
          <span id="atr-${sym}" class="tabular">ATR: $0.0</span>
        </div>
      `;
      container.appendChild(card);
      renderedCards[sym] = card;
      return card;
    }

    // Main Reactive Update Loop
    async function updateDashboard() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();

        // 1. Session & Engine Status
        isHalted = data.halted || false;
        updatePauseButton();
        document.getElementById('session-badge').textContent = data.session_name || 'LONDON';
        const kzBadge = document.getElementById('kz-badge');
        if (data.kill_zone) {
          kzBadge.textContent = data.kill_zone.replace('_KZ', '') + ' KZ';
          kzBadge.style.display = 'inline-flex';
        } else {
          kzBadge.style.display = 'none';
        }

        // 2. Prop Firm Cockpit HUD & Equity Trajectory
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

          // Equity Trajectory SVG
          const eqPts = pf.equity_curve || [pf.initial_balance, pf.virtual_balance];
          const tgtBal = pf.initial_balance + (pf.target_usd || pf.initial_balance * 0.09);
          const ddBal = pf.initial_balance - (pf.max_dd_usd || pf.initial_balance * 0.06);
          document.getElementById('chart-tgt-lbl').textContent = `$${tgtBal.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
          document.getElementById('chart-base-lbl').textContent = `$${pf.initial_balance.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
          document.getElementById('chart-dd-lbl').textContent = `$${ddBal.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
          renderEquityChart(document.getElementById('equity-svg'), eqPts, pf.initial_balance, tgtBal, ddBal);
        } else {
          document.getElementById('pf-title').textContent = 'Personal Account Mode';
          document.getElementById('pf-subtitle').textContent = 'Prop firm risk limits disabled · Running live portfolio risk';
          document.getElementById('pf-equity').textContent = `$${(data.account?.balance || 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
        }

        // 3. Multi-Asset Reactive Updates (Sparklines, MTF Badges, Positions)
        const syms = data.symbols || {};
        const assetsContainer = document.getElementById('assets-grid');
        let openPositionsCount = 0;

        for (const [sym, info] of Object.entries(syms)) {
          getOrCreateAssetCard(sym, assetsContainer);

          if (info.ltp && info.ltp > 0) {
            lastKnownPrices[sym] = info.ltp;
          }
          const displayPrice = lastKnownPrices[sym] > 0 ? lastKnownPrices[sym] : (info.ltp || 0);
          document.getElementById(`price-${sym}`).textContent = `$${Number(displayPrice).toLocaleString(undefined, { minimumFractionDigits: (sym === 'BTCUSDT' ? 1 : 2), maximumFractionDigits: 2 })}`;

          // Render Sparkline
          if (info.sparkline && info.sparkline.length >= 2) {
            renderSparkline(document.getElementById(`spark-${sym}`), info.sparkline);
          }

          // Render MTF Matrix Badges (1D / 4H / 1H / 5m)
          const mtf = info.mtf || {};
          const m1d = document.getElementById(`mtf-1d-${sym}`);
          const m4h = document.getElementById(`mtf-4h-${sym}`);
          const m1h = document.getElementById(`mtf-1h-${sym}`);
          const m5m = document.getElementById(`mtf-5m-${sym}`);

          if (m1d) {
            const isBull = mtf['1d'] === 'BULL';
            const isBear = mtf['1d'] === 'BEAR';
            m1d.className = `mtf-cell ${isBull ? 'bull' : (isBear ? 'bear' : 'neut')}`;
            m1d.textContent = `1D ${isBull ? '▲' : (isBear ? '▼' : '—')}`;
          }
          if (m4h) {
            const isBull = mtf['4h'] === 'BULL';
            const isBear = mtf['4h'] === 'BEAR';
            m4h.className = `mtf-cell ${isBull ? 'bull' : (isBear ? 'bear' : 'neut')}`;
            m4h.textContent = `4H ${isBull ? '▲' : (isBear ? '▼' : '—')}`;
          }
          if (m1h) {
            const isBull = mtf['1h'] === 'BULL';
            const isBear = mtf['1h'] === 'BEAR';
            m1h.className = `mtf-cell ${isBull ? 'bull' : (isBear ? 'bear' : 'neut')}`;
            m1h.textContent = `1H ${isBull ? '▲' : (isBear ? '▼' : '—')}`;
          }
          if (m5m) {
            const isBull = mtf['5m'] === 'BULL';
            const isBear = mtf['5m'] === 'BEAR';
            m5m.className = `mtf-cell ${isBull ? 'bull' : (isBear ? 'bear' : 'neut')}`;
            m5m.textContent = `5m ${isBull ? '▲' : (isBear ? '▼' : '—')}`;
          }

          // Check Position Transitions for Toasts & Chimes
          const isPosition = info.position === 'OPEN';
          if (isPosition) openPositionsCount++;

          const prevPos = knownPositions[sym];
          if (!prevPos && isPosition) {
            // New position opened!
            showToast('ORDER FILLED', `${info.direction} ${sym} @ $${Number(info.entry_price).toLocaleString()}`, 'info');
            playChime(true);
          }
          knownPositions[sym] = isPosition;

          const posStrip = document.getElementById(`pos-strip-${sym}`);
          posStrip.className = `pos-strip ${isPosition ? (info.direction === 'LONG' ? 'long' : 'short') : 'flat'}`;
          document.getElementById(`pos-label-${sym}`).textContent = isPosition ? `${info.direction} @ $${Number(info.entry_price || 0).toLocaleString()}` : 'FLAT';

          const pnlVal = info.unrealized_pnl || 0;
          const pnlSign = pnlVal >= 0 ? '+' : '';
          const pnlEl = document.getElementById(`pos-pnl-${sym}`);
          pnlEl.textContent = isPosition ? `PnL: ${pnlSign}$${pnlVal.toFixed(2)}` : 'Scanning 5m setup...';
          pnlEl.style.color = isPosition ? (pnlVal > 0 ? 'var(--green)' : (pnlVal < 0 ? 'var(--red)' : 'var(--text-primary)')) : '';
          pnlEl.style.fontWeight = isPosition ? '700' : '';
          document.getElementById(`pos-close-${sym}`).style.display = isPosition ? 'inline-block' : 'none';

          // Quant Center-Zero Delta Gauge (uses real independent prob_long & prob_short edge across all symbols)
          const pLong = Number(info.prob_long ?? 0.5);
          const pShort = Number(info.prob_short ?? (1.0 - pLong));
          document.getElementById(`prob-long-${sym}`).textContent = `P(L) ${pLong.toFixed(2)}`;
          document.getElementById(`prob-short-${sym}`).textContent = `P(S) ${pShort.toFixed(2)}`;

          const gLong = document.getElementById(`gauge-long-${sym}`);
          const gShort = document.getElementById(`gauge-short-${sym}`);
          const bTag = document.getElementById(`bias-tag-${sym}`);
          const edge = pLong - pShort;

          if (edge >= 0.015) {
            const deltaPct = Math.min(50, Math.max(4, edge * 250));
            gLong.style.width = `${deltaPct}%`;
            gShort.style.width = `0%`;
            bTag.textContent = `+${Math.round(deltaPct * 2)}% ${edge >= 0.04 ? 'BULLISH BIAS' : 'SLIGHT BULL'}`;
            bTag.style.color = 'var(--green)';
          } else if (edge <= -0.015) {
            const deltaPct = Math.min(50, Math.max(4, Math.abs(edge) * 250));
            gShort.style.width = `${deltaPct}%`;
            gLong.style.width = `0%`;
            bTag.textContent = `-${Math.round(deltaPct * 2)}% ${edge <= -0.04 ? 'BEARISH BIAS' : 'SLIGHT BEAR'}`;
            bTag.style.color = 'var(--red)';
          } else {
            gLong.style.width = `0%`;
            gShort.style.width = `0%`;
            bTag.textContent = 'NEUTRAL CONSENSUS';
            bTag.style.color = 'var(--text-muted)';
          }

          document.getElementById(`regime-${sym}`).textContent = `REGIME: ${info.regime || 'TRENDING'}`;
          document.getElementById(`atr-${sym}`).textContent = `ATR: $${(info.atr || 0).toFixed(1)}`;
        }

        document.getElementById('open-risk-badge').textContent = `POSITIONS: ${openPositionsCount}/2 MAX`;

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
            sweepBadge = `<span class="pill-tag bull">Sweep Bull (${l.sweep_level || 'PDL'})</span>`;
          } else if (l.sweep_bearish) {
            sweepBadge = `<span class="pill-tag bear">Sweep Bear (${l.sweep_level || 'PDH'})</span>`;
          } else if (l.lbma_pm_fix > 0) {
            sweepBadge = `<span style="color:var(--blue); font-size:11px;" class="tabular">Fix $${l.lbma_pm_fix.toFixed(1)}</span>`;
          }

          const row = document.createElement('tr');
          row.innerHTML = `
            <td style="font-weight:600;">${sym}</td>
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
            document.getElementById('next-event-text').textContent = `High-Impact Event in ${data.intel.next_event_min} mins`;
          }
        }
        if (data.cross_market) {
          document.getElementById('dxy-val').textContent = data.cross_market.dxy_trend || 'USD_NEUTRAL';
        }
        if (data.lead_lag) {
          document.getElementById('spill-val').textContent = data.lead_lag.eth_spillover_bias || 'NEUTRAL';
          document.getElementById('gold-bias-val').textContent = data.lead_lag.gold_dxy_bias || 'NEUTRAL';
        }

        const rSched = data.retrain_status || {};
        const rParts = [];
        for (const [s, st] of Object.entries(rSched)) {
          rParts.push(`${s.replace('USDT', '')}: ${st.next_retrain_in || 2000}c`);
        }
        if (rParts.length > 0) {
          document.getElementById('retrain-text').textContent = `Retrain in ${rParts.join(' · ')}`;
        }

        // 6. Trade Journal & Exit Detection
        const jList = data.journal || [];
        document.getElementById('journal-stats').textContent = `${data.account?.total || jList.length} Trades · ${data.account?.wins || 0}W / ${data.account?.losses || 0}L`;

        if (lastKnownTradeCount !== null && jList.length > lastKnownTradeCount) {
          const newest = jList[0];
          if (newest) {
            const pnlNum = Number(newest.pnl_usd || 0);
            const isWin = pnlNum >= 0;
            showToast(isWin ? 'TAKE PROFIT HIT' : 'STOP LOSS HIT', `${newest.action}: ${isWin ? '+' : ''}$${pnlNum.toFixed(2)} (${newest.exit_reason || ''})`, isWin ? 'success' : 'danger');
            playChime(isWin);
          }
        }
        lastKnownTradeCount = jList.length;

        const jBody = document.getElementById('journal-tbody');
        jBody.innerHTML = '';
        if (jList.length === 0) {
          jBody.innerHTML = '<tr><td colspan="8" style="text-align:center; color:var(--text-dim); padding:16px;">No closed trades recorded yet.</td></tr>';
        } else {
          jList.slice(0, 15).forEach(t => {
            const pnl = Number(t.pnl_usd || 0);
            const pnlTag = pnl >= 0 ? `<span class="pill-tag bull">+$${pnl.toFixed(2)}</span>` : `<span class="pill-tag bear">-$${Math.abs(pnl).toFixed(2)}</span>`;
            const symLabel = (t.symbol || '—').replace('USDT', '');
            const row = document.createElement('tr');
            row.innerHTML = `
              <td style="color:var(--text-muted);">#${t.id}</td>
              <td style="color:var(--text-secondary);">${t.open_ts}</td>
              <td style="font-weight:600; color:var(--text-primary);">${symLabel}</td>
              <td style="font-weight:600;">${t.action}</td>
              <td>$${Number(t.entry_price).toLocaleString()}</td>
              <td>$${Number(t.exit_price).toLocaleString()}</td>
              <td>${pnlTag}</td>
              <td style="color:var(--text-secondary); font-size:11px;">${t.exit_reason || ''}</td>
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
</html>"""

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
            lines = ["ID,Time,Symbol,Action,Entry Price,Exit Price,PnL USD,Exit Reason"]
            for t in journal:
                lines.append(f"{t.get('id', '')},{t.get('open_ts', '')},{t.get('symbol', '')},{t.get('action', '')},{t.get('entry_price', 0.0)},{t.get('exit_price', 0.0)},{t.get('pnl_usd', 0.0)},\"{t.get('exit_reason', '')}\"")
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
