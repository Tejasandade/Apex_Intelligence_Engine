"""
Apex Indian Engine — Live Terminal ASCII HUD Dashboard
======================================================
Renders a real-time, zero-lag console status matrix showing:
- NIFTY, BANKNIFTY, and INDIA VIX
- Volatility regime and OI sentiment
- Active Option Contract, PnL in ₹, Greeks, and Trailing SL
- Daily Net PnL after STT and capital status
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional


class IndianTerminalHUD:
    """
    Renders high-frequency ASCII dashboard to the terminal.
    """

    def render(self, metrics: dict[str, Any]) -> str:
        mode = metrics.get("mode", "PAPER")
        status = metrics.get("status", "READY")
        cap = metrics.get("capital_inr", 100_000.0)
        daily_pnl = metrics.get("daily_realized_pnl", 0.0)
        unrealized = metrics.get("daily_unrealized_pnl", 0.0)
        net_pnl = daily_pnl + unrealized
        friction = metrics.get("daily_friction_cost", 0.0)
        trades = metrics.get("trades_today", 0)
        max_trades = metrics.get("max_trades", 4)
        vix = metrics.get("vix", 14.5)
        vix_regime = metrics.get("vix_regime", "GOLDEN_ZONE")

        bn_spot = metrics.get("banknifty_spot", 0.0)
        bn_vwap = metrics.get("banknifty_vwap", 0.0)
        nifty_spot = metrics.get("nifty_spot", 0.0)
        nifty_vwap = metrics.get("nifty_vwap", 0.0)

        pcr = metrics.get("pcr", 1.0)
        oi_flow = metrics.get("oi_flow", "BALANCED")
        call_wall = metrics.get("call_wall", 0.0)
        put_wall = metrics.get("put_wall", 0.0)

        open_pos = metrics.get("open_position")

        sign = "+" if net_pnl >= 0 else ""
        t_str = time.strftime("%H:%M:%S IST")

        lines = []
        lines.append("╔" + "═" * 78 + "╗")
        lines.append(f"║  🇮🇳 APEX INDIAN INTELLIGENCE ENGINE v1.0 — OPTIONS CORE                      ║")
        lines.append(f"║  Time: {t_str} | Mode: {mode:<6} | Status: {status:<10}                        ║")
        lines.append("╠" + "═" * 78 + "╣")
        lines.append(
            f"║  CAPITAL: ₹{cap:>10,.2f} | Net Daily PnL: {sign}₹{net_pnl:>8,.2f} | Trades: {trades}/{max_trades} | Fees: ₹{friction:>6.2f} ║"
        )
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║  MARKET POSTURE & VOLATILITY:                                                ║")
        lines.append(
            f"║   BANKNIFTY: ₹{bn_spot:>9,.2f} (VWAP: ₹{bn_vwap:>9,.2f})                                ║"
        )
        lines.append(
            f"║   NIFTY 50:  ₹{nifty_spot:>9,.2f} (VWAP: ₹{nifty_vwap:>9,.2f})                                ║"
        )
        lines.append(
            f"║   INDIA VIX: {vix:>5.2f} | Regime: {vix_regime:<14}                                      ║"
        )
        lines.append("╠" + "═" * 78 + "╣")
        lines.append("║  DERIVATIVES & INSTITUTIONAL OI ALPHA:                                       ║")
        lines.append(
            f"║   PCR: {pcr:>4.2f} | Flow: {oi_flow:<20} | Walls: {put_wall:.0f} (Floor) / {call_wall:.0f} (Ceil) ║"
        )
        lines.append("╠" + "═" * 78 + "╣")

        if open_pos:
            sym = open_pos.get("symbol", "")
            lots = open_pos.get("lots", 1)
            qty = open_pos.get("quantity", 30)
            entry = open_pos.get("entry_price", 0.0)
            sl = open_pos.get("sl_price", 0.0)
            tp1 = open_pos.get("tp1_price", 0.0)
            tp2 = open_pos.get("tp2_price", 0.0)
            u_pnl = open_pos.get("unrealized_pnl", 0.0)
            u_sign = "+" if u_pnl >= 0 else ""

            lines.append("║  ACTIVE OPTION POSITION:                                                     ║")
            lines.append(f"║   Symbol: {sym:<34} | Lots: {lots} ({qty} Qty)                     ║")
            lines.append(f"║   Entry: ₹{entry:>7.2f} | SL: ₹{sl:>7.2f} | TP1: ₹{tp1:>7.2f} | TP2: ₹{tp2:>7.2f}           ║")
            lines.append(f"║   Unrealized PnL: {u_sign}₹{u_pnl:>8.2f}                                                    ║")
        else:
            lines.append("║  ACTIVE OPTION POSITION: FLAT (Monitoring 5m council setups...)              ║")

        lines.append("╚" + "═" * 78 + "╝")
        return "\n".join(lines)

