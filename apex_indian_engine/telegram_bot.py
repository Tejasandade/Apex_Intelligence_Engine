"""
Apex Indian Engine — Dedicated Telegram Alerts & Remote Control Bridge
======================================================================
Provides dedicated real-time Telegram alerts and interactive command
control specifically for the Indian Options Engine.
Uses INDIAN_TELEGRAM_BOT_TOKEN & INDIAN_TELEGRAM_CHAT_ID from .env.
Completely isolated from the MT5 Prop Firm bot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable, Optional
import urllib.parse
import urllib.request
import json

logger = logging.getLogger("apex.indian.telegram")


class IndianTelegramNotifier:
    """
    Dedicated Telegram bot for Indian NIFTY & BANKNIFTY Options.
    """

    def __init__(self):
        # Dedicated credentials for Indian system
        self.bot_token = os.getenv("INDIAN_TELEGRAM_BOT_TOKEN") or ""
        self.chat_id = os.getenv("INDIAN_TELEGRAM_CHAT_ID") or ""
        
        self.is_configured = bool(self.bot_token and self.chat_id)
        if not self.is_configured:
            logger.info(
                "Indian Telegram Bot not configured. Add INDIAN_TELEGRAM_BOT_TOKEN "
                "and INDIAN_TELEGRAM_CHAT_ID to .env to receive dedicated alerts."
            )

        self._running = False
        self._last_update_id = 0
        self._poll_thread: Optional[threading.Thread] = None

        # Command hooks (injected by engine)
        self.cmd_get_status: Optional[Callable[[], str]] = None
        self.cmd_close_position: Optional[Callable[[], str]] = None
        self.cmd_pause_trading: Optional[Callable[[], str]] = None
        self.cmd_resume_trading: Optional[Callable[[], str]] = None

    def send_message(self, text: str) -> bool:
        """Sends a message via the Telegram Bot API synchronously (safe for background threads)."""
        if not self.is_configured:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to send Indian Telegram alert: {e}")
            return False

    def notify_trade_opened(
        self,
        symbol: str,
        index_name: str,
        option_type: str,
        strike: float,
        lots: int,
        qty: int,
        entry_px: float,
        sl_px: float,
        tp1_px: float,
        tp2_px: float,
        score: float,
        conviction: float,
        vix: float,
        regime: str,
        strike_type: str,
    ) -> None:
        """Alert on new option trade opened."""
        icon = "🟢 CALL (CE)" if option_type == "CE" else "🔴 PUT (PE)"
        msg = (
            f"<b>🇮🇳 [APEX INDIAN OPTIONS] NEW POSITION</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Action:</b> BUY {icon}\n"
            f"<b>Contract:</b> <code>{symbol}</code> ({strike_type})\n"
            f"<b>Quantity:</b> {lots} Lots ({qty} Qty)\n"
            f"<b>Entry Premium:</b> ₹{entry_px:.2f}\n"
            f"<b>Stop Loss:</b> ₹{sl_px:.2f} (-₹{(entry_px - sl_px):.1f} pts)\n"
            f"<b>Target TP1:</b> ₹{tp1_px:.2f} (+1.5R)\n"
            f"<b>Target TP2:</b> ₹{tp2_px:.2f} (+2.5R)\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Council Score:</b> {score:+.2f} (Conviction: {conviction*100:.0f}%)\n"
            f"<b>India VIX:</b> {vix:.2f} ({regime})\n"
            f"<b>Time:</b> {time.strftime('%H:%M:%S IST')}"
        )
        self.send_message(msg)

    def notify_sl_ratchet(self, symbol: str, tier: int, new_sl: float, r_mult: float, reason: str) -> None:
        """Alert on dynamic trailing stop ratchet."""
        msg = (
            f"<b>🛡️ [APEX INDIAN OPTIONS] SL RATCHET</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Contract:</b> <code>{symbol}</code>\n"
            f"<b>Trailing Tier:</b> Tier {tier} (+{r_mult:.1f}R)\n"
            f"<b>New SL:</b> ₹{new_sl:.2f}\n"
            f"<b>Status:</b> {reason}\n"
            f"<b>Time:</b> {time.strftime('%H:%M:%S IST')}"
        )
        self.send_message(msg)

    def notify_trade_closed(
        self,
        symbol: str,
        entry_px: float,
        exit_px: float,
        qty: int,
        gross_pnl: float,
        friction_cost: float,
        net_pnl: float,
        reason: str,
        today_net_pnl: float,
        capital_inr: float,
    ) -> None:
        """Alert on trade closed and friction settled."""
        pnl_icon = "🟢" if net_pnl >= 0 else "🔴"
        sign = "+" if net_pnl >= 0 else ""
        msg = (
            f"<b>{pnl_icon} [APEX INDIAN OPTIONS] POSITION CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Contract:</b> <code>{symbol}</code>\n"
            f"<b>Exit Price:</b> ₹{exit_px:.2f} (Entry: ₹{entry_px:.2f})\n"
            f"<b>Exit Reason:</b> {reason}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Gross PnL:</b> ₹{gross_pnl:+,.2f}\n"
            f"<b>Taxes & Fees:</b> -₹{friction_cost:,.2f} (STT/GST/Brk)\n"
            f"<b>Net Cash PnL:</b> <b>{sign}₹{net_pnl:,.2f}</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Today's Net Realized:</b> ₹{today_net_pnl:+,.2f}\n"
            f"<b>Account Capital:</b> ₹{capital_inr:,.2f}\n"
            f"<b>Time:</b> {time.strftime('%H:%M:%S IST')}"
        )
        self.send_message(msg)

    def notify_daily_circuit_breaker(self, reason: str, daily_loss: float, capital: float) -> None:
        """Alert when daily loss safety buffer or circuit breaker trips."""
        msg = (
            f"<b>🚨 [APEX INDIAN OPTIONS] CIRCUIT BREAKER TRIGGERED</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"<b>Halt Reason:</b> {reason}\n"
            f"<b>Daily Net PnL:</b> -₹{abs(daily_loss):,.2f}\n"
            f"<b>Capital Preserved:</b> ₹{capital:,.2f}\n"
            f"<b>Action:</b> Trading paused for rest of session to protect capital.\n"
            f"<b>Time:</b> {time.strftime('%H:%M:%S IST')}"
        )
        self.send_message(msg)

    def start_polling(self) -> None:
        """Starts background long-polling for commands."""
        if not self.is_configured:
            return

        self._running = True

        def _poll_worker():
            while self._running:
                try:
                    url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates?offset={self._last_update_id + 1}&timeout=10"
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            for result in data.get("result", []):
                                update_id = result.get("update_id", 0)
                                self._last_update_id = max(self._last_update_id, update_id)
                                message = result.get("message", {})
                                text = str(message.get("text", "")).strip()
                                from_id = str(message.get("chat", {}).get("id", ""))

                                if from_id == self.chat_id:
                                    self._handle_command(text)
                except Exception as e:
                    logger.debug(f"Telegram polling tick error: {e}")
                    time.sleep(2.0)

        self._poll_thread = threading.Thread(target=_poll_worker, daemon=True, name="IndianTelegramPoll")
        self._poll_thread.start()
        logger.info("Indian Telegram command listener started.")

    def stop_polling(self) -> None:
        self._running = False

    def _handle_command(self, cmd_text: str) -> None:
        cmd = cmd_text.lower().split()[0] if cmd_text else ""
        if cmd in ("/start", "/help", "/indian_help"):
            reply = (
                "<b>🇮🇳 APEX Indian Options Bot Commands:</b>\n"
                "/status or /indian_status — View engine status & metrics\n"
                "/pnl or /indian_pnl — Today's realized and net INR PnL\n"
                "/close or /indian_close — Instantly close active option trade\n"
                "/pause or /indian_pause — Pause new trade entries\n"
                "/resume or /indian_resume — Resume trading"
            )
            self.send_message(reply)
        elif cmd in ("/status", "/indian_status"):
            if self.cmd_get_status:
                self.send_message(self.cmd_get_status())
            else:
                self.send_message("Engine status handler not connected.")
        elif cmd in ("/close", "/indian_close"):
            if self.cmd_close_position:
                self.send_message(self.cmd_close_position())
        elif cmd in ("/pause", "/indian_pause"):
            if self.cmd_pause_trading:
                self.send_message(self.cmd_pause_trading())
        elif cmd in ("/resume", "/indian_resume"):
            if self.cmd_resume_trading:
                self.send_message(self.cmd_resume_trading())

