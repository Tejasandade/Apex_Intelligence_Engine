"""
Apex Overnight Autonomous Supervisor
=====================================
1. Monitors MT5 Prop Firm Bot (scripts/apex_v7.py) overnight.
   - Verifies process is alive; auto-restarts if ever terminated.
   - Checks data/prop_firm_state.json for Target Achievement ($5,450.00).
   - Logs health status every 30 minutes to data/logs/overnight_monitor.log.
2. At 09:05 AM IST:
   - Checks if Indian Options Engine (apex_indian_engine) is active.
   - If not running, autonomously launches it.
   - Sends a Telegram notification confirming Monday market open deployment.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
import psutil
from dotenv import load_dotenv

load_dotenv()

# Setup logging
log_dir = Path("data/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "overnight_monitor.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("apex.supervisor")


def send_telegram(text: str) -> None:
    """Sends notification to configured Telegram channels."""
    import urllib.request
    import urllib.parse

    tokens = []
    # Dedicated Indian bot token
    ind_token = os.getenv("INDIAN_TELEGRAM_BOT_TOKEN")
    ind_chat = os.getenv("INDIAN_TELEGRAM_CHAT_ID")
    if ind_token and ind_chat:
        tokens.append((ind_token, ind_chat))

    # Main MT5 bot token
    mt5_token = os.getenv("TELEGRAM_BOT_TOKEN")
    mt5_chat = os.getenv("TELEGRAM_CHAT_ID")
    if mt5_token and mt5_chat and (mt5_token, mt5_chat) not in tokens:
        tokens.append((mt5_token, mt5_chat))

    for tok, chat in tokens:
        try:
            url = f"https://api.telegram.org/bot{tok}/sendMessage"
            payload = json.dumps({
                "chat_id": chat,
                "text": text,
                "parse_mode": "HTML",
            }).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                pass
        except Exception as e:
            logger.debug(f"Telegram send failed: {e}")


def is_process_running(keyword: str) -> bool:
    """Checks if any active process has the keyword in its command line."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmd = " ".join(proc.info["cmdline"] or [])
            if keyword.lower() in cmd.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def get_prop_firm_status() -> dict[str, Any]:
    state_file = Path("data/prop_firm_state.json")
    if state_file.exists():
        try:
            with open(state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def main():
    logger.info("=" * 65)
    logger.info("🌙 APEX OVERNIGHT SUPERVISOR ACTIVE")
    logger.info("Monitoring MT5 Prop Firm Bot + Scheduled Indian Engine 09:05 AM Launch")
    logger.info("=" * 65)

    indian_engine_launched = False
    target_alert_sent = False
    last_health_log = 0.0

    while True:
        try:
            now = datetime.now()
            now_t = now.time()

            # ── 1. MT5 Bot Process Watchdog ──
            mt5_alive = is_process_running("apex_v7.py")
            if not mt5_alive:
                logger.warning("⚠️ MT5 Bot (apex_v7.py) not detected! Restarting autonomously...")
                subprocess.Popen(
                    [sys.executable, "scripts/apex_v7.py"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                )
                time.sleep(5)
                send_telegram("⚠️ <b>[APEX WATCHDOG]</b> MT5 Prop Firm Bot was restarted automatically.")

            # ── 2. Check Prop Firm Challenge Progress ──
            pf_state = get_prop_firm_status()
            cur_bal = float(pf_state.get("virtual_balance", 5000.0))
            wins = pf_state.get("wins", 0)
            losses = pf_state.get("losses", 0)

            if cur_bal >= 5450.0 and not target_alert_sent:
                target_alert_sent = True
                msg = (
                    f"🎉 <b>[APEX PROP FIRM VICTORY!]</b>\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"<b>Target Achieved:</b> ${cur_bal:,.2f} / $5,450.00\n"
                    f"<b>Record:</b> {wins}W / {losses}L\n"
                    f"<b>Status:</b> SharkFunded $5,000 Challenge PASSED!\n"
                    f"<b>Time:</b> {now.strftime('%H:%M:%S IST')}"
                )
                logger.info(f"Target reached: ${cur_bal:,.2f}")
                send_telegram(msg)

            # ── 3. Indian Engine 09:05 AM Launch Trigger ──
            # Target window: 09:05:00 to 09:14:59 IST
            if not indian_engine_launched:
                if dtime(9, 5) <= now_t <= dtime(9, 14):
                    indian_running = is_process_running("run_indian")
                    if not indian_running:
                        logger.info("⏰ 09:05 AM IST reached. Launching Indian Options Engine...")
                        subprocess.Popen(
                            [sys.executable, "-m", "apex_indian_engine.run_indian", "--mode", "paper"],
                            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
                        )
                        indian_engine_launched = True
                        msg = (
                            f"🇮🇳 <b>[APEX SUPERVISOR] 09:05 AM IST</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"<b>Action:</b> Indian Options Engine autonomously started!\n"
                            f"<b>Mode:</b> PAPER (Pre-Open Sync Active)\n"
                            f"<b>Instruments:</b> NIFTY & BANKNIFTY Options\n"
                            f"<b>Trading Starts:</b> 09:15 AM IST"
                        )
                        send_telegram(msg)
                        logger.info("Indian Options Engine launched successfully.")
                    else:
                        logger.info("Indian Options Engine already running.")
                        indian_engine_launched = True

            # ── 4. Periodic 30-Minute Status Log ──
            if (time.time() - last_health_log) >= 1800.0:
                last_health_log = time.time()
                logger.info(
                    f"Health Check: MT5 Bot Alive: {mt5_alive} | Balance: ${cur_bal:,.2f} "
                    f"({wins}W/{losses}L) | Remaining to Target: ${max(0.0, 5450.0 - cur_bal):.2f}"
                )

            # Poll every 10 seconds
            time.sleep(10.0)

        except Exception as e:
            logger.error(f"Supervisor error: {e}")
            time.sleep(10.0)


if __name__ == "__main__":
    main()

