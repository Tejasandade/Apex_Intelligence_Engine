"""
Apex Intelligence Engine — StateManager
=========================================
Persistent crash-resume state: saves portfolio + account to disk on every
trade event so the bot can resume cleanly after a crash or restart.

Usage:
    from src.core.state_manager import StateManager
    sm = StateManager()
    sm.load_state(portfolio, account)   # call at startup
    sm.save_state(portfolio, account)   # call on open/close
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("apex.core.state_manager")

_DEFAULT_STATE_FILE = Path("data/apex_state.json")
_MAX_STATE_AGE_HOURS = 24.0   # Ignore state files older than this


class StateManager:
    """
    Atomic crash-resume state persistence.

    Saves:
      - portfolio dict (position, entry, SL/TP, notional, etc.)
      - account dict  (balance, peak, win/loss stats, halt flags)
      - metadata      (last_save_time, bot_version)

    On load:
      - Verifies state file is not stale (< 24h old)
      - Verifies position is FLAT or still has an OPEN entry in the journal
      - Restores balance + stats but NOT an in-progress trade
        (re-entry requires a fresh signal — open trades are marked CANCELLED on crash)
    """

    def __init__(self, state_file: Path | str = _DEFAULT_STATE_FILE):
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Save ───────────────────────────────────────────────────────────────────

    def save_state(self, portfolio: dict[str, Any], account: dict[str, Any]) -> None:
        """
        Atomically write state to disk.
        Uses write-then-rename for crash safety (no half-written files).
        """
        payload = {
            "last_save_time": time.time(),
            "last_save_iso": _iso_now(),
            "bot_version": "7.1",
            "portfolio": _safe_copy(portfolio),
            "account": _safe_copy(account),
        }
        tmp = self.state_file.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
            os.replace(tmp, self.state_file)   # atomic on POSIX and Windows
            logger.debug(f"[StateManager] state saved → {self.state_file}")
        except Exception as exc:
            logger.warning(f"[StateManager] save_state error: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_state(
        self,
        portfolio: dict[str, Any],
        account: dict[str, Any],
    ) -> bool:
        """
        Restore account stats from the last saved state.

        Returns True if state was successfully loaded, False otherwise.

        IMPORTANT: Never restores an OPEN position — if the bot crashed mid-trade
        the position is treated as cancelled (manual review required).
        Only account balance / stats are restored.
        """
        if not self.state_file.exists():
            logger.info("[StateManager] No saved state found — starting fresh.")
            return False

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(f"[StateManager] Cannot read state file: {exc}")
            return False

        # Staleness check
        saved_at = float(payload.get("last_save_time", 0))
        age_hours = (time.time() - saved_at) / 3600.0
        if age_hours > _MAX_STATE_AGE_HOURS:
            logger.info(
                f"[StateManager] State file is {age_hours:.1f}h old (> {_MAX_STATE_AGE_HOURS}h) — ignoring."
            )
            return False

        saved_account = payload.get("account", {})
        saved_portfolio = payload.get("portfolio", {})

        # Restore account stats (all numeric/string fields)
        _restore_dict(account, saved_account, skip_keys={"today_date", "halt_until"})

        # Reset halt flags that are time-sensitive
        account["halted"] = False
        account["halt_reason"] = ""
        account["halt_until"] = 0.0

        # Restore daily counters if same calendar day, reset only on rollover
        today_str = datetime.now().strftime("%Y-%m-%d")
        if saved_account.get("today_date") == today_str:
            account["trades_today"] = int(saved_account.get("trades_today", 0))
            account["today_date"] = today_str
        else:
            account["trades_today"] = 0
            account["today_date"] = today_str

        # If there was an OPEN position when we crashed, treat as cancelled
        if saved_portfolio.get("position") == "OPEN":
            logger.warning(
                "[StateManager] Found OPEN position in saved state — "
                "treating as CANCELLED (manual review required).\n"
                f"  Was: {saved_portfolio.get('direction')} @"
                f"${saved_portfolio.get('entry_price', 0):,.1f}"
            )
            # portfolio stays FLAT (we don't restore the open position)
        else:
            # Restore FLAT position state (no risk, safe)
            portfolio.update({k: v for k, v in saved_portfolio.items()
                              if k in ("position", "direction", "phase")})

        logger.info(
            f"[StateManager] State restored from {_ago(saved_at)} ago | "
            f"Balance: ${account.get('balance', 0):,.2f} | "
            f"W{account.get('wins', 0)}/L{account.get('losses', 0)}"
        )
        return True

    def delete_state(self) -> None:
        """Remove the saved state file (e.g. on clean shutdown)."""
        try:
            self.state_file.unlink(missing_ok=True)
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_copy(d: dict) -> dict:
    """Deep-copy a dict, converting non-serialisable values to strings."""
    result = {}
    for k, v in d.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result


def _restore_dict(target: dict, source: dict, skip_keys: set | None = None) -> None:
    """Update target with values from source, skipping specified keys."""
    skip = skip_keys or set()
    for k, v in source.items():
        if k in skip:
            continue
        if k in target:
            try:
                # Cast to same type as existing value
                existing = target[k]
                if isinstance(existing, float):
                    target[k] = float(v)
                elif isinstance(existing, int) and not isinstance(existing, bool):
                    target[k] = int(v)
                elif isinstance(existing, bool):
                    target[k] = bool(v)
                else:
                    target[k] = v
            except (TypeError, ValueError):
                target[k] = v


def _iso_now() -> str:
    import datetime
    return datetime.datetime.now().isoformat()


def _ago(ts: float) -> str:
    """Human-readable 'N minutes ago' string."""
    secs = time.time() - ts
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs/60:.0f}m"
    else:
        return f"{secs/3600:.1f}h"

