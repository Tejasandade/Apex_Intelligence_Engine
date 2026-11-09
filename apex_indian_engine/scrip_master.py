"""
Apex Indian Engine — Scrip Master & Strike Contract Resolver
============================================================
Downloads, caches, and indexes Angel One's OpenAPIScripMaster.
Provides O(1) resolution for weekly/monthly option strike tokens, lot sizes, and expiries.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

from apex_indian_engine.config import INDEX_SPECS, IndexSpec

logger = logging.getLogger("apex.indian.scrip_master")

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"


class ScripMaster:
    """
    Manages instrument definitions for NSE Indices and NFO Options.
    """

    def __init__(self, cache_path: str = "data/angel_scrip_master.json"):
        self.cache_path = Path(cache_path)
        self._raw_scrip_list: list[dict[str, Any]] = []
        
        # Indexed lookups:
        # { (index_name, expiry_str, strike_float, 'CE'/'PE'): contract_dict }
        self._options_lookup: dict[tuple[str, str, float, str], dict[str, Any]] = {}
        # { index_name: sorted list of distinct expiry dates }
        self._expiries_by_index: dict[str, list[str]] = {}
        # { token_str: contract_dict }
        self._by_token: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load_or_download(self, force_download: bool = False) -> bool:
        """
        Loads scrip master from local cache if fresh (< 24h old); otherwise downloads.
        """
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        need_download = force_download or (not self.cache_path.exists())

        if not need_download:
            # Check file age (must be from today)
            file_mtime = datetime.fromtimestamp(self.cache_path.stat().st_mtime).date()
            if file_mtime < date.today():
                need_download = True

        if need_download:
            logger.info(f"Downloading Scrip Master from {SCRIP_MASTER_URL}...")
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                req = urllib.request.Request(SCRIP_MASTER_URL, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw_data = resp.read()
                    with open(self.cache_path, "wb") as f:
                        f.write(raw_data)
                logger.info(f"Scrip Master saved to {self.cache_path}")
            except Exception as e:
                logger.error(f"Scrip Master download failed: {e}")
                if not self.cache_path.exists():
                    return False

        return self._index_scrip_file()

    def _index_scrip_file(self) -> bool:
        """Index NFO option contracts and NSE indices into fast hash lookups."""
        try:
            logger.info(f"Indexing Scrip Master from {self.cache_path}...")
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._options_lookup.clear()
            self._expiries_by_index.clear()
            self._by_token.clear()

            valid_names = set(INDEX_SPECS.keys())
            today = date.today()

            for item in data:
                exch = item.get("exch_seg")
                name = item.get("name")
                token = str(item.get("token"))

                # 1. Index Spot tokens (NSE Cash Indices)
                if exch == "NSE":
                    for idx_name, spec in INDEX_SPECS.items():
                        if item.get("symbol") == spec.spot_symbol or name == spec.spot_symbol:
                            self._by_token[token] = item

                # 2. Index NFO Options
                if exch == "NFO" and name in valid_names:
                    symbol_str = item.get("symbol", "")
                    opt_type = ""
                    if symbol_str.endswith("CE"):
                        opt_type = "CE"
                    elif symbol_str.endswith("PE"):
                        opt_type = "PE"

                    if not opt_type:
                        continue  # Skip futures or non-CE/PE contracts

                    # Parse strike price (Angel One stores strike in rupees or paise)
                    try:
                        raw_strike = float(item.get("strike", 0.0))
                        strike = raw_strike / 100.0 if raw_strike >= 100000 else raw_strike
                    except ValueError:
                        continue

                    # Expiry date parsing (e.g. "26NOV2026" or "12-NOV-2026")
                    expiry_raw = item.get("expiry", "")
                    if not expiry_raw:
                        continue

                    key = (name, expiry_raw, strike, opt_type)
                    contract_info = {
                        "token": token,
                        "symbol": symbol_str,
                        "name": name,
                        "expiry": expiry_raw,
                        "strike": strike,
                        "option_type": opt_type,
                        "lot_size": int(item.get("lotsize", INDEX_SPECS[name].default_lot_size)),
                        "tick_size": float(item.get("tick_size", 0.05)),
                        "exch_seg": "NFO",
                    }

                    self._options_lookup[key] = contract_info
                    self._by_token[token] = contract_info

                    if name not in self._expiries_by_index:
                        self._expiries_by_index[name] = []
                    if expiry_raw not in self._expiries_by_index[name]:
                        self._expiries_by_index[name].append(expiry_raw)

            # Sort expiries chronologically
            for name in self._expiries_by_index:
                self._expiries_by_index[name] = self._sort_expiries(self._expiries_by_index[name])

            self._loaded = True
            logger.info(f"Indexed {len(self._options_lookup)} options contracts across {list(self._expiries_by_index.keys())}")
            return True
        except Exception as e:
            logger.error(f"Scrip Master indexing error: {e}")
            return False

    def _sort_expiries(self, expiries: list[str]) -> list[str]:
        """Sort date strings into chronological order."""
        def _parse_d(exp_str: str) -> datetime:
            for fmt in ("%d%b%Y", "%d-%b-%Y", "%Y-%m-%d"):
                try:
                    return datetime.strptime(exp_str.upper(), fmt)
                except ValueError:
                    pass
            return datetime.max

        return sorted(expiries, key=_parse_d)

    def get_nearest_expiry(self, index_name: str) -> Optional[str]:
        """Returns the current week's active expiry date string."""
        exp_list = self._expiries_by_index.get(index_name.upper(), [])
        return exp_list[0] if exp_list else None

    def get_next_expiry(self, index_name: str) -> Optional[str]:
        """Returns the following week's active expiry date string (0DTE shield)."""
        exp_list = self._expiries_by_index.get(index_name.upper(), [])
        return exp_list[1] if len(exp_list) > 1 else (exp_list[0] if exp_list else None)

    def get_all_expiries(self, index_name: str) -> list[str]:
        """Returns all upcoming expiry dates for the index."""
        return self._expiries_by_index.get(index_name.upper(), [])

    def resolve_strike_contract(
        self,
        index_name: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> Optional[dict[str, Any]]:
        """O(1) lookup of exact strike contract."""
        key = (index_name.upper(), expiry, float(strike), option_type.upper())
        return self._options_lookup.get(key)

    def resolve_atm_strike(
        self,
        index_name: str,
        spot_price: float,
        option_type: str,
        expiry: Optional[str] = None,
        offset_steps: int = 0,
    ) -> Optional[dict[str, Any]]:
        """
        Dynamically round spot price to nearest strike step and resolve contract.
        offset_steps = 0  -> ATM (At-The-Money)
        offset_steps = 1  -> 1 Strike ITM (In-The-Money for higher Delta ~ 0.55-0.60)
        offset_steps = -1 -> 1 Strike OTM (Out-The-Money)
        """
        spec = INDEX_SPECS.get(index_name.upper())
        if not spec:
            return None

        exp = expiry or self.get_nearest_expiry(index_name)
        if not exp:
            return None

        step = spec.step_size
        atm_strike = round(spot_price / step) * step

        opt_upper = option_type.upper()
        if opt_upper == "CE":
            # For Calls: Lower strike is ITM
            target_strike = atm_strike - (offset_steps * step)
        else:
            # For Puts: Higher strike is ITM
            target_strike = atm_strike + (offset_steps * step)

        return self.resolve_strike_contract(index_name, exp, target_strike, opt_upper)

    def get_atm_strike(self, index_name: str, spot_price: float) -> float:
        """Returns the rounded ATM strike price float."""
        spec = INDEX_SPECS.get(index_name.upper())
        step = spec.step_size if spec else 100.0
        return float(round(spot_price / step) * step)

    get_option_contract = resolve_strike_contract
