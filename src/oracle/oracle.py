"""
Apex Intelligence Engine V6 — Higher Timeframe Oracle
=======================================================
Macro structural memory layer. Downloads HTF data at boot and maps
persistent structural levels (Order Blocks, FVGs, Daily/Weekly levels)
that the 1-minute engine would otherwise be blind to.

Architecture:
    Boot: REST API → 1H/4H/1D candles → compute structure → StructuralLevel[]
    Live: StructureAdvisor queries Oracle on every signal → confluence/rejection
    Refresh: Background task re-downloads HTF data every 4 hours

Design principles:
    1. Read-only context layer — the Oracle does NOT trade or vote
    2. Enhancement, not dependency — if boot fails, engine runs without it
    3. Levels are invalidated (mitigated) in real-time on every candle close
    4. Touch count degrades level strength — tested 3+ times = weakened
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.data.providers.binance import BinanceHistoricalProvider
from src.features.indicators.structure import (
    compute_bos,
    compute_fvg,
    compute_liquidity_sweep,
)

logger = get_logger("apex.oracle")


class OracleBootError(Exception):
    """Raised when the Oracle cannot download HTF data at boot."""
    pass


@dataclass
class StructuralLevel:
    """A persistent macro structural level detected on HTF charts."""

    level_type: str         # "ORDER_BLOCK", "FVG", "DAILY_HIGH", "DAILY_LOW",
                            # "WEEKLY_OPEN", "WEEKLY_HIGH", "WEEKLY_LOW",
                            # "PREV_DAILY_HIGH", "PREV_DAILY_LOW"
    zone_low: float
    zone_high: float
    direction: str          # "BULLISH" or "BEARISH"
    timeframe: str          # "1h", "4h", "1d"
    strength: float         # 0.0 - 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    mitigated: bool = False
    touch_count: int = 0    # Incremented on each price interaction


class HTFOracle:
    """
    Higher Timeframe Oracle — macro structural memory.

    Downloads HTF candle data at boot, computes structural levels,
    and provides a query interface for the StructureAdvisor.
    """

    def __init__(self, symbol: str = "btcusdt"):
        self.symbol = symbol.upper()
        self._levels: list[StructuralLevel] = []
        self._provider: BinanceHistoricalProvider | None = None
        self._booted = False
        self._refresh_task: asyncio.Task | None = None

        # Raw history kept for MTF buffer seeding
        self._history_1m: pd.DataFrame = pd.DataFrame()
        self._history_1h: pd.DataFrame = pd.DataFrame()
        self._history_4h: pd.DataFrame = pd.DataFrame()
        self._history_1d: pd.DataFrame = pd.DataFrame()

    @property
    def is_booted(self) -> bool:
        return self._booted

    @property
    def level_count(self) -> int:
        return len([l for l in self._levels if not l.mitigated])

    # ── Boot Sequence ─────────────────────────────────────────────────────

    async def boot(self, symbol: str | None = None) -> None:
        """
        Download HTF data and compute all structural levels.

        Downloads:
        - 1500 × 1m candles (~25 hours) for MTF buffer seeding
        - 720  × 1h candles (~30 days)  for hourly structure
        - 180  × 4h candles (~30 days)  for macro structure
        - 90   × 1d candles (~90 days)  for daily/weekly levels

        Raises:
            OracleBootError: If data download fails.
        """
        if symbol:
            self.symbol = symbol.upper()

        logger.info("oracle_boot_started", symbol=self.symbol)

        self._provider = BinanceHistoricalProvider()
        try:
            await self._provider.connect()

            # Download all timeframes
            self._history_1m = await self._provider.fetch_historical_candles(
                symbol=self.symbol, interval="1m", limit=1500,
            )
            self._history_1h = await self._provider.fetch_historical_candles(
                symbol=self.symbol, interval="1h", limit=720,
            )
            self._history_4h = await self._provider.fetch_historical_candles(
                symbol=self.symbol, interval="4h", limit=180,
            )
            self._history_1d = await self._provider.fetch_historical_candles(
                symbol=self.symbol, interval="1d", limit=90,
            )

        except Exception as e:
            logger.error("oracle_boot_download_failed", error=str(e))
            raise OracleBootError(f"Failed to download HTF data: {e}") from e
        finally:
            if self._provider:
                await self._provider.disconnect()

        # Validate minimum data
        if len(self._history_1h) < 50:
            raise OracleBootError(
                f"Insufficient 1H data: {len(self._history_1h)} candles (need 50+)"
            )

        # Build structural levels from all timeframes
        self._levels = []
        self._build_structure_levels(self._history_1h, "1h")
        self._build_structure_levels(self._history_4h, "4h")
        self._build_daily_weekly_levels(self._history_1d)

        self._booted = True

        active_levels = [l for l in self._levels if not l.mitigated]
        logger.info(
            "oracle_boot_complete",
            symbol=self.symbol,
            total_levels=len(self._levels),
            active_levels=len(active_levels),
            candles_1m=len(self._history_1m),
            candles_1h=len(self._history_1h),
            candles_4h=len(self._history_4h),
            candles_1d=len(self._history_1d),
            level_breakdown={
                lt: len([l for l in active_levels if l.level_type == lt])
                for lt in set(l.level_type for l in active_levels)
            },
        )

    # ── Level Construction ────────────────────────────────────────────────

    def _build_structure_levels(self, df: pd.DataFrame, timeframe: str) -> None:
        """
        Detect Order Blocks and FVGs on a given timeframe DataFrame.
        """
        if df.empty or len(df) < 20:
            return

        # Compute FVGs
        fvg_signal, fvg_pct = compute_fvg(df)
        for i in range(len(df)):
            sig = fvg_signal.iloc[i]
            pct = fvg_pct.iloc[i]
            if sig == 0.0 or pct < 0.001:
                continue

            row = df.iloc[i]
            if sig > 0:  # Bullish FVG
                zone_low = float(df.iloc[i - 2]["high"]) if i >= 2 else float(row["low"])
                zone_high = float(row["low"])
                direction = "BULLISH"
            else:  # Bearish FVG
                zone_low = float(row["high"])
                zone_high = float(df.iloc[i - 2]["low"]) if i >= 2 else float(row["high"])
                direction = "BEARISH"

            # Determine strength from gap size
            strength = min(1.0, float(pct) * 20)  # 5% gap → 1.0 strength

            ts = self._get_timestamp(row)
            self._levels.append(StructuralLevel(
                level_type="FVG",
                zone_low=min(zone_low, zone_high),
                zone_high=max(zone_low, zone_high),
                direction=direction,
                timeframe=timeframe,
                strength=strength,
                created_at=ts,
            ))

        # Compute BOS → derive Order Blocks
        bos_signal, bos_strength = compute_bos(df, window=20)
        for i in range(2, len(df)):
            sig = bos_signal.iloc[i]
            if sig == 0.0:
                continue

            # The Order Block is the candle BODY that preceded the BOS
            ob_candle = df.iloc[i - 1]
            ob_open = float(ob_candle["open"])
            ob_close = float(ob_candle["close"])

            if sig > 0:  # Bullish BOS → bullish OB (last bearish candle before break)
                direction = "BULLISH"
            else:  # Bearish BOS → bearish OB
                direction = "BEARISH"

            strength = min(1.0, float(bos_strength.iloc[i]) * 10)
            ts = self._get_timestamp(ob_candle)

            self._levels.append(StructuralLevel(
                level_type="ORDER_BLOCK",
                zone_low=min(ob_open, ob_close),
                zone_high=max(ob_open, ob_close),
                direction=direction,
                timeframe=timeframe,
                strength=strength,
                created_at=ts,
            ))

    def _build_daily_weekly_levels(self, df_daily: pd.DataFrame) -> None:
        """
        Extract Daily and Weekly open/high/low/close levels.
        These are static reference levels that never expire within their period.
        """
        if df_daily.empty or len(df_daily) < 2:
            return

        # Previous Daily High/Low (most commonly referenced retail level)
        prev_day = df_daily.iloc[-2]
        self._levels.append(StructuralLevel(
            level_type="PREV_DAILY_HIGH",
            zone_low=float(prev_day["high"]) - 10,  # Small zone around the level
            zone_high=float(prev_day["high"]) + 10,
            direction="BEARISH",  # Previous high acts as resistance
            timeframe="1d",
            strength=0.7,
            created_at=self._get_timestamp(prev_day),
        ))
        self._levels.append(StructuralLevel(
            level_type="PREV_DAILY_LOW",
            zone_low=float(prev_day["low"]) - 10,
            zone_high=float(prev_day["low"]) + 10,
            direction="BULLISH",  # Previous low acts as support
            timeframe="1d",
            strength=0.7,
            created_at=self._get_timestamp(prev_day),
        ))

        # Current Daily Open/High/Low
        today = df_daily.iloc[-1]
        self._levels.append(StructuralLevel(
            level_type="DAILY_HIGH",
            zone_low=float(today["high"]) - 5,
            zone_high=float(today["high"]) + 5,
            direction="BEARISH",
            timeframe="1d",
            strength=0.6,
            created_at=self._get_timestamp(today),
        ))
        self._levels.append(StructuralLevel(
            level_type="DAILY_LOW",
            zone_low=float(today["low"]) - 5,
            zone_high=float(today["low"]) + 5,
            direction="BULLISH",
            timeframe="1d",
            strength=0.6,
            created_at=self._get_timestamp(today),
        ))

        # Weekly levels — aggregate from daily data
        if len(df_daily) >= 7:
            # Find the start of the current week (Monday)
            df_daily_copy = df_daily.copy()
            df_daily_copy["datetime"] = pd.to_datetime(
                df_daily_copy["timestamp"], unit="ms"
            )

            # Current week's data
            current_week_start = df_daily_copy["datetime"].iloc[-1] - pd.Timedelta(days=6)
            week_data = df_daily_copy[df_daily_copy["datetime"] >= current_week_start]

            if len(week_data) >= 2:
                weekly_open = float(week_data.iloc[0]["open"])
                weekly_high = float(week_data["high"].max())
                weekly_low = float(week_data["low"].min())

                self._levels.append(StructuralLevel(
                    level_type="WEEKLY_OPEN",
                    zone_low=weekly_open - 15,
                    zone_high=weekly_open + 15,
                    direction="BULLISH",  # Weekly open is a magnet, not directional
                    timeframe="1w",
                    strength=0.8,
                    created_at=self._get_timestamp(week_data.iloc[0]),
                ))
                self._levels.append(StructuralLevel(
                    level_type="WEEKLY_HIGH",
                    zone_low=weekly_high - 10,
                    zone_high=weekly_high + 10,
                    direction="BEARISH",
                    timeframe="1w",
                    strength=0.8,
                    created_at=self._get_timestamp(week_data.iloc[-1]),
                ))
                self._levels.append(StructuralLevel(
                    level_type="WEEKLY_LOW",
                    zone_low=weekly_low - 10,
                    zone_high=weekly_low + 10,
                    direction="BULLISH",
                    timeframe="1w",
                    strength=0.8,
                    created_at=self._get_timestamp(week_data.iloc[-1]),
                ))

            # Previous week
            prev_week_end = current_week_start - pd.Timedelta(days=1)
            prev_week_start = prev_week_end - pd.Timedelta(days=6)
            prev_week = df_daily_copy[
                (df_daily_copy["datetime"] >= prev_week_start)
                & (df_daily_copy["datetime"] <= prev_week_end)
            ]
            if len(prev_week) >= 2:
                pw_high = float(prev_week["high"].max())
                pw_low = float(prev_week["low"].min())
                self._levels.append(StructuralLevel(
                    level_type="PREV_WEEKLY_HIGH",
                    zone_low=pw_high - 15,
                    zone_high=pw_high + 15,
                    direction="BEARISH",
                    timeframe="1w",
                    strength=0.75,
                    created_at=self._get_timestamp(prev_week.iloc[-1]),
                ))
                self._levels.append(StructuralLevel(
                    level_type="PREV_WEEKLY_LOW",
                    zone_low=pw_low - 15,
                    zone_high=pw_low + 15,
                    direction="BULLISH",
                    timeframe="1w",
                    strength=0.75,
                    created_at=self._get_timestamp(prev_week.iloc[-1]),
                ))

    # ── Query Interface ───────────────────────────────────────────────────

    def query_levels(
        self, price: float, atr: float, max_distance_atr: float = 0.5
    ) -> list[StructuralLevel]:
        """
        Return all active (non-mitigated) levels within proximity of price.

        Args:
            price: Current market price.
            atr: Current ATR value.
            max_distance_atr: Maximum distance in ATR multiples.

        Returns:
            List of StructuralLevel within range. Touch counts are incremented.
        """
        if not self._booted or atr <= 0:
            return []

        proximity = atr * max_distance_atr
        result: list[StructuralLevel] = []

        for level in self._levels:
            if level.mitigated:
                continue

            # Check if price is within proximity of the level zone
            dist_to_zone = 0.0
            if price < level.zone_low:
                dist_to_zone = level.zone_low - price
            elif price > level.zone_high:
                dist_to_zone = price - level.zone_high
            # else: price is inside the zone, dist = 0

            if dist_to_zone <= proximity:
                level.touch_count += 1
                result.append(level)

        if result:
            logger.debug(
                "oracle_query",
                price=f"${price:.2f}",
                atr=f"${atr:.2f}",
                levels_found=len(result),
                types=[l.level_type for l in result],
            )

        return result

    # ── Level Invalidation ────────────────────────────────────────────────

    def invalidate_levels(self, candle: dict[str, Any]) -> None:
        """
        Check if the current candle has mitigated any active levels.

        Invalidation rules:
        - ORDER_BLOCK: Full candle body closes beyond the zone
        - FVG: Any candle closes on the other side of the gap
        - DAILY/WEEKLY: Never expire within their period
        """
        if not self._booted:
            return

        candle_open = float(candle.get("open", 0))
        candle_close = float(candle.get("close", 0))
        body_high = max(candle_open, candle_close)
        body_low = min(candle_open, candle_close)

        mitigated_count = 0

        for level in self._levels:
            if level.mitigated:
                continue

            # Daily/Weekly levels don't get mitigated by candles
            if level.level_type in (
                "DAILY_HIGH", "DAILY_LOW", "PREV_DAILY_HIGH", "PREV_DAILY_LOW",
                "WEEKLY_OPEN", "WEEKLY_HIGH", "WEEKLY_LOW",
                "PREV_WEEKLY_HIGH", "PREV_WEEKLY_LOW",
            ):
                continue

            if level.level_type == "ORDER_BLOCK":
                # Mitigated if full candle body closes beyond the zone
                if level.direction == "BULLISH" and body_low < level.zone_low:
                    # Price body is entirely below bullish OB → broken
                    if body_high < level.zone_low:
                        level.mitigated = True
                        mitigated_count += 1
                elif level.direction == "BEARISH" and body_high > level.zone_high:
                    # Price body is entirely above bearish OB → broken
                    if body_low > level.zone_high:
                        level.mitigated = True
                        mitigated_count += 1

            elif level.level_type == "FVG":
                # Mitigated if candle closes on the other side
                if level.direction == "BULLISH" and candle_close < level.zone_low:
                    level.mitigated = True
                    mitigated_count += 1
                elif level.direction == "BEARISH" and candle_close > level.zone_high:
                    level.mitigated = True
                    mitigated_count += 1

        if mitigated_count > 0:
            logger.info(
                "oracle_levels_mitigated",
                count=mitigated_count,
                remaining=self.level_count,
            )

    # ── History Access (for MTF buffer seeding) ───────────────────────────

    def get_1m_history(self) -> pd.DataFrame:
        """Return the 1m candle history downloaded at boot."""
        return self._history_1m.copy()

    def get_1h_history(self) -> pd.DataFrame:
        """Return the 1h candle history downloaded at boot."""
        return self._history_1h.copy()

    # ── Background Refresh ────────────────────────────────────────────────

    async def refresh_loop(self, interval_hours: int = 4) -> None:
        """
        Background task that refreshes HTF data periodically.
        Re-downloads, rebuilds levels, preserves touch counts and
        mitigation state for levels that still exist.
        """
        while True:
            await asyncio.sleep(interval_hours * 3600)
            try:
                logger.info("oracle_refresh_started")

                # Save existing touch counts and mitigation state
                old_levels = {
                    (l.level_type, round(l.zone_low, 2), round(l.zone_high, 2)): l
                    for l in self._levels
                }

                # Re-download and rebuild
                self._provider = BinanceHistoricalProvider()
                await self._provider.connect()

                self._history_1h = await self._provider.fetch_historical_candles(
                    symbol=self.symbol, interval="1h", limit=720,
                )
                self._history_4h = await self._provider.fetch_historical_candles(
                    symbol=self.symbol, interval="4h", limit=180,
                )
                self._history_1d = await self._provider.fetch_historical_candles(
                    symbol=self.symbol, interval="1d", limit=90,
                )
                await self._provider.disconnect()

                # Rebuild levels
                self._levels = []
                self._build_structure_levels(self._history_1h, "1h")
                self._build_structure_levels(self._history_4h, "4h")
                self._build_daily_weekly_levels(self._history_1d)

                # Restore touch counts for levels that still exist
                for level in self._levels:
                    key = (level.level_type, round(level.zone_low, 2), round(level.zone_high, 2))
                    if key in old_levels:
                        old = old_levels[key]
                        level.touch_count = old.touch_count
                        level.mitigated = old.mitigated

                logger.info(
                    "oracle_refresh_complete",
                    active_levels=self.level_count,
                    total_levels=len(self._levels),
                )

            except Exception as e:
                logger.error("oracle_refresh_failed", error=str(e))
                # Non-fatal — existing levels remain valid

    # ── Diagnostics ───────────────────────────────────────────────────────

    def get_summary(self) -> dict[str, Any]:
        """Return Oracle state summary for dashboard/logging."""
        active = [l for l in self._levels if not l.mitigated]
        return {
            "booted": self._booted,
            "symbol": self.symbol,
            "total_levels": len(self._levels),
            "active_levels": len(active),
            "mitigated_levels": len(self._levels) - len(active),
            "by_type": {
                lt: len([l for l in active if l.level_type == lt])
                for lt in set(l.level_type for l in active)
            } if active else {},
            "by_timeframe": {
                tf: len([l for l in active if l.timeframe == tf])
                for tf in set(l.timeframe for l in active)
            } if active else {},
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_timestamp(row: pd.Series) -> datetime:
        """Extract datetime from a candle row."""
        ts = row.get("timestamp", 0)
        try:
            return pd.to_datetime(ts, unit="ms").to_pydatetime().replace(tzinfo=timezone.utc)
        except Exception:
            return datetime.now(timezone.utc)
