"""
Apex Intelligence Engine V5 — Historical Data Downloader
==========================================================
Downloads and caches historical candle data for all configured markets.

Usage:
    python -m scripts.download_data
    python -m scripts.download_data --symbol btcusdt --bars 50000
    python -m scripts.download_data --symbol ethusdt --bars 50000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import DATA_DIR, get_market, get_active_symbols, get_model_params
from src.core.logging import get_logger
from src.data.providers.binance import BinanceHistoricalProvider
from src.data.validation.validator import DataValidator

logger = get_logger("apex.scripts.download_data")


async def download_crypto(symbol: str, total_bars: int = 50000) -> None:
    """Download historical data for a crypto symbol."""
    provider = BinanceHistoricalProvider()
    validator = DataValidator()

    try:
        await provider.connect()

        logger.info(
            "download_started",
            symbol=symbol,
            bars=total_bars,
            source="binance_futures",
        )

        df = await provider.fetch_historical_candles(
            symbol=symbol.upper(),
            interval="1m",
            limit=total_bars,
        )

        if df.empty:
            logger.error("download_empty", symbol=symbol)
            return

        # Validate
        df = validator.validate_dataframe(df)

        # Save as Parquet (efficient) and CSV (human-readable)
        save_dir = DATA_DIR / "historical" / "crypto"
        save_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = save_dir / f"{symbol.lower()}.parquet"
        csv_path = save_dir / f"{symbol.lower()}.csv"

        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

        logger.info(
            "download_complete",
            symbol=symbol,
            rows=len(df),
            parquet=str(parquet_path),
            csv=str(csv_path),
            date_range=f"{df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}",
        )

    finally:
        await provider.disconnect()


async def main():
    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Symbol to download (e.g., btcusdt). Downloads all if not specified.",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=50000,
        help="Number of 1-minute bars to download (default: 50000 ≈ 35 days)",
    )
    parser.add_argument(
        "--market",
        type=str,
        default="crypto",
        choices=["crypto", "india_equity"],
        help="Market type to download",
    )

    args = parser.parse_args()

    if args.market == "crypto":
        if args.symbol:
            await download_crypto(args.symbol, args.bars)
        else:
            # Download all crypto symbols
            symbols = get_active_symbols("crypto")
            for sym in symbols:
                await download_crypto(sym, args.bars)
    elif args.market == "india_equity":
        logger.info(
            "india_download_deferred",
            msg="India market data download requires Dhan API (Phase 4). "
            "Use manual CSV upload for now.",
        )


if __name__ == "__main__":
    asyncio.run(main())
