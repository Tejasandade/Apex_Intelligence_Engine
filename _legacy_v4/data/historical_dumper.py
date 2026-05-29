import os
import glob
import pandas as pd
import datetime
from binance_historical_data import BinanceDataDumper
from loguru import logger

DUMP_DIR = "D:/Apex_Intelligence_Engine/data/historical"
CLEANED_DIR = os.path.join(DUMP_DIR, "cleaned")

def download_historical_data():
    """Downloads the last 6 months of 1m klines for BTCUSDT."""
    logger.info("Initializing BinanceDataDumper for USD-M Futures (1m klines)...")
    
    # Calculate 6 months ago (approx 180 days)
    date_start = datetime.date.today() - datetime.timedelta(days=180)
    
    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=DUMP_DIR,
        asset_class="um",  # USD-M Futures
        data_type="klines",
        data_frequency="1m",
    )
    
    logger.info(f"Starting download for BTCUSDT from {date_start} to today...")
    
    data_dumper.dump_data(
        tickers=["BTCUSDT"],
        date_start=date_start,
        date_end=datetime.date.today(),
        is_to_update_existing=False
    )
    
    logger.success("Download complete.")

def convert_to_timescaledb_format():
    """
    Reads the raw downloaded zip files and converts them 
    into a cleaner format suitable for TimescaleDB insertion.
    """
    logger.info("Converting raw historical data for TimescaleDB...")
    
    os.makedirs(CLEANED_DIR, exist_ok=True)
    
    # binance-historical-data dumps files in a specific directory structure
    search_pattern = os.path.join(DUMP_DIR, "**", "*.zip")
    raw_files = glob.glob(search_pattern, recursive=True)
    
    if not raw_files:
        logger.warning("No raw zip files found. Checking for CSVs...")
        search_pattern = os.path.join(DUMP_DIR, "**", "*.csv")
        raw_files = glob.glob(search_pattern, recursive=True)
        
    if not raw_files:
        logger.error("No historical data files found to convert.")
        return

    all_data = []
    
    # Standard Binance kline columns
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    for file in raw_files:
        try:
            # Pandas can natively read CSVs inside zip files
            df = pd.read_csv(file, names=columns, header=None)
            
            # Binance sometimes includes headers in the CSV. If the first row is 'open_time', drop it.
            if df.iloc[0]['open_time'] == 'open_time':
                df = df.iloc[1:].copy()
                
            # Convert timestamp from milliseconds to proper datetime
            df["time"] = pd.to_datetime(df["open_time"].astype(float), unit="ms")
            
            # Keep only the essential columns for market_data
            df["symbol"] = "BTCUSDT"
            df = df[["time", "symbol", "open", "high", "low", "close", "volume"]]
            
            # Ensure numeric types
            numeric_cols = ["open", "high", "low", "close", "volume"]
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            
            all_data.append(df)
            logger.debug(f"Processed {os.path.basename(file)}")
            
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            
    if all_data:
        # Concatenate and sort by time
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values(by="time").drop_duplicates(subset=["time"])
        
        output_file = os.path.join(CLEANED_DIR, "BTCUSDT_1m_cleaned.csv")
        final_df.to_csv(output_file, index=False)
        logger.success(f"Successfully converted and saved cleaned data to {output_file}")
        logger.info(f"Total historical rows: {len(final_df)}")
    else:
        logger.warning("No data was successfully converted.")

if __name__ == "__main__":
    download_historical_data()
    convert_to_timescaledb_format()
