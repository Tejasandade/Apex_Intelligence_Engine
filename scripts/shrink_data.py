import pandas as pd
from pathlib import Path

def shrink_data(market="crypto", symbol="btcusdt", rows=10000):
    csv_path = Path("data/historical") / market / f"{symbol}.csv"
    if not csv_path.exists():
        return
        
    print(f"Reading {csv_path}...")
    # Read chunk by chunk to avoid RAM explosion
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        chunks.append(chunk)
        
    df = pd.concat(chunks)
    
    df = df.iloc[-rows:]
        
    backup_path = csv_path.parent / f"{symbol}_backup.csv"
    if not backup_path.exists():
        import shutil
        shutil.copy(csv_path, backup_path)
        print(f"Backed up to {backup_path}")
        
    df.to_csv(csv_path, index=False)
    print(f"Overwrote {csv_path} with {len(df)} rows")

if __name__ == "__main__":
    shrink_data()
