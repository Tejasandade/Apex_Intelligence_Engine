import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def run_proof_of_edge():
    print("Loading historical data...")
    df = pd.read_csv("data/historical/cleaned/BTCUSDT_1m_cleaned.csv")
    df = df.iloc[-50000:].copy()
    df.reset_index(drop=True, inplace=True)
    
    print(f"Data loaded: {len(df):,} 1-minute candles.")
    
    # --- STRATEGY PARAMETERS (TURTLE BREAKOUT) ---
    ENTRY_WINDOW = 55
    EXIT_WINDOW = 20
    FEE_BPS = 0.07 # 0.14% Round Trip
    
    print(f"\nCalculating Momentum Breakout (Entry: {ENTRY_WINDOW}, Exit: {EXIT_WINDOW})...")
    
    # 1. Calculate Channels
    df['highest_high'] = df['high'].rolling(window=ENTRY_WINDOW).max()
    df['lowest_low'] = df['low'].rolling(window=ENTRY_WINDOW).min()
    
    df['exit_high'] = df['high'].rolling(window=EXIT_WINDOW).max()
    df['exit_low'] = df['low'].rolling(window=EXIT_WINDOW).min()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # --- BACKTEST ENGINE ---
    position = 0 
    entry_price = 0.0
    trades = []
    
    balance = 10000.0
    initial_balance = 10000.0
    peak_balance = 10000.0
    max_drawdown = 0.0
    
    closes = df['close'].values
    highest_high = df['highest_high'].values
    lowest_low = df['lowest_low'].values
    exit_high = df['exit_high'].values
    exit_low = df['exit_low'].values
    
    for i in range(1, len(closes)):
        current_price = closes[i]
        
        # We check yesterday's highest_high to avoid lookahead bias
        prev_highest = highest_high[i-1]
        prev_lowest = lowest_low[i-1]
        prev_exit_high = exit_high[i-1]
        prev_exit_low = exit_low[i-1]
        
        # Check Exits
        if position == 1:
            if current_price <= prev_exit_low: # Trailing Stop Hit
                pnl_pct = (current_price - entry_price) / entry_price
                net_pnl_pct = pnl_pct - (FEE_BPS * 2 / 100)
                profit = balance * net_pnl_pct
                balance += profit
                trades.append({'type': 'LONG', 'pnl_pct': net_pnl_pct, 'profit': profit})
                position = 0
                peak_balance = max(peak_balance, balance)
                max_drawdown = min(max_drawdown, (balance - peak_balance) / peak_balance)
                
        elif position == -1:
            if current_price >= prev_exit_high: # Trailing Stop Hit
                pnl_pct = (entry_price - current_price) / entry_price
                net_pnl_pct = pnl_pct - (FEE_BPS * 2 / 100)
                profit = balance * net_pnl_pct
                balance += profit
                trades.append({'type': 'SHORT', 'pnl_pct': net_pnl_pct, 'profit': profit})
                position = 0
                peak_balance = max(peak_balance, balance)
                max_drawdown = min(max_drawdown, (balance - peak_balance) / peak_balance)
                
        # Check Entries
        if position == 0:
            if current_price > prev_highest:
                position = 1
                entry_price = current_price
            elif current_price < prev_lowest:
                position = -1
                entry_price = current_price
                
    # --- STATISTICS ---
    print("\n" + "="*50)
    print("       PROOF OF EDGE: TREND FOLLOWING (TURTLE)")
    print("="*50)
    
    if len(trades) == 0:
        print("0 trades taken.")
        return
        
    wins = [t for t in trades if t['pnl_pct'] > 0]
    losses = [t for t in trades if t['pnl_pct'] <= 0]
    
    win_rate = len(wins) / len(trades)
    total_net_profit = balance - initial_balance
    total_net_pct = (total_net_profit / initial_balance) * 100
    
    avg_win = np.mean([t['pnl_pct'] for t in wins]) * 100 if wins else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losses]) * 100 if losses else 0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    print(f"Total Trades:     {len(trades)}")
    print(f"Win Rate:         {win_rate:.2%}")
    print(f"Avg Win:          {avg_win:.2f}%")
    print(f"Avg Loss:         {avg_loss:.2f}%")
    print(f"Reward:Risk:      {abs(avg_win/avg_loss):.2f}")
    print(f"Expectancy:       {expectancy:.3f}% per trade")
    print(f"Max Drawdown:     {max_drawdown:.2%}")
    print("-"*50)
    print(f"Starting Balance: ${initial_balance:.2f}")
    print(f"Final Balance:    ${balance:.2f}")
    print(f"Net P&L:          ${total_net_profit:.2f} ({total_net_pct:.2f}%)")
    
    if expectancy > 0:
        print("\n[YES] EDGE VERIFIED: Strategy has positive mathematical expectancy after fees.")
    else:
        print("\n[NO] EDGE FAILED: Strategy loses to fees. Do not build.")

if __name__ == "__main__":
    run_proof_of_edge()
