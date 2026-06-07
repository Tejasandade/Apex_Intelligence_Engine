import json
with open(r'd:\Apex_Intelligence_Engine\data\state\position_manager.json') as f:
    state = json.load(f)
    for t in state['trade_history']:
        print(f"Side: {t['side']}, PnL: {t['pnl']:.2f}")
