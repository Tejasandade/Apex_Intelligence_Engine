"""
Apex Intelligence Engine — XM MT5 Quick Auth Tester
Run: python scripts/test_mt5_auth.py
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

try:
    import MetaTrader5 as mt5
except ImportError:
    print("❌ MetaTrader5 python package not installed. Run 'pip install MetaTrader5'")
    sys.exit(1)

login = os.getenv("MT5_LOGIN", "")
password = os.getenv("MT5_PASSWORD", "")
server = os.getenv("MT5_SERVER", "")
path = os.getenv("MT5_PATH", r"C:\Program Files\XM Global MT5\terminal64.exe")

print("=" * 60)
print("  APEX XM / METATRADER 5 CONNECTION TEST")
print("=" * 60)
print(f"  Terminal Path: {path}")
print(f"  Account Login: {login if login else 'Not set in .env'}")
print(f"  Server:        {server if server else 'Not set in .env'}")
print("-" * 60)

init_kwargs = {}
if os.path.exists(path):
    init_kwargs["path"] = path
if login and login.isdigit():
    init_kwargs["login"] = int(login)
if password:
    init_kwargs["password"] = password
if server:
    init_kwargs["server"] = server

if not mt5.initialize(**init_kwargs):
    err = mt5.last_error()
    print(f"❌ Failed to connect to MT5. Error: {err}")
    print("\n💡 Troubleshooting Tips:")
    print("  1. Make sure XM Global MT5 is open or terminal path is correct.")
    print("  2. In MT5: Tools -> Options -> Expert Advisors -> check 'Allow Algo Trading'.")
    print("  3. Double check MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file.")
    sys.exit(1)

acc = mt5.account_info()
if acc:
    print(f"✅ CONNECTED TO XM MT5 SUCCESSFULLY!")
    print(f"   Name:     {acc.name}")
    print(f"   Login:    {acc.login}")
    print(f"   Server:   {acc.server}")
    print(f"   Company:  {acc.company}")
    print(f"   Balance:  ${acc.balance:,.2f} {acc.currency}")
    print(f"   Equity:   ${acc.equity:,.2f} {acc.currency}")
    print(f"   Leverage: 1:{acc.leverage}")
    
    # Check BTC symbol
    symbols = [s.name for s in (mt5.symbols_get() or []) if "BTC" in s.name.upper()]
    print(f"   Available BTC Symbols: {symbols[:5]}")
else:
    print("⚠️ Connected to terminal, but no active trading account logged in.")

mt5.shutdown()
print("=" * 60)
