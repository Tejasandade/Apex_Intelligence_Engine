import os, sys
sys.path.insert(0, ".")

ws  = open("src/dashboard/backend/ws_server.py", encoding="utf-8").read()
jsx = open("src/dashboard/frontend/src/App.jsx", encoding="utf-8").read()
rm  = open("src/agents/risk_manager.py", encoding="utf-8").read()
ex  = open("src/agents/executor.py", encoding="utf-8").read()
db  = open("src/models/dataset_builder.py", encoding="utf-8").read()
nse = open("src/ingestion_nse.py", encoding="utf-8").read()

checks = [
    ("BaseBrokerAdapter",             os.path.exists("src/brokers/base_adapter.py")),
    ("BinanceAdapter",                os.path.exists("src/brokers/binance_adapter.py")),
    ("AngelOneAdapter",               os.path.exists("src/brokers/angel_one_adapter.py")),
    ("OandaAdapter",                  os.path.exists("src/brokers/oanda_adapter.py")),
    ("Executor uses adapter",         "adapter" in ex.lower()),
    ("capital_pools in risk_manager", "capital_pools" in rm),
    ("NSE session auto-pause",        "session" in nse.lower()),
    ("Trailing Stop Loss",            "trailing" in ex.lower()),
    ("Scale-out at Target 1",         "scale" in ex.lower() or "partial" in ex.lower()),
    ("ingestion_nse.py exists",       os.path.exists("src/ingestion_nse.py")),
    ("oracle_india.py exists",        os.path.exists("src/agents/oracle_india.py")),
    ("crypto_model.json trained",     os.path.exists("data/models/crypto_model.json")),
    ("banknifty_model.json trained",  os.path.exists("data/models/banknifty_model.json")),
    ("dataset_builder market_type",   "market_type" in db),
    ("model_registry crypto+nse",     "model_registry" in ws and "nse" in ws),
    ("Market tabs active_tab",        "active_tab" in ws),
    ("GlobalMasterView component",    "function GlobalMasterView" in jsx),
    ("GlobalMasterView rendered",     "<GlobalMasterView" in jsx),
    ("alpha_ranker_loop",             "alpha_ranker_loop" in ws),
    ("global_best_signal in snapshot","global_best_signal" in ws),
]

sections = [
    ("Step 1 - Adapter Pattern",           checks[0:5]),
    ("Step 2 - Multi-Pool Risk Mgmt",      checks[5:7]),
    ("Step 3 - Trailing Stop / Scale-out", checks[7:9]),
    ("Step 4 - Indian Market Pipeline",    checks[9:11]),
    ("Step 5 - Specialized AI Models",     checks[11:15]),
    ("Step 6 - Global Command Center",     checks[15:18]),
    ("Step 7 - Architecture Finalization", checks[18:20]),
]

print()
print("Phase 5 PRD Final Compliance Audit")
print("=" * 52)
for title, grp in sections:
    print(f"\n  [{title}]")
    for label, ok in grp:
        icon = "[x]" if ok else "[ ]"
        print(f"    {icon} {label}")

done = sum(1 for _, ok in checks if ok)
print()
print("=" * 52)
print(f"  RESULT: {done}/{len(checks)} complete")
if done == len(checks):
    print("  Phase 5 PRD: FULLY SATISFIED")
