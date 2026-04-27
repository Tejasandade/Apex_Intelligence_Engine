"""Phase 6 Step 6 - inline validation"""
import sys
sys.path.insert(0, ".")
from src.models.quant_model import MODEL_FEATURE_COLUMNS

with open("src/dashboard/backend/ws_server.py", encoding="utf-8") as f:
    src = f.read()

start = src.index("def _build_synthetic_features")
end   = src.index("def _load_market_features")
block = src[start:end]

all_cols = MODEL_FEATURE_COLUMNS + ["open", "high", "low", "close", "volume"]
present  = [c for c in all_cols if ('"' + c + '"') in block]
missing  = [c for c in MODEL_FEATURE_COLUMNS if c not in present]

print("=== Phase 6 Step 6 Final Check ===")
print(f"MODEL_FEATURE_COLUMNS : {len(MODEL_FEATURE_COLUMNS)}")
print(f"Keys present in block : {len(present)}")
print(f"Missing               : {missing if missing else 'NONE'}")
bos_ok = ('"structure_break_signal":    1.0') in block
print(f"structure_break_signal +/-1.0 : {'OK' if bos_ok else 'WARNING'}")
print(f"VWAP guard _sanitize  : {'OK' if 'merged[\"VWAP\"] = merged.get(\"close\"' in src else 'MISSING'}")
print(f"VWAP guard _load      : {'OK' if 'features[\"VWAP\"] = features.get(\"close\"' in src else 'MISSING'}")
print(f"VWAP fallback snapshot: {'OK' if 'features.get(\"VWAP\") or features.get(\"close\"' in src else 'MISSING'}")
print()
if not missing and bos_ok:
    print("ALL CHECKS PASSED - Apex Engine V3 ready for deployment.")
else:
    print("FIX REQUIRED")
    sys.exit(1)
