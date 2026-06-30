"""Tentukan fitur Tier-1 (yang di-shift di simulasi_drift) + cek skala Amount.
Jalankan dari root fraud_deploy/:
    python tests/utils/find_drift_features.py
"""
import json
import pandas as pd

NORMAL   = "simulation/simulasi_normal.csv"
DRIFT    = "simulation/simulasi_drift.csv"
BASELINE = "monitoring_config/static_allbaseline.json"

normal = pd.read_csv(NORMAL)
drift  = pd.read_csv(DRIFT)
drop = [c for c in ["Class", "Time"] if c in normal.columns]
normal = normal.drop(columns=drop)
drift  = drift.drop(columns=[c for c in drop if c in drift.columns])

print("=== Perbandingan mean per kolom (normal vs drift) ===")
print(f"{'feature':<10}{'mean_normal':>14}{'mean_drift':>14}{'delta':>14}")
shifted = []
for col in normal.columns:
    mn, md = normal[col].mean(), drift[col].mean()
    delta = md - mn
    mark = ""
    if abs(delta) > 1e-6:
        mark = "  <-- SHIFTED"
        shifted.append(col)
    print(f"{col:<10}{mn:>14.6f}{md:>14.6f}{delta:>14.6f}{mark}")

print(f"\n>>> Fitur yang di-shift (kandidat Tier-1): {shifted}")
print(f">>> Untuk monitoring.py: TIER1_FEATURES = {[c.lower() for c in shifted]}")

# ── Bonus: cek konsistensi skala Amount (baseline vs simulasi) ──
print("\n=== Cek skala Amount ===")
try:
    baseline = json.load(open(BASELINE))
    amt = baseline.get("features", {}).get("amount", {})
    b_mean = amt.get("stats", {}).get("mean")
    b_std  = amt.get("stats", {}).get("std")
    s_mean = normal["Amount"].mean() if "Amount" in normal.columns else None
    print(f"  baseline amount mean / std : {b_mean} / {b_std}")
    print(f"  simulasi amount mean        : {s_mean}")
    if b_mean is not None and s_mean is not None:
        if abs(b_mean) < 5 and abs(s_mean) < 5:
            print("  -> keduanya tampak SCALED (mean ~0). Konsisten.")
        elif abs(b_mean) > 20 or abs(s_mean) > 20:
            print("  -> ada yang RAW (mean jauh dari 0). Periksa konsistensi!")
        else:
            print("  -> periksa manual.")
except FileNotFoundError:
    print(f"  (Baseline {BASELINE} tidak ditemukan.)")
