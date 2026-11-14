"""
make_drift_neg.py — Membuat simulasi_drift_neg.csv (shift -3 sigma).

Tujuan: validasi POSITIF jalur prediction drift. Pada simulasi_drift.csv yang
asli, shift +3 sigma pada V4/V12/V14 tidak mengubah proporsi prediksi fraud
(Fisher p = 1,0). Hipotesis: shift ke arah SEBALIKNYA (-3 sigma) mendorong
sampel ke wilayah fraud (V12 dan V14 pada data fraud bernilai sangat negatif),
sehingga fraud rate produksi melonjak dan prediction drift terdeteksi.

Konvensi mengikuti train_and_generate.py:
  - shift dihitung sebagai 3 x std per fitur
  - diterapkan pada fitur Tier-1 (dibaca dari baseline, bukan hardcode)
  - Amount tetap mentah
Catatan: std di sini dihitung dari simulasi_normal.csv, bukan X_train.
Keduanya berasal dari split stratified dataset yang sama, jadi nilainya
hampir identik. Untuk percobaan validasi ini perbedaannya tidak material.

Jalankan dari root fraud_deploy/:
    python tests/scenario2_drift/make_drift_neg.py
"""
import json
import pandas as pd
from pathlib import Path

NORMAL   = Path("simulation/simulasi_normal.csv")
BASELINE = Path("monitoring_config/static_allbaseline.json")
OUT      = Path("simulation/simulasi_drift_neg.csv")

df_full = pd.read_csv(NORMAL)

with open(BASELINE) as f:
    baseline = json.load(f)
tier1 = baseline["meta"]["tier1_features"]

colmap = {c.lower(): c for c in df_full.columns}

# Subset 2000 baris pertama, konsisten dengan simulasi_normal_2k.csv di S2A.
# Sigma tetap dihitung dari file PENUH agar besaran shift konsisten dengan
# metode train_and_generate.py, bukan dari subset kecil.
df = df_full.head(2000).copy()

print(f"Fitur Tier-1 dari baseline: {tier1}")
for feat in tier1:
    col = colmap[feat]
    sigma = df_full[col].std()          # sigma dari data penuh
    df[col] = df[col] - 3 * sigma
    print(f"  -3 sigma shift '{col}': -{3 * sigma:.4f}")

df.to_csv(OUT, index=False)
print(f"\nTersimpan: {OUT} ({len(df)} baris)")
print("Selanjutnya:")
print("  1. python tests/utils/reset_state.py")
print("  2. python simulation/simulate_traffic.py simulasi_drift_neg.csv")
print("  3. curl.exe -X POST \"http://localhost:5000/run-monitoring?limit=1000&window_minutes=60\"")
print("  4. Setelah selesai: reset state lagi agar tidak mencemari pengujian lain")