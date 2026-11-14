"""make_prior_shift.py — Uji positif prediction drift via prior probability shift.
Oversample baris fraud ASLI dari simulasi_normal.csv hingga prevalensi ~5%.
Jalankan dari root fraud_deploy/:
    python tests/scenario2_drift/make_prior_shift.py
"""
import pandas as pd

NORMAL = "simulation/simulasi_normal.csv"
OUT    = "simulation/simulasi_prior_shift.csv"
N_OUT  = 2000
FRAUD_TARGET = 0.05          # prevalensi target 5% (baseline 0,17%)

df = pd.read_csv(NORMAL)
fraud  = df[df["Class"] == 1]
normal = df[df["Class"] == 0]
print(f"Tersedia: {len(fraud)} fraud, {len(normal)} normal")

n_fraud  = int(N_OUT * FRAUD_TARGET)          # 100 baris fraud
n_normal = N_OUT - n_fraud                    # 1900 baris normal

fraud_s  = fraud.sample(n=n_fraud, replace=len(fraud) < n_fraud, random_state=42)
normal_s = normal.sample(n=n_normal, random_state=42)

out = pd.concat([fraud_s, normal_s]).sample(frac=1, random_state=42)  # shuffle
out.to_csv(OUT, index=False)
print(f"Tersimpan: {OUT} ({len(out)} baris, fraud rate {out['Class'].mean():.4f})")