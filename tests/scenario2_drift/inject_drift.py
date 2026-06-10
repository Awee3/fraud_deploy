"""Inject 3σ positive mean shift pada Tier-1 features (V14, V10, V12)."""
import pandas as pd
import json

TIER1 = ["V14", "V10", "V12"]
with open("monitoring_config/static_baseline.json") as f:
    baseline = json.load(f)

df = pd.read_csv("simulation/simulation_set.csv")
df_drift = df.sample(n=5000, random_state=123).reset_index(drop=True)

print(f"{'Fitur':<6}{'baseline_μ':>12}{'baseline_σ':>12}{'shift':>10}{'old_μ':>12}{'new_μ':>12}")
for feat in TIER1:
    mu, sigma = baseline[feat]["mean"], baseline[feat]["std"]   # sesuaikan struktur JSON
    shift = 3 * sigma
    old_mu = df_drift[feat].mean()
    df_drift[feat] = df_drift[feat] + shift
    print(f"{feat:<6}{mu:>12.4f}{sigma:>12.4f}{shift:>10.4f}{old_mu:>12.4f}{df_drift[feat].mean():>12.4f}")

df_drift.to_csv("simulation/simulasi_drift.csv", index=False)
print(f"\nSaved {len(df_drift)} rows -> simulation/simulasi_drift.csv")