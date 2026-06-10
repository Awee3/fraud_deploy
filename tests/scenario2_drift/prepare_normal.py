"""Ambil 5000 sampel acak dari simulation set (holdout) untuk skenario normal."""
import pandas as pd

df = pd.read_csv("simulation/simulation_set.csv")   # holdout 20% yg di-export ke fraud_deploy
df_normal = df.sample(n=5000, random_state=42).reset_index(drop=True)
df_normal.to_csv("simulation/simulasi_normal.csv", index=False)
print(f"Saved {len(df_normal)} rows -> simulation/simulasi_normal.csv")
print(df_normal[["V14", "V10", "V12"]].mean())