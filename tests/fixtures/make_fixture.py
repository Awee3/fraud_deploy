"""Buat sample_transaction.json dari 1 baris simulasi_normal.csv (Amount MENTAH).
Payload dikirim ke /predict yang akan men-scale Amount sendiri.
Jalankan dari root fraud_deploy/.
"""
import pandas as pd
import json

df = pd.read_csv("simulation/simulasi_normal.csv")
drop = [c for c in ["Time", "Class"] if c in df.columns]
row = df.drop(columns=drop).iloc[0].to_dict()
with open("tests/fixtures/sample_transaction.json", "w") as f:
    json.dump(row, f, indent=2)
print(f"Saved sample_transaction.json ({len(row)} fitur). Amount(mentah)={row.get('Amount')}")
