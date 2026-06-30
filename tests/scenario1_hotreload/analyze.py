"""Agregasi hasil load + reload. Jalankan dari root fraud_deploy/."""
import json
import pandas as pd

df = pd.read_json("tests/results/s1_load.jsonl", lines=True)
info = json.load(open("tests/results/s1_reload.json"))
df["ts_send"] = pd.to_datetime(df["ts_send"])
t_start = pd.to_datetime(info["trigger_start"])
t_end = pd.to_datetime(info["trigger_end"])
win = df[(df["ts_send"] >= t_start - pd.Timedelta(seconds=2)) &
         (df["ts_send"] <= t_end + pd.Timedelta(seconds=3))]

total, ok = len(df), int(df["success"].sum())
print("=== Overall ===")
print(f"  Total {total} | Success {ok} ({ok/total*100:.2f}%) | Fail {total-ok}")
print(f"  Mean {df['latency_ms'].mean():.2f} | p50 {df['latency_ms'].quantile(.5):.2f} | "
      f"p95 {df['latency_ms'].quantile(.95):.2f} | p99 {df['latency_ms'].quantile(.99):.2f} ms")
print("=== Reload Window ===")
print(f"  Requests {len(win)} | Failures {int((~win['success']).sum())} | "
      f"Max latency {win['latency_ms'].max():.2f} ms")
print(f"=== Version: {info['model_path_before']} -> {info['model_path_after']} ===")
