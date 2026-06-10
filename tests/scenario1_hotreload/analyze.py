"""Agregasi hasil load + reload jadi laporan ringkas."""
import json
import pandas as pd

df = pd.read_json("tests/results/s1_load.jsonl", lines=True)
with open("tests/results/s1_reload.json") as f:
    reload_info = json.load(f)

df["ts_send"] = pd.to_datetime(df["ts_send"])
t_start = pd.to_datetime(reload_info["trigger_start"])
t_end = pd.to_datetime(reload_info["trigger_end"])
win = df[(df["ts_send"] >= t_start - pd.Timedelta(seconds=2)) &
         (df["ts_send"] <= t_end + pd.Timedelta(seconds=3))]

total, success = len(df), int(df["success"].sum())
print("=== Overall ===")
print(f"  Total: {total} | Success: {success} ({success/total*100:.2f}%) | Fail: {total-success}")
print(f"  Mean: {df['latency_ms'].mean():.2f} ms | p50: {df['latency_ms'].quantile(.5):.2f} | "
      f"p95: {df['latency_ms'].quantile(.95):.2f} | p99: {df['latency_ms'].quantile(.99):.2f}")
print("\n=== Reload Window ===")
print(f"  Requests: {len(win)} | Failures: {int((~win['success']).sum())} | "
      f"Max latency: {win['latency_ms'].max():.2f} ms")
print("\n=== Version Transition ===")
ver = df.dropna(subset=["model_version"]).sort_values("ts_send")
print(ver.groupby("model_version").agg(
    first_seen=("ts_send", "min"), last_seen=("ts_send", "max"), count=("request_id", "count")))