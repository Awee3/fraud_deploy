"""Replay CSV ke /predict dengan throttling; catat waktu request terakhir."""
import pandas as pd
import requests
import time
import sys
import json
from datetime import datetime

CSV_FILE = sys.argv[1]          # path csv
TAG = sys.argv[2]               # "normal" / "drift"
API_URL = "http://localhost:5000/predict"
RATE = 20
OUTPUT_FILE = f"tests/results/s2_replay_{TAG}.json"

df = pd.read_csv(CSV_FILE)
if "Class" in df.columns:
    df = df.drop(columns=["Class"])

success = fail = 0
t_start = datetime.now()
print(f"[{t_start}] Replay {len(df)} rows from {CSV_FILE} @ {RATE} req/s")
for idx, row in df.iterrows():
    try:
        r = requests.post(API_URL, json=row.to_dict(), timeout=5)
        success += int(r.status_code == 200); fail += int(r.status_code != 200)
    except Exception:
        fail += 1
    if idx % 500 == 0 and idx > 0:
        print(f"  {idx}/{len(df)} | ok={success} fail={fail}")
    time.sleep(1.0 / RATE)
t_end = datetime.now()

with open(OUTPUT_FILE, "w") as f:
    json.dump({"csv": CSV_FILE, "tag": TAG,
               "ts_start": t_start.isoformat(timespec="milliseconds"),
               "ts_end": t_end.isoformat(timespec="milliseconds"),
               "duration_s": (t_end - t_start).total_seconds(),
               "success": success, "fail": fail}, f, indent=2)
print(f"\n[{t_end}] Done. ok={success} fail={fail}")
print(f">>> Time to Detect: ts_end = {t_end.isoformat(timespec='milliseconds')} (catat jam Telegram alert)")