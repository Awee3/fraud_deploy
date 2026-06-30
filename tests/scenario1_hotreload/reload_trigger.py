"""Trigger /reload ke model baru setelah DELAY_SECONDS; rekam model_path before/after.
Butuh model kedua di models/ (mis. xgboost_model_v2.pkl). Untuk sekadar menguji
MEKANISME reload, boleh menyalin xgboost_model.pkl jadi xgboost_model_v2.pkl.
Jalankan dari root fraud_deploy/.
"""
import requests
import time
import json
import os
from datetime import datetime

HEALTH_URL = "http://localhost:5000/health"
RELOAD_URL = "http://localhost:5000/reload"
NEW_MODEL  = "xgboost_model_v2.pkl"      # file harus ada di models/
DELAY_SECONDS = 30
OUTPUT_FILE = "tests/results/s1_reload.json"

os.makedirs("tests/results", exist_ok=True)
print(f"[{datetime.now()}] Sleeping {DELAY_SECONDS}s…")
time.sleep(DELAY_SECONDS)

before = requests.get(HEALTH_URL, timeout=3).json()
print(f"[{datetime.now()}] model_path BEFORE: {before.get('model_path')}")

t_start = datetime.now()
r = requests.post(RELOAD_URL, params={"model_name": NEW_MODEL}, timeout=30)
t_end = datetime.now()
print(f"[{t_end}] /reload -> {r.status_code} {r.json()}")

after = requests.get(HEALTH_URL, timeout=3).json()
print(f"[{datetime.now()}] model_path AFTER: {after.get('model_path')}")

json.dump({"trigger_start": t_start.isoformat(timespec="milliseconds"),
           "trigger_end": t_end.isoformat(timespec="milliseconds"),
           "trigger_duration_ms": (t_end - t_start).total_seconds() * 1000,
           "model_path_before": before.get("model_path"),
           "model_path_after": after.get("model_path"),
           "reload_status_code": r.status_code}, open(OUTPUT_FILE, "w"), indent=2)
print("Kembalikan model aktif setelah uji: "
      'curl -X POST "http://localhost:5000/reload?model_name=xgboost_model.pkl"')
