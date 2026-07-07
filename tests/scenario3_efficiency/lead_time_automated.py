"""Trigger webhook n8n & deteksi record deployment baru di /metrics/deployment.
Jalankan dari root fraud_deploy/.
Usage: python lead_time_automated.py <nama_file_model.pkl>
"""
import requests
import time
import sys
from datetime import datetime

N8N_WEBHOOK = "http://localhost:5678/webhook/deploy-model"
DEPLOY_HIST = "http://localhost:5000/metrics/deployment"
MAX_WAIT, POLL = 300, 0.1

model_file = sys.argv[1] if len(sys.argv) > 1 else "xgboost_model.pkl"
PAYLOAD = {"commits": [{"modified": [f"models/{model_file}"], "added": []}]}

def latest_record():
    try:
        hist = requests.get(DEPLOY_HIST, params={"limit": 1}, timeout=3).json()
        if isinstance(hist, list) and hist:
            return hist[0]
    except Exception:
        pass
    return None

base = latest_record()
base_ts = base.get("timestamp") if base else None
t0 = datetime.now()
print(f"[{t0}] Trigger webhook untuk {model_file} (baseline deploy ts={base_ts})")
requests.post(N8N_WEBHOOK, json=PAYLOAD, timeout=10)

detected = None; rec = None
while (datetime.now() - t0).total_seconds() < MAX_WAIT:
    cur = latest_record()
    if cur and cur.get("timestamp") != base_ts:
        detected = datetime.now(); rec = cur; break
    time.sleep(POLL)

if detected is None:
    print("TIMEOUT — tidak ada record deployment baru. Cek workflow n8n.")
else:
    print("\n=== Lead Time AUTOMATED ===")
    print(f"  Model: {model_file}")
    print(f"  External wall-clock (T0 -> record): {(detected - t0).total_seconds():.2f}s")
    print(f"  Pipeline self-reported lead_time_seconds: {rec.get('lead_time_seconds')}")
    print(f"  Record: {rec}")