"""Trigger webhook n8n & poll /health sampai versi target aktif."""
import requests
import time
import json
from datetime import datetime

N8N_WEBHOOK = "http://localhost:5678/webhook/deploy-model"
HEALTH_URL  = "http://localhost:5000/health"
PREDICT_URL = "http://localhost:5000/predict"
TARGET_VERSION = "rf_model_v4"     # set ke versi yg akan di-deploy
MAX_WAIT, POLL = 300, 0.5

with open("tests/fixtures/sample_transaction.json") as f:
    sample = json.load(f)

print(f"[{datetime.now()}] BEFORE: {requests.get(HEALTH_URL, timeout=3).json()}")
t0 = datetime.now()
print(f"[{t0}] T0: trigger n8n webhook")
print(f"  Webhook: {requests.post(N8N_WEBHOOK, timeout=10).status_code}")

t_active = None
while (datetime.now() - t0).total_seconds() < MAX_WAIT:
    try:
        h = requests.get(HEALTH_URL, timeout=2).json()
        if h.get("model_version") == TARGET_VERSION:
            t_active = datetime.now(); break
    except Exception:
        pass
    time.sleep(POLL)

if t_active is None:
    print(f"TIMEOUT after {MAX_WAIT}s")
else:
    smoke = requests.post(PREDICT_URL, json=sample, timeout=5)
    t_smoke = datetime.now()
    print(f"\n=== Lead Time AUTOMATED ===")
    print(f"  T0 -> active : {(t_active - t0).total_seconds():.2f}s")
    print(f"  T0 -> smoke  : {(t_smoke - t0).total_seconds():.2f}s (smoke {smoke.status_code})")