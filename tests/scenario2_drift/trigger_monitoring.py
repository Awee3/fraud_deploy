"""Trigger workflow monitoring n8n dan capture response."""
import requests
import json
import sys
from datetime import datetime

TAG = sys.argv[1]
N8N_URL = "http://localhost:5678/webhook/run-monitoring"   # sesuaikan path webhook
OUTPUT_FILE = f"tests/results/s2_monitoring_{TAG}.json"

t_trigger = datetime.now()
print(f"[{t_trigger}] Trigger monitoring webhook…")
r = requests.post(N8N_URL, timeout=120)
t_resp = datetime.now()

payload = {"tag": TAG,
           "ts_trigger": t_trigger.isoformat(timespec="milliseconds"),
           "ts_response": t_resp.isoformat(timespec="milliseconds"),
           "trigger_to_response_s": (t_resp - t_trigger).total_seconds(),
           "status_code": r.status_code,
           "body": r.json() if r.headers.get("content-type","").startswith("application/json") else r.text}
with open(OUTPUT_FILE, "w") as f:
    json.dump(payload, f, indent=2, default=str)
print(json.dumps(payload, indent=2, default=str))