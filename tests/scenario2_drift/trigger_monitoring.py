"""Panggil /run-monitoring LANGSUNG (deterministik) untuk capture report + p-value.
Telegram alert dikirim oleh workflow n8n (IF status==DRIFT_DETECTED); untuk screenshot,
jalankan workflow monitoring n8n manual (Execute Workflow) setelah replay.
Pakai: python tests/scenario2_drift/trigger_monitoring.py normal|drift
Jalankan dari root fraud_deploy/.
"""
import requests
import json
import sys
import os
from datetime import datetime

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
MON_URL = "http://localhost:5000/run-monitoring"
os.makedirs("tests/results", exist_ok=True)
OUTPUT_FILE = f"tests/results/s2_monitoring_{TAG}.json"

t = datetime.now()
r = requests.post(MON_URL, params={"limit": 1000, "window_minutes": 60}, timeout=120)
report = r.json()
json.dump({"tag": TAG, "ts": t.isoformat(timespec="milliseconds"),
           "status_code": r.status_code, "report": report},
          open(OUTPUT_FILE, "w"), indent=2, default=str)
print(f"[{datetime.now()}] status_code={r.status_code} | status={report.get('status')} "
      f"| alert_level={report.get('alert_level')}")
print("tier1_drifted:", report.get("tier1_drifted"))
print("tier2_drifted:", report.get("tier2_drifted"), f"(butuh >={report.get('tier2_min_required')})")
print("trigger_reason:", report.get("trigger_reason"))
