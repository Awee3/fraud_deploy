"""Trigger /reload setelah DELAY_SECONDS; rekam versi sebelum & sesudah."""
import requests
import time
import json
from datetime import datetime

HEALTH_URL = "http://localhost:5000/health"
RELOAD_URL = "http://localhost:5000/reload"
DELAY_SECONDS = 30
OUTPUT_FILE = "tests/results/s1_reload.json"

print(f"[{datetime.now()}] Sleeping {DELAY_SECONDS}s before trigger…")
time.sleep(DELAY_SECONDS)

v_before = requests.get(HEALTH_URL, timeout=3).json()
print(f"[{datetime.now()}] /health BEFORE: {v_before}")

t_start = datetime.now()
r = requests.post(RELOAD_URL, timeout=30)
t_end = datetime.now()
print(f"[{t_end}] /reload returned {r.status_code}")

v_after = requests.get(HEALTH_URL, timeout=3).json()
print(f"[{datetime.now()}] /health AFTER: {v_after}")

with open(OUTPUT_FILE, "w") as f:
    json.dump({
        "trigger_start": t_start.isoformat(timespec="milliseconds"),
        "trigger_end": t_end.isoformat(timespec="milliseconds"),
        "trigger_duration_ms": (t_end - t_start).total_seconds() * 1000,
        "version_before": v_before, "version_after": v_after,
        "reload_status_code": r.status_code,
    }, f, indent=2)