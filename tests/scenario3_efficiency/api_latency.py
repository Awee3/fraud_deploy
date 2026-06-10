"""1000 request sekuensial; hitung p50/p95/p99."""
import requests
import time
import statistics
import json
import sys

MODE = sys.argv[1]      # "with_middleware" / "without_middleware"
API_URL = "http://localhost:5000/predict"
N = 1000
OUTPUT_FILE = f"tests/results/s3_latency_{MODE}.json"

with open("tests/fixtures/sample_transaction.json") as f:
    payload = json.load(f)

lat, err = [], 0
for i in range(N):
    start = time.time()
    try:
        r = requests.post(API_URL, json=payload, timeout=5)
        if r.status_code == 200:
            lat.append((time.time() - start) * 1000)
        else:
            err += 1
    except Exception:
        err += 1
    if i % 100 == 0 and i > 0:
        print(f"  {i}/{N}")

s = sorted(lat)
res = {"mode": MODE, "n_success": len(lat), "n_errors": err,
       "mean_ms": statistics.mean(lat), "median_ms": statistics.median(lat),
       "stdev_ms": statistics.stdev(lat) if len(lat) > 1 else 0,
       "p50_ms": s[int(len(s)*0.50)], "p95_ms": s[int(len(s)*0.95)],
       "p99_ms": s[int(len(s)*0.99)], "max_ms": max(lat)}
with open(OUTPUT_FILE, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))