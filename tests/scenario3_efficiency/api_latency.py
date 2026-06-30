"""1000 request sekuensial ke /predict; hitung p50/p95/p99.
Pakai: python tests/scenario3_efficiency/api_latency.py with_middleware|without_middleware
Toggle middleware: comment/uncomment app.add_middleware(MetricsMiddleware) di main.py lalu
    docker compose up -d --build fraud-api
Jalankan dari root fraud_deploy/.
"""
import requests
import time
import statistics
import json
import sys
import os

MODE = sys.argv[1] if len(sys.argv) > 1 else "with_middleware"
API_URL = "http://localhost:5000/predict"
N = 1000
os.makedirs("tests/results", exist_ok=True)
OUTPUT_FILE = f"tests/results/s3_latency_{MODE}.json"

payload = json.load(open("tests/fixtures/sample_transaction.json"))
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
       "mean_ms": round(statistics.mean(lat), 3),
       "median_ms": round(statistics.median(lat), 3),
       "stdev_ms": round(statistics.stdev(lat), 3) if len(lat) > 1 else 0,
       "p50_ms": round(s[int(len(s)*0.50)], 3),
       "p95_ms": round(s[int(len(s)*0.95)], 3),
       "p99_ms": round(s[int(len(s)*0.99)], 3),
       "max_ms": round(max(lat), 3)}
json.dump(res, open(OUTPUT_FILE, "w"), indent=2)
print(json.dumps(res, indent=2))
