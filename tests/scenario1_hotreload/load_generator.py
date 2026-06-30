"""Beban konkuren ke /predict selama DURATION_SECONDS. Jalankan dari root fraud_deploy/."""
import asyncio
import httpx
import time
import json
from datetime import datetime

API_URL = "http://localhost:5000/predict"
DURATION_SECONDS = 90
REQUESTS_PER_SECOND = 20
OUTPUT_FILE = "tests/results/s1_load.jsonl"

with open("tests/fixtures/sample_transaction.json") as f:
    SAMPLE_PAYLOAD = json.load(f)

async def send_request(client, request_id):
    start = time.time()
    rec = {"request_id": request_id,
           "ts_send": datetime.now().isoformat(timespec="milliseconds"),
           "status_code": None, "latency_ms": None, "success": False, "error": None}
    try:
        resp = await client.post(API_URL, json=SAMPLE_PAYLOAD, timeout=10.0)
        rec["status_code"] = resp.status_code
        rec["latency_ms"] = (time.time() - start) * 1000
        rec["success"] = resp.status_code == 200
    except Exception as e:
        rec["latency_ms"] = (time.time() - start) * 1000
        rec["error"] = str(e)
    return rec

async def main():
    import os
    os.makedirs("tests/results", exist_ok=True)
    async with httpx.AsyncClient() as client:
        results, request_id = [], 0
        t0 = time.time()
        while time.time() - t0 < DURATION_SECONDS:
            batch_start = time.time()
            batch = []
            for _ in range(REQUESTS_PER_SECOND):
                request_id += 1
                batch.append(send_request(client, request_id))
            results.extend(await asyncio.gather(*batch))
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                await asyncio.sleep(1.0 - elapsed)
    with open(OUTPUT_FILE, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    total = len(results); ok = sum(1 for r in results if r["success"])
    print(f"Total: {total} | Success: {ok} | Failed: {total - ok}")

if __name__ == "__main__":
    asyncio.run(main())
