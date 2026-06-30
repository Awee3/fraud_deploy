"""
measure_detection_latency.py — Ukur latensi komputasi /run-monitoring.
Memanggil endpoint N kali dan menghitung statistik waktu respons (ms):
mean, median, p95, min, max. Ini = waktu sistem menjalankan KS-Test 29 fitur +
Fisher's Exact pada 1000 log terakhir (BUKAN interval jadwal monitoring).

PRASYARAT: isi dulu DB dengan data (mis. replay simulasi_drift) supaya monitoring
benar-benar memproses 1000 log — agar latensi yang diukur realistis.

Jalankan dari root fraud_deploy/:
    python tests/scenario2_drift/measure_detection_latency.py
    python tests/scenario2_drift/measure_detection_latency.py 30   # 30 panggilan
"""
import sys
import json
import time
import statistics
from pathlib import Path
from datetime import datetime

import requests

N_CALLS  = int(sys.argv[1]) if len(sys.argv) > 1 else 20
WARMUP   = 2                      # panggilan pemanasan (tidak dihitung)
MON_URL  = "http://localhost:5000/run-monitoring"
PARAMS   = {"limit": 1000, "window_minutes": 60}
OUT_FILE = Path("tests/results/s2_detection_latency.json")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)


def one_call() -> tuple[float, str, int]:
    t0 = time.perf_counter()
    r = requests.post(MON_URL, params=PARAMS, timeout=120)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    status = r.json().get("status", "?") if r.status_code == 200 else f"HTTP{r.status_code}"
    return elapsed_ms, status, r.status_code


# ── Pre-check ──────────────────────────────────────────────────────────────────
try:
    requests.get("http://localhost:5000/health", timeout=3)
except Exception:
    print("[X] API tidak merespons di http://localhost:5000. Pastikan container up.")
    sys.exit(1)

print(f"=== Latensi /run-monitoring — {N_CALLS} panggilan (+{WARMUP} warmup) ===\n")

# warmup
for _ in range(WARMUP):
    one_call()

lat, statuses = [], []
for i in range(1, N_CALLS + 1):
    ms, status, code = one_call()
    lat.append(ms)
    statuses.append(status)
    print(f"  [{i:>2}/{N_CALLS}] {ms:8.1f} ms | status={status}")
    time.sleep(0.2)

# ── Statistik ──────────────────────────────────────────────────────────────────
s = sorted(lat)
def pct(p): return s[min(int(len(s) * p / 100), len(s) - 1)]

summary = {
    "n_calls":   N_CALLS,
    "mean_ms":   round(statistics.mean(lat), 1),
    "median_ms": round(statistics.median(lat), 1),
    "p95_ms":    round(pct(95), 1),
    "min_ms":    round(min(lat), 1),
    "max_ms":    round(max(lat), 1),
    "stdev_ms":  round(statistics.stdev(lat), 1) if len(lat) > 1 else 0,
    "statuses":  list(set(statuses)),
    "measured_at": datetime.now().isoformat(timespec="seconds"),
}

print("\n" + "=" * 48)
print(f"  Mean   : {summary['mean_ms']} ms")
print(f"  Median : {summary['median_ms']} ms")
print(f"  p95    : {summary['p95_ms']} ms")
print(f"  Min    : {summary['min_ms']} ms")
print(f"  Max    : {summary['max_ms']} ms")
print(f"  Stdev  : {summary['stdev_ms']} ms")
print(f"  Status report: {summary['statuses']}")
print("=" * 48)

json.dump(summary, open(OUT_FILE, "w"), indent=2)
print(f"\nTersimpan: {OUT_FILE}")
print("Catatan: ini latensi KOMPUTASI monitoring, bukan interval jadwal (harian).")