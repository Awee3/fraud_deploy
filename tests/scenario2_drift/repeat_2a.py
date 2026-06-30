"""
repeat_2a.py — Jalankan Skenario 2A (negative control) berulang N kali.
Tiap ulangan: reset DB → replay simulasi_normal → /run-monitoring → catat status.
Tujuan: mengukur tingkat false-positive secara empiris (idealnya 0 alarm dari N run).

Jalankan dari root fraud_deploy/:
    python tests/scenario2_drift/repeat_2a.py
    python tests/scenario2_drift/repeat_2a.py 10        # 10 ulangan
    python tests/scenario2_drift/repeat_2a.py 10 simulasi_normal_2k.csv

Catatan:
- Memanggil reset_state.py & simulate_traffic.py sebagai subprocess (identik dgn manual).
- Subprocess dipaksa UTF-8 agar emoji di simulate_traffic.py tidak crash di Windows.
- Atur delay replay (PowerShell): $env:DELAY_SECONDS="0.02"
"""
import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

import requests

# ── Konfigurasi ────────────────────────────────────────────────────────────────
N_RUNS    = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CSV_NAME  = sys.argv[2] if len(sys.argv) > 2 else "simulasi_normal_2k.csv"
MON_URL   = "http://localhost:5000/run-monitoring"
PY        = sys.executable
RESET     = ["tests/utils/reset_state.py"]
REPLAY    = ["simulation/simulate_traffic.py", CSV_NAME]
CSV_PATH  = Path("simulation") / CSV_NAME
OUT_FILE  = Path("tests/results/s2a_repeat.json")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Paksa subprocess Python memakai UTF-8 (perbaikan emoji-crash di Windows saat output dialihkan)
ENV = os.environ.copy()
ENV["PYTHONUTF8"] = "1"
ENV["PYTHONIOENCODING"] = "utf-8"


def run_step(args: list, label: str) -> None:
    """Jalankan subprocess; jika gagal, tampilkan output asli lalu hentikan."""
    res = subprocess.run([PY] + args, env=ENV, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print(f"\n[X] Langkah '{label}' GAGAL (exit {res.returncode}).")
        if res.stdout.strip():
            print("--- stdout ---\n" + res.stdout.strip())
        if res.stderr.strip():
            print("--- stderr ---\n" + res.stderr.strip())
        sys.exit(1)


# ── Pre-check ──────────────────────────────────────────────────────────────────
if not CSV_PATH.exists():
    print(f"[X] File tidak ditemukan: {CSV_PATH}")
    print("   Buat subset dulu, mis:")
    print("   python -c \"import pandas as pd; "
          f"pd.read_csv('simulation/simulasi_normal.csv').head(2000)"
          f".to_csv('simulation/{CSV_NAME}', index=False)\"")
    sys.exit(1)

try:
    requests.get("http://localhost:5000/health", timeout=3)
except Exception:
    print("[X] API di http://localhost:5000 tidak merespons. Pastikan container up.")
    sys.exit(1)

# ── Loop ───────────────────────────────────────────────────────────────────────
results = []
print(f"=== Skenario 2A — {N_RUNS} ulangan (CSV: {CSV_NAME}) ===\n")

for i in range(1, N_RUNS + 1):
    print(f"[Run {i}/{N_RUNS}] {datetime.now().strftime('%H:%M:%S')}")
    run_step(RESET,  "reset_state")
    run_step(REPLAY, "replay")

    r = requests.post(MON_URL, params={"limit": 1000, "window_minutes": 60}, timeout=120)
    rep = r.json()
    status      = rep.get("status")
    alert_level = rep.get("alert_level")
    t1 = rep.get("tier1_drifted", [])
    t2 = rep.get("tier2_drifted", [])
    alarm = (status == "DRIFT_DETECTED")

    results.append({"run": i, "status": status, "alert_level": alert_level,
                    "tier1_drifted": t1, "tier2_drifted": t2,
                    "n_tier2_drift": len(t2), "alarm": alarm})

    flag = "ALARM (false positive)" if alarm else "OK (tidak ada alarm)"
    print(f"   status={status} | alert={alert_level} | "
          f"Tier-1={len(t1)} | Tier-2={len(t2)} | {flag}")

# ── Ringkasan ──────────────────────────────────────────────────────────────────
n_alarm = sum(1 for x in results if x["alarm"])
print("\n" + "=" * 56)
print(f"RINGKASAN: {n_alarm} alarm dari {N_RUNS} ulangan "
      f"({n_alarm / N_RUNS * 100:.1f}% false-positive)")
if n_alarm:
    print("Run yang memicu alarm:")
    for x in results:
        if x["alarm"]:
            print(f"  - Run {x['run']}: Tier-2 drift {x['n_tier2_drift']} -> {x['tier2_drifted']}")
print("=" * 56)

json.dump({"n_runs": N_RUNS, "csv": CSV_NAME, "n_alarm": n_alarm,
           "false_positive_rate": round(n_alarm / N_RUNS, 4),
           "runs": results}, open(OUT_FILE, "w"), indent=2)
print(f"\nDetail tersimpan: {OUT_FILE}")