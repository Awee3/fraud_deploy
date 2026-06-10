"""Stopwatch manual dengan checkpoint per langkah deployment manual."""
from datetime import datetime

stages = [
    "T0: commit/tag siap (start)",
    "T1: SSH ke server",
    "T2: git pull selesai",
    "T3: model file di-copy ke models/",
    "T4: docker compose restart selesai",
    "T5: /health 200 + versi sesuai",
    "T6: /predict smoke test PASS",
]
print("Lead Time MANUAL — tekan ENTER tiap milestone:\n")
times = []
for s in stages:
    input(f"  [{s}] ENTER…")
    t = datetime.now(); times.append(t)
    print(f"    @ {t.isoformat(timespec='milliseconds')}")
print("\n=== Durasi per stage ===")
for i in range(1, len(times)):
    print(f"  {stages[i-1][:3]} -> {stages[i][:3]}: {(times[i]-times[i-1]).total_seconds():.2f}s")
print(f"\n=== TOTAL Lead Time (Manual): {(times[-1]-times[0]).total_seconds():.2f}s ===")