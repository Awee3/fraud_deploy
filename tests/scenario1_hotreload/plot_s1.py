"""
plot_s1.py — Plot latency vs waktu untuk Skenario 1 (Hot-Reload), siap slide.
Membaca tests/results/s1_load.jsonl + tests/results/s1_reload.json.
Output: tests/results/s1_latency_plot.png

Jalankan dari root fraud_deploy/:
    python tests/scenario1_hotreload/plot_s1.py
(butuh: pandas, matplotlib  →  python -m pip install matplotlib)
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")               # tidak butuh display
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

LOAD_FILE   = Path("tests/results/s1_load.jsonl")
RELOAD_FILE = Path("tests/results/s1_reload.json")
OUT_FILE    = Path("tests/results/s1_latency_plot.png")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_json(LOAD_FILE, lines=True)
df["ts_send"] = pd.to_datetime(df["ts_send"])
df = df.sort_values("ts_send").reset_index(drop=True)
# detik relatif sejak request pertama (sumbu-x lebih mudah dibaca dari timestamp)
t0 = df["ts_send"].iloc[0]
df["t_rel"] = (df["ts_send"] - t0).dt.total_seconds()

reload_info  = json.load(open(RELOAD_FILE))
reload_start = (pd.to_datetime(reload_info["trigger_start"]) - t0).total_seconds()
reload_end   = (pd.to_datetime(reload_info["trigger_end"])   - t0).total_seconds()

ok   = df[df["success"]]
fail = df[~df["success"]]

# ── Statistik untuk anotasi ─────────────────────────────────────────────────────
total   = len(df)
n_ok    = int(df["success"].sum())
p99     = df["latency_ms"].quantile(0.99)
mean_ms = df["latency_ms"].mean()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.5))

# titik latensi tiap request
ax.scatter(ok["t_rel"], ok["latency_ms"], s=12, alpha=0.55,
           color="#2563eb", label="Request sukses", zorder=3)
if len(fail):
    ax.scatter(fail["t_rel"], fail["latency_ms"], s=40, color="#dc2626",
               marker="x", label="Request gagal", zorder=4)

# pita area saat reload berlangsung
ax.axvspan(reload_start, reload_end, color="#f59e0b", alpha=0.18, zorder=1)
ax.axvline(reload_start, color="#d97706", ls="--", lw=1.6,
           label="Hot-reload model", zorder=2)
ax.annotate("Reload model\n(tanpa downtime)",
            xy=(reload_start, ax.get_ylim()[1] * 0.92),
            xytext=(reload_start + 4, ax.get_ylim()[1] * 0.92),
            fontsize=9, color="#92400e", va="top")

# garis p99 (label di tengah dengan latar putih agar terbaca & tak bertabrakan)
ax.axhline(p99, color="#6b7280", ls=":", lw=1.2, zorder=2)
x_mid = df["t_rel"].min() + 0.62 * (df["t_rel"].max() - df["t_rel"].min())
ax.annotate(f"p99 = {p99:.0f} ms", xy=(x_mid, p99),
            xytext=(x_mid, p99 + ax.get_ylim()[1] * 0.02),
            fontsize=9, color="#374151", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set_xlabel("Waktu sejak mulai (detik)", fontsize=11)
ax.set_ylabel("Latency (ms)", fontsize=11)
ax.set_title("Skenario 1 — Latency vs Waktu Selama Hot-Reload Model",
             fontsize=13, fontweight="bold", pad=12)

# kotak ringkasan hasil
summary = (f"Total request : {total}\n"
           f"Sukses        : {n_ok} ({n_ok/total*100:.1f}%)\n"
           f"Gagal         : {total - n_ok}\n"
           f"Mean latency  : {mean_ms:.0f} ms")
ax.text(0.015, 0.97, summary, transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f3f4f6", ec="#9ca3af", alpha=0.95))

ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax.grid(True, alpha=0.25, zorder=0)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT_FILE, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_FILE}  ({total} request, {n_ok} sukses, {total-n_ok} gagal)")