"""Kosongkan tabel DB sebelum tiap skenario (backup dulu).
Default: kosongkan predictions + request_metrics (deployment_metrics DIPERTAHANKAN).
Untuk hapus semua: python tests/utils/reset_state.py predictions request_metrics deployment_metrics
Jalankan dari root fraud_deploy/.
"""
import sqlite3
import shutil
import sys
from pathlib import Path
from datetime import datetime

DB_PATH = "logs/predictions.db"
BACKUP_DIR = Path("tests/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

tables = sys.argv[1:] or ["predictions", "request_metrics"]
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(DB_PATH, BACKUP_DIR / f"predictions_{ts}.db")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
for t in tables:
    cur.execute(f"DELETE FROM {t};")
    print(f"  cleared: {t}")
conn.commit()
conn.close()
print(f"[{datetime.now()}] Reset done. Backup: predictions_{ts}.db | tables: {tables}")
