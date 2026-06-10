"""Reset state DB sebelum tiap skenario (dengan backup otomatis)."""
import sqlite3
import shutil
from pathlib import Path
from datetime import datetime

DB_PATH = "api/logs/metrics.db"   # SESUAIKAN dgn volume di docker-compose
BACKUP_DIR = Path("tests/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
shutil.copy(DB_PATH, BACKUP_DIR / f"metrics_{ts}.db")   # backup dulu

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute("DELETE FROM request_metrics WHERE 1=1;")    # sesuaikan nama tabel
cur.execute("DELETE FROM prediction_logs WHERE 1=1;")
conn.commit()
conn.close()
print(f"[{datetime.now()}] State reset. Backup: metrics_{ts}.db")