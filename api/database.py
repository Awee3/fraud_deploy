# database.py
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("logs/predictions.db")

# Kolom fitur yang diharapkan dari dataset credit card
FEATURE_COLS = [f"v{i}" for i in range(1, 29)] + ["amount"]

_db_lock = threading.Lock()

# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,

    v1  REAL, v2  REAL, v3  REAL, v4  REAL, v5  REAL, v6  REAL, v7  REAL,
    v8  REAL, v9  REAL, v10 REAL, v11 REAL, v12 REAL, v13 REAL, v14 REAL,
    v15 REAL, v16 REAL, v17 REAL, v18 REAL, v19 REAL, v20 REAL, v21 REAL,
    v22 REAL, v23 REAL, v24 REAL, v25 REAL, v26 REAL, v27 REAL, v28 REAL,

    amount      REAL,
    prediction  INTEGER NOT NULL,
    probability REAL
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
    ON predictions(timestamp DESC);
"""


# ── Inisialisasi ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Membuat file database dan tabel jika belum ada.
    Dipanggil sekali saat FastAPI startup.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)
    print(f"[DB] Database siap di: {DB_PATH.resolve()}")


# ── Context Manager ────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    """
    Membuka koneksi SQLite dengan:
    - Threading lock untuk mencegah race condition
    - Auto commit/rollback
    - Auto close
    """
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row  # row bisa diakses seperti dict
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ── Write ──────────────────────────────────────────────────────────────────────

def log_prediction(data: dict, prediction: int, probability: float | None) -> int:
    """
    Menyimpan satu record prediksi ke SQLite.

    Menerima dict dengan key apapun (case-insensitive).
    Key yang tidak dikenal (bukan v1-v28/amount) akan diabaikan secara aman.

    Return: ID record yang baru dibuat.
    """
    # Normalisasi semua key ke lowercase (V1 → v1, Amount → amount, dst.)
    normalized = {k.lower(): v for k, v in data.items()}

    all_cols  = ["timestamp"] + FEATURE_COLS + ["prediction", "probability"]
    all_vals  = (
        [datetime.now(timezone.utc).isoformat()]
        + [normalized.get(col)  for col in FEATURE_COLS]
        + [prediction, probability]
    )

    placeholders = ", ".join(["?"] * len(all_cols))
    col_names    = ", ".join(all_cols)

    with _get_conn() as conn:
        cursor = conn.execute(
            f"INSERT INTO predictions ({col_names}) VALUES ({placeholders})",
            all_vals,
        )
        return cursor.lastrowid


# ── Read ───────────────────────────────────────────────────────────────────────

def get_recent_logs(limit: int = 1000) -> list[dict]:
    """
    Mengambil N record terbaru, diurutkan dari yang terlama.
    (DESC untuk ambil N terbaru, lalu dibalik agar kronologis)
    """
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # Balik urutan: terlama → terbaru (lebih natural untuk analisis)
    return [dict(row) for row in reversed(rows)]


def get_log_count() -> int:
    """Jumlah total record, untuk health check dan monitoring."""
    with _get_conn() as conn:
        result = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return result[0] if result else 0