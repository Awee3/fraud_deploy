# database.py
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("logs/predictions.db")

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

-- ✅ BARU: Tabel latency & error rate per request
CREATE TABLE IF NOT EXISTS request_metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    endpoint    TEXT    NOT NULL,
    method      TEXT    NOT NULL,
    status_code INTEGER NOT NULL,
    latency_ms  REAL    NOT NULL,
    is_error    INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_request_metrics_timestamp
    ON request_metrics(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_request_metrics_endpoint
    ON request_metrics(endpoint, timestamp DESC);

-- ✅ BARU: Tabel lead time deployment
CREATE TABLE IF NOT EXISTS deployment_metrics (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    model_name        TEXT    NOT NULL,
    lead_time_seconds REAL    NOT NULL,
    status            TEXT    NOT NULL,
    triggered_by      TEXT    DEFAULT 'github_webhook'
);
"""


# ── Init ───────────────────────────────────────────────────────────────────────

def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)
    print(f"[DB] Database siap di: {DB_PATH.resolve()}")


# ── Context Manager ────────────────────────────────────────────────────────────

@contextmanager
def _get_conn():
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


# ── Predictions CRUD ───────────────────────────────────────────────────────────

def log_prediction(data: dict, prediction: int, probability: Optional[float]) -> int:
    normalized = {k.lower(): v for k, v in data.items()}
    all_cols = ["timestamp"] + FEATURE_COLS + ["prediction", "probability"]
    all_vals = (
        [datetime.now(timezone.utc).isoformat()]
        + [normalized.get(col) for col in FEATURE_COLS]
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


def get_recent_logs(limit: int = 1000) -> list:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def get_log_count() -> int:
    with _get_conn() as conn:
        result = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()
        return result[0] if result else 0


# ── Request Metrics CRUD ───────────────────────────────────────────────────────

def log_request_metric(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
    is_error: int,
) -> None:
    """Dipanggil oleh middleware FastAPI setiap request masuk."""
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO request_metrics
                (timestamp, endpoint, method, status_code, latency_ms, is_error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                datetime.now(timezone.utc).isoformat(),
                endpoint,
                method,
                status_code,
                round(latency_ms, 3),
                is_error,
            ],
        )


def get_operational_metrics(window_minutes: int = 60) -> dict:
    """
    Menghitung metrik operasional dari N menit terakhir.
    Digunakan oleh endpoint /metrics dan /run-monitoring.
    """
    with _get_conn() as conn:
        # Ambil semua request dalam window waktu
        rows = conn.execute(
            """
            SELECT latency_ms, status_code, is_error, endpoint
            FROM request_metrics
            WHERE timestamp >= datetime('now', ? || ' minutes')
            """,
            (f"-{window_minutes}",),
        ).fetchall()

    if not rows:
        return {
            "window_minutes":     window_minutes,
            "total_requests":     0,
            "error_rate_percent": 0.0,
            "latency_ms": {
                "avg": 0.0, "min": 0.0,
                "max": 0.0, "p50": 0.0,
                "p90": 0.0, "p95": 0.0,
            },
            "status": "NO_DATA",
        }

    latencies   = [r["latency_ms"] for r in rows]
    errors      = [r["is_error"]   for r in rows]
    total       = len(rows)
    error_count = sum(errors)

    # Hitung percentile secara manual
    sorted_lat  = sorted(latencies)
    def percentile(data: list, pct: float) -> float:
        idx = int(len(data) * pct / 100)
        return round(data[min(idx, len(data) - 1)], 3)

    # Hitung per-endpoint breakdown
    endpoint_counts = {}
    for r in rows:
        ep = r["endpoint"]
        endpoint_counts[ep] = endpoint_counts.get(ep, 0) + 1

    error_rate  = round((error_count / total) * 100, 2)
    status      = "DEGRADED" if error_rate > 5.0 else "HEALTHY"

    return {
        "window_minutes":     window_minutes,
        "total_requests":     total,
        "error_count":        error_count,
        "error_rate_percent": error_rate,
        "requests_per_minute": round(total / window_minutes, 2),
        "latency_ms": {
            "avg": round(sum(latencies) / total, 3),
            "min": round(min(latencies), 3),
            "max": round(max(latencies), 3),
            "p50": percentile(sorted_lat, 50),
            "p90": percentile(sorted_lat, 90),
            "p95": percentile(sorted_lat, 95),
        },
        "endpoint_breakdown": endpoint_counts,
        "status": status,
    }


# ── Deployment Metrics CRUD ────────────────────────────────────────────────────

def log_deployment_metric(
    model_name: str,
    lead_time_seconds: float,
    status: str,
    triggered_by: str = "github_webhook",
) -> int:
    """Dipanggil oleh n8n setelah deployment selesai."""
    with _get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO deployment_metrics
                (timestamp, model_name, lead_time_seconds, status, triggered_by)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                datetime.now(timezone.utc).isoformat(),
                model_name,
                round(lead_time_seconds, 3),
                status,
                triggered_by,
            ],
        )
        return cursor.lastrowid


def get_deployment_history(limit: int = 10) -> list:
    """Mengambil riwayat deployment terbaru."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM deployment_metrics
            ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_latest_deployment() -> Optional[dict]:
    """Mengambil deployment terakhir saja."""
    history = get_deployment_history(limit=1)
    return history[0] if history else None