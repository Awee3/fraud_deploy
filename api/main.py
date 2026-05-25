# main.py
import time
from fastapi import FastAPI, HTTPException, Query, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional

from database import (
    init_db,
    log_prediction, get_recent_logs, get_log_count,
    log_request_metric, get_operational_metrics,
    log_deployment_metric, get_deployment_history, get_latest_deployment,
)
from monitoring import run_full_monitoring

app = FastAPI(title="Fraud Detection API - MLOps Edition")

MODEL_PATH = Path("models/rf_model.pkl")
model: Any = None

# Endpoint yang tidak perlu dicatat di metrics (noise)
_SKIP_METRICS_ENDPOINTS = {"/health", "/metrics", "/docs", "/openapi.json"}


# ── Middleware: Catat Latency & Error Rate ─────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Secara otomatis mencatat latency dan status setiap request.
    Berjalan transparan — tidak mengubah request/response apapun.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()
        status_code = 500
        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            raise exc
        finally:
            # Hitung latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            endpoint   = request.url.path

            # Skip endpoint yang tidak relevan
            if endpoint not in _SKIP_METRICS_ENDPOINTS:
                is_error = 1 if status_code >= 400 else 0
                try:
                    log_request_metric(
                        endpoint=endpoint,
                        method=request.method,
                        status_code=status_code,
                        latency_ms=latency_ms,
                        is_error=is_error,
                    )
                except Exception:
                    pass  # Jangan sampai metrics logging crash API utama


app.add_middleware(MetricsMiddleware)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _load_model() -> Any:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ── Lifecycle ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup_event() -> None:
    global model
    init_db()
    model = _load_model()
    if model is None:
        print(f"[WARN] Model tidak ditemukan di {MODEL_PATH}.")
    else:
        print(f"[API] Model loaded: {MODEL_PATH}")


# ── Endpoints: Deployment ──────────────────────────────────────────────────────

@app.post("/reload")
def trigger_reload(
    model_name: str = Query(default=None)
):
    global model, MODEL_PATH
    if model_name:
        new_path = Path(f"models/{model_name}")
        if not new_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File model '{new_path}' tidak ditemukan.",
            )
        MODEL_PATH = new_path
    try:
        new_model = _load_model()
        if new_model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model gagal dimuat dari '{MODEL_PATH}'.",
            )
        model = new_model
        return {
            "message":    "Model reloaded successfully.",
            "model_path": str(MODEL_PATH),
            "status":     "success",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoints: Core ────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    op_metrics     = get_operational_metrics(window_minutes=60)
    last_deploy    = get_latest_deployment()
    return {
        "status":            "ok",
        "model_loaded":      model is not None,
        "model_path":        str(MODEL_PATH),
        "total_predictions": get_log_count(),
        # ✅ Preview metrik operasional di health check
        "last_1h": {
            "requests":         op_metrics["total_requests"],
            "error_rate":       f"{op_metrics['error_rate_percent']}%",
            "avg_latency_ms":   op_metrics["latency_ms"]["avg"],
            "p95_latency_ms":   op_metrics["latency_ms"]["p95"],
        },
        "last_deployment": {
            "model":       last_deploy["model_name"]        if last_deploy else None,
            "lead_time_s": last_deploy["lead_time_seconds"] if last_deploy else None,
            "status":      last_deploy["status"]            if last_deploy else None,
            "timestamp":   last_deploy["timestamp"]         if last_deploy else None,
        },
    }


@app.post("/predict")
def predict(data: dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia.")
    if not isinstance(data, dict) or not data:
        raise HTTPException(status_code=422, detail="Payload harus JSON object non-empty.")

    try:
        df_input       = pd.DataFrame([data])
        raw_prediction = model.predict(df_input)[0]
        prediction     = _normalize_scalar(raw_prediction)

        probability = None
        if hasattr(model, "predict_proba"):
            proba_row = model.predict_proba(df_input)[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                idx     = classes.index(1) if 1 in classes else int(proba_row.argmax())
            else:
                idx = int(proba_row.argmax())
            probability = float(proba_row[idx])

        record_id = log_prediction(data, int(prediction), probability)

        return {
            "record_id":   record_id,
            "prediction":  prediction,
            "probability": probability,
            "status":      "logged",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses prediksi: {e}")


@app.get("/logs")
def get_logs(limit: int = Query(default=500, ge=1, le=5000)):
    try:
        return get_recent_logs(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Endpoints: Monitoring ──────────────────────────────────────────────────────

@app.post("/run-monitoring")
def run_monitoring(
    limit: int = Query(default=1000, ge=100, le=5000),
    window_minutes: int = Query(default=60, ge=10, le=1440),
):
    """
    Monitoring lengkap: Data Drift + Prediction Drift + Operational Metrics.
    Dipanggil oleh n8n secara terjadwal.
    """
    try:
        # 1. Drift detection
        logs   = get_recent_logs(limit=limit)
        report = run_full_monitoring(logs)

        # 2. ✅ Tambahkan operational metrics ke dalam report
        op_metrics  = get_operational_metrics(window_minutes=window_minutes)
        last_deploy = get_latest_deployment()

        report["operational_metrics"] = op_metrics
        report["last_deployment"]     = last_deploy

        # 3. Update status jika service degraded
        if op_metrics["status"] == "DEGRADED":
            report["status"] = "DRIFT_DETECTED"  # Trigger alert juga

        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_metrics(
    window_minutes: int = Query(default=60, ge=1, le=1440)
):
    """Endpoint dedicated untuk melihat metrik operasional."""
    return get_operational_metrics(window_minutes=window_minutes)


@app.post("/metrics/deployment")
def record_deployment(data: dict[str, Any]):
    """
    ✅ Dipanggil oleh n8n setelah deployment selesai.
    Mencatat lead time ke database.
    """
    required = {"model_name", "lead_time_seconds", "status"}
    if not required.issubset(data.keys()):
        raise HTTPException(
            status_code=422,
            detail=f"Field wajib: {required}",
        )
    try:
        record_id = log_deployment_metric(
            model_name        = str(data["model_name"]),
            lead_time_seconds = float(data["lead_time_seconds"]),
            status            = str(data["status"]),
            triggered_by      = str(data.get("triggered_by", "github_webhook")),
        )
        return {
            "record_id": record_id,
            "message":   "Deployment metric recorded.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/deployment")
def get_deployment_metrics(limit: int = Query(default=10, ge=1, le=100)):
    """Riwayat deployment beserta lead time masing-masing."""
    return get_deployment_history(limit=limit)