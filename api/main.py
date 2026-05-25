# main.py
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib
from pathlib import Path
from typing import Any

# ✅ Import modul baru pengganti CSV
from database import init_db, log_prediction, get_recent_logs, get_log_count
from monitoring import run_full_monitoring

app = FastAPI(title="Fraud Detection API - MLOps Edition")

MODEL_PATH = Path("models/xgboost_model_2.pkl")
model: Any = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _normalize_scalar(value: Any) -> Any:
    """Ubah numpy scalar/object ke tipe JSON-friendly."""
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

    # ✅ Inisialisasi SQLite saat server pertama kali nyala
    init_db()

    model = _load_model()
    if model is None:
        print(
            f"[WARN] Model tidak ditemukan di {MODEL_PATH}. "
            "Endpoint /predict akan mengembalikan 503 sampai model tersedia."
        )
    else:
        print(f"[API] Model loaded: {MODEL_PATH}")


# ── Endpoints: Deployment ──────────────────────────────────────────────────────

@app.post("/reload")
def trigger_reload(
    model_name: str = Query(
        default=None, description="Nama file model baru, cth: rf_model_v2.pkl"
    )
):
    """
    Hot-reload model ke dalam memori tanpa mematikan server.
    Dipanggil oleh n8n setelah model baru berhasil diunduh dari GitHub.
    """
    global model, MODEL_PATH

    if model_name:
        new_path = Path(f"models/{model_name}")
        if not new_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File model '{new_path}' tidak ditemukan di server.",
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
        raise HTTPException(
            status_code=500, detail=f"Failed to reload model: {e}"
        )


@app.get("/health")
def health() -> dict:
    return {
        "status":            "ok",
        "model_loaded":      model is not None,
        "model_path":        str(MODEL_PATH),
        # ✅ Sekarang menampilkan jumlah record dari database
        "total_predictions": get_log_count(),
    }


@app.post("/predict")
def predict(data: dict[str, Any]):
    """
    Menerima data transaksi, mengembalikan prediksi fraud,
    dan menyimpan log ke SQLite secara otomatis.
    """
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model belum tersedia di server."
        )
    if not isinstance(data, dict) or not data:
        raise HTTPException(
            status_code=422, detail="Payload harus berupa JSON object non-empty."
        )

    try:
        df_input        = pd.DataFrame([data])
        raw_prediction  = model.predict(df_input)[0]
        prediction      = _normalize_scalar(raw_prediction)

        probability = None
        if hasattr(model, "predict_proba"):
            proba_row = model.predict_proba(df_input)[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                idx     = classes.index(1) if 1 in classes else int(proba_row.argmax())
            else:
                idx = int(proba_row.argmax())
            probability = float(proba_row[idx])

        # ✅ Simpan ke SQLite (menggantikan logika CSV lama)
        record_id = log_prediction(
            data=data,
            prediction=int(prediction),
            probability=probability,
        )

        return {
            "record_id":  record_id,    # ✅ ID dari database untuk traceability
            "prediction": prediction,
            "probability": probability,
            "status":     "logged",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Gagal memproses prediksi: {e}"
        )


@app.get("/logs")
def get_logs(limit: int = Query(default=500, ge=1, le=5000)):
    """
    Mengembalikan N record terbaru dari database.
    Digunakan oleh n8n atau untuk inspeksi manual.
    """
    try:
        # ✅ Baca dari SQLite (menggantikan pd.read_csv)
        return get_recent_logs(limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Gagal membaca log: {e}"
        )


# ── Endpoint Baru: Monitoring ──────────────────────────────────────────────────

@app.post("/run-monitoring")
def run_monitoring(
    limit: int = Query(
        default=1000,
        ge=100,
        le=5000,
        description="Jumlah record terbaru yang dianalisis.",
    )
):
    """
    Menjalankan analisis statistik lengkap (KS-Test + Chi-Squared/Fisher's).

    Dipanggil oleh n8n secara terjadwal. n8n cukup memeriksa field 'status':
      - 'OK'             → tidak ada drift, lanjut ke siklus berikutnya
      - 'DRIFT_DETECTED' → kirim alert ke Telegram
      - 'ERROR'          → ada masalah infrastruktur, perlu investigasi

    Menggantikan seluruh logika JavaScript statistik di n8n.
    """
    try:
        logs   = get_recent_logs(limit=limit)
        report = run_full_monitoring(logs)
        return report
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Monitoring gagal dieksekusi: {e}"
        )