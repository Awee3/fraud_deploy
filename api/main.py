from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone
from threading import Lock
from typing import Any

app = FastAPI(title="Fraud Detection API - MLOps Edition")

# Path relatif terhadap lokasi Docker WORKDIR (/app)
MODEL_PATH = Path("models/rf_model.pkl")
LOG_FILE = Path("logs/prediction_logs.csv")

model: Any = None
log_lock = Lock()


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


@app.on_event("startup")
def startup_event() -> None:
    global model
    model = _load_model()
    if model is None:
        print(f"CRITICAL: Model {MODEL_PATH} tidak ditemukan! /predict akan mengembalikan 503.")

@app.post("/reload")
def trigger_reload(model_name: str = Query(default=None, description="Nama file model baru, cth: rf_model_test.pkl")):
    """
    Endpoint untuk melakukan hot-reload model ke dalam memori (RAM) tanpa mematikan server.
    """
    global model
    global MODEL_PATH
    
    # 1. Jika n8n mengirimkan parameter nama model baru, perbarui MODEL_PATH
    if model_name:
        new_path = Path(f"models/{model_name}")
        if not new_path.exists():
            raise HTTPException(status_code=404, detail=f"File model {new_path} tidak ditemukan di server!")
        MODEL_PATH = new_path
        
    # 2. Coba load ulang model menggunakan fungsi _load_model() milikmu
    try:
        new_model = _load_model()
        if new_model is None:
            raise HTTPException(status_code=404, detail=f"Model gagal dimuat dari {MODEL_PATH}.")
            
        # 3. Timpa model lama di memori dengan model yang baru
        model = new_model
        
        return {
            "message": "Model reloaded successfully", 
            "model_path": str(MODEL_PATH),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")
    
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
def predict(data: dict[str, Any]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model belum tersedia di server.")

    if not isinstance(data, dict) or not data:
        raise HTTPException(status_code=422, detail="Payload harus object JSON non-empty.")

    try:
        df_input = pd.DataFrame([data])

        # Prediksi kelas
        raw_prediction = model.predict(df_input)[0]
        prediction = _normalize_scalar(raw_prediction)

        # Prediksi probabilitas (jika tersedia)
        probability = None
        if hasattr(model, "predict_proba"):
            proba_row = model.predict_proba(df_input)[0]
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                idx = classes.index(1) if 1 in classes else int(proba_row.argmax())
            else:
                idx = int(proba_row.argmax())
            probability = float(proba_row[idx])

        # Catat log
        log_entry = data.copy()
        log_entry.update({
            "prediction": prediction,
            "probability": probability,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_log = pd.DataFrame([log_entry])

        with log_lock:
            df_log.to_csv(
                LOG_FILE,
                mode="a",
                header=not LOG_FILE.exists(),
                index=False
            )

        return {
            "prediction": prediction,
            "probability": probability,
            "status": "logged"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses prediksi: {e}")


@app.get("/logs")
def get_logs(limit: int = Query(default=500, ge=1, le=5000)):
    if not LOG_FILE.exists():
        return []

    try:
        df = pd.read_csv(LOG_FILE)
        if df.empty:
            return []
        df = df.tail(limit)
        df = df.astype(object).where(pd.notnull(df), None)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal membaca log: {e}")