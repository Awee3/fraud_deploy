import os
import random
import time
from pathlib import Path

import pandas as pd
import requests

# 1. Konfigurasi
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "test_data_sample.csv"
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")
DELAY_SECONDS = float(os.getenv("DELAY_SECONDS", "1.5"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
DROP_COLUMNS = [c.strip() for c in os.getenv("DROP_COLUMNS", "Class").split(",") if c.strip()]


def _format_prob(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


# 2. Fungsi Simulasi
def run_simulation() -> None:
    if not DATA_FILE.exists():
        print(f"❌ Error: File tidak ditemukan: {DATA_FILE}")
        return

    try:
        print(f"Membaca data dari {DATA_FILE}...")
        df = pd.read_csv(DATA_FILE)

        if df.empty:
            print("⚠️ Dataset kosong. Tidak ada transaksi untuk dikirim.")
            return

        # Buang kolom label/kolom yang tidak boleh dikirim ke model
        cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
        if cols_to_drop:
            print(f"Menghapus kolom non-feature: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        if df.shape[1] == 0:
            print("❌ Semua kolom terhapus. Tidak ada feature untuk diprediksi.")
            return

        records = df.to_dict(orient="records")

        print(f"Memulai simulasi. Mengirim data ke {API_URL}")
        print("-" * 50)

        with requests.Session() as session:
            for index, payload in enumerate(records):
                try:
                    response = session.post(
                        API_URL,
                        json=payload,
                        timeout=REQUEST_TIMEOUT
                    )

                    if response.ok:
                        try:
                            result = response.json()
                        except ValueError:
                            print(f"[{index}] ❌ GAGAL | Respons bukan JSON: {response.text}")
                            continue

                        prediksi = result.get("prediction", "-")
                        prob = _format_prob(result.get("probability"))
                        print(f"[{index}] ✅ BERHASIL | Prediksi: {prediksi} (Prob: {prob})")
                    else:
                        print(f"[{index}] ❌ GAGAL | HTTP {response.status_code}: {response.text}")

                except requests.exceptions.RequestException as exc:
                    print(f"[{index}] ❌ GAGAL | Request error: {exc}")
                    break  # stop jika API tidak stabil/tidak bisa dijangkau

                sleep_time = random.uniform(DELAY_SECONDS * 0.8, DELAY_SECONDS * 1.2)
                time.sleep(sleep_time)

    except Exception as e:
        print(f"❌ Error tak terduga: {e}")


# 3. Eksekusi
if __name__ == "__main__":
    run_simulation()