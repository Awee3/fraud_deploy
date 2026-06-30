# simulation/simulate_traffic.py
import os
import random
import time
from pathlib import Path
import pandas as pd
import requests
# ── Konfigurasi ────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
API_URL         = os.getenv("API_URL",         "http://localhost:5000/predict")
DELAY_SECONDS   = float(os.getenv("DELAY_SECONDS",   "0.5"))   # Dipercepat
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
DROP_COLUMNS    = [
    c.strip()
    for c in os.getenv("DROP_COLUMNS", "Class").split(",")
    if c.strip()
]


def _format_prob(value) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run_simulation(csv_filename: str = "test_data_sample.csv") -> None:
    data_file = BASE_DIR / csv_filename

    if not data_file.exists():
        print(f"❌ File tidak ditemukan: {data_file}")
        print(f"   File yang tersedia di {BASE_DIR}:")
        for f in BASE_DIR.glob("*.csv"):
            print(f"   - {f.name}")
        return

    print(f"📂 Membaca data dari: {data_file}")
    df = pd.read_csv(data_file)

    if df.empty:
        print("⚠️ Dataset kosong.")
        return

    # Buang kolom label
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        print(f"🗑️  Menghapus kolom: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    records    = df.to_dict(orient="records")
    total      = len(records)
    success    = 0
    failed     = 0

    print(f"🚀 Mengirim {total} transaksi ke {API_URL}")
    print(f"⏱️  Delay antar request: {DELAY_SECONDS}s")
    print("-" * 60)

    with requests.Session() as session:
        for index, payload in enumerate(records):
            try:
                response = session.post(
                    API_URL,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )

                if response.ok:
                    result   = response.json()
                    prediksi = result.get("prediction", "-")
                    prob     = _format_prob(result.get("probability"))
                    rec_id   = result.get("record_id", "-")
                    print(f"[{index+1}/{total}] ✅ ID:{rec_id} | Pred:{prediksi} | Prob:{prob}")
                    success += 1
                else:
                    print(f"[{index+1}/{total}] ❌ HTTP {response.status_code}: {response.text}")
                    failed += 1

            except requests.exceptions.RequestException as exc:
                print(f"[{index+1}/{total}] ❌ Request error: {exc}")
                failed += 1
                # ✅ GANTI break → continue agar tetap lanjut
                continue

            time.sleep(random.uniform(DELAY_SECONDS * 0.8, DELAY_SECONDS * 1.2))

    print("-" * 60)
    print(f"✅ Berhasil : {success}/{total}")
    print(f"❌ Gagal    : {failed}/{total}")


if __name__ == "__main__":
    import sys
    # Bisa terima argumen filename: python simulate_traffic.py simulasi_normal.csv
    filename = sys.argv[1] if len(sys.argv) > 1 else "test_data_sample.csv"
    run_simulation(filename)