# monitoring.py
import json
import os
import numpy as np
from scipy import stats
from pathlib import Path

BASELINE_PATH = Path(
    os.getenv("BASELINE_PATH", "monitoring_config/static_allbaseline.json")
)

# Semua fitur yang akan dimonitor (29 fitur, bukan hanya 10)
ALL_FEATURES  = [f"v{i}" for i in range(1, 29)] + ["amount"]
ALPHA         = 0.05   # Significance threshold
MIN_SAMPLES   = 30     # Minimum sampel agar KS-Test valid secara statistik


# ── Helper ─────────────────────────────────────────────────────────────────────

def _load_baseline() -> dict:
    """Load file baseline JSON. Raise FileNotFoundError jika tidak ada."""
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def _classify_severity(p_value: float) -> str:
    """
    Mengklasifikasikan tingkat keparahan drift berdasarkan p-value.
      p >= 0.05  → NONE     (tidak ada drift)
      p >= 0.01  → WARNING  (drift mulai terdeteksi)
      p <  0.01  → CRITICAL (drift signifikan)
    """
    if p_value >= ALPHA:
        return "NONE"
    elif p_value >= 0.01:
        return "WARNING"
    else:
        return "CRITICAL"


# ── Statistical Tests ──────────────────────────────────────────────────────────

def _run_ks_test(
    baseline_samples: list[float],
    prod_samples: list[float],
) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov Test.
    H0: Kedua sampel berasal dari distribusi yang sama.
    Reject H0 jika p_value < ALPHA → drift terdeteksi.
    """
    ks_stat, p_value = stats.ks_2samp(baseline_samples, prod_samples)
    return float(ks_stat), float(p_value)


def _run_prediction_drift(
    baseline_pred_dist: dict,
    predictions: list[int],
) -> dict:
    """
    Mendeteksi pergeseran distribusi prediksi (fraud vs. legitimate).

    Menggunakan Chi-Squared jika expected_fraud >= 5,
    Fisher's Exact Test jika tidak (lebih aman untuk data imbalanced).

    Ini mengatasi kelemahan Chi-Squared standar pada kasus fraud detection
    di mana proporsi fraud sangat kecil (~0.17%).
    """
    total = len(predictions)
    if total == 0:
        return {"error": "Tidak ada data prediksi tersedia."}

    fraud_count = sum(predictions)
    legit_count = total - fraud_count

    baseline_fraud_rate = baseline_pred_dist.get("fraud_rate", 0.0017)
    expected_fraud      = baseline_fraud_rate * total
    expected_legit      = (1 - baseline_fraud_rate) * total

    # ── Pilih test yang tepat ──────────────────────────────────────────────────
    if expected_fraud < 5:
        # Chi-Squared tidak valid jika expected frequency < 5
        # Gunakan Fisher's Exact Test sebagai fallback
        baseline_scale  = 10_000
        b_fraud         = round(baseline_fraud_rate * baseline_scale)
        b_legit         = baseline_scale - b_fraud

        _, p_value  = stats.fisher_exact(
            [[fraud_count, legit_count], [b_fraud, b_legit]]
        )
        test_method = "Fisher's Exact Test"
        chi2_stat   = None
    else:
        observed         = np.array([fraud_count,  legit_count])
        expected         = np.array([expected_fraud, expected_legit])
        chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
        test_method      = "Chi-Squared Test"
        chi2_stat        = round(float(chi2_stat), 6)

    is_drift = bool(p_value < ALPHA)

    return {
        "test_method":         test_method,
        "chi2_stat":           chi2_stat,
        "p_value":             round(float(p_value), 6),
        "observed_fraud_rate": round(fraud_count / total, 6),
        "baseline_fraud_rate": round(baseline_fraud_rate, 6),
        "is_drift":            is_drift,
        "severity":            _classify_severity(float(p_value)),
    }


# ── Main Runner ────────────────────────────────────────────────────────────────

def run_full_monitoring(logs: list[dict]) -> dict:
    """
    Entry point utama. Dipanggil oleh endpoint /run-monitoring di main.py.

    Menjalankan:
    1. KS-Test pada semua 29 fitur numerik
    2. Chi-Squared / Fisher's Exact Test pada distribusi prediksi

    Return: dict laporan lengkap siap dikembalikan sebagai JSON response.
    """
    # ── Guard: tidak ada data produksi ────────────────────────────────────────
    if not logs:
        return {
            "status":             "ERROR",
            "total_samples":      0,
            "features_monitored": 0,
            "drifted_features":   [],
            "critical_features":  [],
            "feature_results":    [],
            "prediction_drift":   {},
            "summary":            "Tidak ada data produksi untuk dianalisis.",
        }

    # ── Guard: baseline tidak ditemukan ───────────────────────────────────────
    try:
        baseline = _load_baseline()
    except FileNotFoundError:
        return {
            "status":             "ERROR",
            "total_samples":      len(logs),
            "features_monitored": 0,
            "drifted_features":   [],
            "critical_features":  [],
            "feature_results":    [],
            "prediction_drift":   {},
            "summary":            f"Baseline tidak ditemukan di: {BASELINE_PATH}",
        }

    # ── 1. Data Drift: KS-Test per fitur ──────────────────────────────────────
    feature_results  = []
    drifted_features = []

    for feature in ALL_FEATURES:
        # Skip jika baseline tidak punya data untuk fitur ini
        if feature not in baseline.get("features", {}):
            continue

        baseline_samples = baseline["features"][feature].get("samples", [])
        prod_samples     = [
            float(row[feature])
            for row in logs
            if row.get(feature) is not None
        ]

        # KS-Test tidak reliable dengan sampel terlalu sedikit
        if len(prod_samples) < MIN_SAMPLES:
            continue

        ks_stat, p_value = _run_ks_test(baseline_samples, prod_samples)
        severity         = _classify_severity(p_value)
        is_drift         = bool(p_value < ALPHA)

        result = {
            "feature":       feature,
            "ks_stat":       round(ks_stat, 6),
            "p_value":       round(p_value, 6),
            "is_drift":      is_drift,
            "severity":      severity,
            "prod_mean":     round(float(np.mean(prod_samples)), 6),
            "baseline_mean": round(
                float(np.mean(baseline_samples)), 6
            ),
            "prod_n":        len(prod_samples),
        }

        feature_results.append(result)
        if is_drift:
            drifted_features.append(feature)

    # ── 2. Prediction Drift ────────────────────────────────────────────────────
    predictions    = [
        int(row["prediction"])
        for row in logs
        if row.get("prediction") is not None
    ]
    pred_drift     = _run_prediction_drift(
        baseline.get("prediction_distribution", {}),
        predictions,
    )

    # ── 3. Tentukan status keseluruhan ─────────────────────────────────────────
    has_feature_drift    = len(drifted_features) > 0
    has_prediction_drift = pred_drift.get("is_drift", False)
    overall_status       = (
        "DRIFT_DETECTED"
        if (has_feature_drift or has_prediction_drift)
        else "OK"
    )

    critical_features = [
        r["feature"] for r in feature_results if r["severity"] == "CRITICAL"
    ]

    summary = (
        f"Monitored {len(feature_results)} features | "
        f"Drifted: {len(drifted_features)} | "
        f"Critical: {len(critical_features)} | "
        f"Prediction drift: {has_prediction_drift}"
    )

    return {
        "status":             overall_status,
        "total_samples":      len(logs),
        "features_monitored": len(feature_results),
        "drifted_features":   drifted_features,
        "critical_features":  critical_features,
        "feature_results":    feature_results,
        "prediction_drift":   pred_drift,
        "summary":            summary,
    }