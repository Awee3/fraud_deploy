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
ALPHA         = 0.05   # Significance threshold (umum)
MIN_SAMPLES   = 30     # Minimum sampel agar KS-Test valid secara statistik

# ── 2-Tier Alerting Config ──────────────────────────────────────────────────
# Tier-1 (vital)  : fitur paling berpengaruh. Drift 1 fitur saja sudah CRITICAL.
# Tier-2 (minor)  : sisanya. Perlu >= TIER2_MIN_DRIFT fitur drift SIMULTAN agar
#                   memicu alert — ini mitigasi masalah multiple-testing
#                   (29 KS-Test @ alpha=0.05 punya peluang ~77% minimal 1 false
#                    positive; aturan Tier-2 menahan noise acak satuan).
#
# PENTING: set TIER1_FEATURES = 3 fitur terpenting (huruf kecil), yaitu fitur
# yang sama dengan yang di-shift oleh simulasi_drift.csv (top10[:3] di skrip
# training). Dengan begitu kasus drift memicu Tier-1 CRITICAL secara bersih.
TIER1_FEATURES  = ["v14", "v10", "v12"]   # <-- VERIFIKASI ke top-3 aktualmu
TIER1_ALPHA     = 0.01    # ambang ketat untuk fitur vital (CRITICAL-level)
TIER2_ALPHA     = 0.05    # ambang standar untuk fitur minor
TIER2_MIN_DRIFT = 3       # minimal fitur minor drift simultan agar memicu alert


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
    """
    total = len(predictions)
    if total == 0:
        return {"error": "Tidak ada data prediksi tersedia."}

    fraud_count = sum(predictions)
    legit_count = total - fraud_count

    baseline_fraud_rate = baseline_pred_dist.get("fraud_rate", 0.0017)
    expected_fraud      = baseline_fraud_rate * total
    expected_legit      = (1 - baseline_fraud_rate) * total

    if expected_fraud < 5:
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

def _empty_report(status: str, n_logs: int, summary: str) -> dict:
    return {
        "status":             status,
        "alert_level":        "NONE",
        "total_samples":      n_logs,
        "features_monitored": 0,
        "drifted_features":   [],
        "critical_features":  [],
        "tier1_drifted":      [],
        "tier2_drifted":      [],
        "tier2_min_required": TIER2_MIN_DRIFT,
        "tier1_features":     TIER1_FEATURES,
        "feature_results":    [],
        "prediction_drift":   {},
        "summary":            summary,
    }


def run_full_monitoring(logs: list[dict]) -> dict:
    """
    Entry point utama. Dipanggil oleh endpoint /run-monitoring di main.py.

    Menjalankan:
    1. KS-Test pada semua 29 fitur numerik
    2. Chi-Squared / Fisher's Exact Test pada distribusi prediksi
    3. Evaluasi 2-Tier Alerting untuk menentukan status & alert_level

    Return: dict laporan lengkap siap dikembalikan sebagai JSON response.
    """
    # ── Guard: tidak ada data produksi ────────────────────────────────────────
    if not logs:
        return _empty_report("ERROR", 0, "Tidak ada data produksi untuk dianalisis.")

    # ── Guard: baseline tidak ditemukan ───────────────────────────────────────
    try:
        baseline = _load_baseline()
    except FileNotFoundError:
        return _empty_report("ERROR", len(logs), f"Baseline tidak ditemukan di: {BASELINE_PATH}")

    # ── 1. Data Drift: KS-Test per fitur ──────────────────────────────────────
    feature_results  = []
    drifted_features = []

    for feature in ALL_FEATURES:
        if feature not in baseline.get("features", {}):
            continue

        baseline_samples = baseline["features"][feature].get("samples", [])
        prod_samples     = [
            float(row[feature])
            for row in logs
            if row.get(feature) is not None
        ]

        if len(prod_samples) < MIN_SAMPLES or len(baseline_samples) < MIN_SAMPLES:
            continue

        ks_stat, p_value = _run_ks_test(baseline_samples, prod_samples)
        severity         = _classify_severity(p_value)
        is_drift         = bool(p_value < ALPHA)
        tier             = "TIER1" if feature in TIER1_FEATURES else "TIER2"

        result = {
            "feature":       feature,
            "tier":          tier,
            "ks_stat":       round(ks_stat, 6),
            "p_value":       round(p_value, 6),
            "is_drift":      is_drift,
            "severity":      severity,
            "prod_mean":     round(float(np.mean(prod_samples)), 6),
            "baseline_mean": round(float(np.mean(baseline_samples)), 6),
            "prod_n":        len(prod_samples),
        }
        feature_results.append(result)
        if is_drift:
            drifted_features.append(feature)

    # ── 2. Prediction Drift ────────────────────────────────────────────────────
    predictions = [
        int(row["prediction"])
        for row in logs
        if row.get("prediction") is not None
    ]
    pred_drift  = _run_prediction_drift(
        baseline.get("prediction_distribution", {}),
        predictions,
    )

    # ── 3. Evaluasi 2-Tier Alerting ────────────────────────────────────────────
    # Tier-1: fitur vital. Drift 1 fitur (p < TIER1_ALPHA) → CRITICAL.
    tier1_drifted = [
        r["feature"] for r in feature_results
        if r["tier"] == "TIER1" and r["p_value"] < TIER1_ALPHA
    ]
    # Tier-2: fitur minor. Perlu >= TIER2_MIN_DRIFT yang drift (p < TIER2_ALPHA).
    tier2_drifted = [
        r["feature"] for r in feature_results
        if r["tier"] == "TIER2" and r["p_value"] < TIER2_ALPHA
    ]
    has_prediction_drift = pred_drift.get("is_drift", False)

    tier1_triggered = len(tier1_drifted) >= 1
    tier2_triggered = len(tier2_drifted) >= TIER2_MIN_DRIFT

    if tier1_triggered:
        alert_level = "CRITICAL"
    elif tier2_triggered or has_prediction_drift:
        alert_level = "WARNING"
    else:
        alert_level = "NONE"

    overall_status = "DRIFT_DETECTED" if alert_level != "NONE" else "OK"

    critical_features = [
        r["feature"] for r in feature_results if r["severity"] == "CRITICAL"
    ]

    # Alasan eksplisit (berguna untuk pesan Telegram & dokumentasi Bab 4.6)
    reasons = []
    if tier1_triggered:
        reasons.append(f"Tier-1 drift: {', '.join(tier1_drifted)}")
    if tier2_triggered:
        reasons.append(f"Tier-2 drift {len(tier2_drifted)}>={TIER2_MIN_DRIFT}: {', '.join(tier2_drifted)}")
    if has_prediction_drift:
        reasons.append("Prediction drift terdeteksi")
    trigger_reason = " | ".join(reasons) if reasons else "Tidak ada drift signifikan"

    summary = (
        f"Monitored {len(feature_results)} features | "
        f"Alert: {alert_level} | "
        f"Tier-1 drift: {len(tier1_drifted)} | "
        f"Tier-2 drift: {len(tier2_drifted)}/{TIER2_MIN_DRIFT} | "
        f"Prediction drift: {has_prediction_drift}"
    )

    return {
        "status":             overall_status,
        "alert_level":        alert_level,
        "trigger_reason":     trigger_reason,
        "total_samples":      len(logs),
        "features_monitored": len(feature_results),
        "drifted_features":   drifted_features,     # semua p<0.05 (dipakai Telegram)
        "critical_features":  critical_features,     # semua p<0.01
        "tier1_features":     TIER1_FEATURES,
        "tier1_drifted":      tier1_drifted,
        "tier2_drifted":      tier2_drifted,
        "tier2_min_required": TIER2_MIN_DRIFT,
        "feature_results":    feature_results,
        "prediction_drift":   pred_drift,
        "summary":            summary,
    }