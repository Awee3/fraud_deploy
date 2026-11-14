# monitoring.py
import json
import os
import numpy as np
from scipy import stats
from pathlib import Path
from typing import Optional

BASELINE_PATH = Path(
    os.getenv("BASELINE_PATH", "monitoring_config/static_allbaseline.json")
)

# Semua fitur yang akan dimonitor (29 fitur: v1..v28 + amount)
ALL_FEATURES = [f"v{i}" for i in range(1, 29)] + ["amount"]

MIN_SAMPLES      = 30   # minimum sampel agar KS-Test valid secara statistik
MIN_PRED_SAMPLES = 30   # minimum sampel agar uji prediction drift bermakna

# ── 2-Tier Alerting Config ──────────────────────────────────────────────────
# Tier-1 (vital)  : fitur paling berpengaruh. Drift 1 fitur saja sudah CRITICAL.
#                   Karena satu fitur sudah cukup memicu CRITICAL, taraf yang
#                   dipakai lebih ketat (alpha = 0.01). Konsekuensinya, fitur
#                   Tier-1 dengan 0.01 <= p < 0.05 dianggap belum cukup bukti.
#                   Fitur seperti itu tetap DILAPORKAN pada tier1_subthreshold
#                   agar operator bisa melihatnya, tetapi tidak mengubah
#                   alert_level. Menaikkannya jadi pemicu WARNING akan membuat
#                   laju alarm palsu Tier-1 melonjak dari 2,97% ke 14,3%.
#
# Tier-2 (minor)  : sisanya (26 fitur). Perlu >= TIER2_MIN_DRIFT fitur drift
#                   SIMULTAN agar memicu alert. Ini mitigasi praktis masalah
#                   multiple-testing. Di bawah H0, 26 fitur @ alpha=0.05
#                   menghasilkan ~1,3 false-positive (ekspektasi). Ambang >=5
#                   menekan probabilitas false-alarm agregat Tier-2 ke ~0,85%
#                   (asumsi independensi antar fitur, wajar untuk komponen PCA).
#
# Anggaran alarm palsu total di bawah H0:
#   Tier-1      : 1 - (1 - 0.01)^3            ~ 2,97%
#   Tier-2      : P(X >= 5), X ~ Binom(26,.05) ~ 0,85%
#   Pred drift  : alpha = 0.01 (uji tunggal)   ~ 1,00%
#   Gabungan    : 1 - (0.9703 * 0.9915 * 0.99) ~ 4,75%  (di bawah 5%)
# CATATAN: sebelumnya prediction drift memakai alpha 0.05, yang membuat total
# naik ke ~8,6% dan membatalkan klaim "di bawah 5%". Itu sudah diperbaiki.
#
# Daftar fitur Tier-1 TIDAK di-hardcode. Dibaca saat runtime dari baseline JSON
# (meta.tier1_features), yang ditulis otomatis oleh train_and_generate.py dari
# top-3 XGBoost feature importance. Satu sumber kebenaran:
#   training -> baseline -> monitoring.
TIER1_ALPHA     = 0.01   # ambang ketat untuk fitur vital (CRITICAL-level)
TIER2_ALPHA     = 0.05   # ambang standar untuk fitur minor
TIER2_MIN_DRIFT = 5      # minimal fitur minor drift simultan agar memicu alert
PRED_ALPHA      = 0.01   # ambang untuk prediction drift (uji tunggal)

# Ambang pelaporan umum (dipakai untuk daftar drifted_features di pesan Telegram).
ALPHA = TIER2_ALPHA

# Ukuran sampel baseline default untuk tabel kontingensi Fisher, dipakai HANYA
# jika baseline JSON tidak memuat prediction_distribution.n_samples. Angka ini
# arbitrer dan memengaruhi power uji, jadi sebaiknya baseline diregenerasi agar
# memuat n_samples yang sebenarnya.
DEFAULT_BASELINE_N = 10_000


# ── Helper ─────────────────────────────────────────────────────────────────────

def _load_baseline() -> dict:
    """Load file baseline JSON. Raise FileNotFoundError jika tidak ada."""
    with open(BASELINE_PATH, "r") as f:
        return json.load(f)


def _classify_severity(p_value: float, tier: str = "TIER2") -> str:
    """
    Klasifikasi tingkat keparahan drift, konsisten dengan kebijakan alerting.

    TIER1 (alpha = 0.01, satu fitur sudah memicu CRITICAL):
      p <  0.01              -> CRITICAL
      0.01 <= p < 0.05       -> WARNING  (sub-threshold, tidak memicu alarm)
      p >= 0.05              -> NONE

    TIER2 (alpha = 0.05, hanya memicu WARNING dan hanya secara agregat):
      p <  0.05              -> WARNING
      p >= 0.05              -> NONE
    Fitur Tier-2 TIDAK PERNAH diberi label CRITICAL, karena secara desain
    fitur minor memang tidak bisa membawa sistem ke status CRITICAL sendirian.
    Sebelumnya label CRITICAL bocor ke fitur Tier-2 dan membuat pesan Telegram
    melaporkan "sekian fitur critical" padahal status sistem OK.
    """
    if tier == "TIER1":
        if p_value < TIER1_ALPHA:
            return "CRITICAL"
        if p_value < TIER2_ALPHA:
            return "WARNING"
        return "NONE"

    return "WARNING" if p_value < TIER2_ALPHA else "NONE"


# ── Statistical Tests ──────────────────────────────────────────────────────────

def _run_ks_test(
    baseline_samples: list[float],
    prod_samples: list[float],
) -> tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov Test.
    H0: Kedua sampel berasal dari distribusi yang sama.
    Reject H0 jika p_value < alpha tier terkait -> drift terdeteksi.
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

    Keputusan drift memakai PRED_ALPHA (0.01), bukan 0.05, agar jalur ini tidak
    menambah laju alarm palsu sistem melebihi anggaran yang dijelaskan di atas.
    """
    total = len(predictions)
    if total == 0:
        return {
            "tested": False,
            "is_drift": False,
            "severity": "NONE",
            "error": "Tidak ada data prediksi tersedia.",
        }

    if total < MIN_PRED_SAMPLES:
        return {
            "tested": False,
            "is_drift": False,
            "severity": "NONE",
            "total_samples": total,
            "error": (
                f"Sampel prediksi terlalu sedikit ({total} < {MIN_PRED_SAMPLES}), "
                "uji tidak dijalankan."
            ),
        }

    fraud_count = sum(predictions)
    legit_count = total - fraud_count

    baseline_fraud_rate = baseline_pred_dist.get("fraud_rate", 0.0017)
    expected_fraud      = baseline_fraud_rate * total
    expected_legit      = (1 - baseline_fraud_rate) * total

    # Ukuran sampel baseline: pakai nilai sebenarnya kalau tersedia.
    baseline_n = int(
        baseline_pred_dist.get("n_samples")
        or baseline_pred_dist.get("total")
        or DEFAULT_BASELINE_N
    )
    baseline_n_is_assumed = not (
        baseline_pred_dist.get("n_samples") or baseline_pred_dist.get("total")
    )

    if expected_fraud < 5:
        b_fraud = round(baseline_fraud_rate * baseline_n)
        b_legit = baseline_n - b_fraud
        _, p_value = stats.fisher_exact(
            [[fraud_count, legit_count], [b_fraud, b_legit]]
        )
        test_method = "Fisher's Exact Test"
        chi2_stat   = None
    else:
        observed = np.array([fraud_count, legit_count])
        expected = np.array([expected_fraud, expected_legit])
        chi2_stat, p_value = stats.chisquare(observed, f_exp=expected)
        test_method = "Chi-Squared Test"
        chi2_stat   = round(float(chi2_stat), 6)

    p_value  = float(p_value)
    is_drift = bool(p_value < PRED_ALPHA)

    return {
        "tested":                True,
        "test_method":           test_method,
        "alpha":                 PRED_ALPHA,
        "chi2_stat":             chi2_stat,
        "p_value":               round(p_value, 6),
        "observed_fraud_rate":   round(fraud_count / total, 6),
        "baseline_fraud_rate":   round(baseline_fraud_rate, 6),
        "baseline_n":            baseline_n,
        "baseline_n_is_assumed": baseline_n_is_assumed,
        "total_samples":         total,
        "is_drift":              is_drift,
        "severity":              "CRITICAL" if is_drift else "NONE",
    }


# ── Main Runner ────────────────────────────────────────────────────────────────

def _empty_report(status: str, n_logs: int, summary: str,
                  alert_level: str = "NONE",
                  tier1: Optional[list[str]] = None) -> dict:
    """
    Laporan kosong untuk jalur error. Struktur key HARUS identik dengan return
    run_full_monitoring(), supaya ekspresi n8n (mis. $json.trigger_reason) tidak
    pecah saat jalur error terjadi.
    """
    return {
        "status":             status,
        "alert_level":        alert_level,
        "trigger_reason":     summary,
        "total_samples":      n_logs,
        "features_monitored": 0,
        "skipped_features":   [],
        "drifted_features":   [],
        "critical_features":  [],
        "tier1_features":     tier1 or [],
        "tier1_drifted":      [],
        "tier1_subthreshold": [],
        "tier2_drifted":      [],
        "tier2_min_required": TIER2_MIN_DRIFT,
        "feature_results":    [],
        "prediction_drift":   {"tested": False, "is_drift": False},
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
        return _empty_report(
            "ERROR", 0,
            "Tidak ada data produksi untuk dianalisis.",
            alert_level="ERROR",
        )

    # ── Guard: baseline tidak ditemukan ───────────────────────────────────────
    try:
        baseline = _load_baseline()
    except FileNotFoundError:
        return _empty_report(
            "ERROR", len(logs),
            f"Baseline tidak ditemukan di: {BASELINE_PATH}",
            alert_level="ERROR",
        )

    baseline_features = baseline.get("features", {})

    # ── Tier-1 dari baseline (single source of truth) ──────────────────────────
    tier1_declared = [
        f.lower() for f in baseline.get("meta", {}).get("tier1_features", [])
    ]
    if not tier1_declared:
        return _empty_report(
            "ERROR", len(logs),
            "Baseline tidak memuat meta.tier1_features. Regenerate dengan "
            "train_and_generate.py. Monitoring dihentikan agar klasifikasi "
            "tier tidak salah.",
            alert_level="ERROR",
        )

    # Validasi: nama fitur Tier-1 harus benar-benar ada di baseline dan di daftar
    # fitur yang dimonitor. Kalau ada typo, tanpa validasi ini jalur CRITICAL
    # akan mati diam-diam karena tidak ada fitur yang pernah diklasifikasi TIER1.
    tier1_invalid = [
        f for f in tier1_declared
        if f not in ALL_FEATURES or f not in baseline_features
    ]
    if tier1_invalid:
        return _empty_report(
            "ERROR", len(logs),
            "Fitur Tier-1 pada baseline tidak dikenali: "
            f"{', '.join(tier1_invalid)}. Monitoring dihentikan agar jalur "
            "CRITICAL tidak mati diam-diam.",
            alert_level="ERROR",
            tier1=tier1_declared,
        )

    tier1_features = tier1_declared

    # ── 1. Data Drift: KS-Test per fitur ──────────────────────────────────────
    feature_results  = []
    drifted_features = []
    skipped_features = []   # fitur yang tidak sempat diuji, harus terlihat

    for feature in ALL_FEATURES:
        tier = "TIER1" if feature in tier1_features else "TIER2"

        if feature not in baseline_features:
            skipped_features.append({
                "feature": feature, "tier": tier,
                "reason": "tidak ada di baseline",
            })
            continue

        baseline_samples = baseline_features[feature].get("samples", [])
        prod_samples = [
            float(row[feature])
            for row in logs
            if row.get(feature) is not None
        ]

        if len(prod_samples) < MIN_SAMPLES or len(baseline_samples) < MIN_SAMPLES:
            skipped_features.append({
                "feature": feature, "tier": tier,
                "reason": (
                    f"sampel kurang (prod={len(prod_samples)}, "
                    f"baseline={len(baseline_samples)}, min={MIN_SAMPLES})"
                ),
            })
            continue

        ks_stat, p_value = _run_ks_test(baseline_samples, prod_samples)
        alpha_used = TIER1_ALPHA if tier == "TIER1" else TIER2_ALPHA
        severity   = _classify_severity(p_value, tier)

        result = {
            "feature":       feature,
            "tier":          tier,
            "alpha":         alpha_used,
            "ks_stat":       round(ks_stat, 6),
            "p_value":       round(p_value, 6),
            "is_drift":      bool(p_value < alpha_used),
            "severity":      severity,
            "prod_mean":     round(float(np.mean(prod_samples)), 6),
            "baseline_mean": round(float(np.mean(baseline_samples)), 6),
            "prod_n":        len(prod_samples),
        }
        feature_results.append(result)

        # drifted_features tetap memakai ambang pelaporan umum (0.05) supaya
        # daftar di pesan Telegram tidak berubah maknanya.
        if p_value < ALPHA:
            drifted_features.append(feature)

    # ── Guard: monitoring buta ────────────────────────────────────────────────
    # Kalau tidak ada fitur yang berhasil diuji, atau tidak ada satu pun fitur
    # Tier-1 yang teruji, sistem TIDAK BOLEH melaporkan OK. Melaporkan sehat
    # ketika sebenarnya buta adalah kegagalan monitoring yang paling mahal.
    tier1_tested = [r for r in feature_results if r["tier"] == "TIER1"]

    if not feature_results:
        rep = _empty_report(
            "ERROR", len(logs),
            "Tidak ada fitur yang berhasil diuji. Periksa nama kolom log dan "
            "isi baseline.",
            alert_level="ERROR",
            tier1=tier1_features,
        )
        rep["skipped_features"] = skipped_features
        return rep

    if not tier1_tested:
        rep = _empty_report(
            "ERROR", len(logs),
            "Tidak ada fitur Tier-1 yang berhasil diuji, jalur CRITICAL tidak "
            "aktif. Status OK tidak dapat dipertanggungjawabkan.",
            alert_level="ERROR",
            tier1=tier1_features,
        )
        rep["skipped_features"]   = skipped_features
        rep["feature_results"]    = feature_results
        rep["features_monitored"] = len(feature_results)
        rep["drifted_features"]   = drifted_features
        return rep

    # ── 2. Prediction Drift ────────────────────────────────────────────────────
    predictions = [
        int(row["prediction"])
        for row in logs
        if row.get("prediction") is not None
    ]
    pred_drift = _run_prediction_drift(
        baseline.get("prediction_distribution", {}),
        predictions,
    )

    # ── 3. Evaluasi 2-Tier Alerting ────────────────────────────────────────────
    # Tier-1: drift 1 fitur (p < TIER1_ALPHA) sudah CRITICAL.
    tier1_drifted = [
        r["feature"] for r in feature_results
        if r["tier"] == "TIER1" and r["p_value"] < TIER1_ALPHA
    ]
    # Tier-1 sub-threshold: 0.01 <= p < 0.05. Dilaporkan, tidak memicu alarm.
    tier1_subthreshold = [
        r["feature"] for r in feature_results
        if r["tier"] == "TIER1" and TIER1_ALPHA <= r["p_value"] < TIER2_ALPHA
    ]
    # Tier-2: perlu >= TIER2_MIN_DRIFT fitur drift (p < TIER2_ALPHA).
    tier2_drifted = [
        r["feature"] for r in feature_results
        if r["tier"] == "TIER2" and r["p_value"] < TIER2_ALPHA
    ]
    has_prediction_drift = bool(pred_drift.get("is_drift", False))

    tier1_triggered = len(tier1_drifted) >= 1
    tier2_triggered = len(tier2_drifted) >= TIER2_MIN_DRIFT

    if tier1_triggered:
        alert_level = "CRITICAL"
    elif tier2_triggered or has_prediction_drift:
        alert_level = "WARNING"
    else:
        alert_level = "NONE"

    overall_status = "DRIFT_DETECTED" if alert_level != "NONE" else "OK"

    # Sekarang hanya berisi fitur Tier-1 yang benar-benar CRITICAL, konsisten
    # dengan alert_level. Key-nya dipertahankan supaya template n8n tidak pecah.
    critical_features = [
        r["feature"] for r in feature_results if r["severity"] == "CRITICAL"
    ]

    # ── Alasan eksplisit (dipakai pesan Telegram & dokumentasi Bab 4.6) ────────
    reasons = []
    if tier1_triggered:
        reasons.append(f"Tier-1 drift: {', '.join(tier1_drifted)}")
    if tier2_triggered:
        reasons.append(
            f"Tier-2 drift {len(tier2_drifted)}>={TIER2_MIN_DRIFT}: "
            f"{', '.join(tier2_drifted)}"
        )
    if has_prediction_drift:
        reasons.append(
            f"Prediction drift terdeteksi (p<{PRED_ALPHA})"
        )
    trigger_reason = " | ".join(reasons) if reasons else "Tidak ada drift signifikan"

    notes = []
    if tier1_subthreshold:
        notes.append(
            f"Tier-1 sub-threshold (0.01<=p<0.05, tidak memicu alarm): "
            f"{', '.join(tier1_subthreshold)}"
        )
    if skipped_features:
        notes.append(f"{len(skipped_features)} fitur tidak diuji")
    if pred_drift.get("baseline_n_is_assumed"):
        notes.append(
            f"baseline_n prediction drift diasumsikan {DEFAULT_BASELINE_N}"
        )

    summary = (
        f"Monitored {len(feature_results)}/{len(ALL_FEATURES)} features | "
        f"Alert: {alert_level} | "
        f"Tier-1 drift: {len(tier1_drifted)} | "
        f"Tier-2 drift: {len(tier2_drifted)}/{TIER2_MIN_DRIFT} | "
        f"Prediction drift: {has_prediction_drift}"
    )
    if notes:
        summary += " | " + " ; ".join(notes)

    return {
        "status":             overall_status,
        "alert_level":        alert_level,
        "trigger_reason":     trigger_reason,
        "total_samples":      len(logs),
        "features_monitored": len(feature_results),
        "skipped_features":   skipped_features,
        "drifted_features":   drifted_features,      # semua p<0.05 (dipakai Telegram)
        "critical_features":  critical_features,     # kini hanya Tier-1 p<0.01
        "tier1_features":     tier1_features,
        "tier1_drifted":      tier1_drifted,
        "tier1_subthreshold": tier1_subthreshold,
        "tier2_drifted":      tier2_drifted,
        "tier2_min_required": TIER2_MIN_DRIFT,
        "feature_results":    feature_results,
        "prediction_drift":   pred_drift,
        "summary":            summary,
    }