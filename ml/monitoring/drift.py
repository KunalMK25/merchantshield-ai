"""
Data drift detection.

Compares the newest transactions against two references:
  * recent   - the trailing window just before them (catches sudden shifts)
  * baseline - the oldest history (catches slow creep a trailing window would absorb)

Severity comes from the Population Stability Index (PSI), using the usual bands:
below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 significant. The two-sample
Kolmogorov-Smirnov statistic is reported alongside as supporting evidence; it is not
used for severity because with thousands of rows it flags changes too small to matter.

Choices that come from measuring the data rather than from convention:
  * `prior_txn_count` and `account_age_days` grow every day by construction. In a dense history
    today's values always sit above the trailing window's range, so judging them raises a false
    alarm daily (found when replaying a long real-style dataset). They are reported for information,
    marked "informational", and never counted in the verdict.
  * `day_of_week` is excluded: a one-day window has a single weekday, which reads as
    enormous drift every day.
  * The model's own score distribution is monitored too, because shifts that are spread
    thinly across many features (more fraud, say) show up there first.
"""

import numpy as np
from scipy import stats

from ml.features.build_features import FEATURE_COLUMNS

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
MIN_CURRENT_ROWS = 200
MIN_REFERENCE_ROWS = 500
BINS = 10
_EPS = 1e-4

MATURING_FEATURES = ("prior_txn_count", "account_age_days")
DRIFT_FEATURES = [f for f in FEATURE_COLUMNS if f != "day_of_week"]


def _clean(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def psi(reference, current, bins: int = BINS) -> float:
    """Population Stability Index of `current` against `reference`."""
    ref, cur = _clean(reference), _clean(current)
    if len(ref) == 0 or len(cur) == 0:
        raise ValueError("psi needs at least one finite value on each side")

    if len(np.unique(ref)) <= bins:
        # flags and other low-cardinality features: one bucket per distinct value
        cats = np.unique(np.concatenate([ref, cur]))
        r = np.array([(ref == c).mean() for c in cats])
        c = np.array([(cur == c_).mean() for c_ in cats])
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)

    r, c = np.clip(r, _EPS, None), np.clip(c, _EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_test(reference, current) -> dict:
    res = stats.ks_2samp(_clean(reference), _clean(current))
    return {"statistic": float(res.statistic), "p_value": float(res.pvalue)}


INFORMATIONAL = "informational"


def severity(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_MODERATE:
        return "moderate"
    return "stable"


def _feature_row(name, recent, baseline, current):
    psi_recent = psi(recent[name], current[name])
    ks = ks_test(recent[name], current[name])
    if name in MATURING_FEATURES:
        return {"feature": name, "psi": psi_recent, "psi_recent": psi_recent, "psi_baseline": None,
                "ks_statistic": ks["statistic"], "ks_p_value": ks["p_value"], "severity": INFORMATIONAL}
    psi_base = psi(baseline[name], current[name])
    worst = max(psi_recent, psi_base)
    return {"feature": name, "psi": worst, "psi_recent": psi_recent, "psi_baseline": psi_base,
            "ks_statistic": ks["statistic"], "ks_p_value": ks["p_value"], "severity": severity(worst)}


def drift_report(recent_reference, baseline_reference, current, score_reference=None, score_current=None) -> dict:
    """
    `recent_reference`, `baseline_reference`, `current` are frames with the model features.
    Optional score arrays are the model's fraud probabilities on the reference and current rows.
    """
    n_ref, n_cur = len(recent_reference), len(current)
    report = {
        "status": "stable", "max_psi": 0.0, "n_reference": n_ref, "n_current": n_cur,
        "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
        "features": [], "score": None, "flagged_features": [],
    }
    if n_cur < MIN_CURRENT_ROWS or n_ref < MIN_REFERENCE_ROWS:
        report["status"] = "insufficient_data"
        return report

    report["features"] = [_feature_row(f, recent_reference, baseline_reference, current)
                          for f in DRIFT_FEATURES]
    judged = [f for f in report["features"] if f["severity"] != INFORMATIONAL]
    severities = [f["severity"] for f in judged]
    max_psi = max(f["psi"] for f in judged)
    report["flagged_features"] = [f["feature"] for f in judged if f["severity"] == "significant"]

    if score_reference is not None and score_current is not None:
        s = psi(score_reference, score_current)
        ks = ks_test(score_reference, score_current)
        report["score"] = {"psi": s, "ks_statistic": ks["statistic"], "ks_p_value": ks["p_value"],
                           "severity": severity(s)}
        severities.append(report["score"]["severity"])
        max_psi = max(max_psi, s)

    report["max_psi"] = max_psi
    if "significant" in severities:
        report["status"] = "drifted"
    elif "moderate" in severities:
        report["status"] = "watch"
    return report
