"""
Per-account anomaly detection.

Looks at each customer's day as a whole and asks "is this unusual for THIS customer?",
so a naturally busy shopper is not flagged for being busy. Two independent views are
combined:

  * Isolation Forest (unsupervised, no fraud labels) over account-day behaviour,
    fitted on earlier days only and thresholded at the reference's 99.5th percentile.
  * Five plain-language signals a bank analyst can read and challenge:
      takeover       new device AND new region in the same day
      amount_spike   a payment far above the customer's usual amount
      card_testing   several failed payments in one day
      activity_burst far more payments than this customer usually makes
      spend_surge    far more total spend than this customer usually has in a day

Severity is a simple count: the forest flag plus the number of signals. Three or more is
critical, two is high, one is watch. On the demo data, high-or-critical days were almost
all genuine compromises while the forest and signals each alone were noisier.

This module only RECOMMENDS. It never blocks, holds or closes anything; turning a flag into
an account hold is a human decision handled by the alert workflow.

Limits worth knowing: relative signals need a few earlier days of history per customer
(until then they read as neutral), and a compromise that looks exactly like the customer's
normal behaviour (a single modest purchase from the usual device) is invisible by design.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

SEVERITIES = ("critical", "high", "watch")
ACTIONS = {
    "critical": "Recommend temporary account hold pending analyst review",
    "high": "Recommend analyst review",
    "watch": "Monitor",
}

MIN_PRIOR_DAYS = 3          # earlier account-days needed before "usual" levels are trusted
MIN_REFERENCE_ROWS = 200
REFERENCE_QUANTILE = 0.995
MODEL_COLUMNS = ["n", "total", "max_ratio", "max_z", "new_dev", "new_geo", "failed", "night",
                 "n_vs_usual", "spend_vs_usual"]

OUTPUT_COLUMNS = ["customer_id", "day", "severity", "points", "iso_score", "iso_flag", "signals",
                  "n_txns", "total_amount", "recommended_action"]


# ------------------------------------------------------------------- table

def account_day_table(feats: pd.DataFrame, origin: pd.Timestamp = None) -> pd.DataFrame:
    """One row per customer per day, with the customer's own earlier-days norm for comparison."""
    f = feats.copy()
    ts = pd.to_datetime(f["timestamp"])
    origin = origin if origin is not None else ts.min().normalize()
    f["day"] = (ts.dt.normalize() - origin).dt.days
    f["failed"] = (f["status"] == "failed").astype(int)

    spec = dict(n=("amount", "size"), total=("amount", "sum"),
                max_ratio=("amount_vs_avg_ratio", "max"), max_z=("amount_zscore", "max"),
                new_dev=("new_device_flag", "sum"), new_geo=("new_geo_flag", "sum"),
                failed=("failed", "sum"), night=("is_night", "sum"))
    if "is_fraud" in f.columns:
        spec["fraud"] = ("is_fraud", "max")
    t = f.groupby(["customer_id", "day"]).agg(**spec).reset_index()
    t = t.sort_values(["customer_id", "day"]).reset_index(drop=True)

    grp = t.groupby("customer_id")
    for col, usual in (("n", "usual_n"), ("total", "usual_total")):
        # shift(1): a day never contributes to its own norm, nor do later days
        t[usual] = grp[col].transform(lambda s: s.shift(1).expanding(min_periods=MIN_PRIOR_DAYS).mean())
    t["n_vs_usual"] = (t["n"] / t["usual_n"].clip(lower=0.5)).where(t["usual_n"].notna(), 1.0)
    t["spend_vs_usual"] = (t["total"] / t["usual_total"].clip(lower=1.0)).where(t["usual_total"].notna(), 1.0)
    return t


# ----------------------------------------------------------------- signals

def signals_for(row) -> list:
    """The explainable signals that fire for one account-day row."""
    out = []
    if row["new_dev"] >= 1 and row["new_geo"] >= 1:
        out.append({"name": "takeover", "label": "New device and new region",
                    "detail": f"Used {int(row['new_dev'])} new device(s) from {int(row['new_geo'])} "
                              f"new region(s) in one day"})
    if row["max_ratio"] >= 4:
        out.append({"name": "amount_spike", "label": "Unusually large payment",
                    "detail": f"Largest payment was {row['max_ratio']:.1f}x this customer's usual amount"})
    if row["failed"] >= 3:
        out.append({"name": "card_testing", "label": "Repeated failed payments",
                    "detail": f"{int(row['failed'])} failed payments in one day"})
    if row["n_vs_usual"] >= 4 and row["n"] >= 5:
        out.append({"name": "activity_burst", "label": "Burst of activity",
                    "detail": f"{int(row['n'])} payments, {row['n_vs_usual']:.1f}x this customer's usual daily activity"})
    if row["spend_vs_usual"] >= 6:
        out.append({"name": "spend_surge", "label": "Spend far above normal",
                    "detail": f"Spent {row['spend_vs_usual']:.1f}x the usual daily total"})
    return out


def _severity(points: int):
    if points >= 3:
        return "critical"
    if points == 2:
        return "high"
    if points == 1:
        return "watch"
    return None


# ---------------------------------------------------------------- detector

class AnomalyDetector:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self._forest = None
        self._threshold = None
        self.reference_rate = None

    @staticmethod
    def _matrix(table: pd.DataFrame):
        return np.log1p(table[MODEL_COLUMNS].clip(lower=0))

    def fit(self, reference: pd.DataFrame):
        if len(reference) < MIN_REFERENCE_ROWS:
            raise ValueError(f"Need at least {MIN_REFERENCE_ROWS} reference account-days, got {len(reference)}")
        X = self._matrix(reference)
        self._forest = IsolationForest(n_estimators=300, random_state=self.seed).fit(X)
        scores = -self._forest.score_samples(X)
        self._threshold = float(np.quantile(scores, REFERENCE_QUANTILE))
        self.reference_rate = float((scores > self._threshold).mean())
        return self

    def detect(self, table: pd.DataFrame) -> pd.DataFrame:
        if self._forest is None:
            raise RuntimeError("fit() the detector on reference days first")
        if table.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        scores = -self._forest.score_samples(self._matrix(table))
        rows = []
        for (_, row), score in zip(table.iterrows(), scores):
            signals = signals_for(row)
            iso_flag = bool(score > self._threshold)
            points = int(iso_flag) + len(signals)
            severity = _severity(points)
            if severity is None:
                continue
            rows.append({
                "customer_id": row["customer_id"], "day": int(row["day"]), "severity": severity,
                "points": points, "iso_score": float(score), "iso_flag": iso_flag, "signals": signals,
                "n_txns": int(row["n"]), "total_amount": float(row["total"]),
                "recommended_action": ACTIONS[severity],
            })
        out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
        if out.empty:
            return out
        out["_rank"] = out["severity"].map({s: i for i, s in enumerate(SEVERITIES)})
        return (out.sort_values(["_rank", "iso_score"], ascending=[True, False])
                   .drop(columns="_rank").reset_index(drop=True))


# ----------------------------------------------------------------- summary

def summarize(flags: pd.DataFrame, n_accounts: int, reference_rate: float) -> dict:
    by = {s: int((flags["severity"] == s).sum()) if not flags.empty else 0 for s in SEVERITIES}
    iso = int(flags["iso_flag"].sum()) if not flags.empty else 0
    rate = iso / n_accounts if n_accounts else 0.0
    return {
        "accounts_checked": int(n_accounts),
        "flagged": sum(by.values()),
        "by_severity": by,
        "iso_flag_rate": rate,
        "rate_vs_expected": rate / reference_rate if reference_rate else 0.0,
    }


def to_records(flags: pd.DataFrame, limit: int = None) -> list:
    """JSON-safe records, most severe first."""
    df = flags if limit is None else flags.head(limit)
    return [{k: (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in rec.items()}
            for rec in df.to_dict("records")]
