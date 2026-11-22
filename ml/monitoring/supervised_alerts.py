"""
Account alerts from the supervised model's scores.

An account-day's risk is the highest fraud probability among that day's payments. On the real Sparkov
data this beat the unsupervised account detector on validation days by a wide margin, because the
supervised model sees each transaction (amount, hour, velocity, history) while the unsupervised signals
only see day totals. It needs fraud labels, so it applies where labels exist (a dataset workspace), not
to a brand-new deployment.

Two tier thresholds (critical above high, always distinct) are chosen on VALIDATION days only. They have
to be chosen per model: class weighting inflates probabilities, so useful thresholds sit near 0.99 and
differ between LightGBM, XGBoost and the rest. The grid therefore reaches up to 0.9999; an earlier
version stopped at 0.95 and picked the grid's edge, a sign it was too coarse.

Like the other detector this only RECOMMENDS. Records have the same shape, so the alert queue and the
dashboard pages need no changes.
"""

import numpy as np
import pandas as pd

from ml.monitoring.anomaly import ACTIONS, OUTPUT_COLUMNS

TIER_GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999)
OBJECTIVE = "mean F1 of the critical and critical+high tiers on the validation days"


def account_day_scores(df: pd.DataFrame, probs, origin=None) -> pd.DataFrame:
    """One row per customer per day: the day's highest payment score, with counts and the fraud label if known."""
    probs = np.asarray(probs, dtype=float)
    if len(probs) != len(df):
        raise ValueError(f"Need one score per row: got {len(probs)} scores for {len(df)} rows.")
    ts = pd.to_datetime(df["timestamp"])
    origin = pd.Timestamp(origin) if origin is not None else ts.min().normalize()
    frame = pd.DataFrame({"customer_id": df["customer_id"].to_numpy(),
                          "day": (ts.dt.normalize() - origin).dt.days.to_numpy(),
                          "p": probs, "amount": df["amount"].to_numpy()})
    spec = dict(max_prob=("p", "max"), n_txns=("p", "size"), total=("amount", "sum"))
    if "is_fraud" in df.columns:
        frame["fraud"] = df["is_fraud"].to_numpy()
        spec["fraud"] = ("fraud", "max")
    return frame.groupby(["customer_id", "day"], sort=True).agg(**spec).reset_index()


def _tier_at(scores: np.ndarray, truth: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    tp, n = int((flagged & truth).sum()), int(flagged.sum())
    precision, recall = (tp / n if n else 0.0), (tp / truth.sum() if truth.sum() else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"flagged": n, "precision": precision, "recall": recall, "f1": f1}


def calibrate_thresholds(validation: pd.DataFrame, grid=TIER_GRID) -> dict:
    """Pick (critical, high) with critical strictly above high, by validation days only."""
    if "fraud" not in validation.columns or not (validation["fraud"] == 1).any():
        raise ValueError("The validation window has no compromised account-days, so there is nothing to calibrate on.")
    scores, truth = validation["max_prob"].to_numpy(), validation["fraud"].to_numpy() == 1
    results = []
    for high in grid:
        for critical in grid:
            if critical <= high:
                continue
            c, both = _tier_at(scores, truth, critical), _tier_at(scores, truth, high)
            results.append({"critical": critical, "high": high, "critical_tier": c, "critical_plus_high": both,
                            "score": (c["f1"] + both["f1"]) / 2})
    best = max(results, key=lambda g: g["score"])        # ties go to the first (lower, more sensitive) pair
    return {"critical": best["critical"], "high": best["high"], "score": best["score"], "objective": OBJECTIVE,
            "validation_account_days": int(len(validation)), "validation_compromised": int(truth.sum()),
            "grid": results}


def detect_supervised(scores: pd.DataFrame, critical: float, high: float) -> pd.DataFrame:
    """Critical and high alerts in the same record shape the anomaly detector produces."""
    hit = scores[scores["max_prob"] >= high]
    rows = []
    for r in hit.itertuples():
        severity = "critical" if r.max_prob >= critical else "high"
        rows.append({
            "customer_id": r.customer_id, "day": int(r.day), "severity": severity,
            "points": 3 if severity == "critical" else 2, "iso_score": float(r.max_prob), "iso_flag": False,
            "signals": [{"name": "model_score", "label": "High model fraud score",
                         "detail": f"Highest-risk payment scored {r.max_prob:.1%} fraud probability"}],
            "n_txns": int(r.n_txns), "total_amount": float(r.total), "recommended_action": ACTIONS[severity],
        })
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if out.empty:
        return out
    out["_rank"] = (out["severity"] != "critical").astype(int)
    return out.sort_values(["_rank", "iso_score"], ascending=[True, False]).drop(columns="_rank").reset_index(drop=True)
