"""
Threshold choice and winner selection for the model contest.

The rule is the project's existing one (see ml/models/lgbm_v1_metadata.json):
choose the threshold that minimises total expected cost on the VALIDATION set,
subject to a recall floor. Nothing in this module ever receives test data.
"""

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ml.evaluation.cost_model import CostAssumptions, expected_cost

THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def evaluate_at(y_true, y_prob, amounts, cost: CostAssumptions, threshold: float) -> dict:
    """Metrics for one operating point."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    total_cost = expected_cost(y_true, y_pred, amounts, cost)["total_expected_cost"]
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        roc_auc = pr_auc = None
    return {
        "threshold": float(threshold),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "cost": float(total_cost), "roc_auc": roc_auc, "pr_auc": pr_auc,
    }


def pick_threshold(y_true, y_prob, amounts, cost: CostAssumptions, min_recall: float = 0.8):
    """
    Lowest-cost threshold on `THRESHOLD_GRID` whose recall meets `min_recall`.
    If no threshold reaches the floor, fall back to the overall cheapest one and
    flag it with recall_constraint_met=False so callers can refuse to promote it.
    """
    rows = [evaluate_at(y_true, y_prob, amounts, cost, t) for t in THRESHOLD_GRID]
    feasible = [r for r in rows if r["recall"] >= min_recall]
    pool, met = (feasible, True) if feasible else (rows, False)
    best = min(pool, key=lambda r: r["cost"])  # first minimum wins ties
    best["recall_constraint_met"] = met
    return best["threshold"], best


def select_winner(entries):
    """
    Cheapest validation cost among entries that met the recall floor.
    Ties are broken by name so the result is deterministic. None if nobody qualifies.
    """
    qualified = [e for e in entries if e["val"].get("recall_constraint_met")]
    if not qualified:
        return None
    return min(qualified, key=lambda e: (e["val"]["cost"], e["name"]))
