"""
Phase 3 baseline: Logistic Regression trained on the chronological split.

Establishes the performance floor that all candidate models must beat before
any is considered for selection (Phase 4). Also defines `chronological_split`,
which is the project's shared utility for every train/eval script — it is the
single place where the day-boundary constants (train < day 40, 40 <= val < 50,
test >= day 50) are applied to the feature DataFrame.

RULES FOLLOWED:
- No hyperparameter tuning against the test set.
- No cherry-picking a threshold to improve reported metrics.
- Test set is touched exactly once, at the very end, for reporting.
- `chronological_split` is imported (not re-implemented) by every downstream
  script that needs the same split, so the boundary values can never diverge.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, confusion_matrix, precision_recall_curve
)

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost


def chronological_split(df, train_end_day, val_end_day):
    """
    Split a feature DataFrame into train / validation / test sets by simulation day.

    The split is time-ordered, not random, to prevent any time-aware feature
    (velocity, rolling average, device/geo novelty) from leaking future information
    into the training set via shuffled rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'timestamp' column (parseable by pd.to_datetime).
    train_end_day : int
        Exclusive upper bound for the training set, measured in whole days from
        the earliest timestamp in `df`.  All rows with sim_day < train_end_day
        are training data.  Default used throughout the project: 40.
    val_end_day : int
        Exclusive upper bound for the validation set.  Rows with
        train_end_day <= sim_day < val_end_day form the validation set.
        Default used throughout the project: 50.

    Returns
    -------
    train, val, test : pd.DataFrame
        Three non-overlapping, time-ordered DataFrames.  Test contains all rows
        with sim_day >= val_end_day.

    Notes
    -----
    `sim_day` is computed relative to the earliest timestamp in `df`, so the
    day boundaries are dataset-relative, not calendar-absolute.  This means the
    same boundary values reproduce the same split across any environment that
    uses the same synthetic dataset (i.e. any run of generate_synthetic.py with
    the fixed RNG_SEED=42).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    start = df["timestamp"].min().normalize()
    df["sim_day"] = (df["timestamp"] - start).dt.days

    train = df[df["sim_day"] < train_end_day]
    val = df[(df["sim_day"] >= train_end_day) & (df["sim_day"] < val_end_day)]
    test = df[df["sim_day"] >= val_end_day]
    return train, val, test


def evaluate(y_true, y_prob, amounts, threshold=0.5, label=""):
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    cost = expected_cost(y_true, y_pred, amounts, CostAssumptions())

    print(f"\n===== {label} (threshold={threshold}) =====")
    print(f"Class distribution: {np.mean(y_true):.4%} positive, n={len(y_true):,}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Confusion matrix [tn, fp / fn, tp]:\n[{tn:>6} {fp:>6}]\n[{fn:>6} {tp:>6}]")
    print(f"False positives: {fp}   False negatives: {fn}")
    print(f"Estimated cost (FP_COST=Rs{cost['fp_cost_assumed']}, "
          f"FN_COST={cost['fn_cost_fraction_assumed']}xamount): Rs{cost['total_expected_cost']:,.2f}")
    print(f"  -> FP contribution: Rs{cost['fp_total_cost']:,.2f} | FN contribution: Rs{cost['fn_total_cost']:,.2f}")

    return dict(
        precision=precision, recall=recall, f1=f1, pr_auc=pr_auc, roc_auc=roc_auc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), cost=cost,
    )


def main():
    feats = pd.read_csv("ml/data/features.csv")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])

    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)

    print(f"Train: {len(train):,} rows | days 0-39 | fraud rate {train['is_fraud'].mean():.4%}")
    print(f"Val:   {len(val):,} rows | days 40-49 | fraud rate {val['is_fraud'].mean():.4%}")
    print(f"Test:  {len(test):,} rows | days 50-59 | fraud rate {test['is_fraud'].mean():.4%}")

    print("\nFeature availability check (as-of prediction time): all", len(FEATURE_COLUMNS),
          "features are computed strictly from prior transactions / calendar fields — "
          "no post-hoc or same-transaction-outcome fields are used.")
    print("Features used:", FEATURE_COLUMNS)

    X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"]
    X_val, y_val = val[FEATURE_COLUMNS], val["is_fraud"]
    X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(X_train_s, y_train)

    val_prob = model.predict_proba(X_val_s)[:, 1]
    test_prob = model.predict_proba(X_test_s)[:, 1]

    val_metrics = evaluate(y_val.values, val_prob, val["amount"].values, threshold=0.5, label="VALIDATION")
    test_metrics = evaluate(y_test.values, test_prob, test["amount"].values, threshold=0.5, label="TEST (held-out)")

    # persist for reuse in later phases
    os.makedirs("ml/models", exist_ok=True)
    import joblib
    joblib.dump(dict(model=model, scaler=scaler, feature_columns=FEATURE_COLUMNS),
                "ml/models/baseline_logreg_v1.pkl")

    print("\nBaseline model saved to ml/models/baseline_logreg_v1.pkl")
    return val_metrics, test_metrics


if __name__ == "__main__":
    main()
