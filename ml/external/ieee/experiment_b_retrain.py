"""
Experiment B — External Retraining on IEEE-CIS
===============================================

QUESTION ANSWERED:
  "Can MerchantShield's feature-engineering + training methodology be
   retrained from scratch on IEEE-CIS, and does it outperform baselines
   on that dataset?"

WHAT THIS EXPERIMENT IS:
  - A NEW LightGBM model is trained from scratch using the honest IEEE-CIS
    feature set (8 active + 5 degraded/inert = 15 total).
  - The same model architecture as train_candidates.py (num_leaves, max_depth,
    learning_rate, scale_pos_weight, early stopping).
  - Threshold selection uses the EXTERNAL validation set only — the test set
    is never touched during selection.
  - The same cost-based threshold-selection rule as MerchantShield's original
    pipeline: minimise expected cost subject to recall >= 80%.
  - Baselines: Logistic Regression and a majority-class naive classifier.
  - The test set is scored ONCE at the selected threshold.

WHAT THIS EXPERIMENT IS NOT:
  - It is NOT Experiment A (frozen model evaluation).
  - The retrained model is SEPARATE from lgbm_v1 — stored as
    ml/external/ieee/results/ieee_lgbm_v1.pkl.
  - It does NOT modify lgbm_v1, its threshold, or any existing production code.
  - It does NOT use the frozen 0.40 threshold — a new threshold is selected
    on the external validation set.

NOTE ON PREVALENCE SHIFT:
  IEEE-CIS fraud prevalence is ~3.5% vs ~1.6% in MerchantShield's synthetic
  training data. scale_pos_weight for the retrained model is computed from the
  EXTERNAL training set. The threshold sweep on the external validation set
  will find a different optimal point than 0.40 (expected from the sensitivity
  analysis: higher prevalence → lower optimal threshold).

NOTE ON COST MODEL:
  We retain fp_cost=50, fn_cost_fraction=0.5 for direct comparability with the
  synthetic experiment. IEEE-CIS amounts are USD; MerchantShield's were INR.
  The cost numbers are labelled as "cost units" throughout. Absolute values are
  NOT comparable across experiments; relative ordering within each experiment is.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)
import lightgbm as lgb

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost

EXTERNAL_RESULTS_DIR = os.path.join(_ROOT, "ml", "external", "ieee", "results")
IEEE_MODEL_PATH = os.path.join(EXTERNAL_RESULTS_DIR, "ieee_lgbm_v1.pkl")
MIN_RECALL_FLOOR = 0.80


def _score(y_true, y_prob, amounts, threshold, cost_assumptions):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = expected_cost(y_true, y_pred, amounts, cost_assumptions)
    try:
        roc = float(roc_auc_score(y_true, y_prob))
        pr  = float(average_precision_score(y_true, y_prob))
    except ValueError:
        roc = pr = float("nan")
    return dict(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc, pr_auc=pr,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        fraud_total=int(tp + fn),
        fraud_caught=int(tp),
        fp_cost=float(cost["fp_total_cost"]),
        fn_cost=float(cost["fn_total_cost"]),
        total_cost=float(cost["total_expected_cost"]),
    )


def _threshold_sweep(y_true, y_prob, amounts, cost_assumptions):
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    rows = []
    for th in thresholds:
        rows.append(_score(y_true, y_prob, amounts, th, cost_assumptions))
    return pd.DataFrame(rows)


def _select_threshold(sweep_df, recall_floor=MIN_RECALL_FLOOR):
    eligible = sweep_df[sweep_df["recall"] >= recall_floor]
    if len(eligible) > 0:
        row = eligible.loc[eligible["total_cost"].idxmin()]
        found = True
    else:
        row = sweep_df.loc[sweep_df["recall"].idxmax()]
        found = False
    return float(row["threshold"]), row.to_dict(), found


def run_experiment_b(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment B: retrain LightGBM + LR baseline on IEEE-CIS data.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Output of ieee_split.split_features().

    Returns
    -------
    results : dict
    """
    X_train = train[FEATURE_COLUMNS].values
    X_val   = val[FEATURE_COLUMNS].values
    X_test  = test[FEATURE_COLUMNS].values
    y_train = train["is_fraud"].values
    y_val   = val["is_fraud"].values
    y_test  = test["is_fraud"].values
    amt_train = train["amount"].values
    amt_val   = val["amount"].values
    amt_test  = test["amount"].values

    cost = CostAssumptions()  # fp=50, fn_fraction=0.5

    if verbose:
        print(f"[Exp B] Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
        print(f"[Exp B] Train fraud rate: {y_train.mean():.4%}")
        print(f"[Exp B] External prevalence is {y_train.mean():.4%} vs ~1.59% synthetic. "
              f"scale_pos_weight will be recalculated from external training data.")

    results = {
        "experiment": "B",
        "description": "External retraining on IEEE-CIS (fresh model, new threshold)",
        "dataset": "IEEE-CIS Fraud Detection (Vesta / Kaggle 2019)",
        "feature_contract": {
            "total_features": len(FEATURE_COLUMNS),
            "inert_features": ["new_device_flag", "failed_ratio_trailing10"],
            "degraded_features": ["new_geo_flag", "account_age_days",
                                  "hour_of_day", "is_night", "day_of_week"],
        },
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "train_fraud_rate": float(y_train.mean()),
        "val_fraud_rate": float(y_val.mean()),
        "test_fraud_rate": float(y_test.mean()),
    }

    # ------------------------------------------------------------------ #
    # 1. Naive baseline: predict all-positive
    # ------------------------------------------------------------------ #
    naive_test = _score(y_test, np.ones(len(y_test), dtype=float), amt_test, 0.5, cost)
    if verbose:
        print(f"[Exp B] Naive all-positive baseline: "
              f"P={naive_test['precision']:.3f} R={naive_test['recall']:.3f} "
              f"Cost={naive_test['total_cost']:,.2f}")
    results["naive_baseline_test"] = naive_test

    # ------------------------------------------------------------------ #
    # 2. Logistic Regression baseline (mirrors train_baseline.py approach)
    # ------------------------------------------------------------------ #
    if verbose:
        print("[Exp B] Training Logistic Regression baseline…")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X_train_s, y_train)
    lr_val_prob  = lr.predict_proba(X_val_s)[:, 1]
    lr_test_prob = lr.predict_proba(X_test_s)[:, 1]

    lr_val_sweep = _threshold_sweep(y_val, lr_val_prob, amt_val, cost)
    lr_threshold, lr_val_best_row, lr_found = _select_threshold(lr_val_sweep)
    lr_test_metrics = _score(y_test, lr_test_prob, amt_test, lr_threshold, cost)

    if verbose:
        print(f"  LR threshold (val-selected): {lr_threshold:.2f}  "
              f"val P={lr_val_best_row['precision']:.3f} R={lr_val_best_row['recall']:.3f}")
        print(f"  LR test: P={lr_test_metrics['precision']:.3f}  "
              f"R={lr_test_metrics['recall']:.3f}  F1={lr_test_metrics['f1']:.3f}  "
              f"Cost={lr_test_metrics['total_cost']:,.2f}")

    results["logistic_regression"] = {
        "val_threshold": lr_threshold,
        "val_metrics_at_threshold": lr_val_best_row,
        "test_metrics": lr_test_metrics,
        "threshold_found_within_recall_floor": lr_found,
    }

    # ------------------------------------------------------------------ #
    # 3. LightGBM — external retrain (same architecture as train_candidates.py)
    # ------------------------------------------------------------------ #
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    if verbose:
        print(f"\n[Exp B] Training LightGBM (external)…")
        print(f"  scale_pos_weight={scale_pos_weight:.2f}  "
              f"(pos={n_pos:,}  neg={n_neg:,})")

    lgbm = lgb.LGBMClassifier(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=600,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    lgbm.fit(
        X_train, y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)
        ],
    )
    best_iter = lgbm.best_iteration_
    if verbose:
        print(f"  LightGBM early stopped at iteration {best_iter}")

    lgbm_val_prob  = lgbm.predict_proba(X_val)[:, 1]
    lgbm_test_prob = lgbm.predict_proba(X_test)[:, 1]

    # Threshold selection on VAL ONLY
    lgbm_val_sweep = _threshold_sweep(y_val, lgbm_val_prob, amt_val, cost)
    lgbm_threshold, lgbm_val_best_row, lgbm_found = _select_threshold(lgbm_val_sweep)

    if verbose:
        print(f"  LGBM val-selected threshold: {lgbm_threshold:.2f}  "
              f"P={lgbm_val_best_row['precision']:.3f}  R={lgbm_val_best_row['recall']:.3f}  "
              f"Cost={lgbm_val_best_row['total_cost']:,.2f}")

    # Evaluate ONCE on test at selected threshold
    lgbm_test_metrics = _score(y_test, lgbm_test_prob, amt_test, lgbm_threshold, cost)

    if verbose:
        print(f"\n[Exp B] === LGBM TEST RESULT (PRIMARY) ===")
        print(f"  Threshold: {lgbm_threshold:.2f}  "
              f"P={lgbm_test_metrics['precision']:.3f}  "
              f"R={lgbm_test_metrics['recall']:.3f}  "
              f"F1={lgbm_test_metrics['f1']:.3f}")
        print(f"  ROC-AUC={lgbm_test_metrics['roc_auc']:.3f}  "
              f"PR-AUC={lgbm_test_metrics['pr_auc']:.3f}")
        print(f"  FP={lgbm_test_metrics['fp']}  FN={lgbm_test_metrics['fn']}  "
              f"Caught={lgbm_test_metrics['fraud_caught']}/{lgbm_test_metrics['fraud_total']}")
        print(f"  Cost={lgbm_test_metrics['total_cost']:,.2f} units")

    # ------------------------------------------------------------------ #
    # 4. Feature importance from the retrained model
    # ------------------------------------------------------------------ #
    importance = dict(zip(FEATURE_COLUMNS, lgbm.feature_importances_.tolist()))
    importance_sorted = dict(
        sorted(importance.items(), key=lambda x: x[1], reverse=True)
    )

    if verbose:
        print("\n[Exp B] Feature importances (gain):")
        for feat, imp in importance_sorted.items():
            bar = "█" * int(imp / max(importance.values()) * 30)
            print(f"  {feat:28s}: {imp:6.0f} {bar}")

    # ------------------------------------------------------------------ #
    # 5. Compare LR baseline vs LGBM on test
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[Exp B] === COMPARISON TABLE ===")
        print(f"  {'Model':<30}  {'Threshold':>9}  {'Precision':>9}  {'Recall':>6}  "
              f"{'F1':>5}  {'ROC-AUC':>7}  {'Cost':>10}")
        print(f"  {'-'*30}  {'-'*9}  {'-'*9}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*10}")
        print(f"  {'Naive (all-positive)':<30}  {'0.50':>9}  "
              f"{naive_test['precision']:>9.3f}  {naive_test['recall']:>6.3f}  "
              f"{naive_test['f1']:>5.3f}  {'—':>7}  {naive_test['total_cost']:>10,.2f}")
        print(f"  {'Logistic Regression':<30}  {lr_threshold:>9.2f}  "
              f"{lr_test_metrics['precision']:>9.3f}  {lr_test_metrics['recall']:>6.3f}  "
              f"{lr_test_metrics['f1']:>5.3f}  {lr_test_metrics['roc_auc']:>7.3f}  "
              f"{lr_test_metrics['total_cost']:>10,.2f}")
        print(f"  {'LightGBM (external)':<30}  {lgbm_threshold:>9.2f}  "
              f"{lgbm_test_metrics['precision']:>9.3f}  {lgbm_test_metrics['recall']:>6.3f}  "
              f"{lgbm_test_metrics['f1']:>5.3f}  {lgbm_test_metrics['roc_auc']:>7.3f}  "
              f"{lgbm_test_metrics['total_cost']:>10,.2f}")

    # ------------------------------------------------------------------ #
    # 6. Save the external model (separate from lgbm_v1 — never overwrites it)
    # ------------------------------------------------------------------ #
    os.makedirs(EXTERNAL_RESULTS_DIR, exist_ok=True)
    joblib.dump(lgbm, IEEE_MODEL_PATH)
    if verbose:
        print(f"\n[Exp B] External model saved to: {IEEE_MODEL_PATH}")
        print(f"  (This does NOT overwrite ml/models/candidate_lgbm_v1.pkl)")

    # Verify lgbm_v1 still intact
    frozen_path = os.path.join(_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
    assert os.path.exists(frozen_path), "Frozen model was unexpectedly deleted!"

    # ------------------------------------------------------------------ #
    # 7. Package and save results
    # ------------------------------------------------------------------ #
    results.update({
        "lightgbm_external": {
            "best_iteration": best_iter,
            "val_threshold": lgbm_threshold,
            "val_threshold_found_within_recall_floor": lgbm_found,
            "val_metrics_at_threshold": lgbm_val_best_row,
            "test_metrics": lgbm_test_metrics,
            "val_threshold_sweep": lgbm_val_sweep.to_dict(orient="records"),
            "feature_importance_gain": importance_sorted,
            "model_path": IEEE_MODEL_PATH,
            "note": "This model is separate from lgbm_v1. Do not use for production.",
        },
    })

    out_path = os.path.join(EXTERNAL_RESULTS_DIR, "experiment_b_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"[Exp B] Results saved to {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--txn",  required=True)
    parser.add_argument("--idnt", default=None)
    args = parser.parse_args()

    from ml.external.ieee.ieee_adapter  import load_and_adapt
    from ml.external.ieee.ieee_features import build_ieee_features
    from ml.external.ieee.ieee_split    import split_features

    adapted, _  = load_and_adapt(args.txn, args.idnt, verbose=True)
    features, _ = build_ieee_features(adapted, verbose=True)
    train, val, test = split_features(features, verbose=True)
    run_experiment_b(train, val, test, verbose=True)
