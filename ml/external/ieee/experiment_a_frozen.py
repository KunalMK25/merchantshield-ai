"""
Experiment A — Frozen LightGBM Transfer to IEEE-CIS
====================================================

QUESTION ANSWERED:
  "Does the existing frozen lgbm_v1 model (trained on MerchantShield's
   synthetic data) generalise to the IEEE-CIS external dataset?"

WHAT THIS EXPERIMENT IS:
  - The frozen model (ml/models/candidate_lgbm_v1.pkl) is loaded AS-IS.
  - The frozen threshold (0.40, from lgbm_v1_metadata.json) is applied first.
  - No retraining, no fine-tuning, no threshold re-selection.
  - A threshold sweep on the external validation set is run SEPARATELY as
    Experiment A.5 to answer: "Is the model's probability ranking useful even
    if its calibration is off?" This is logged but never feeds back into any
    model or threshold selection decision.

WHAT THIS EXPERIMENT IS NOT:
  - It is NOT Experiment B (external retraining).
  - Results here do NOT update lgbm_v1 or its threshold.
  - A failed transfer does not mean the methodology failed.

FEATURE CONTRACT:
  The frozen model was trained on all 15 FEATURE_COLUMNS including 2 inert
  features (new_device_flag=0, failed_ratio_trailing10=0.0) that are constant
  in the IEEE-CIS adapter. This means the frozen model sees those features as
  zero for every transaction — consistent with what build_features() produces
  for a customer's very first transaction on the synthetic data. The model
  can still score transactions; those features simply carry no discriminative
  signal in the external experiment.

COST MODEL NOTE:
  MerchantShield's cost model uses INR (Rs50 FP, 0.5×amount FN).
  IEEE-CIS amounts are in USD. We retain the same numeric parameters for
  direct comparability — the absolute cost numbers are reported as "cost
  units" rather than a named currency. The relative ordering of thresholds
  and the cost ratio between FP and FN are what matter for threshold
  selection, not the currency label.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost
from ml.external.ieee.ieee_split import TRAIN_BOUNDARY, VAL_BOUNDARY

FROZEN_MODEL_PATH = os.path.join(_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")
FROZEN_METADATA_PATH = os.path.join(_ROOT, "ml", "models", "lgbm_v1_metadata.json")
EXTERNAL_RESULTS_DIR = os.path.join(_ROOT, "ml", "external", "ieee", "results")
MIN_RECALL_FLOOR = 0.80


def _score_at_threshold(y_true, y_prob, amounts, threshold, cost_assumptions):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = expected_cost(y_true, y_pred, amounts, cost_assumptions)
    return dict(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        fraud_total=int(tp + fn),
        fraud_caught=int(tp),
        fp_cost=float(cost["fp_total_cost"]),
        fn_cost=float(cost["fn_total_cost"]),
        total_cost=float(cost["total_expected_cost"]),
    )


def run_experiment_a(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    verbose: bool = True,
) -> dict:
    """
    Run Experiment A: evaluate the frozen lgbm_v1 model on IEEE-CIS data.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Output of ieee_split.split_features(). Must contain FEATURE_COLUMNS,
        'is_fraud', 'TransactionDT', 'amount'.

    Returns
    -------
    results : dict  — full metrics, threshold sweep, and conclusions.
    """
    # ------------------------------------------------------------------ #
    # 0. Load frozen model — never modify it
    # ------------------------------------------------------------------ #
    if not os.path.exists(FROZEN_MODEL_PATH):
        raise FileNotFoundError(
            f"Frozen model not found at {FROZEN_MODEL_PATH}. "
            "Ensure the MerchantShield repository is intact."
        )
    model = joblib.load(FROZEN_MODEL_PATH)

    with open(FROZEN_METADATA_PATH) as f:
        meta = json.load(f)

    frozen_threshold = float(meta["selected_threshold"])
    assert frozen_threshold == 0.40, (
        f"Expected frozen threshold 0.40, got {frozen_threshold}. "
        "Do not change the frozen threshold."
    )
    assert meta["model_name"] == "lgbm_v1"

    if verbose:
        print(f"[Exp A] Frozen model: {meta['model_name']}  threshold: {frozen_threshold}")
        print(f"[Exp A] Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    cost = CostAssumptions()  # fp=50, fn_fraction=0.5 — same parameters, USD amounts

    # ------------------------------------------------------------------ #
    # 1. Score all three splits with the frozen model
    # ------------------------------------------------------------------ #
    X_train = train[FEATURE_COLUMNS]
    X_val   = val[FEATURE_COLUMNS]
    X_test  = test[FEATURE_COLUMNS]

    y_train, y_val, y_test = (
        train["is_fraud"].values,
        val["is_fraud"].values,
        test["is_fraud"].values,
    )
    amt_train, amt_val, amt_test = (
        train["amount"].values,
        val["amount"].values,
        test["amount"].values,
    )

    if verbose:
        print("[Exp A] Scoring all splits with frozen model…")
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val   = model.predict_proba(X_val)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------------------ #
    # 2. Evaluate at the frozen 0.40 threshold — PRIMARY RESULT
    # ------------------------------------------------------------------ #
    res_val_frozen  = _score_at_threshold(y_val,  prob_val,  amt_val,  frozen_threshold, cost)
    res_test_frozen = _score_at_threshold(y_test, prob_test, amt_test, frozen_threshold, cost)

    try:
        roc_val  = float(roc_auc_score(y_val,  prob_val))
        roc_test = float(roc_auc_score(y_test, prob_test))
        pr_val   = float(average_precision_score(y_val,  prob_val))
        pr_test  = float(average_precision_score(y_test, prob_test))
    except ValueError:
        roc_val = roc_test = pr_val = pr_test = float("nan")

    res_val_frozen.update(roc_auc=roc_val, pr_auc=pr_val)
    res_test_frozen.update(roc_auc=roc_test, pr_auc=pr_test)

    if verbose:
        print(f"\n[Exp A] === FROZEN THRESHOLD ({frozen_threshold}) ON VALIDATION ===")
        print(f"  P={res_val_frozen['precision']:.3f}  R={res_val_frozen['recall']:.3f}  "
              f"F1={res_val_frozen['f1']:.3f}  ROC-AUC={roc_val:.3f}  PR-AUC={pr_val:.3f}")
        print(f"  FP={res_val_frozen['fp']}  FN={res_val_frozen['fn']}  "
              f"Caught={res_val_frozen['fraud_caught']}/{res_val_frozen['fraud_total']}")
        print(f"  Cost={res_val_frozen['total_cost']:,.2f} units")

        print(f"\n[Exp A] === FROZEN THRESHOLD ({frozen_threshold}) ON TEST (PRIMARY RESULT) ===")
        print(f"  P={res_test_frozen['precision']:.3f}  R={res_test_frozen['recall']:.3f}  "
              f"F1={res_test_frozen['f1']:.3f}  ROC-AUC={roc_test:.3f}  PR-AUC={pr_test:.3f}")
        print(f"  FP={res_test_frozen['fp']}  FN={res_test_frozen['fn']}  "
              f"Caught={res_test_frozen['fraud_caught']}/{res_test_frozen['fraud_total']}")
        print(f"  Cost={res_test_frozen['total_cost']:,.2f} units")

    # ------------------------------------------------------------------ #
    # 3. Threshold sweep on VALIDATION (Exp A.5 — ranking utility check)
    # This does NOT touch the test set. It answers: "Is the model's ranking
    # useful even if 0.40 isn't the right operating point externally?"
    # ------------------------------------------------------------------ #
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    val_sweep = []
    for th in thresholds:
        r = _score_at_threshold(y_val, prob_val, amt_val, th, cost)
        val_sweep.append(r)
    val_sweep_df = pd.DataFrame(val_sweep)

    # Best threshold on val by cost (subject to recall floor)
    eligible = val_sweep_df[val_sweep_df["recall"] >= MIN_RECALL_FLOOR]
    if len(eligible) > 0:
        best_val_row = eligible.loc[eligible["total_cost"].idxmin()]
        best_val_threshold = float(best_val_row["threshold"])
        external_best_threshold_found = True
    else:
        best_val_row = val_sweep_df.loc[val_sweep_df["recall"].idxmax()]
        best_val_threshold = float(best_val_row["threshold"])
        external_best_threshold_found = False

    if verbose:
        print(f"\n[Exp A.5] Best external threshold on val (cost-min, recall≥{MIN_RECALL_FLOOR:.0%}): "
              f"{best_val_threshold:.2f}  "
              f"P={best_val_row['precision']:.3f}  R={best_val_row['recall']:.3f}  "
              f"Cost={best_val_row['total_cost']:,.2f}")
        print("  [Note: this does NOT update the frozen model or threshold]")

    # Evaluate the val-best threshold on the test set
    # (purely for informational comparison — this is NOT the model selection step)
    res_test_best_external = _score_at_threshold(
        y_test, prob_test, amt_test, best_val_threshold, cost
    )
    res_test_best_external.update(roc_auc=roc_test, pr_auc=pr_test)

    # ------------------------------------------------------------------ #
    # 4. Naive baseline: predict all-positive (recall=100%, precision=prevalence)
    # ------------------------------------------------------------------ #
    prevalence = float(y_test.mean())
    naive_all_fraud = _score_at_threshold(y_test, np.ones_like(y_test, dtype=float), amt_test, 0.5, cost)

    # ------------------------------------------------------------------ #
    # 5. Probability calibration summary
    # ------------------------------------------------------------------ #
    prob_stats_val = {
        "mean": float(prob_val.mean()),
        "median": float(np.median(prob_val)),
        "p90": float(np.percentile(prob_val, 90)),
        "p99": float(np.percentile(prob_val, 99)),
        "fraction_above_frozen_threshold": float((prob_val >= frozen_threshold).mean()),
        "fraud_mean_prob": float(prob_val[y_val == 1].mean()),
        "legit_mean_prob": float(prob_val[y_val == 0].mean()),
    }

    # ------------------------------------------------------------------ #
    # 6. Package results
    # ------------------------------------------------------------------ #
    results = {
        "experiment": "A",
        "description": "Frozen lgbm_v1 transfer to IEEE-CIS (no retraining)",
        "model": "lgbm_v1",
        "frozen_threshold": frozen_threshold,
        "dataset": "IEEE-CIS Fraud Detection (Vesta / Kaggle 2019)",
        "fraud_prevalence_external": float(prevalence),
        "fraud_prevalence_synthetic_training": 0.0159,  # from lgbm_v1_metadata
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "feature_contract": {
            "total_features": len(FEATURE_COLUMNS),
            "inert_features": ["new_device_flag", "failed_ratio_trailing10"],
            "degraded_features": ["new_geo_flag", "account_age_days",
                                  "hour_of_day", "is_night", "day_of_week"],
        },
        "primary_result": {
            "description": "Frozen 0.40 threshold on held-out test set",
            "val_metrics": res_val_frozen,
            "test_metrics": res_test_frozen,
        },
        "threshold_sweep_val": val_sweep_df.to_dict(orient="records"),
        "exp_a5_best_external_threshold": {
            "threshold": best_val_threshold,
            "found_eligible": external_best_threshold_found,
            "val_metrics": best_val_row.to_dict(),
            "test_metrics": res_test_best_external,
            "note": (
                "This threshold was selected on the external val set. "
                "It does NOT update lgbm_v1 or its frozen 0.40 threshold. "
                "It answers: 'What is the best this frozen model can achieve externally?'"
            ),
        },
        "naive_baseline_test": naive_all_fraud,
        "probability_calibration_val": prob_stats_val,
    }

    # ------------------------------------------------------------------ #
    # 7. Save results
    # ------------------------------------------------------------------ #
    os.makedirs(EXTERNAL_RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(EXTERNAL_RESULTS_DIR, "experiment_a_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\n[Exp A] Results saved to {out_path}")

    return results


if __name__ == "__main__":
    # Run Experiment A standalone (after adapter pipeline has been run)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--txn",  required=True, help="Path to train_transaction.csv")
    parser.add_argument("--idnt", default=None,  help="Path to train_identity.csv (optional)")
    args = parser.parse_args()

    from ml.external.ieee.ieee_adapter  import load_and_adapt
    from ml.external.ieee.ieee_features import build_ieee_features
    from ml.external.ieee.ieee_split    import split_features

    adapted, adapter_meta = load_and_adapt(args.txn, args.idnt, verbose=True)
    features, feat_info   = build_ieee_features(adapted, verbose=True)
    train, val, test       = split_features(features, verbose=True)
    run_experiment_a(train, val, test, verbose=True)
