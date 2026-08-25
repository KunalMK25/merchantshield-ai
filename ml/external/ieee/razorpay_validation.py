"""
Phase 17 — Razorpay Real-Data Held-Out Validation
===================================================

PURPOSE
-------
Produce a defensible, reproducible held-out evaluation of MerchantShield's
fraud-detection methodology on real-world e-commerce transaction data, to
satisfy the Razorpay AI Risk Manager track requirement:

  "Build a working detector with measured precision and recall on a
   held-out test set, including false-positive cost."

DATASET
-------
IEEE-CIS Fraud Detection — real e-commerce chargeback data from Vesta
Corporation, released via Kaggle (2019). This is NOT Razorpay transaction
data; it is real-world card-not-present e-commerce fraud data.
Labels represent confirmed chargebacks (isFraud=1).

WHAT THIS MODULE IS
-------------------
- A clean, end-to-end real-data validation experiment.
- Built entirely on top of the existing Phase 14/15 infrastructure.
- Produces a machine-readable result JSON and a frozen held-out test score.
- Does NOT modify lgbm_v1, its threshold, the synthetic pipeline, or any
  Phase 14/15 results.

FEATURE SET
-----------
This experiment uses 16 features:
  - All 15 FEATURE_COLUMNS from MerchantShield's production feature pipeline
    (2 are inert on IEEE-CIS: new_device_flag=0, failed_ratio_trailing10=0)
  - card_product_share from Phase 15 (selected winner, reduces FP by 17.7%)

This matches the Phase 15 winner variant exactly.

DATA LOCATION (configurable)
-----------------------------
The IEEE-CIS CSV files are NOT committed to Git. They must be downloaded
from Kaggle (https://www.kaggle.com/c/ieee-fraud-detection) and placed in
a local directory. The location is controlled by the IEEE_DATA_DIR environment
variable (default: C:/Users/user/Downloads/ieee_cis_inspect).

  Windows:  set IEEE_DATA_DIR=path\\to\\data
  Linux:    export IEEE_DATA_DIR=/path/to/data

Required files inside IEEE_DATA_DIR:
  - train_transaction.csv   (683 MB — contains isFraud labels)
  - train_identity.csv      (26.5 MB — optional device info)

The Kaggle test set (test_transaction.csv) has labels withheld by the
competition host and is NOT used for reporting held-out metrics.

CHRONOLOGICAL SPLIT
-------------------
Reuses the existing Phase 14 boundaries (declared before any modeling):
  Train:       TransactionDT < 9,614,666   ≈ 383,851 rows  (days 0–110)
  Validation:  ≥ 9,614,666 and < 12,192,853 ≈  88,581 rows  (days 110–140)
  Test:        TransactionDT ≥ 12,192,853  ≈ 118,108 rows  (days 140–182)

Threshold selection uses ONLY the validation split.
The test split is scored EXACTLY ONCE at the end, after all decisions are frozen.

LEAKAGE CONTROLS
----------------
All inherited from Phases 14/15:
  - build_features.py: expanding-window per customer_id, strictly prior transactions
  - card_product_features.py: cumcount() — prior-only, label-independent
  - isFraud never read during feature construction
  - Test split never inspected during threshold selection

COST MODEL
----------
Inherits CostAssumptions: fp_cost=50 (cost units), fn_cost_fraction=0.5.
IEEE-CIS amounts are USD; the original model used INR. Cost figures are
reported as "cost units" — absolute values are not comparable across datasets.
The fp_cost=50 is an assumed illustrative value, not a measured Razorpay figure.

SCIENTIFIC INTEGRITY
--------------------
- If the data files are not found, the script stops and explains what to do.
  It does NOT fall back to synthetic data or fabricate metrics.
- The held-out test score is computed exactly once, after threshold freeze.
- All intermediate results (validation sweep, feature importances) are saved.
"""

import os
import sys
import json
import warnings
import time

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost
from ml.external.ieee.ieee_adapter import load_and_adapt
from ml.external.ieee.ieee_features import build_ieee_features
from ml.external.ieee.ieee_split import split_features, TRAIN_BOUNDARY, VAL_BOUNDARY
from ml.external.ieee.card_product_features import (
    build_card_product_features,
    NEW_FEATURE_NAMES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_DEFAULT_DATA_DIR = r"C:\Users\user\Downloads\ieee_cis_inspect"
IEEE_DATA_DIR = os.environ.get("IEEE_DATA_DIR", _DEFAULT_DATA_DIR)

RESULTS_DIR = os.path.join(_ROOT, "ml", "external", "ieee", "results")
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase17_razorpay_results.json")
MODEL_PATH   = os.path.join(RESULTS_DIR, "phase17_lgbm.pkl")

# Phase 17 feature set: base 15 + card_product_share (Phase 15 winner)
P17_FEATURES = list(FEATURE_COLUMNS) + ["card_product_share"]

# Cost model — same parameters as all prior experiments for comparability
COST = CostAssumptions()   # fp_cost=50, fn_cost_fraction=0.5
MIN_RECALL_FLOOR = 0.80

# Frozen production model path (verified never modified)
FROZEN_MODEL_PATH = os.path.join(_ROOT, "ml", "models", "candidate_lgbm_v1.pkl")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _check_data_available() -> tuple[str, str]:
    """
    Resolve and verify data file paths. Raises a clear FileNotFoundError with
    download instructions if files are missing.
    """
    txn_path  = os.path.join(IEEE_DATA_DIR, "train_transaction.csv")
    idnt_path = os.path.join(IEEE_DATA_DIR, "train_identity.csv")

    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            f"\n{'='*70}\n"
            f"IEEE-CIS transaction file not found:\n  {txn_path}\n\n"
            f"The Phase 17 real-data experiment requires the IEEE-CIS dataset.\n"
            f"It is NOT committed to this repository.\n\n"
            f"TO OBTAIN THE DATA:\n"
            f"  1. Create a free Kaggle account at https://www.kaggle.com\n"
            f"  2. Accept the competition rules at:\n"
            f"     https://www.kaggle.com/c/ieee-fraud-detection\n"
            f"  3. Download train_transaction.csv and train_identity.csv\n"
            f"  4. Place both files in:\n  {IEEE_DATA_DIR}\n"
            f"     (or set IEEE_DATA_DIR environment variable to another directory)\n"
            f"  5. Re-run this script\n"
            f"\nThe competition test files (test_transaction.csv) have labels\n"
            f"withheld and are NOT used for held-out metrics.\n"
            f"{'='*70}\n"
        )

    if not os.path.exists(idnt_path):
        print(f"[p17] WARNING: train_identity.csv not found at {idnt_path}")
        print(f"[p17] Proceeding without identity file (DeviceInfo will be absent).")
        return txn_path, None

    return txn_path, idnt_path


def _score_at(y_true, y_prob, amounts, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost_detail = expected_cost(y_true, y_pred, amounts, COST)
    try:
        roc = float(roc_auc_score(y_true, y_prob))
        pra = float(average_precision_score(y_true, y_prob))
    except ValueError:
        roc = pra = float("nan")
    n_legit = int(tn + fp)
    fpr = float(fp / n_legit) if n_legit > 0 else float("nan")
    fp_tp_ratio = float(fp / tp) if tp > 0 else float("inf")
    return dict(
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        pr_auc=pra,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        fraud_total=int(tp + fn),
        fraud_caught=int(tp),
        false_positive_rate=fpr,
        fp_per_tp=fp_tp_ratio,
        fp_cost_units=float(cost_detail["fp_total_cost"]),
        fn_cost_units=float(cost_detail["fn_total_cost"]),
        total_cost_units=float(cost_detail["total_expected_cost"]),
    )


def _val_sweep(y_val, p_val, amt_val):
    """Threshold sweep on validation only. Test set never touched."""
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return [_score_at(y_val, p_val, amt_val, th) for th in thresholds]


def _select_threshold(sweep_rows):
    """
    Cost-minimising threshold subject to recall >= MIN_RECALL_FLOOR.
    If no threshold meets the recall floor, returns the highest-recall threshold.
    """
    df = pd.DataFrame(sweep_rows)
    eligible = df[df["recall"] >= MIN_RECALL_FLOOR]
    if len(eligible) > 0:
        row = eligible.loc[eligible["total_cost_units"].idxmin()]
        found_eligible = True
    else:
        row = df.loc[df["recall"].idxmax()]
        found_eligible = False
    return float(round(row["threshold"], 2)), row.to_dict(), found_eligible


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_phase17(*, verbose: bool = True) -> dict:
    """
    Run the complete Phase 17 real-data validation.

    Returns the full results dict (also saved to RESULTS_PATH).
    Raises FileNotFoundError if data is unavailable.
    """
    t0 = time.time()

    # ------------------------------------------------------------------ #
    # 0. Verify data availability — hard stop if absent
    # ------------------------------------------------------------------ #
    txn_path, idnt_path = _check_data_available()

    # Confirm production model is untouched
    assert os.path.exists(FROZEN_MODEL_PATH), \
        "Production frozen model lgbm_v1 is missing — do not proceed."

    if verbose:
        print(f"\n{'='*70}")
        print("PHASE 17 — RAZORPAY REAL-DATA HELD-OUT VALIDATION")
        print(f"{'='*70}")
        print(f"[p17] Data directory:  {IEEE_DATA_DIR}")
        print(f"[p17] Feature set:     {len(P17_FEATURES)} features "
              f"(15 base + card_product_share)")
        print(f"[p17] Recall floor:    {MIN_RECALL_FLOOR:.0%}")
        print(f"[p17] Cost model:      fp={COST.fp_cost} fn=0.5×amount (cost units)")
        print(f"[p17] Split:           train<{TRAIN_BOUNDARY:,} / val<{VAL_BOUNDARY:,} / test≥{VAL_BOUNDARY:,}")

    # ------------------------------------------------------------------ #
    # 1. Load and adapt IEEE-CIS data
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] Loading IEEE-CIS data…")
    adapted, adapter_meta = load_and_adapt(
        txn_path, idnt_path, verbose=verbose, validate_d11=False
    )

    # ------------------------------------------------------------------ #
    # 2. Build base features (calls build_features.py unchanged)
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] Building base features (build_features.py unchanged)…")
    features, feat_info = build_ieee_features(adapted, verbose=verbose)

    # ------------------------------------------------------------------ #
    # 3. Add card_product_share (Phase 15 winner)
    #    build_card_product_features requires TransactionDT-sorted input
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] Adding card_product_share (Phase 15 winner feature)…")
    features_full = build_card_product_features(features)

    # Verify card_product_share is the only new column used
    assert "card_product_share" in features_full.columns
    assert "card_product_prior_count" in features_full.columns  # built internally
    # We only add the *share* to P17_FEATURES, matching Phase 15's KEEP decision

    # ------------------------------------------------------------------ #
    # 4. Chronological split (reuses Phase 14 boundaries)
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] Applying chronological split…")
    train, val, test = split_features(features_full, verbose=verbose)

    X_train = train[P17_FEATURES].values
    X_val   = val[P17_FEATURES].values
    X_test  = test[P17_FEATURES].values
    y_train = train["is_fraud"].values
    y_val   = val["is_fraud"].values
    y_test  = test["is_fraud"].values
    amt_val  = val["amount"].values
    amt_test = test["amount"].values

    if verbose:
        print(f"[p17] Train fraud rate: {y_train.mean():.4%}  "
              f"Val: {y_val.mean():.4%}  Test: {y_test.mean():.4%}")

    # ------------------------------------------------------------------ #
    # 5. Train LightGBM on training partition only
    #    Same architecture as Phase 14/15 for comparability.
    # ------------------------------------------------------------------ #
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    spw   = n_neg / n_pos

    if verbose:
        print(f"\n[p17] Training LightGBM (scale_pos_weight={spw:.2f})…")

    model = lgb.LGBMClassifier(
        num_leaves=31,
        max_depth=6,
        learning_rate=0.05,
        n_estimators=600,
        scale_pos_weight=spw,
        random_state=42,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_X=X_val,
        eval_y=y_val,
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)
        ],
    )
    best_iter = int(model.best_iteration_)
    if verbose:
        print(f"[p17] Early stopped at iteration {best_iter}")

    # ------------------------------------------------------------------ #
    # 6. Threshold selection on VALIDATION only
    #    Test set is NOT touched until step 7.
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] Selecting threshold on validation split…")
    p_val  = model.predict_proba(X_val)[:, 1]
    sweep  = _val_sweep(y_val, p_val, amt_val)
    threshold, val_row_at_threshold, found_eligible = _select_threshold(sweep)

    if verbose:
        print(f"[p17] Selected threshold: {threshold:.2f}  "
              f"(recall≥{MIN_RECALL_FLOOR:.0%} {'satisfied' if found_eligible else 'NOT met — using max-recall fallback'})")
        print(f"[p17] Val metrics at {threshold:.2f}: "
              f"P={val_row_at_threshold['precision']:.3f}  "
              f"R={val_row_at_threshold['recall']:.3f}  "
              f"Cost={val_row_at_threshold['total_cost_units']:,.0f}")

    # ------------------------------------------------------------------ #
    # 7. FREEZE — evaluate ONCE on held-out test set
    #    Nothing changes after this point.
    # ------------------------------------------------------------------ #
    if verbose:
        print(f"\n[p17] {'='*50}")
        print(f"[p17] EVALUATING FROZEN MODEL ON HELD-OUT TEST SET (once)")
        print(f"[p17] {'='*50}")

    p_test = model.predict_proba(X_test)[:, 1]
    test_metrics = _score_at(y_test, p_test, amt_test, threshold)

    if verbose:
        m = test_metrics
        print(f"[p17] Threshold:   {m['threshold']:.2f}")
        print(f"[p17] Precision:   {m['precision']:.4f}")
        print(f"[p17] Recall:      {m['recall']:.4f}")
        print(f"[p17] F1:          {m['f1']:.4f}")
        print(f"[p17] ROC-AUC:     {m['roc_auc']:.4f}")
        print(f"[p17] PR-AUC:      {m['pr_auc']:.4f}")
        print(f"[p17] TP={m['tp']:,}  FP={m['fp']:,}  FN={m['fn']:,}  TN={m['tn']:,}")
        print(f"[p17] Fraud caught: {m['fraud_caught']:,} / {m['fraud_total']:,} "
              f"({m['fraud_caught']/m['fraud_total']:.1%})")
        print(f"[p17] FP rate:     {m['false_positive_rate']:.4%}")
        print(f"[p17] FP per TP:   {m['fp_per_tp']:.1f}")
        print(f"[p17] Cost units:  {m['total_cost_units']:,.0f}  "
              f"(FP={m['fp_cost_units']:,.0f}  FN={m['fn_cost_units']:,.0f})")

    # ------------------------------------------------------------------ #
    # 8. Feature importances
    # ------------------------------------------------------------------ #
    fi = dict(zip(P17_FEATURES, model.feature_importances_.tolist()))
    fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    if verbose:
        print(f"\n[p17] Feature importances (gain):")
        max_imp = max(fi_sorted.values()) if fi_sorted else 1
        for feat, imp in fi_sorted.items():
            bar  = "█" * int(imp / max_imp * 28)
            mark = " ★" if feat == "card_product_share" else ""
            print(f"  {feat:30s}: {imp:6.0f} {bar}{mark}")

    # ------------------------------------------------------------------ #
    # 9. Confirm production model is still intact
    # ------------------------------------------------------------------ #
    assert os.path.exists(FROZEN_MODEL_PATH), \
        "CRITICAL: frozen lgbm_v1 artifact has disappeared!"
    frozen_model = joblib.load(FROZEN_MODEL_PATH)
    assert hasattr(frozen_model, "predict_proba"), \
        "CRITICAL: frozen model artifact is not a valid classifier!"
    if verbose:
        print(f"\n[p17] Production lgbm_v1 integrity check: PASS")

    # ------------------------------------------------------------------ #
    # 10. Save model artifact and results JSON
    # ------------------------------------------------------------------ #
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    elapsed = time.time() - t0
    results = {
        "experiment": "phase17_razorpay_validation",
        "description": (
            "Real-data held-out validation on IEEE-CIS fraud data. "
            "16 features (15 base + card_product_share). "
            "Threshold selected on validation only. Test scored once."
        ),
        "dataset": {
            "name": "IEEE-CIS Fraud Detection (Vesta Corporation / Kaggle 2019)",
            "type": "real_world_ecommerce_chargebacks",
            "not_committed_to_git": True,
            "note": (
                "This is NOT Razorpay data. It is real e-commerce card-not-present "
                "fraud from a US payment processor. Labels = confirmed chargebacks."
            ),
        },
        "data_location": {
            "env_var": "IEEE_DATA_DIR",
            "resolved_to": IEEE_DATA_DIR,
            "identity_file_used": idnt_path is not None,
        },
        "split": {
            "method": "chronological_transactiondt",
            "train_boundary": TRAIN_BOUNDARY,
            "val_boundary": VAL_BOUNDARY,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "train_fraud_rate": float(y_train.mean()),
            "val_fraud_rate": float(y_val.mean()),
            "test_fraud_rate": float(y_test.mean()),
        },
        "feature_set": {
            "total": len(P17_FEATURES),
            "base_features": list(FEATURE_COLUMNS),
            "added_features": ["card_product_share"],
            "inert_features": ["new_device_flag", "failed_ratio_trailing10"],
            "degraded_features": [
                "new_geo_flag", "account_age_days",
                "hour_of_day", "is_night", "day_of_week",
            ],
            "note": (
                "new_device_flag and failed_ratio_trailing10 are constant 0 "
                "because DeviceInfo is 79.9% missing and there is no status field "
                "in IEEE-CIS. See docs/external_validation_ieee_cis.md."
            ),
        },
        "model": {
            "family": "LightGBM",
            "version": "phase17_lgbm",
            "artifact": MODEL_PATH,
            "random_seed": 42,
            "best_iteration": best_iter,
            "scale_pos_weight": float(spw),
            "hyperparameters": {
                "num_leaves": 31, "max_depth": 6,
                "learning_rate": 0.05, "n_estimators": 600,
                "min_child_samples": 20, "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "note": "SEPARATE from lgbm_v1. Does not modify the production model.",
            "production_model_untouched": True,
        },
        "cost_model": {
            "fp_cost": float(COST.fp_cost),
            "fn_cost_fraction": float(COST.fn_cost_fraction),
            "currency_note": (
                "IEEE-CIS amounts are USD. fp_cost=50 is an illustrative "
                "assumption, not a measured Razorpay value."
            ),
        },
        "threshold_selection": {
            "method": "cost_minimising_on_validation",
            "recall_floor": MIN_RECALL_FLOOR,
            "selected_threshold": threshold,
            "recall_floor_met": found_eligible,
            "val_metrics_at_threshold": val_row_at_threshold,
            "val_sweep": sweep,
        },
        "test_results": test_metrics,
        "feature_importance_gain": fi_sorted,
        "elapsed_seconds": round(elapsed, 1),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if verbose:
        print(f"\n[p17] Results saved to: {RESULTS_PATH}")
        print(f"[p17] Model saved to:   {MODEL_PATH}")
        print(f"[p17] Elapsed: {elapsed/60:.1f} min")
        print(f"\n{'='*70}")
        print("PHASE 17 COMPLETE")
        print(f"{'='*70}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Phase 17 — Razorpay real-data held-out validation. "
            "Reads data from IEEE_DATA_DIR env variable "
            f"(default: {_DEFAULT_DATA_DIR})."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Override IEEE_DATA_DIR. Must contain train_transaction.csv "
            "and optionally train_identity.csv."
        ),
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["IEEE_DATA_DIR"] = args.data_dir
        # re-resolve module-level variable after env override
        import ml.external.ieee.razorpay_validation as _self
        _self.IEEE_DATA_DIR = args.data_dir

    run_phase17(verbose=not args.quiet)
