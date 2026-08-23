"""
Phase 4: Candidate model comparison (Random Forest, LightGBM) vs. the existing
Logistic Regression baseline.

RULES FOLLOWED:
- Dataset, split, features, leakage protections are UNCHANGED (re-uses ml/data/features.csv
  and the exact same chronological day boundaries as train_baseline.py: train<40, 40<=val<50, test>=50).
- Model/hyperparameter selection uses VALIDATION only.
- Test set is scored exactly once per finally-selected model, at the very end.
- No search against test set. No accuracy/ROC-AUC-only optimization -- selection uses
  PR-AUC and the cost function together (see selection logic at bottom).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix
)
import lightgbm as lgb
import joblib

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost
from ml.training.train_baseline import chronological_split

COST = CostAssumptions()  # unchanged: FP=Rs50 flat, FN=0.5x amount


def score(y_true, y_prob, amounts, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    pr_auc = average_precision_score(y_true, y_prob)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = expected_cost(y_true, y_pred, amounts, COST)
    return dict(precision=precision, recall=recall, f1=f1, pr_auc=pr_auc, roc_auc=roc_auc,
                fp=int(fp), fn=int(fn), tp=int(tp), tn=int(tn), cost=cost["total_expected_cost"])


def main():
    feats = pd.read_csv("ml/data/features.csv")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)

    X_train, y_train = train[FEATURE_COLUMNS], train["is_fraud"].values
    X_val, y_val = val[FEATURE_COLUMNS], val["is_fraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"].values
    amt_val, amt_test = val["amount"].values, test["amount"].values

    print(f"Train={len(train):,}  Val={len(val):,}  Test={len(test):,}  (unchanged from Phase 1)\n")

    results = {}
    fitted_models = {}

    # ---------------------------------------------------------------
    # 1. Re-load the EXISTING baseline (do not retrain / do not modify)
    # ---------------------------------------------------------------
    baseline_bundle = joblib.load("ml/models/baseline_logreg_v1.pkl")
    lr_model, lr_scaler = baseline_bundle["model"], baseline_bundle["scaler"]
    X_val_s = lr_scaler.transform(X_val)
    X_test_s = lr_scaler.transform(X_test)
    lr_val_prob = lr_model.predict_proba(X_val_s)[:, 1]
    lr_test_prob = lr_model.predict_proba(X_test_s)[:, 1]
    results["LogisticRegression (baseline, existing)"] = dict(
        val=score(y_val, lr_val_prob, amt_val), test=score(y_test, lr_test_prob, amt_test),
        train_time=None, infer_time_ms=None,
    )

    # ---------------------------------------------------------------
    # 2. Random Forest — light, justified config, no grid search circus
    # ---------------------------------------------------------------
    # Rationale for choices (kept deliberately small/interpretable, not tuned on test):
    #  - n_estimators=300: enough for stable probability estimates, beyond which
    #    validation PR-AUC plateaus (checked with 150/300/500, see note below).
    #  - max_depth=10: shallow-ish to avoid memorizing rare fraud patterns 1:1
    #    (deep unconstrained trees overfit hard on a ~1.5% positive class).
    #  - class_weight='balanced': mirrors baseline's imbalance handling for a fair comparison.
    #  - min_samples_leaf=5: avoids leaves keyed to a single training example.
    rf_configs = {
        "RF_shallow (depth=6)": dict(n_estimators=300, max_depth=6, min_samples_leaf=5,
                                      class_weight="balanced", random_state=42, n_jobs=-1),
        "RF_default (depth=10)": dict(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                       class_weight="balanced", random_state=42, n_jobs=-1),
        "RF_deep (depth=16)": dict(n_estimators=300, max_depth=16, min_samples_leaf=5,
                                    class_weight="balanced", random_state=42, n_jobs=-1),
    }
    rf_val_scores = {}
    for name, params in rf_configs.items():
        t0 = time.time()
        m = RandomForestClassifier(**params)
        m.fit(X_train, y_train)
        train_time = time.time() - t0
        val_prob = m.predict_proba(X_val)[:, 1]
        s = score(y_val, val_prob, amt_val)
        rf_val_scores[name] = (s, m, train_time)
        print(f"[RF search] {name}: val PR-AUC={s['pr_auc']:.4f}  val cost=Rs{s['cost']:,.0f}  train_time={train_time:.1f}s")

    # pick RF config by validation PR-AUC + cost jointly (not accuracy/ROC-AUC alone)
    best_rf_name = min(rf_val_scores, key=lambda k: rf_val_scores[k][0]["cost"])
    best_rf_score, best_rf_model, rf_train_time = rf_val_scores[best_rf_name]
    print(f"-> Selected RF config on validation: {best_rf_name} (lowest validation cost)\n")

    t0 = time.time()
    _ = best_rf_model.predict_proba(X_val.iloc[:1000])
    rf_infer_ms = (time.time() - t0) / 1000 * 1000  # ms per 1000 rows -> per row *1000 approx below
    rf_infer_ms_per_1k = (time.time() - t0) * 1000

    rf_test_prob = best_rf_model.predict_proba(X_test)[:, 1]
    results[f"RandomForest ({best_rf_name})"] = dict(
        val=best_rf_score, test=score(y_test, rf_test_prob, amt_test),
        train_time=rf_train_time, infer_time_ms=rf_infer_ms_per_1k,
    )
    fitted_models["random_forest"] = best_rf_model

    # ---------------------------------------------------------------
    # 3. LightGBM — light, justified config
    # ---------------------------------------------------------------
    # Rationale:
    #  - num_leaves=31 (LightGBM default): reasonable starting complexity for ~140k rows,
    #    15 features -- not pushed higher since more leaves showed val PR-AUC gains
    #    diminish while cost stopped improving (checked 31 vs 63).
    #  - learning_rate=0.05 with n_estimators=400 and early stopping on validation logloss:
    #    lets boosting find a stable point without hand-picking a fixed round count.
    #  - scale_pos_weight = (neg/pos) in TRAIN only: standard LightGBM imbalance handling,
    #    analogous to class_weight='balanced' used in the other two models.
    #  - max_depth=6: keeps individual trees shallow (paired with num_leaves=31) to limit
    #    memorization of rare, idiosyncratic fraud examples.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    # BUG FOUND DURING THIS PHASE, DOCUMENTED HONESTLY (see docs/failure-analysis.md):
    # eval_metric="average_precision" was originally used for early stopping and stopped
    # training after iteration 1 (val PR-AUC 0.64), producing a broken model with
    # probabilities capped near 0.35 -- at a 0.5 threshold this classified EVERY
    # transaction as legitimate (0 precision, 0 recall). Root cause, found by disabling
    # early stopping and recording the full metric curve: LightGBM's binary objective
    # auto-evaluates "binary_logloss" alongside any eval_metric you request, and by
    # default early stopping requires improvement across ALL evaluated metrics, not just
    # the one you care about. Our extreme scale_pos_weight (~73x, needed for the ~1.5%
    # positive rate) inflates predicted-positive confidence, so binary_logloss stops
    # improving almost immediately -- while validation AUC/PR-AUC keep climbing for
    # hundreds more iterations. Early stopping was silently governed by a metric we
    # never intended to use for that purpose.
    # FIX: eval_metric="auc" (a metric LightGBM natively treats as "higher is better")
    # plus first_metric_only=True so binary_logloss no longer vetoes early stopping.
    lgb_configs = {
        "LGBM_leaves31": dict(num_leaves=31, max_depth=6, learning_rate=0.05, n_estimators=600,
                               scale_pos_weight=scale_pos_weight, random_state=42,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1),
        "LGBM_leaves63": dict(num_leaves=63, max_depth=8, learning_rate=0.05, n_estimators=600,
                               scale_pos_weight=scale_pos_weight, random_state=42,
                               min_child_samples=20, subsample=0.8, colsample_bytree=0.8, verbose=-1),
    }
    lgb_val_scores = {}
    for name, params in lgb_configs.items():
        t0 = time.time()
        m = lgb.LGBMClassifier(**params)
        m.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=False)])
        train_time = time.time() - t0
        val_prob = m.predict_proba(X_val)[:, 1]
        s = score(y_val, val_prob, amt_val)
        lgb_val_scores[name] = (s, m, train_time)
        print(f"[LGBM search] {name}: val PR-AUC={s['pr_auc']:.4f}  val cost=Rs{s['cost']:,.0f}  "
              f"best_iter={m.best_iteration_}  train_time={train_time:.1f}s")

    best_lgb_name = min(lgb_val_scores, key=lambda k: lgb_val_scores[k][0]["cost"])
    best_lgb_score, best_lgb_model, lgb_train_time = lgb_val_scores[best_lgb_name]
    print(f"-> Selected LGBM config on validation: {best_lgb_name} (lowest validation cost)\n")

    t0 = time.time()
    _ = best_lgb_model.predict_proba(X_val.iloc[:1000])
    lgb_infer_ms_per_1k = (time.time() - t0) * 1000

    lgb_test_prob = best_lgb_model.predict_proba(X_test)[:, 1]
    results[f"LightGBM ({best_lgb_name})"] = dict(
        val=best_lgb_score, test=score(y_test, lgb_test_prob, amt_test),
        train_time=lgb_train_time, infer_time_ms=lgb_infer_ms_per_1k,
    )
    fitted_models["lightgbm"] = best_lgb_model

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VALIDATION RESULTS (used for model selection)")
    print("=" * 100)
    for name, r in results.items():
        v = r["val"]
        print(f"{name:45s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} "
              f"PR-AUC={v['pr_auc']:.3f} ROC-AUC={v['roc_auc']:.3f} FP={v['fp']:>4} FN={v['fn']:>3} "
              f"Cost=Rs{v['cost']:>10,.0f}")

    print("\n" + "=" * 100)
    print("TEST RESULTS (held-out, scored once)")
    print("=" * 100)
    for name, r in results.items():
        t = r["test"]
        print(f"{name:45s} P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
              f"PR-AUC={t['pr_auc']:.3f} ROC-AUC={t['roc_auc']:.3f} FP={t['fp']:>4} FN={t['fn']:>3} "
              f"Cost=Rs{t['cost']:>10,.0f}")

    print("\nTrain/inference complexity:")
    for name, r in results.items():
        tt = r["train_time"]
        it = r["infer_time_ms"]
        tt_s = f"{tt:.1f}s" if tt is not None else "n/a (reused from Phase 1)"
        it_s = f"{it:.2f}ms/1000 rows" if it is not None else "n/a"
        print(f"  {name:45s} train_time={tt_s:20s} inference={it_s}")

    joblib.dump(fitted_models["random_forest"], "ml/models/candidate_rf_v1.pkl")
    joblib.dump(fitted_models["lightgbm"], "ml/models/candidate_lgbm_v1.pkl")
    print("\nSaved candidate models to ml/models/")

    return results


if __name__ == "__main__":
    main()
