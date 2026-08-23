"""
Phase 5: Threshold selection + cost sensitivity analysis.

RULES FOLLOWED:
- LightGBM model from Phase 4 (ml/models/candidate_lgbm_v1.pkl) is loaded, NOT retrained.
- Feature pipeline, dataset, split are unchanged (same chronological boundaries as before).
- Threshold is chosen using VALIDATION predictions only.
- Test set is scored exactly once, at the single frozen threshold, at the very end.
- Sensitivity analysis on FP cost / FN cost / prevalence is run on VALIDATION data only
  (it's a "what if our assumptions were different" exercise, not a test-set search).
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from ml.features.build_features import FEATURE_COLUMNS
from ml.evaluation.cost_model import CostAssumptions, expected_cost
from ml.training.train_baseline import chronological_split

MIN_ACCEPTABLE_RECALL = 0.80  # operational floor: project stance is we don't want to
                              # let fraud recall fall below this just to save on FP cost


def threshold_table(y_true, y_prob, amounts, thresholds, cost_assumptions):
    rows = []
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = expected_cost(y_true, y_pred, amounts, cost_assumptions)
        n_fraud = int(np.sum(y_true))
        rows.append(dict(
            threshold=round(th, 2), precision=precision, recall=recall, f1=f1,
            fp=int(fp), fn=int(fn), fraud_caught=int(tp), fraud_total=n_fraud,
            fp_cost=cost["fp_total_cost"], fn_cost=cost["fn_total_cost"],
            total_cost=cost["total_expected_cost"],
        ))
    return pd.DataFrame(rows)


def main():
    feats = pd.read_csv("ml/data/features.csv")
    feats["timestamp"] = pd.to_datetime(feats["timestamp"])
    train, val, test = chronological_split(feats, train_end_day=40, val_end_day=50)

    X_val, y_val = val[FEATURE_COLUMNS], val["is_fraud"].values
    X_test, y_test = test[FEATURE_COLUMNS], test["is_fraud"].values
    amt_val, amt_test = val["amount"].values, test["amount"].values

    model = joblib.load("ml/models/candidate_lgbm_v1.pkl")
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    base_cost = CostAssumptions()  # FP=Rs50, FN=0.5x amount -- unchanged

    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)
    val_table = threshold_table(y_val, val_prob, amt_val, thresholds, base_cost)

    print("=" * 118)
    print("THRESHOLD SWEEP ON VALIDATION SET (used for selection only)")
    print("=" * 118)
    print(val_table.to_string(index=False, formatters={
        "precision": "{:.3f}".format, "recall": "{:.3f}".format, "f1": "{:.3f}".format,
        "fp_cost": "Rs{:,.0f}".format, "fn_cost": "Rs{:,.0f}".format, "total_cost": "Rs{:,.0f}".format,
    }))

    # ---- Selection rule: minimize total cost subject to recall >= MIN_ACCEPTABLE_RECALL ----
    eligible = val_table[val_table["recall"] >= MIN_ACCEPTABLE_RECALL]
    if len(eligible) == 0:
        print(f"\nWARNING: no threshold in sweep achieves recall >= {MIN_ACCEPTABLE_RECALL}. "
              f"Falling back to the highest-recall threshold available.")
        chosen_row = val_table.loc[val_table["recall"].idxmax()]
    else:
        chosen_row = eligible.loc[eligible["total_cost"].idxmin()]

    chosen_threshold = float(chosen_row["threshold"])

    # for comparison: what would pure-F1-maximization have picked?
    f1_argmax_row = val_table.loc[val_table["f1"].idxmax()]
    default_row = val_table.iloc[(val_table["threshold"] - 0.50).abs().argsort().iloc[0]]

    print("\n" + "=" * 118)
    print(f"SELECTION RULE: minimize total expected cost subject to recall >= {MIN_ACCEPTABLE_RECALL:.0%}")
    print("=" * 118)
    print(f"Chosen threshold:            {chosen_threshold:.2f}  "
          f"(precision={chosen_row['precision']:.3f}, recall={chosen_row['recall']:.3f}, "
          f"cost=Rs{chosen_row['total_cost']:,.0f})")
    print(f"Pure F1-maximizing threshold would be: {f1_argmax_row['threshold']:.2f}  "
          f"(precision={f1_argmax_row['precision']:.3f}, recall={f1_argmax_row['recall']:.3f}, "
          f"cost=Rs{f1_argmax_row['total_cost']:,.0f})  <- NOT selected, see rationale below")
    print(f"Default 0.50 threshold for reference:  "
          f"(precision={default_row['precision']:.3f}, recall={default_row['recall']:.3f}, "
          f"cost=Rs{default_row['total_cost']:,.0f})")

    cost_saved_vs_default = default_row["total_cost"] - chosen_row["total_cost"]
    print(f"\nEstimated validation-set cost reduction vs default 0.50 threshold: "
          f"Rs{cost_saved_vs_default:,.0f} "
          f"({cost_saved_vs_default / default_row['total_cost']:.1%} lower)")

    # ---- FREEZE threshold, evaluate ONCE on test ----
    y_test_pred = (test_prob >= chosen_threshold).astype(int)
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred, labels=[0, 1]).ravel()
    test_cost = expected_cost(y_test, y_test_pred, amt_test, base_cost)

    print("\n" + "=" * 118)
    print(f"FROZEN THRESHOLD ({chosen_threshold:.2f}) EVALUATED ONCE ON HELD-OUT TEST SET")
    print("=" * 118)
    print(f"Precision: {test_precision:.3f}  Recall: {test_recall:.3f}  F1: {test_f1:.3f}")
    print(f"FP: {fp}  FN: {fn}  Fraud caught: {tp}/{tp+fn}")
    print(f"FP cost: Rs{test_cost['fp_total_cost']:,.0f}  FN cost: Rs{test_cost['fn_total_cost']:,.0f}  "
          f"Total: Rs{test_cost['total_expected_cost']:,.0f}")

    # for reference only: what default 0.5 gives on test (NOT used to pick anything)
    y_test_pred_default = (test_prob >= 0.50).astype(int)
    default_test_cost = expected_cost(y_test, y_test_pred_default, amt_test, base_cost)
    print(f"\n[Reference only] Default 0.50 threshold on test set: "
          f"cost=Rs{default_test_cost['total_expected_cost']:,.0f}  "
          f"(chosen threshold {chosen_threshold:.2f} is "
          f"Rs{default_test_cost['total_expected_cost'] - test_cost['total_expected_cost']:,.0f} cheaper on test)")

    # ---------------------------------------------------------------
    # Sensitivity analysis (validation data only)
    # ---------------------------------------------------------------
    print("\n" + "=" * 118)
    print("SENSITIVITY ANALYSIS #1: FP cost varies (FN cost fraction fixed at 0.5)")
    print("=" * 118)
    fp_cost_variants = [10, 50, 100, 250]
    sens_fp_results = []
    for fpc in fp_cost_variants:
        ca = CostAssumptions(fp_cost=fpc, fn_cost_fraction=0.5)
        tbl = threshold_table(y_val, val_prob, amt_val, thresholds, ca)
        elig = tbl[tbl["recall"] >= MIN_ACCEPTABLE_RECALL]
        best = (elig.loc[elig["total_cost"].idxmin()] if len(elig) else tbl.loc[tbl["recall"].idxmax()])
        sens_fp_results.append((fpc, best["threshold"], best["total_cost"]))
        print(f"  FP_COST=Rs{fpc:>4}  -> optimal threshold={best['threshold']:.2f}  "
              f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, cost=Rs{best['total_cost']:,.0f})")

    print("\n" + "=" * 118)
    print("SENSITIVITY ANALYSIS #2: FN cost fraction varies (FP cost fixed at Rs50)")
    print("=" * 118)
    fn_fraction_variants = [0.25, 0.50, 0.75, 1.00]
    sens_fn_results = []
    for fnf in fn_fraction_variants:
        ca = CostAssumptions(fp_cost=50, fn_cost_fraction=fnf)
        tbl = threshold_table(y_val, val_prob, amt_val, thresholds, ca)
        elig = tbl[tbl["recall"] >= MIN_ACCEPTABLE_RECALL]
        best = (elig.loc[elig["total_cost"].idxmin()] if len(elig) else tbl.loc[tbl["recall"].idxmax()])
        sens_fn_results.append((fnf, best["threshold"], best["total_cost"]))
        print(f"  FN_COST_FRACTION={fnf:.2f}  -> optimal threshold={best['threshold']:.2f}  "
              f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, cost=Rs{best['total_cost']:,.0f})")

    print("\n" + "=" * 118)
    print("SENSITIVITY ANALYSIS #3: fraud prevalence varies (via stratified resampling of validation set)")
    print("=" * 118)
    rng = np.random.default_rng(7)
    pos_idx = np.where(y_val == 1)[0]
    neg_idx = np.where(y_val == 0)[0]
    actual_prevalence = len(pos_idx) / len(y_val)
    prevalence_variants = [0.005, 0.01, actual_prevalence, 0.03, 0.05]
    sens_prev_results = []
    for target_prev in prevalence_variants:
        # hold positives fixed, resample negatives (with replacement if needed) to hit target prevalence
        n_pos = len(pos_idx)
        n_neg_needed = int(round(n_pos * (1 - target_prev) / target_prev))
        replace = n_neg_needed > len(neg_idx)
        sampled_neg = rng.choice(neg_idx, size=n_neg_needed, replace=replace)
        combined_idx = np.concatenate([pos_idx, sampled_neg])
        y_sub = y_val[combined_idx]
        prob_sub = val_prob[combined_idx]
        amt_sub = amt_val[combined_idx]

        tbl = threshold_table(y_sub, prob_sub, amt_sub, thresholds, base_cost)
        elig = tbl[tbl["recall"] >= MIN_ACCEPTABLE_RECALL]
        best = (elig.loc[elig["total_cost"].idxmin()] if len(elig) else tbl.loc[tbl["recall"].idxmax()])
        sens_prev_results.append((target_prev, best["threshold"], best["total_cost"], best["precision"]))
        tag = " (actual)" if abs(target_prev - actual_prevalence) < 1e-6 else ""
        print(f"  prevalence={target_prev:.3%}{tag}  -> optimal threshold={best['threshold']:.2f}  "
              f"(precision={best['precision']:.3f}, recall={best['recall']:.3f}, cost=Rs{best['total_cost']:,.0f})")

    # ---------------------------------------------------------------
    # Save chart data + model metadata
    # ---------------------------------------------------------------
    val_table.to_csv("ml/models/threshold_sweep_validation.csv", index=False)

    metadata = {
        "model_name": "lgbm_v1",  # must match ml.evaluation.decision_engine.MODEL_VERSION
        "model_file": "ml/models/candidate_lgbm_v1.pkl",
        "selected_threshold": chosen_threshold,
        "selection_rule": f"minimize total expected cost on validation subject to recall >= {MIN_ACCEPTABLE_RECALL:.0%}",
        "selection_data": "validation set only; test set untouched during selection",
        "cost_assumptions": {"fp_cost": base_cost.fp_cost, "fn_cost_fraction": base_cost.fn_cost_fraction},
        "validation_metrics_at_threshold": {
            "precision": float(chosen_row["precision"]), "recall": float(chosen_row["recall"]),
            "f1": float(chosen_row["f1"]), "fp": int(chosen_row["fp"]), "fn": int(chosen_row["fn"]),
            "total_cost": float(chosen_row["total_cost"]),
        },
        "test_metrics_at_frozen_threshold": {
            "precision": float(test_precision), "recall": float(test_recall), "f1": float(test_f1),
            "fp": int(fp), "fn": int(fn), "fraud_caught": int(tp), "fraud_total": int(tp + fn),
            "total_cost": float(test_cost["total_expected_cost"]),
        },
        "why_not_default_0_5": (
            f"Default 0.50 gives validation cost Rs{default_row['total_cost']:,.0f} vs "
            f"Rs{chosen_row['total_cost']:,.0f} at the chosen threshold "
            f"({cost_saved_vs_default/default_row['total_cost']:.1%} lower). "
            "0.50 is an arbitrary classification convention with no connection to the "
            "project's actual cost function."
        ),
        "why_not_f1_argmax": (
            f"Pure F1-maximizing threshold ({f1_argmax_row['threshold']:.2f}) gives F1="
            f"{f1_argmax_row['f1']:.3f} vs the chosen threshold's F1={chosen_row['f1']:.3f}, "
            "but F1 weights precision and recall equally by construction. Our cost function is "
            "asymmetric (FN cost scales with transaction amount, typically far larger than the "
            "flat FP cost), so the cost-minimizing threshold is a distinct, deliberate choice, "
            "not a proxy for F1."
        ),
        "sensitivity_fp_cost": [{"fp_cost": f, "optimal_threshold": t, "cost": c} for f, t, c in sens_fp_results],
        "sensitivity_fn_cost_fraction": [{"fn_cost_fraction": f, "optimal_threshold": t, "cost": c} for f, t, c in sens_fn_results],
        "sensitivity_prevalence": [{"prevalence": p, "optimal_threshold": t, "cost": c, "precision": pr} for p, t, c, pr in sens_prev_results],
    }
    with open("ml/models/lgbm_v1_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\nSaved threshold sweep to ml/models/threshold_sweep_validation.csv")
    print("Saved model metadata (incl. frozen threshold + rationale) to ml/models/lgbm_v1_metadata.json")

    return val_table, metadata


if __name__ == "__main__":
    main()
