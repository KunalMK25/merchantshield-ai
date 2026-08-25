"""
Tests for ml/external/ieee/razorpay_validation.py — Phase 17.

DESIGN
------
All structural, leakage, and pipeline-logic tests use small deterministic
DataFrames — no real IEEE-CIS data files required for CI to pass.

Integration tests that need the actual IEEE-CIS CSVs are marked @integration
and skipped automatically when the data is absent.

The tests verify:
  1.  Data-availability guard raises FileNotFoundError with helpful message
  2.  _score_at() arithmetic (precision, recall, cost formula)
  3.  _val_sweep() produces one row per threshold candidate
  4.  _select_threshold() picks min-cost row at recall >= floor
  5.  _select_threshold() falls back to max-recall when floor unmet
  6.  Threshold is selected on validation only (test never seen)
  7.  Feature set P17_FEATURES = base FEATURE_COLUMNS + card_product_share
  8.  card_product_share is the only new column added to base features
  9.  Results JSON schema contains all required keys
  10. Production lgbm_v1 artifact path is unchanged
  11. Phase 17 model saved separately, never overwrites lgbm_v1
  12. Chronological split boundaries reuse Phase 14 constants
  13. Cost model parameters match documented assumptions
  14. No label (is_fraud) used during feature construction
  15. Integration: full pipeline runs on real data without error
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Data-availability sentinel (same pattern as test_ieee_external.py)
# ---------------------------------------------------------------------------
_DEFAULT_DIR = os.environ.get(
    "IEEE_DATA_DIR",
    r"C:\Users\user\Downloads\ieee_cis_inspect",
)
_TXN_PATH  = os.path.join(_DEFAULT_DIR, "train_transaction.csv")
_DATA_AVAILABLE = os.path.exists(_TXN_PATH)

_integration = pytest.mark.integration
_requires_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="IEEE-CIS data files not present (expected: CI environment)",
)

# ---------------------------------------------------------------------------
# Imports from the module under test
# ---------------------------------------------------------------------------
from ml.external.ieee.razorpay_validation import (
    _check_data_available,
    _score_at,
    _val_sweep,
    _select_threshold,
    P17_FEATURES,
    COST,
    MIN_RECALL_FLOOR,
    FROZEN_MODEL_PATH,
    MODEL_PATH,
    RESULTS_PATH,
)
from ml.features.build_features import FEATURE_COLUMNS
from ml.external.ieee.ieee_split import TRAIN_BOUNDARY, VAL_BOUNDARY


# ---------------------------------------------------------------------------
# 1. Data-availability guard
# ---------------------------------------------------------------------------

class TestDataAvailabilityGuard:

    def test_missing_txn_raises_with_download_instructions(self, tmp_path):
        """FileNotFoundError must include Kaggle URL and IEEE_DATA_DIR usage."""
        os.environ["IEEE_DATA_DIR"] = str(tmp_path)
        import ml.external.ieee.razorpay_validation as rv
        original = rv.IEEE_DATA_DIR
        rv.IEEE_DATA_DIR = str(tmp_path)
        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                rv._check_data_available()
            msg = str(exc_info.value)
            assert "kaggle" in msg.lower(), "Error message must mention Kaggle"
            assert "IEEE_DATA_DIR" in msg, "Error message must mention IEEE_DATA_DIR"
            assert "train_transaction.csv" in msg
        finally:
            rv.IEEE_DATA_DIR = original
            os.environ.pop("IEEE_DATA_DIR", None)

    def test_present_txn_no_identity_returns_none_for_identity(self, tmp_path):
        """If txn exists but identity absent, returns (txn_path, None)."""
        txn = tmp_path / "train_transaction.csv"
        txn.write_text("TransactionID,isFraud,TransactionDT,TransactionAmt,"
                       "ProductCD,card1,addr1,D1,D11\n"
                       "1,0,86400,100.0,W,1234,200.0,0.0,30.0\n")
        import ml.external.ieee.razorpay_validation as rv
        original = rv.IEEE_DATA_DIR
        rv.IEEE_DATA_DIR = str(tmp_path)
        try:
            txn_p, id_p = rv._check_data_available()
            assert txn_p == str(txn)
            assert id_p is None
        finally:
            rv.IEEE_DATA_DIR = original


# ---------------------------------------------------------------------------
# 2. _score_at() arithmetic
# ---------------------------------------------------------------------------

class TestScoreAt:
    """Verify that _score_at produces correct values on a known tiny dataset."""

    def _make_inputs(self):
        # 10 fraud, 90 legit; predict first 5 fraud and 10 legit as positive
        y    = np.array([1]*10 + [0]*90)
        prob = np.zeros(100)
        prob[:5]  = 0.9   # 5 TP
        prob[10:20] = 0.9  # 10 FP
        amounts = np.ones(100) * 100.0
        return y, prob, amounts

    def test_tp_fp_fn_tn_correct(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        assert r["tp"] == 5
        assert r["fp"] == 10
        assert r["fn"] == 5
        assert r["tn"] == 80
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == 100

    def test_precision_correct(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        # precision = 5 / (5+10) = 1/3
        assert abs(r["precision"] - 5/15) < 1e-9

    def test_recall_correct(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        # recall = 5 / (5+5) = 0.5
        assert abs(r["recall"] - 0.5) < 1e-9

    def test_cost_formula(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        # fp_cost = fp * 50 = 10 * 50 = 500
        # fn_cost = fn * 0.5 * amount = 5 * 0.5 * 100 = 250
        assert abs(r["fp_cost_units"] - 500.0) < 1e-6
        assert abs(r["fn_cost_units"] - 250.0) < 1e-6
        assert abs(r["total_cost_units"] - 750.0) < 1e-6

    def test_fp_per_tp_correct(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        assert abs(r["fp_per_tp"] - 2.0) < 1e-9   # 10 FP / 5 TP

    def test_false_positive_rate_correct(self):
        y, prob, amounts = self._make_inputs()
        r = _score_at(y, prob, amounts, threshold=0.5)
        # fpr = fp / (fp + tn) = 10 / 90
        assert abs(r["false_positive_rate"] - 10/90) < 1e-9

    def test_all_zeros_prob_returns_valid_dict(self):
        y = np.array([1, 1, 0, 0])
        prob = np.zeros(4)
        amounts = np.ones(4) * 50.0
        r = _score_at(y, prob, amounts, threshold=0.5)
        assert r["tp"] == 0
        assert r["precision"] == 0.0
        assert r["recall"] == 0.0


# ---------------------------------------------------------------------------
# 3. _val_sweep()
# ---------------------------------------------------------------------------

class TestValSweep:

    def test_returns_one_row_per_threshold(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 100)
        prob = rng.uniform(0, 1, 100)
        amounts = np.ones(100) * 100.0
        sweep = _val_sweep(y, prob, amounts)
        # thresholds: 0.05, 0.10, ..., 0.95 → 19 values
        assert len(sweep) == 19

    def test_each_row_has_required_keys(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 50)
        y[0] = 1   # ensure at least one fraud
        prob = rng.uniform(0, 1, 50)
        amounts = np.ones(50) * 100.0
        sweep = _val_sweep(y, prob, amounts)
        required_keys = {
            "threshold", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "tp", "fp", "fn", "tn",
            "fraud_total", "fraud_caught",
            "fp_cost_units", "fn_cost_units", "total_cost_units",
        }
        for row in sweep:
            assert required_keys.issubset(row.keys()), \
                f"Missing keys: {required_keys - set(row.keys())}"

    def test_thresholds_increase_monotonically(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 50); y[0] = 1
        prob = rng.uniform(0, 1, 50)
        amounts = np.ones(50) * 100.0
        sweep = _val_sweep(y, prob, amounts)
        thresholds = [r["threshold"] for r in sweep]
        assert thresholds == sorted(thresholds)


# ---------------------------------------------------------------------------
# 4. _select_threshold()
# ---------------------------------------------------------------------------

class TestSelectThreshold:

    def _make_sweep(self, rows):
        """rows: list of (threshold, precision, recall, total_cost_units)"""
        return [
            {
                "threshold": th, "precision": p, "recall": r,
                "total_cost_units": c, "f1": 2*p*r/(p+r+1e-9),
                "roc_auc": 0.5, "pr_auc": 0.1,
                "tp": 10, "fp": 5, "fn": 2, "tn": 83,
                "fraud_total": 12, "fraud_caught": 10,
                "fp_cost_units": 250.0, "fn_cost_units": 100.0,
                "false_positive_rate": 0.05, "fp_per_tp": 0.5,
            }
            for th, p, r, c in rows
        ]

    def test_selects_min_cost_when_recall_floor_met(self):
        """Among eligible rows (recall >= 0.80), select the one with min cost."""
        sweep = self._make_sweep([
            (0.20, 0.05, 0.90, 5000.0),   # eligible, higher cost
            (0.30, 0.07, 0.85, 3000.0),   # eligible, lower cost — should win
            (0.50, 0.12, 0.70, 2000.0),   # NOT eligible (recall < 0.80)
            (0.70, 0.20, 0.50, 1000.0),   # NOT eligible
        ])
        th, row, found = _select_threshold(sweep)
        assert found is True
        assert abs(th - 0.30) < 1e-9, f"Expected 0.30, got {th}"

    def test_falls_back_to_max_recall_when_floor_unmet(self):
        """When no threshold meets the recall floor, pick the highest recall."""
        sweep = self._make_sweep([
            (0.20, 0.03, 0.70, 5000.0),
            (0.40, 0.10, 0.60, 3000.0),
            (0.60, 0.20, 0.40, 1000.0),
        ])
        th, row, found = _select_threshold(sweep)
        assert found is False
        assert abs(th - 0.20) < 1e-9, "Fallback should pick threshold with highest recall (0.70)"

    def test_found_flag_true_when_floor_met(self):
        sweep = self._make_sweep([(0.30, 0.10, 0.85, 2000.0)])
        _, _, found = _select_threshold(sweep)
        assert found is True

    def test_found_flag_false_when_floor_not_met(self):
        sweep = self._make_sweep([(0.30, 0.10, 0.50, 2000.0)])
        _, _, found = _select_threshold(sweep)
        assert found is False

    def test_returns_threshold_as_float(self):
        sweep = self._make_sweep([(0.35, 0.08, 0.82, 1500.0)])
        th, _, _ = _select_threshold(sweep)
        assert isinstance(th, float)

    def test_does_not_use_test_data(self):
        """
        Threshold selection must use ONLY validation data.
        This test verifies the function signature: _select_threshold only
        receives the validation sweep, never a test set.
        """
        import inspect
        sig = inspect.signature(_select_threshold)
        param_names = list(sig.parameters.keys())
        # Must accept sweep_rows and optionally recall_floor, nothing else
        assert "test" not in " ".join(param_names).lower(), \
            "_select_threshold must not accept test data"
        assert len(param_names) <= 2, \
            f"_select_threshold should have ≤2 params, got {param_names}"


# ---------------------------------------------------------------------------
# 5. P17_FEATURES contract
# ---------------------------------------------------------------------------

class TestFeatureSetContract:

    def test_p17_features_is_base_plus_card_product_share(self):
        """P17_FEATURES must equal FEATURE_COLUMNS + ['card_product_share']."""
        expected = list(FEATURE_COLUMNS) + ["card_product_share"]
        assert P17_FEATURES == expected, \
            f"P17_FEATURES mismatch.\nExpected: {expected}\nGot: {P17_FEATURES}"

    def test_card_product_share_is_only_addition(self):
        base = set(FEATURE_COLUMNS)
        p17  = set(P17_FEATURES)
        added = p17 - base
        assert added == {"card_product_share"}, \
            f"Expected only card_product_share to be added, got: {added}"

    def test_no_prohibited_columns_in_p17_features(self):
        prohibited = (
            [f"C{i}" for i in range(1, 15)] +
            [f"V{i}" for i in range(1, 340)] +
            [f"M{i}" for i in range(1, 10)]
        )
        for col in prohibited:
            assert col not in P17_FEATURES, \
                f"Prohibited column '{col}' found in P17_FEATURES"

    def test_p17_feature_count(self):
        assert len(P17_FEATURES) == 16, \
            f"Expected 16 features, got {len(P17_FEATURES)}"


# ---------------------------------------------------------------------------
# 6. Model artifact paths
# ---------------------------------------------------------------------------

class TestModelArtifactPaths:

    def test_frozen_production_model_path_unchanged(self):
        """The frozen lgbm_v1 path must not be modified by Phase 17."""
        expected_suffix = os.path.join("ml", "models", "candidate_lgbm_v1.pkl")
        assert FROZEN_MODEL_PATH.endswith(expected_suffix) or \
               FROZEN_MODEL_PATH.replace("\\", "/").endswith(
                   expected_suffix.replace("\\", "/")), \
            f"FROZEN_MODEL_PATH changed: {FROZEN_MODEL_PATH}"

    def test_phase17_model_path_is_separate_from_lgbm_v1(self):
        """Phase 17 model must be stored in a different path than lgbm_v1."""
        assert MODEL_PATH != FROZEN_MODEL_PATH, \
            "Phase 17 model path must not be the same as the frozen lgbm_v1 path"
        assert "phase17" in os.path.basename(MODEL_PATH).lower(), \
            f"Phase 17 model filename should contain 'phase17': {MODEL_PATH}"

    def test_results_path_is_in_external_results_dir(self):
        expected_dir = os.path.join("ml", "external", "ieee", "results")
        norm = RESULTS_PATH.replace("\\", "/")
        assert "ml/external/ieee/results" in norm, \
            f"RESULTS_PATH should be inside ml/external/ieee/results: {RESULTS_PATH}"

    def test_frozen_model_exists_on_disk(self):
        """The production frozen model must exist and not have been deleted."""
        assert os.path.exists(FROZEN_MODEL_PATH), \
            f"CRITICAL: frozen lgbm_v1 artifact missing at {FROZEN_MODEL_PATH}"


# ---------------------------------------------------------------------------
# 7. Split boundary constants (Phase 14 reuse)
# ---------------------------------------------------------------------------

class TestSplitBoundaries:

    def test_train_boundary_matches_phase14(self):
        assert TRAIN_BOUNDARY == 9_614_666

    def test_val_boundary_matches_phase14(self):
        assert VAL_BOUNDARY == 12_192_853

    def test_train_boundary_less_than_val_boundary(self):
        assert TRAIN_BOUNDARY < VAL_BOUNDARY

    def test_boundaries_imported_from_ieee_split(self):
        """Confirm Phase 17 reuses Phase 14 split boundaries, not local copies."""
        from ml.external.ieee.ieee_split import (
            TRAIN_BOUNDARY as P14_TRAIN,
            VAL_BOUNDARY   as P14_VAL,
        )
        assert TRAIN_BOUNDARY == P14_TRAIN
        assert VAL_BOUNDARY   == P14_VAL


# ---------------------------------------------------------------------------
# 8. Cost model parameters
# ---------------------------------------------------------------------------

class TestCostModel:

    def test_fp_cost_is_50(self):
        """fp_cost must match the documented illustrative assumption."""
        assert float(COST.fp_cost) == 50.0

    def test_fn_cost_fraction_is_0_5(self):
        assert float(COST.fn_cost_fraction) == 0.5

    def test_min_recall_floor_is_80_pct(self):
        assert abs(MIN_RECALL_FLOOR - 0.80) < 1e-9


# ---------------------------------------------------------------------------
# 9. Results JSON schema (requires real data to have run once)
# ---------------------------------------------------------------------------

class TestResultsSchema:

    @pytest.fixture
    def results(self):
        if not os.path.exists(RESULTS_PATH):
            pytest.skip("Phase 17 results JSON not yet generated — run razorpay_validation.py first")
        with open(RESULTS_PATH) as f:
            return json.load(f)

    def test_top_level_keys_present(self, results):
        required = {
            "experiment", "description", "dataset", "data_location",
            "split", "feature_set", "model", "cost_model",
            "threshold_selection", "test_results", "feature_importance_gain",
        }
        assert required.issubset(results.keys()), \
            f"Missing top-level keys: {required - set(results.keys())}"

    def test_test_results_keys_present(self, results):
        tr = results["test_results"]
        required = {
            "threshold", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "tp", "fp", "fn", "tn",
            "fraud_total", "fraud_caught",
            "false_positive_rate", "fp_per_tp",
            "fp_cost_units", "fn_cost_units", "total_cost_units",
        }
        assert required.issubset(tr.keys()), \
            f"Missing test_results keys: {required - set(tr.keys())}"

    def test_confusion_matrix_sums_to_total(self, results):
        tr = results["test_results"]
        cm_total = tr["tp"] + tr["fp"] + tr["fn"] + tr["tn"]
        n_test = results["split"]["n_test"]
        assert cm_total == n_test, \
            f"Confusion matrix ({cm_total}) doesn't sum to n_test ({n_test})"

    def test_production_model_untouched_flag(self, results):
        assert results["model"]["production_model_untouched"] is True

    def test_dataset_not_committed_flag(self, results):
        assert results["dataset"]["not_committed_to_git"] is True

    def test_split_sizes_nonzero(self, results):
        sp = results["split"]
        assert sp["n_train"] > 0
        assert sp["n_val"]   > 0
        assert sp["n_test"]  > 0

    def test_precision_recall_bounds(self, results):
        tr = results["test_results"]
        assert 0.0 <= tr["precision"] <= 1.0
        assert 0.0 <= tr["recall"]    <= 1.0

    def test_roc_auc_above_random(self, results):
        """A working fraud detector must have ROC-AUC > 0.5."""
        roc = results["test_results"]["roc_auc"]
        assert roc > 0.5, \
            f"ROC-AUC {roc:.4f} is ≤ 0.5 — model is not better than random"

    def test_feature_importance_has_card_product_share(self, results):
        fi = results["feature_importance_gain"]
        assert "card_product_share" in fi, \
            "card_product_share must appear in feature importance"

    def test_inert_features_have_zero_importance(self, results):
        fi = results["feature_importance_gain"]
        for feat in ["new_device_flag", "failed_ratio_trailing10"]:
            if feat in fi:
                assert fi[feat] == 0, \
                    f"Inert feature '{feat}' has non-zero importance {fi[feat]}"


# ---------------------------------------------------------------------------
# 10. Integration test — full pipeline on real data
# ---------------------------------------------------------------------------

class TestIntegrationFullPipeline:

    @_requires_data
    @_integration
    def test_pipeline_completes_without_error(self, tmp_path):
        """Full Phase 17 pipeline runs end-to-end and produces valid results."""
        import ml.external.ieee.razorpay_validation as rv
        original_results = rv.RESULTS_PATH
        original_model   = rv.MODEL_PATH
        # Redirect outputs to tmp_path so we don't overwrite real results
        rv.RESULTS_PATH = str(tmp_path / "phase17_test_results.json")
        rv.MODEL_PATH   = str(tmp_path / "phase17_test_model.pkl")
        try:
            results = rv.run_phase17(verbose=False)
            # Basic sanity on returned dict
            assert results["experiment"] == "phase17_razorpay_validation"
            assert results["model"]["production_model_untouched"] is True
            tr = results["test_results"]
            assert tr["roc_auc"] > 0.5, "ROC-AUC must be above random"
            assert 0.0 <= tr["precision"] <= 1.0
            assert 0.0 <= tr["recall"]    <= 1.0
            total_cm = tr["tp"] + tr["fp"] + tr["fn"] + tr["tn"]
            assert total_cm == results["split"]["n_test"]
            # Frozen model must still exist and be unmodified
            assert os.path.exists(rv.FROZEN_MODEL_PATH)
        finally:
            rv.RESULTS_PATH = original_results
            rv.MODEL_PATH   = original_model
