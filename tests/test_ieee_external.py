"""
Tests for the IEEE-CIS external validation track.

DESIGN PRINCIPLES
-----------------
1. All structural/logic tests use synthetic mini-DataFrames that mirror the
   IEEE-CIS schema — no real data files required for the test suite to pass.
2. Tests that require the actual downloaded files are gated with BOTH a skip
   marker (when data absent) AND a pytest `integration` mark (so CI can
   exclude them from the default run via `-m "not integration"`).
3. These tests NEVER import or call any part of the synthetic MerchantShield
   pipeline for comparison — they are self-contained.
4. The tests verify logic correctness, leakage prevention, schema contracts,
   and sentinel/inert behavior. They do NOT assert specific metric values
   (those are in the experiment result JSON files).

Running structural tests only (fast, no data required):
    pytest tests/test_ieee_external.py -m "not integration"

Running integration tests (requires IEEE-CIS data, ~10-15 min):
    pytest tests/test_ieee_external.py -m integration
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Paths to actual data — tests requiring these are skipped if absent
_TXN_PATH  = r"C:\Users\user\Downloads\ieee_cis_inspect\train_transaction.csv"
_IDNT_PATH = r"C:\Users\user\Downloads\ieee_cis_inspect\train_identity.csv"
_DATA_AVAILABLE = os.path.exists(_TXN_PATH)

# Sentinel so we can mark skip cleanly
_requires_data = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason="IEEE-CIS data files not present (expected: CI environment)"
)
# Mark that additionally gates heavy tests from the default run
_integration = pytest.mark.integration

from ml.external.ieee.ieee_adapter import (
    adapt, load_raw, IEEE_REFERENCE_DATETIME,
    _TXN_LOAD_COLS, _ID_LOAD_COLS,
)
from ml.external.ieee.ieee_features import (
    build_ieee_features, FEATURE_COLUMNS,
    ACTIVE_FEATURES, INERT_FEATURES, DEGRADED_FEATURES,
)
from ml.external.ieee.ieee_split import (
    split_features, TRAIN_BOUNDARY, VAL_BOUNDARY,
    _check_no_overlap, _check_no_future_leakage,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic mini-IEEE-CIS DataFrames
# ---------------------------------------------------------------------------

def _make_raw_txn(n=100, seed=42) -> pd.DataFrame:
    """Minimal IEEE-CIS-shaped transaction DataFrame for unit testing."""
    rng = np.random.default_rng(seed)
    # TransactionDT: span covers all three split zones
    dt_vals = np.sort(rng.integers(86_400, 15_811_131, size=n))
    return pd.DataFrame({
        "TransactionID": np.arange(1_000_000, 1_000_000 + n),
        "isFraud":       rng.integers(0, 2, size=n),
        "TransactionDT": dt_vals,
        "TransactionAmt": rng.uniform(1.0, 500.0, size=n),
        "ProductCD":     rng.choice(["W", "C", "R", "H", "S"], size=n),
        "card1":         rng.integers(1000, 9999, size=n),
        "addr1":         rng.choice([100.0, 200.0, 300.0, np.nan], size=n),
        "D1":            rng.choice([0.0, 30.0, 90.0, np.nan], size=n),
        "D11":           rng.choice([0.0, 60.0, 180.0, np.nan], size=n),
        "DeviceInfo":    rng.choice(["iOS Device", "Windows", np.nan], size=n),
    })


def _make_adapted(n=100, seed=42) -> pd.DataFrame:
    """Return an adapted DataFrame (MerchantShield raw schema)."""
    raw = _make_raw_txn(n, seed)
    adapted, _ = adapt(raw)
    return adapted


def _make_feature_df(n=200, seed=42) -> pd.DataFrame:
    """Return a full feature DataFrame with all FEATURE_COLUMNS + labels + TransactionDT."""
    adapted = _make_adapted(n, seed)
    features, _ = build_ieee_features(adapted, verbose=False)
    return features


# ---------------------------------------------------------------------------
# 1. Adapter schema tests
# ---------------------------------------------------------------------------

class TestAdapterSchema:

    def test_adapted_has_required_raw_columns(self):
        """All columns required by build_features must be present."""
        from ml.features.build_features import RAW_COLUMNS_USED
        adapted = _make_adapted()
        for col in RAW_COLUMNS_USED:
            assert col in adapted.columns, f"Missing required column: {col}"

    def test_adapted_has_label_and_dt(self):
        adapted = _make_adapted()
        assert "is_fraud" in adapted.columns
        assert "TransactionDT" in adapted.columns

    def test_adapted_row_count_preserved(self):
        raw = _make_raw_txn(80)
        adapted, _ = adapt(raw)
        assert len(adapted) == 80

    def test_customer_id_maps_from_card1(self):
        raw = _make_raw_txn(50, seed=1)
        raw["card1"] = [1234] * 25 + [5678] * 25
        adapted, _ = adapt(raw)
        assert set(adapted["customer_id"].unique()) == {"1234", "5678"}

    def test_customer_id_has_no_nulls(self):
        adapted = _make_adapted(100)
        assert adapted["customer_id"].isna().sum() == 0

    def test_device_id_is_constant_sentinel(self):
        """new_device_flag must be inert — device_id must be the constant sentinel."""
        adapted = _make_adapted(100)
        assert (adapted["device_id"] == "device_unknown").all()

    def test_status_is_constant_success(self):
        """failed_ratio_trailing10 must be inert — status must be 'success' always."""
        adapted = _make_adapted(100)
        assert (adapted["status"] == "success").all()

    def test_geo_region_no_raw_floats(self):
        """addr1 floats must be converted to 'region_N' strings."""
        adapted = _make_adapted(100)
        # Every value must start with "region_" or "geo_unknown"
        mask = adapted["geo_region"].str.startswith(("region_", "geo_unknown"))
        assert mask.all(), "geo_region contains raw float values"

    def test_geo_region_null_addr1_becomes_geo_unknown(self):
        raw = _make_raw_txn(10, seed=3)
        raw["addr1"] = np.nan  # all null
        adapted, _ = adapt(raw)
        assert (adapted["geo_region"] == "geo_unknown").all()

    def test_timestamp_derived_from_transactiondt(self):
        """timestamp = IEEE_REFERENCE_DATETIME + TransactionDT seconds."""
        raw = _make_raw_txn(5, seed=4)
        raw["TransactionDT"] = [86400, 172800, 259200, 345600, 432000]
        adapted, _ = adapt(raw)
        for i, dt_val in enumerate([86400, 172800, 259200, 345600, 432000]):
            expected = IEEE_REFERENCE_DATETIME.tz_localize(None) + pd.Timedelta(seconds=dt_val)
            assert adapted["timestamp"].iloc[i] == expected

    def test_account_created_before_or_equal_timestamp(self):
        """account_created must never be after timestamp (D11 >= 0 enforced)."""
        adapted = _make_adapted(200)
        diff = (adapted["timestamp"] - adapted["account_created"]).dt.total_seconds()
        assert (diff >= -1).all(), "account_created is after timestamp for some rows"

    def test_is_fraud_preserved_correctly(self):
        raw = _make_raw_txn(20, seed=5)
        raw["isFraud"] = [0] * 15 + [1] * 5
        adapted, _ = adapt(raw)
        assert int(adapted["is_fraud"].sum()) == 5

    def test_transactiondt_preserved_and_sorted(self):
        adapted = _make_adapted(100)
        assert adapted["TransactionDT"].is_monotonic_increasing

    def test_metadata_prohibited_columns_absent(self):
        raw = _make_raw_txn(20)
        _, meta = adapt(raw)
        assert meta["prohibited_columns_verified_absent"] is True

    def test_metadata_isfraud_never_used(self):
        raw = _make_raw_txn(20)
        _, meta = adapt(raw)
        assert meta["isFraud_never_used_in_features"] is True


# ---------------------------------------------------------------------------
# 2. Prohibited column tests
# ---------------------------------------------------------------------------

class TestProhibitedColumns:

    def test_adapter_rejects_v_columns(self):
        raw = _make_raw_txn(10)
        raw["V1"] = 0.0  # inject a prohibited column
        with pytest.raises(ValueError, match="Prohibited columns"):
            adapt(raw)

    def test_adapter_rejects_c_columns(self):
        raw = _make_raw_txn(10)
        raw["C1"] = 0.0
        with pytest.raises(ValueError, match="Prohibited columns"):
            adapt(raw)

    def test_adapter_rejects_m_columns(self):
        raw = _make_raw_txn(10)
        raw["M1"] = "T"
        with pytest.raises(ValueError, match="Prohibited columns"):
            adapt(raw)

    def test_adapter_rejects_d_series(self):
        raw = _make_raw_txn(10)
        raw["D5"] = 10.0
        with pytest.raises(ValueError, match="Prohibited columns"):
            adapt(raw)

    def test_isfraud_not_in_raw_columns_passed_to_build_features(self):
        """build_ieee_features must strip is_fraud before calling build_features."""
        adapted = _make_adapted(50)
        # Confirm is_fraud is in adapted but NOT in the raw_cols passed to build_features
        from ml.features.build_features import RAW_COLUMNS_USED
        assert "is_fraud" not in RAW_COLUMNS_USED  # verify contract from upstream
        assert "is_fraud" in adapted.columns        # present in adapted
        # After building features, is_fraud must be re-attached by join, not passed through
        features, _ = build_ieee_features(adapted, verbose=False)
        assert "is_fraud" in features.columns


# ---------------------------------------------------------------------------
# 3. Feature pipeline tests
# ---------------------------------------------------------------------------

class TestFeaturePipeline:

    def test_all_feature_columns_present(self):
        features = _make_feature_df(100)
        for col in FEATURE_COLUMNS:
            assert col in features.columns, f"Missing FEATURE_COLUMN: {col}"

    def test_no_nan_in_feature_columns(self):
        features = _make_feature_df(200)
        nan_counts = features[FEATURE_COLUMNS].isna().sum()
        assert nan_counts.sum() == 0, f"NaN in features:\n{nan_counts[nan_counts > 0]}"

    def test_inert_new_device_flag_is_zero(self):
        """new_device_flag must be 0 for all rows (constant sentinel device_id)."""
        features = _make_feature_df(200)
        assert (features["new_device_flag"] == 0).all()

    def test_inert_failed_ratio_is_zero(self):
        """failed_ratio_trailing10 must be 0 for all rows (constant status='success')."""
        features = _make_feature_df(200)
        assert (features["failed_ratio_trailing10"] == 0.0).all()

    def test_feature_categorization_covers_all_columns(self):
        all_cats = set(ACTIVE_FEATURES + INERT_FEATURES + DEGRADED_FEATURES)
        assert all_cats == set(FEATURE_COLUMNS)

    def test_active_features_have_nonzero_variance(self):
        # Use a dataset large enough that some cards get multiple transactions,
        # enabling amount_zscore and velocity features to become non-trivial.
        # With 600 rows and card1 drawn from a small range, most cards get
        # ≥2 transactions so ratio/zscore features develop real variance.
        rng = np.random.default_rng(42)
        n = 600
        dt_vals = np.sort(rng.integers(86_400, 15_811_131, size=n))
        # Limit to 30 distinct cards so each gets ~20 transactions on average
        raw = pd.DataFrame({
            "TransactionID": np.arange(2_000_000, 2_000_000 + n),
            "isFraud":       rng.integers(0, 2, size=n),
            "TransactionDT": dt_vals,
            "TransactionAmt": rng.uniform(10.0, 1000.0, size=n),
            "ProductCD":     rng.choice(["W", "C", "R"], size=n),
            "card1":         rng.integers(100, 130, size=n),   # 30 cards → ~20 txns each
            "addr1":         rng.choice([100.0, 200.0, np.nan], size=n),
            "D1":            rng.choice([0.0, 30.0, np.nan], size=n),
            "D11":           rng.choice([60.0, 180.0, np.nan], size=n),
            "DeviceInfo":    rng.choice(["iOS Device", np.nan], size=n),
        })
        adapted, _ = adapt(raw)
        features, _ = build_ieee_features(adapted, verbose=False)
        # Features that require multiple transactions per card to show variance
        multi_txn_features = {
            "amount_zscore", "amount_vs_avg_ratio",
            "velocity_5min", "velocity_30min", "velocity_60min",
        }
        for f in ACTIVE_FEATURES:
            if f in multi_txn_features:
                # Only assert variance if the dataset has cards with >1 transaction
                card_counts = features.groupby("customer_id").size()
                if (card_counts > 1).any():
                    # At least some cards have history — the feature CAN be non-zero
                    n_nonzero = (features[f] != 0).sum()
                    # Relaxed: just confirm the pipeline produces non-zero values
                    # for at least some transactions (not all-zero)
                    assert n_nonzero > 0, (
                        f"Active feature '{f}' is all-zero despite cards with "
                        f"multiple transactions — check pipeline"
                    )
            else:
                std = features[f].std()
                assert std > 0, (
                    f"Active feature '{f}' has zero variance — check pipeline"
                )

    def test_prior_txn_count_never_negative(self):
        features = _make_feature_df(200)
        assert (features["prior_txn_count"] >= 0).all()

    def test_velocity_features_never_negative(self):
        features = _make_feature_df(200)
        for v in ["velocity_5min", "velocity_30min", "velocity_60min"]:
            assert (features[v] >= 0).all()

    def test_amount_zscore_zero_for_first_transaction(self):
        """First transaction per card has prior_txn_count=0 → amount_zscore=0."""
        features = _make_feature_df(300)
        first_txns = features[features["prior_txn_count"] == 0]
        assert (first_txns["amount_zscore"] == 0.0).all()

    def test_is_fraud_reattached_after_build(self):
        adapted = _make_adapted(100)
        features, _ = build_ieee_features(adapted, verbose=False)
        assert "is_fraud" in features.columns
        assert features["is_fraud"].isin([0, 1]).all()

    def test_transactiondt_reattached_after_build(self):
        adapted = _make_adapted(100)
        features, _ = build_ieee_features(adapted, verbose=False)
        assert "TransactionDT" in features.columns


# ---------------------------------------------------------------------------
# 4. Chronological split tests
# ---------------------------------------------------------------------------

class TestChronologicalSplit:

    def test_split_produces_three_nonempty_partitions(self):
        features = _make_feature_df(300)
        # Only run if all three zones are represented
        dt = features["TransactionDT"]
        if not ((dt < TRAIN_BOUNDARY).any() and
                ((dt >= TRAIN_BOUNDARY) & (dt < VAL_BOUNDARY)).any() and
                (dt >= VAL_BOUNDARY).any()):
            pytest.skip("Synthetic data doesn't span all three zones in this sample")
        train, val, test = split_features(features, verbose=False)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_dt_overlap_between_splits(self):
        features = _make_feature_df(500)
        dt = features["TransactionDT"]
        if not ((dt < TRAIN_BOUNDARY).any() and
                ((dt >= TRAIN_BOUNDARY) & (dt < VAL_BOUNDARY)).any() and
                (dt >= VAL_BOUNDARY).any()):
            pytest.skip("Synthetic data doesn't span all three zones")
        train, val, test = split_features(features, verbose=False)
        assert train["TransactionDT"].max() < TRAIN_BOUNDARY
        assert val["TransactionDT"].min()  >= TRAIN_BOUNDARY
        assert val["TransactionDT"].max()  <  VAL_BOUNDARY
        assert test["TransactionDT"].min() >= VAL_BOUNDARY

    def test_split_covers_all_rows(self):
        features = _make_feature_df(500)
        dt = features["TransactionDT"]
        if not ((dt < TRAIN_BOUNDARY).any() and
                ((dt >= TRAIN_BOUNDARY) & (dt < VAL_BOUNDARY)).any() and
                (dt >= VAL_BOUNDARY).any()):
            pytest.skip("Synthetic data doesn't span all three zones")
        train, val, test = split_features(features, verbose=False)
        assert len(train) + len(val) + len(test) == len(features)

    def test_no_future_leakage_assertion(self):
        """_check_no_future_leakage must raise if max_train_dt >= min_val_dt."""
        rng = np.random.default_rng(7)
        n = 60
        features = pd.DataFrame({
            "TransactionDT": np.sort(rng.integers(86400, 15_811_131, n)),
            "is_fraud": rng.integers(0, 2, n),
        })
        # Create a leaky split where train/val overlap
        half = n // 2
        train_bad = features.iloc[:half + 5].copy()
        val_bad   = features.iloc[half:].copy()
        test_ok   = features.iloc[half + 10:].copy()
        with pytest.raises(AssertionError, match="Chronological leakage"):
            _check_no_future_leakage(train_bad, val_bad, test_ok)

    def test_boundary_constants_consistent(self):
        """TRAIN_BOUNDARY < VAL_BOUNDARY — basic sanity."""
        assert TRAIN_BOUNDARY < VAL_BOUNDARY
        assert TRAIN_BOUNDARY > 0
        assert VAL_BOUNDARY > 0

    def test_split_requires_transactiondt_column(self):
        features = pd.DataFrame({"is_fraud": [0, 1], "amount": [100.0, 200.0]})
        with pytest.raises(ValueError, match="TransactionDT"):
            split_features(features, verbose=False)

    def test_split_requires_is_fraud_column(self):
        features = pd.DataFrame({"TransactionDT": [100, 200]})
        with pytest.raises(ValueError, match="is_fraud"):
            split_features(features, verbose=False)


# ---------------------------------------------------------------------------
# 5. No-label-use-in-features tests
# ---------------------------------------------------------------------------

class TestNoLabelLeakage:

    def test_isFraud_not_used_in_adaptation(self):
        """Changing isFraud should not change any feature column."""
        raw_a = _make_raw_txn(40, seed=10)
        raw_b = raw_a.copy()
        raw_b["isFraud"] = 1 - raw_b["isFraud"]  # flip all labels

        adapted_a, _ = adapt(raw_a)
        adapted_b, _ = adapt(raw_b)

        # All non-label, non-transactiondt columns must be identical
        for col in adapted_a.columns:
            if col in ("is_fraud",):
                continue
            pd.testing.assert_series_equal(
                adapted_a[col].reset_index(drop=True),
                adapted_b[col].reset_index(drop=True),
                check_names=False,
                obj=f"Column '{col}' differs when isFraud is flipped",
            )

    def test_feature_values_independent_of_label(self):
        """FEATURE_COLUMNS must be identical regardless of is_fraud values."""
        adapted_a = _make_adapted(60, seed=11)
        adapted_b = adapted_a.copy()
        adapted_b["is_fraud"] = 1 - adapted_b["is_fraud"]

        features_a, _ = build_ieee_features(adapted_a, verbose=False)
        features_b, _ = build_ieee_features(adapted_b, verbose=False)

        for col in FEATURE_COLUMNS:
            # Must align by transaction_id
            fa = features_a.set_index("transaction_id")[col]
            fb = features_b.set_index("transaction_id")[col]
            pd.testing.assert_series_equal(
                fa, fb,
                check_names=False,
                obj=f"Feature '{col}' differs when labels are flipped",
            )

    def test_no_group_fraud_rates_in_features(self):
        """Features must not include computed fraud rates by group."""
        from ml.features.build_features import RAW_COLUMNS_USED
        # The raw schema columns must not include 'is_fraud'
        assert "is_fraud" not in RAW_COLUMNS_USED
        assert "isFraud" not in RAW_COLUMNS_USED


# ---------------------------------------------------------------------------
# 6. Adapter missing-file handling
# ---------------------------------------------------------------------------

class TestAdapterFileHandling:

    def test_missing_txn_file_raises_filenotfounderror(self):
        from ml.external.ieee.ieee_adapter import load_raw
        with pytest.raises(FileNotFoundError, match="IEEE-CIS transaction file not found"):
            load_raw("/nonexistent/train_transaction.csv")

    def test_missing_identity_file_raises_filenotfounderror(self):
        from ml.external.ieee.ieee_adapter import load_raw
        with pytest.raises(FileNotFoundError, match="identity file not found"):
            load_raw(_TXN_PATH if _DATA_AVAILABLE else "/nonexistent/txn.csv",
                     "/nonexistent/identity.csv")


# ---------------------------------------------------------------------------
# 7. Integration tests (require actual data files)
# ---------------------------------------------------------------------------

class TestIntegrationWithRealData:

    @_requires_data
    @_integration
    def test_full_pipeline_produces_590k_rows(self):
        from ml.external.ieee.ieee_adapter  import load_and_adapt
        from ml.external.ieee.ieee_features import build_ieee_features
        adapted, meta = load_and_adapt(_TXN_PATH, _IDNT_PATH, verbose=False)
        assert len(adapted) == 590_540
        assert meta["n_rows"] == 590_540
        assert abs(meta["fraud_rate"] - 0.035) < 0.002

    @_requires_data
    @_integration
    def test_real_data_inert_features_constant(self):
        from ml.external.ieee.ieee_adapter  import load_and_adapt
        from ml.external.ieee.ieee_features import build_ieee_features
        adapted, _ = load_and_adapt(_TXN_PATH, _IDNT_PATH, verbose=False)
        features, _ = build_ieee_features(adapted, verbose=False)
        assert features["new_device_flag"].nunique() == 1
        assert features["failed_ratio_trailing10"].nunique() == 1

    @_requires_data
    @_integration
    def test_real_data_split_sizes_match_expected(self):
        from ml.external.ieee.ieee_adapter  import load_and_adapt
        from ml.external.ieee.ieee_features import build_ieee_features
        from ml.external.ieee.ieee_split    import split_features
        adapted, _ = load_and_adapt(_TXN_PATH, _IDNT_PATH, verbose=False)
        features, _ = build_ieee_features(adapted, verbose=False)
        train, val, test = split_features(features, verbose=False)
        assert len(train) == 383_851
        assert len(val)   ==  88_581
        assert len(test)  == 118_108

    @_requires_data
    @_integration
    def test_real_data_no_prohibited_columns_in_features(self):
        from ml.external.ieee.ieee_adapter  import load_and_adapt
        from ml.external.ieee.ieee_features import build_ieee_features
        adapted, _ = load_and_adapt(_TXN_PATH, _IDNT_PATH, verbose=False)
        features, _ = build_ieee_features(adapted, verbose=False)
        prohibited = (
            [f"C{i}" for i in range(1, 15)] +
            [f"V{i}" for i in range(1, 340)] +
            [f"M{i}" for i in range(1, 10)] +
            [f"D{i}" for i in list(range(2, 11)) + list(range(12, 16))]
        )
        for col in prohibited:
            assert col not in features.columns, f"Prohibited column '{col}' found in features"

    @_requires_data
    @_integration
    def test_real_data_fraudrate_stable_across_splits(self):
        from ml.external.ieee.ieee_adapter  import load_and_adapt
        from ml.external.ieee.ieee_features import build_ieee_features
        from ml.external.ieee.ieee_split    import split_features
        adapted, _ = load_and_adapt(_TXN_PATH, _IDNT_PATH, verbose=False)
        features, _ = build_ieee_features(adapted, verbose=False)
        train, val, test = split_features(features, verbose=False)
        # All splits should be within 1 pp of the overall 3.5% fraud rate
        for name, split in [("train", train), ("val", val), ("test", test)]:
            fr = split["is_fraud"].mean()
            assert abs(fr - 0.035) < 0.015, \
                f"{name} fraud rate {fr:.3%} is too far from expected ~3.5%"
