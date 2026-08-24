"""
IEEE-CIS chronological split.

Split boundaries are declared here and must NOT be changed after the
experiment begins. They were determined during the dataset inspection phase
(before any modeling) from TransactionDT quantiles:

  TRAIN_BOUNDARY  = 9,614,666   → Q65 of TransactionDT ≈ day 110.3
  VAL_BOUNDARY    = 12,192,853  → Q80 of TransactionDT ≈ day 140.1

  Train:  TransactionDT <  TRAIN_BOUNDARY           ≈ 383,851 rows  65%
  Val:    TRAIN_BOUNDARY ≤ TransactionDT < VAL_BOUNDARY ≈  88,581 rows  15%
  Test:   TransactionDT ≥ VAL_BOUNDARY              ≈ 118,108 rows  20%

LEAKAGE NOTE: The split is applied AFTER feature construction. Because
build_features uses an expanding-window per customer_id, applying the split
after feature construction is correct: a training-set transaction's feature
values already only use prior transactions for that card (leakage-safe by
the causal construction in build_features). Splitting before feature
construction would discard all prior history for the first training-set
transaction — losing the very historical context the features are designed
to capture.

However: cards that appear in the test set with prior transactions that fall
in the training set are handled correctly — those prior-transaction features
will have been built from the full chronological history up to each
transaction's timestamp. This mirrors how MerchantShield's synthetic split
works (days-based boundary applied to an already-built feature frame).
"""

import pandas as pd

# Declared boundaries — set before any modeling, do not modify.
TRAIN_BOUNDARY: int = 9_614_666
VAL_BOUNDARY: int   = 12_192_853

# Expected approximate sizes from inspection
EXPECTED_TRAIN_N = 383_851
EXPECTED_VAL_N   =  88_581
EXPECTED_TEST_N  = 118_108

# Tolerance for size check (±2% to account for rounding in quantile boundaries)
_SIZE_TOLERANCE = 0.02


def split_features(
    features: pd.DataFrame,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply the declared chronological split to a feature DataFrame.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ieee_features.build_ieee_features() — must contain
        'TransactionDT' and 'is_fraud' columns.
    verbose : bool

    Returns
    -------
    train, val, test : pd.DataFrame
        Non-overlapping chronological splits. Each contains FEATURE_COLUMNS
        plus is_fraud, customer_id, transaction_id, TransactionDT, amount.
    """
    if "TransactionDT" not in features.columns:
        raise ValueError(
            "features DataFrame must contain 'TransactionDT'. "
            "Ensure ieee_features.build_ieee_features() re-attached it."
        )
    if "is_fraud" not in features.columns:
        raise ValueError("features DataFrame must contain 'is_fraud'.")

    dt = features["TransactionDT"]
    train = features[dt <  TRAIN_BOUNDARY].copy()
    val   = features[(dt >= TRAIN_BOUNDARY) & (dt < VAL_BOUNDARY)].copy()
    test  = features[dt >= VAL_BOUNDARY].copy()

    if verbose:
        for name, split in [("Train", train), ("Val", val), ("Test", test)]:
            print(
                f"[ieee_split] {name:5s}: {len(split):>7,} rows  "
                f"fraud_rate={split['is_fraud'].mean():.3%}  "
                f"dt_range=[{split['TransactionDT'].min():,}, {split['TransactionDT'].max():,}]"
            )

    # Sanity checks
    _check_no_overlap(train, val, test)
    _check_sizes(train, val, test)
    _check_no_future_leakage(train, val, test)

    return train, val, test


def _check_no_overlap(train, val, test):
    """Assert TransactionDT ranges are strictly non-overlapping."""
    assert train["TransactionDT"].max() < TRAIN_BOUNDARY, \
        f"Train set contains TransactionDT >= TRAIN_BOUNDARY ({TRAIN_BOUNDARY})"
    assert val["TransactionDT"].min() >= TRAIN_BOUNDARY, \
        f"Val set contains TransactionDT < TRAIN_BOUNDARY"
    assert val["TransactionDT"].max() < VAL_BOUNDARY, \
        f"Val set contains TransactionDT >= VAL_BOUNDARY ({VAL_BOUNDARY})"
    assert test["TransactionDT"].min() >= VAL_BOUNDARY, \
        f"Test set contains TransactionDT < VAL_BOUNDARY"


def _check_sizes(train, val, test):
    """Warn if split sizes differ substantially from expected."""
    for name, split, expected in [
        ("Train", train, EXPECTED_TRAIN_N),
        ("Val",   val,   EXPECTED_VAL_N),
        ("Test",  test,  EXPECTED_TEST_N),
    ]:
        actual = len(split)
        if abs(actual - expected) / expected > _SIZE_TOLERANCE:
            # Warning, not an error — feature construction may drop rows
            print(
                f"[ieee_split] WARNING: {name} split size {actual:,} differs from "
                f"expected ~{expected:,} by >{_SIZE_TOLERANCE:.0%}. "
                "This may be expected if some rows were dropped during feature construction."
            )


def _check_no_future_leakage(train, val, test):
    """
    Verify that the latest training TransactionDT is strictly less than the
    earliest validation/test TransactionDT. This is the chronological boundary
    check — it confirms no test or val transaction's timestamp appears in train.
    """
    max_train_dt = train["TransactionDT"].max()
    min_val_dt   = val["TransactionDT"].min()
    min_test_dt  = test["TransactionDT"].min()
    assert max_train_dt < min_val_dt, (
        f"Chronological leakage: max train DT ({max_train_dt:,}) "
        f">= min val DT ({min_val_dt:,})"
    )
    assert max_train_dt < min_test_dt, (
        f"Chronological leakage: max train DT ({max_train_dt:,}) "
        f">= min test DT ({min_test_dt:,})"
    )
