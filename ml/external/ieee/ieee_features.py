"""
IEEE-CIS feature builder.

Applies ml/features/build_features.build_features() — UNCHANGED — to the
adapted IEEE-CIS DataFrame produced by ieee_adapter.adapt().

The existing build_features function is called with no modification. This
module only:
  1. Validates the input schema before calling it.
  2. Calls it.
  3. Returns the feature-complete DataFrame plus the feature-availability
     summary so callers know exactly which of the 15 features are inert.

IMPORTANT: the two inert features (new_device_flag, failed_ratio_trailing10)
and the two degraded features (new_geo_flag, account_age_days) are computed
by build_features exactly as coded — they will produce zeros/constant values
because of the sentinel inputs supplied by the adapter. This is intentional
and documented. Do NOT try to post-hoc "fix" them.
"""

import os
import sys

import pandas as pd

# Ensure project root is on path so ml.features imports cleanly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ml.features.build_features import build_features, FEATURE_COLUMNS, RAW_COLUMNS_USED

# Which features are fully active on IEEE-CIS data (non-trivial signal expected)
ACTIVE_FEATURES = [
    "amount",
    "amount_zscore",
    "amount_vs_avg_ratio",
    "prior_txn_count",
    "time_since_prev_txn_min",
    "velocity_5min",
    "velocity_30min",
    "velocity_60min",
]

# Which features are computed but carry no signal (constant sentinel input)
INERT_FEATURES = [
    "new_device_flag",        # device_id = "device_unknown" for all rows
    "failed_ratio_trailing10", # status = "success" for all rows
]

# Which features are computed but with degraded inputs
DEGRADED_FEATURES = [
    "new_geo_flag",       # addr1 proxy, 11.1% null → 0
    "account_age_days",   # D11 proxy, 47.3% null → 0
    "hour_of_day",        # inferred reference datetime
    "is_night",           # same
    "day_of_week",        # same
]

# Sanity: all 15 features must be accounted for
assert set(ACTIVE_FEATURES + INERT_FEATURES + DEGRADED_FEATURES) == set(FEATURE_COLUMNS), (
    "Feature availability mapping does not cover all FEATURE_COLUMNS — check for mismatches."
)


def build_ieee_features(
    adapted: pd.DataFrame,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Build MerchantShield features from the IEEE-CIS adapted DataFrame.

    Parameters
    ----------
    adapted : pd.DataFrame
        Output of ieee_adapter.adapt() — must contain all RAW_COLUMNS_USED
        plus is_fraud and TransactionDT.
    verbose : bool

    Returns
    -------
    features : pd.DataFrame
        Full output of build_features(), with is_fraud and TransactionDT
        re-attached for downstream use.
    feature_info : dict
        Summary of active / inert / degraded features, plus basic stats.
    """
    # --- Schema validation ---
    required = set(RAW_COLUMNS_USED) | {"is_fraud", "TransactionDT"}
    missing = required - set(adapted.columns)
    if missing:
        raise ValueError(
            f"Adapted DataFrame is missing required columns: {missing}. "
            "Run ieee_adapter.adapt() first."
        )

    # Preserve labels and split key before calling build_features (it may drop cols)
    labels = adapted["is_fraud"].copy()
    txn_dt = adapted["TransactionDT"].copy()

    # Drop non-raw columns so build_features gets exactly what it expects
    raw_cols = [c for c in RAW_COLUMNS_USED if c in adapted.columns]
    raw_df = adapted[raw_cols].copy()

    if verbose:
        print(f"[ieee_features] Calling build_features on {len(raw_df):,} rows, "
              f"{raw_df['customer_id'].nunique():,} unique cards…")

    features = build_features(raw_df)

    # Re-attach labels and split key (aligned by transaction_id)
    # build_features sorts by (customer_id, timestamp) then timestamp — we need
    # to re-join on transaction_id to keep alignment correct.
    label_map = adapted.set_index("transaction_id")[["is_fraud", "TransactionDT"]]
    features = features.join(label_map, on="transaction_id", how="left")

    if verbose:
        print(f"[ieee_features] Feature build complete: {len(features):,} rows × "
              f"{len(FEATURE_COLUMNS)} feature columns.")
        # Verify inert features really are constant
        for f in INERT_FEATURES:
            n_unique = features[f].nunique()
            if n_unique > 1:
                print(f"  WARNING: inert feature '{f}' has {n_unique} unique values "
                      f"(expected 1). Check adapter sentinel logic.")
            else:
                print(f"  INERT OK: '{f}' = {features[f].iloc[0]} (constant, as expected)")

    # --- Feature availability summary ---
    feature_info = {
        "total_features": len(FEATURE_COLUMNS),
        "active": ACTIVE_FEATURES,
        "inert": INERT_FEATURES,
        "degraded": DEGRADED_FEATURES,
        "n_rows": len(features),
        "n_cards": int(features["customer_id"].nunique()),
        "fraud_rate": float(features["is_fraud"].mean()),
        "inert_feature_values": {
            f: float(features[f].iloc[0]) for f in INERT_FEATURES
        },
        "feature_stats": {
            f: {
                "mean": float(features[f].mean()),
                "std": float(features[f].std()),
                "null_pct": float(features[f].isna().mean()),
            }
            for f in FEATURE_COLUMNS
        },
    }

    return features, feature_info
