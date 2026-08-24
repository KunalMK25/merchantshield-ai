"""
IEEE-CIS to MerchantShield raw-schema adapter.

PURPOSE
-------
Map the IEEE-CIS Fraud Detection dataset (Vesta / Kaggle 2019) into the
exact column contract expected by ml/features/build_features.py, so the
existing MerchantShield feature-engineering pipeline can be applied
without modification.

WHAT THIS MODULE IS AND IS NOT
-------------------------------
- It IS a data-transformation layer that honestly maps available fields.
- It IS NOT a model, a feature extractor, or a performance optimizer.
- It does NOT touch the frozen lgbm_v1 model or the 0.40 threshold.
- It does NOT modify ml/data/, ml/features/, ml/evaluation/, or any
  existing production code.

LEAKAGE PROHIBITIONS (enforced by this adapter):
  - isFraud is NEVER used to construct any output column.
  - C1–C14 are NEVER loaded or referenced (aggregate counts, undisclosed
    definition, likely post-transaction signals).
  - V1–V339 are NEVER loaded or referenced (Vesta-engineered opaque features).
  - M1–M9 are NEVER loaded or referenced (match flags, >45% missing,
    undisclosed semantics).
  - D2–D10, D12–D15 are NEVER loaded or referenced (undisclosed timedeltas).
  - No group-level fraud-rate statistics are computed as features.
  - Transactions are not filtered by label before or during feature construction.

HONEST FEATURE DEGRADATIONS (documented here, visible in output metadata):
  - new_device_flag:       INERT. DeviceInfo is 79.9% missing after join, and
                           its missingness is fraud-correlated (not MAR). Using
                           a constant sentinel means this feature carries zero
                           signal in the external experiment. It is set to the
                           string "device_unknown" in device_id so build_features
                           computes new_device_flag=0 for all transactions.
  - failed_ratio_trailing10: INERT. No transaction success/failure field exists
                           in IEEE-CIS. status is set to "success" for all rows.
  - account_age_days:      DEGRADED. D11 is used as a proxy ('days since account
                           opened' per community documentation). 47.3% of rows
                           have D11=NaN; those rows get account_created=timestamp
                           which yields account_age_days=0.0. D11 is NOT verified
                           ground-truth account age — see validate_d11_proxy().
  - new_geo_flag:          DEGRADED. addr1 (billing region code) is used as a
                           proxy. 11.1% of rows have addr1=NaN; those rows get
                           geo_region="geo_unknown", yielding new_geo_flag=0.
                           Billing region changes rarely — this is not equivalent
                           to MerchantShield's per-transaction geo region.
  - hour_of_day/is_night/day_of_week: APPROXIMATE. TransactionDT is a timedelta
                           in seconds from an undisclosed reference datetime. We
                           use an inferred reference (IEEE_REFERENCE_DATETIME) to
                           reconstruct wall-clock time. This is clearly labelled
                           throughout. The relative ordering of transactions is
                           exact; only the absolute hour/day values are approximate.

COLUMN CONTRACT REQUIRED BY build_features.py:
  transaction_id, customer_id, merchant_id, merchant_category,
  timestamp, amount, device_id, geo_region, payment_method,
  status, account_created
  PLUS: is_fraud (preserved separately, never passed to build_features)
  PLUS: TransactionDT (preserved for chronological split)
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference datetime for TransactionDT reconstruction.
# TransactionDT is seconds from this undisclosed reference. The minimum value
# in the dataset is 86,400 (= exactly 1 day after the reference), which is
# consistent with the community-documented reference of approximately
# 2017-11-30. We anchor to 2017-11-30 00:00:00 UTC.
# THIS IS AN INFERENCE, NOT GROUND TRUTH. Features derived from absolute
# wall-clock time (hour_of_day, is_night, day_of_week) are APPROXIMATE.
# ---------------------------------------------------------------------------
IEEE_REFERENCE_DATETIME = pd.Timestamp("2017-11-30", tz="UTC")

# Columns we explicitly load from the transaction file (minimise memory).
# V1-V339, C1-C14, M1-M9, D2-D10, D12-D15 are excluded.
_TXN_LOAD_COLS = [
    "TransactionID", "isFraud", "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card1",
    "addr1",
    "D1",
    "D11",
]

# Columns loaded from the identity file for the optional device join.
_ID_LOAD_COLS = ["TransactionID", "DeviceInfo"]


def load_raw(
    txn_path: str,
    identity_path: str | None = None,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load IEEE-CIS transaction (and optionally identity) files.

    Parameters
    ----------
    txn_path : str
        Path to train_transaction.csv (or equivalent labeled file).
    identity_path : str | None
        Path to train_identity.csv. Optional. If None, device information is
        not joined, and device_id will be the constant sentinel for all rows.
    verbose : bool
        Print loading progress to stdout.

    Returns
    -------
    pd.DataFrame
        Raw IEEE-CIS data with only the columns needed by this adapter,
        sorted ascending by TransactionDT. The DataFrame is NOT yet adapted
        to the MerchantShield schema — call adapt() for that.
    """
    if not os.path.exists(txn_path):
        raise FileNotFoundError(
            f"IEEE-CIS transaction file not found: {txn_path!r}\n"
            "Download from https://www.kaggle.com/c/ieee-fraud-detection "
            "(requires Kaggle account and competition rules acceptance). "
            "The file must not be committed to Git."
        )

    if verbose:
        print(f"[ieee_adapter] Loading transaction file: {txn_path}")
    txn = pd.read_csv(txn_path, usecols=_TXN_LOAD_COLS)

    if verbose:
        print(f"[ieee_adapter] Loaded {len(txn):,} transaction rows.")
        fraud_count = int(txn["isFraud"].sum())
        print(f"[ieee_adapter] Fraud rate: {fraud_count:,}/{len(txn):,} = {fraud_count/len(txn):.4%}")

    # Sort by TransactionDT — confirmed monotonically increasing in inspection,
    # but enforced here defensively.
    txn = txn.sort_values("TransactionDT").reset_index(drop=True)

    # Optionally join DeviceInfo from the identity file.
    if identity_path is not None:
        if not os.path.exists(identity_path):
            raise FileNotFoundError(
                f"IEEE-CIS identity file not found: {identity_path!r}"
            )
        if verbose:
            print(f"[ieee_adapter] Joining identity file: {identity_path}")
        idf = pd.read_csv(identity_path, usecols=_ID_LOAD_COLS)
        if verbose:
            print(f"[ieee_adapter] Identity rows: {len(idf):,} "
                  f"({len(idf)/len(txn):.1%} of transactions)")
        txn = txn.merge(idf, on="TransactionID", how="left")
        device_coverage = txn["DeviceInfo"].notna().sum()
        if verbose:
            print(f"[ieee_adapter] DeviceInfo present after join: "
                  f"{device_coverage:,}/{len(txn):,} ({device_coverage/len(txn):.1%})")
    else:
        txn["DeviceInfo"] = np.nan
        if verbose:
            print("[ieee_adapter] No identity file provided. DeviceInfo=NaN for all rows.")

    return txn


def validate_d11_proxy(df: pd.DataFrame, *, verbose: bool = True) -> dict:
    """
    Empirically validate whether D11 ('days since account opened') is a
    plausible proxy for account age.

    Checks:
      1. D11 null rate (documented: ~47.3%)
      2. D11 range and distribution
      3. D11 values should be non-negative for valid rows
      4. Correlation between D11 and card1 first-appearance order
         (older cards should have higher D11 values on average)
      5. D11 is NOT used to select or filter rows here — this is inspection only.

    Returns a dict of validation findings.
    """
    d11 = df["D11"].copy()
    null_rate = d11.isna().mean()
    neg_rate = (d11.dropna() < 0).mean()
    q = d11.dropna().quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).to_dict()

    # Check monotonicity proxy: for cards with multiple transactions, D11 should
    # tend to increase over time (later transactions = older account).
    # We check this on a sample of cards with >=5 non-null D11 observations.
    mono_check_cards = (
        df[df["D11"].notna()]
        .groupby("card1")["D11"]
        .count()
        .pipe(lambda s: s[s >= 5])
        .index
    )
    if len(mono_check_cards) > 0:
        sample_cards = mono_check_cards[:200]
        sub = df[df["card1"].isin(sample_cards) & df["D11"].notna()].copy()
        sub = sub.sort_values("TransactionDT")
        # Per-card Spearman correlation between position (time) and D11
        from scipy.stats import spearmanr
        rhos = []
        for _, g in sub.groupby("card1"):
            if len(g) >= 5:
                rho, _ = spearmanr(np.arange(len(g)), g["D11"].values)
                rhos.append(rho)
        median_rho = float(np.median(rhos)) if rhos else float("nan")
    else:
        median_rho = float("nan")

    finding = {
        "null_rate": float(null_rate),
        "negative_rate": float(neg_rate),
        "quantiles": {str(k): float(v) for k, v in q.items()},
        "median_per_card_spearman_rho_with_time": median_rho,
        "assessment": (
            "USABLE_WITH_CAVEATS"
            if null_rate < 0.55 and neg_rate < 0.05 and not np.isnan(median_rho) and median_rho > 0.3
            else "DEGRADED"
        ),
        "notes": (
            f"D11 null rate={null_rate:.1%}. "
            f"Negative values={neg_rate:.1%} (clamped to 0). "
            f"Median per-card Spearman(D11, time)={median_rho:.3f} "
            f"({'positive monotonic trend — consistent with account-age interpretation' if median_rho > 0.3 else 'weak or unclear trend'})."
        ),
    }

    if verbose:
        print(f"[ieee_adapter] D11 proxy validation:")
        print(f"  null_rate={null_rate:.1%}  neg_rate={neg_rate:.1%}  "
              f"median_spearman_rho={median_rho:.3f}  assessment={finding['assessment']}")

    return finding


def adapt(df: pd.DataFrame, *, d11_assessment: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Transform the raw IEEE-CIS DataFrame into the MerchantShield raw-schema
    column contract required by build_features.build_features().

    Does NOT call build_features — that is done by ieee_features.build_ieee_features().

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_raw().
    d11_assessment : dict | None
        Output of validate_d11_proxy(). If None, D11 is used as a proxy
        regardless (degraded but documented).

    Returns
    -------
    adapted : pd.DataFrame
        Columns: transaction_id, customer_id, merchant_id, merchant_category,
                 timestamp, amount, device_id, geo_region, payment_method,
                 status, account_created, is_fraud, TransactionDT
    metadata : dict
        Machine-readable record of every transformation and sentinel decision.
    """
    # Safety: confirm isFraud is present but will NOT be used in feature construction.
    assert "isFraud" in df.columns, "isFraud column required (kept separate, never used in features)"
    assert "TransactionDT" in df.columns

    # --- LEAKAGE GUARD: confirm prohibited columns are absent ---
    prohibited = (
        [f"C{i}" for i in range(1, 15)]
        + [f"V{i}" for i in range(1, 340)]
        + [f"M{i}" for i in range(1, 10)]
        + [f"D{i}" for i in list(range(2, 11)) + list(range(12, 16))]
    )
    present_prohibited = [c for c in prohibited if c in df.columns]
    if present_prohibited:
        raise ValueError(
            f"Prohibited columns present in DataFrame: {present_prohibited}. "
            "These must not enter the feature pipeline."
        )

    n = len(df)
    out = pd.DataFrame(index=df.index)

    # ------------------------------------------------------------------ #
    # 1. transaction_id  <- TransactionID (string)
    # ------------------------------------------------------------------ #
    out["transaction_id"] = df["TransactionID"].astype(str)

    # ------------------------------------------------------------------ #
    # 2. customer_id  <- card1 (string)
    # card1 has 0.0% null (confirmed in inspection).
    # Represents a credit/debit card number — used as behavioral grouping key.
    # ------------------------------------------------------------------ #
    assert df["card1"].isna().sum() == 0, "card1 should have zero nulls"
    out["customer_id"] = df["card1"].astype(str)

    # ------------------------------------------------------------------ #
    # 3. merchant_id  <- ProductCD
    # ProductCD (W/H/C/S/R) is a product category, not a merchant ID.
    # It is the closest available coarse merchant-category proxy.
    # ------------------------------------------------------------------ #
    out["merchant_id"] = df["ProductCD"].fillna("UNKNOWN")

    # ------------------------------------------------------------------ #
    # 4. merchant_category  <- ProductCD (same field)
    # ------------------------------------------------------------------ #
    out["merchant_category"] = df["ProductCD"].fillna("UNKNOWN")

    # ------------------------------------------------------------------ #
    # 5. timestamp  <- IEEE_REFERENCE_DATETIME + TransactionDT seconds
    # TransactionDT is seconds from an UNDISCLOSED reference datetime.
    # We use IEEE_REFERENCE_DATETIME = 2017-11-30 00:00:00 UTC.
    # This is an INFERENCE. Relative ordering is exact; absolute clock
    # values (hour, day) are APPROXIMATE.
    # ------------------------------------------------------------------ #
    out["timestamp"] = IEEE_REFERENCE_DATETIME + pd.to_timedelta(
        df["TransactionDT"], unit="s"
    )
    # Strip timezone for build_features compatibility (it uses naive datetimes)
    out["timestamp"] = out["timestamp"].dt.tz_localize(None)

    # ------------------------------------------------------------------ #
    # 6. amount  <- TransactionAmt (USD)
    # MerchantShield's original training used INR amounts. The cost model
    # uses amount * fn_cost_fraction — the currency symbol changes but the
    # relative ordering and ratio features are unaffected. Experiment B will
    # use USD amounts throughout consistently.
    # ------------------------------------------------------------------ #
    assert df["TransactionAmt"].isna().sum() == 0
    out["amount"] = df["TransactionAmt"].astype(float)

    # ------------------------------------------------------------------ #
    # 7. device_id  <- CONSTANT SENTINEL "device_unknown"
    # DeviceInfo: 79.9% missing after identity join. Missingness is fraud-
    # correlated (7.25% fraud rate when present vs 2.56% when absent).
    # Using a constant means new_device_flag=0 for ALL transactions because:
    #   - First transaction for a card: build_features sets flag=0 always.
    #   - Subsequent transactions: "device_unknown" is in seen_devices already,
    #     so flag=0 again.
    # This feature is INERT in the external experiment. It is NOT fabricated —
    # the 0 value is the truthful result of "no device information available".
    # ------------------------------------------------------------------ #
    out["device_id"] = "device_unknown"
    device_coverage = df["DeviceInfo"].notna().sum() if "DeviceInfo" in df.columns else 0

    # ------------------------------------------------------------------ #
    # 8. geo_region  <- addr1 (billing region code, cast to string)
    # addr1: 11.1% null. Missingness is fraud-correlated (11.78% vs 2.46%).
    # Null rows get geo_region="geo_unknown" — they do NOT form a behavioral
    # group with each other; the novelty flag for these rows will be 0.
    # addr1 is billing region (rarely changes), NOT per-transaction location.
    # ------------------------------------------------------------------ #
    def _map_geo(v):
        if pd.isna(v):
            return "geo_unknown"
        return f"region_{int(v)}"

    out["geo_region"] = df["addr1"].apply(_map_geo)
    geo_null_count = (out["geo_region"] == "geo_unknown").sum()

    # ------------------------------------------------------------------ #
    # 9. payment_method  <- CONSTANT "card_payment"
    # card4 (visa/mastercard/discover/amex) could serve as a proxy, but
    # it has 0.3% null and MerchantShield's feature pipeline does not use
    # payment_method directly in FEATURE_COLUMNS. Setting a constant avoids
    # any incidental grouping or novelty detection on this field.
    # ------------------------------------------------------------------ #
    out["payment_method"] = "card_payment"

    # ------------------------------------------------------------------ #
    # 10. status  <- CONSTANT "success"
    # No transaction success/failure field exists in IEEE-CIS.
    # failed_ratio_trailing10 = 0.0 for all rows as a result. INERT.
    # ------------------------------------------------------------------ #
    out["status"] = "success"

    # ------------------------------------------------------------------ #
    # 11. account_created  <- timestamp - D11 days (proxy)
    # D11: community-documented as 'days since account opened'. 47.3% null.
    # - Where D11 >= 0: account_created = timestamp - D11 days.
    # - Where D11 <  0: clamp to 0 (treat as account_created = timestamp,
    #   yielding account_age_days = 0.0).
    # - Where D11 is NaN: account_created = timestamp (age = 0.0).
    # This is a DEGRADED proxy. account_age_days = 0.0 for ~47% of rows.
    # ------------------------------------------------------------------ #
    d11 = df["D11"].copy()
    d11_null_count = int(d11.isna().sum())
    d11_neg_count = int((d11.dropna() < 0).sum())
    d11_clamped = d11.fillna(0.0).clip(lower=0.0)
    out["account_created"] = out["timestamp"] - pd.to_timedelta(d11_clamped, unit="D")

    # ------------------------------------------------------------------ #
    # 12. Preserve is_fraud and TransactionDT for split / evaluation use.
    # These are NEVER passed to build_features.build_features().
    # ------------------------------------------------------------------ #
    out["is_fraud"] = df["isFraud"].astype(int)
    out["TransactionDT"] = df["TransactionDT"].astype(int)

    # ------------------------------------------------------------------ #
    # Metadata: machine-readable record of every transformation decision.
    # ------------------------------------------------------------------ #
    metadata = {
        "n_rows": n,
        "fraud_count": int(out["is_fraud"].sum()),
        "fraud_rate": float(out["is_fraud"].mean()),
        "reference_datetime": str(IEEE_REFERENCE_DATETIME),
        "reference_datetime_note": (
            "Inferred; not officially published by Vesta. "
            "hour_of_day, is_night, day_of_week are APPROXIMATE."
        ),
        "customer_id_source": "card1",
        "customer_id_null_rate": 0.0,
        "feature_degradations": {
            "new_device_flag": {
                "status": "INERT",
                "reason": (
                    "DeviceInfo 79.9% missing after identity join. "
                    "Missingness is fraud-correlated (not MAR). "
                    "Constant sentinel 'device_unknown' used: new_device_flag=0 for all rows."
                ),
                "device_info_coverage": float(device_coverage / n),
            },
            "failed_ratio_trailing10": {
                "status": "INERT",
                "reason": "No transaction status field in IEEE-CIS. status='success' for all rows.",
            },
            "account_age_days": {
                "status": "DEGRADED",
                "source": "D11 (community-documented 'days since account opened')",
                "d11_null_rate": float(d11_null_count / n),
                "d11_negative_clamped_count": d11_neg_count,
                "rows_with_age_zero": d11_null_count,
                "d11_assessment": d11_assessment.get("assessment") if d11_assessment else "not_validated",
            },
            "new_geo_flag": {
                "status": "DEGRADED",
                "source": "addr1 (billing region code)",
                "addr1_null_rate": float(geo_null_count / n),
                "note": (
                    "addr1 is billing region, not per-transaction location. "
                    "Null rows get geo_region='geo_unknown' (new_geo_flag=0)."
                ),
            },
            "hour_of_day_is_night_day_of_week": {
                "status": "APPROXIMATE",
                "note": "Derived from inferred reference datetime. See reference_datetime_note.",
            },
        },
        "prohibited_columns_verified_absent": len(present_prohibited) == 0,
        "isFraud_never_used_in_features": True,
    }

    return out, metadata


def load_and_adapt(
    txn_path: str,
    identity_path: str | None = None,
    *,
    verbose: bool = True,
    validate_d11: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Convenience wrapper: load_raw() → validate_d11_proxy() → adapt().

    Returns
    -------
    adapted : pd.DataFrame  — MerchantShield raw-schema columns + is_fraud + TransactionDT
    metadata : dict         — all transformation decisions and quality findings
    """
    df = load_raw(txn_path, identity_path, verbose=verbose)

    d11_assessment = None
    if validate_d11:
        d11_assessment = validate_d11_proxy(df, verbose=verbose)

    adapted, meta = adapt(df, d11_assessment=d11_assessment)
    if d11_assessment:
        meta["d11_proxy_validation"] = d11_assessment

    return adapted, meta
