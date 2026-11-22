"""
The common raw-transaction schema every dataset is mapped into, plus validation and profiling.

Everything downstream (feature engineering, the contest, retraining, drift, anomaly detection)
reads this one schema, so a new dataset only needs a loader that produces it.

`profile` also answers a question the numbers alone hide: can this dataset support MerchantShield's
behavioural features at all? Those features describe a customer's earlier transactions, so a dataset
where almost every customer appears once (PaySim, for example) cannot be modelled this way, however
many rows it has. The profile rates the fit and names the features a dataset leaves inert.
"""

import numpy as np
import pandas as pd

from ml.data.daily_batch import RAW_COLUMNS

MIN_PRIOR_FOR_ZSCORE = 3          # build_features needs 3 earlier transactions for a z-score
GOOD_SHARE = 0.70                 # share of rows with enough history to be called a good fit
WEAK_SHARE = 0.25


class DatasetError(ValueError):
    """A dataset file is missing, malformed, or unsuitable. The message says what and why."""


def validate_raw(df: pd.DataFrame) -> None:
    """Raise DatasetError listing every problem, or return None if `df` fits the raw schema."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"missing column(s): {', '.join(missing)}")

    problems = []
    for col in ("transaction_id", "customer_id", "timestamp", "amount", "account_created", "is_fraud"):
        n = int(df[col].isna().sum())
        if n:
            problems.append(f"{col} has {n} missing value(s)")
    if df["transaction_id"].duplicated().any():
        problems.append("transaction_id must be unique")
    if (pd.to_numeric(df["amount"], errors="coerce") <= 0).any():
        problems.append(f"amount must be positive ({int((df['amount'] <= 0).sum())} rows are not)")
    if not df["is_fraud"].isin([0, 1]).all():
        problems.append("is_fraud must be 0 or 1")
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    created = pd.to_datetime(df["account_created"], errors="coerce")
    if ts.isna().any() or created.isna().any():
        problems.append("timestamp / account_created must be valid dates")
    elif (created > ts).any():
        problems.append(f"account_created is after timestamp for {int((created > ts).sum())} row(s)")
    if problems:
        raise DatasetError("; ".join(problems))


def profile(df: pd.DataFrame) -> dict:
    """Size, history depth, which features the data leaves inert, and an honest fit rating."""
    ts = pd.to_datetime(df["timestamp"])
    first, last = ts.min(), ts.max()
    counts = df.groupby("customer_id").size()
    with_history = int(np.maximum(counts - MIN_PRIOR_FOR_ZSCORE, 0).sum())
    share = with_history / len(df) if len(df) else 0.0

    inert = []
    if df.groupby("customer_id")["device_id"].nunique().max() <= 1:
        inert.append("new_device_flag")
    if df.groupby("customer_id")["geo_region"].nunique().max() <= 1:
        inert.append("new_geo_flag")
    if not (df["status"] == "failed").any():
        inert.append("failed_ratio_trailing10")

    if share >= GOOD_SHARE:
        fit, reason = "good", f"{share:.0%} of rows have enough customer history for behavioural features."
    elif share >= WEAK_SHARE:
        fit, reason = "weak", (f"only {share:.0%} of rows have enough customer history; behavioural "
                               f"features will be informative for part of the data.")
    else:
        fit, reason = "poor", (f"only {share:.0%} of rows have enough customer history, so the behavioural "
                               f"features (built from a customer's earlier transactions) carry almost no signal.")
    return {
        "rows": int(len(df)), "customers": int(counts.size), "fraud_rate": float(df["is_fraud"].mean()),
        "first_day": first.normalize().isoformat(), "last_day": last.normalize().isoformat(),
        "days": int((last.normalize() - first.normalize()).days) + 1,
        "txns_per_customer_median": float(counts.median()), "median_amount": float(df["amount"].median()),
        "share_rows_with_history": float(share), "inert_features": inert,
        "fit": fit, "fit_reason": reason,
    }
