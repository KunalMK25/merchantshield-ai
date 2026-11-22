"""
Loaders that map public datasets into MerchantShield's raw schema.

Each loader returns (raw_frame, notes). `notes` records what was assumed and which features the
dataset leaves inert, in the same spirit as the IEEE-CIS adapter: nothing is quietly invented.

Shared decisions:
  * Transaction ids are replaced with sequential ids (txn_000000001, ...) in time order. The daily
    generator continues the numbering, and original ids are not needed by any feature.
  * `account_created` is not in these datasets, so it is the customer's first day in the data. Account
    age therefore means "tenure observed here", not true account age.
  * Personal columns (names, addresses, date of birth, job) are never read.
  * Where a dataset has no device or region, a constant is used so the corresponding feature is inert
    (always 0) instead of being filled with made-up values.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.daily_batch import RAW_COLUMNS
from ml.datasets.schema import DatasetError
from ml.external.ieee.ieee_adapter import load_and_adapt as _ieee_load_and_adapt

UNKNOWN_DEVICE = "device_unknown"
UNKNOWN_GEO = "geo_unknown"


def _check_columns(path, needed):
    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in needed if c not in header]
    if missing:
        raise DatasetError(f"{Path(path).name}: missing column(s) {missing}. "
                           f"This does not match the layout the loader expects.")


def _sample(ids, k, seed):
    ids = sorted(set(ids))
    if k is None or k >= len(ids):
        return set(ids)
    rng = np.random.default_rng(seed)
    return set(rng.choice(ids, size=k, replace=False).tolist())


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Time order, sequential ids, tenure-based account_created, schema column order."""
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df["transaction_id"] = [f"txn_{i:09d}" for i in range(1, len(df) + 1)]
    first_day = df.groupby("customer_id")["timestamp"].transform("min").dt.normalize()
    df["account_created"] = first_day
    df["is_fraud"] = df["is_fraud"].astype(int)
    df["amount"] = df["amount"].astype(float)
    return df[RAW_COLUMNS]


# ---------------------------------------------------------------- Sparkov

SPARKOV_COLUMNS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "trans_num",
                   "merch_lat", "merch_long", "is_fraud"]


def load_sparkov(paths, max_customers=None, seed=0):
    """Kaggle 'kartik2112/fraud-detection' (fraudTrain.csv / fraudTest.csv). USD amounts."""
    frames = []
    for p in paths:
        _check_columns(p, SPARKOV_COLUMNS)
        frames.append(pd.read_csv(p, usecols=SPARKOV_COLUMNS))
    df = pd.concat(frames, ignore_index=True)

    keep = _sample(df["cc_num"].tolist(), max_customers, seed)
    df = df[df["cc_num"].isin(keep)]

    out = pd.DataFrame({
        "transaction_id": df["trans_num"].astype(str),
        "customer_id": "cc_" + df["cc_num"].astype(str),
        "merchant_id": df["merchant"].astype(str).str.replace(r"^fraud_", "", regex=True),
        "merchant_category": df["category"],
        "timestamp": pd.to_datetime(df["trans_date_trans_time"]),
        "amount": df["amt"],
        # no device in the data; the merchant's location, bucketed to one degree (~110 km), stands in for region
        "device_id": df["cc_num"].astype(str).radd("device_unknown_"),
        "geo_region": "g" + df["merch_lat"].round().astype(int).astype(str) + "_" + df["merch_long"].round().astype(int).astype(str),
        "payment_method": "card", "status": "success", "account_created": pd.NaT, "is_fraud": df["is_fraud"],
    })
    out = _finalize(out)
    notes = {
        "dataset": "sparkov", "rows": len(out), "sampled_customers": out["customer_id"].nunique() if max_customers else None,
        "currency": "USD",
        "assumptions": [
            "No device information: one constant device per customer, so new_device_flag is inert.",
            "Region is the merchant's location rounded to one degree of latitude/longitude, not an administrative region.",
            "No payment status: every transaction is 'success', so failed_ratio_trailing10 is inert.",
            "account_created is the customer's first day in the data (tenure observed here).",
        ],
        "inert_features": ["new_device_flag", "failed_ratio_trailing10"],
    }
    return out, notes


# --------------------------------------------------------------- TabFormer

TABFORMER_COLUMNS = ["User", "Card", "Year", "Month", "Day", "Time", "Amount", "Use Chip",
                     "Merchant Name", "Merchant City", "Merchant State", "MCC", "Errors?", "Is Fraud?"]


def load_tabformer(path, max_users=None, seed=0, chunksize=500_000):
    """IBM TabFormer synthetic credit-card transactions (card_transaction.v1.csv). Read in chunks."""
    _check_columns(path, TABFORMER_COLUMNS)

    keep = None
    if max_users is not None:                         # first pass reads only the id columns
        pairs = set()
        for chunk in pd.read_csv(path, usecols=["User", "Card"], chunksize=chunksize):
            pairs.update(zip(chunk["User"], chunk["Card"]))
        pairs = sorted(pairs)
        idx = _sample(range(len(pairs)), max_users, seed)
        keep = {pairs[i] for i in idx}

    parts, dropped = [], 0
    for chunk in pd.read_csv(path, usecols=TABFORMER_COLUMNS, chunksize=chunksize):
        if keep is not None:
            chunk = chunk[[(u, c) in keep for u, c in zip(chunk["User"], chunk["Card"])]]
        amount = pd.to_numeric(chunk["Amount"].astype(str).str.replace("$", "", regex=False), errors="coerce")
        bad = ~(amount > 0)                            # refunds (negative) and unparseable amounts
        dropped += int(bad.sum())
        chunk, amount = chunk[~bad], amount[~bad]
        if chunk.empty:
            continue
        online = chunk["Use Chip"].astype(str).str.contains("Online", case=False)
        geo = chunk["Merchant State"].where(chunk["Merchant State"].notna(), chunk["Merchant City"]).astype(str)
        stamp = (chunk["Year"].astype(int).astype(str) + "-" + chunk["Month"].astype(int).astype(str).str.zfill(2) + "-"
                 + chunk["Day"].astype(int).astype(str).str.zfill(2) + " " + chunk["Time"].astype(str))
        errors = chunk["Errors?"].fillna("").astype(str).str.strip()
        parts.append(pd.DataFrame({
            "transaction_id": "", "customer_id": "u" + chunk["User"].astype(str) + "_c" + chunk["Card"].astype(str),
            "merchant_id": "m" + chunk["Merchant Name"].astype(str),
            "merchant_category": "mcc_" + chunk["MCC"].astype(str),
            "timestamp": pd.to_datetime(stamp), "amount": amount.astype(float),
            "device_id": "device_unknown_" + chunk["User"].astype(str) + "_" + chunk["Card"].astype(str),
            "geo_region": geo, "payment_method": np.where(online, "online", "card"),
            "status": np.where(errors != "", "failed", "success"), "account_created": pd.NaT,
            "is_fraud": (chunk["Is Fraud?"].astype(str).str.lower() == "yes").astype(int),
        }))
    if not parts:
        raise DatasetError(f"{Path(path).name}: no usable transactions found")
    out = _finalize(pd.concat(parts, ignore_index=True))
    notes = {
        "dataset": "tabformer", "rows": len(out), "currency": "USD",
        "sampled_customers": out["customer_id"].nunique() if max_users else None,
        "dropped_non_positive_amounts": dropped,
        "assumptions": [
            "No device information: one constant device per card, so new_device_flag is inert.",
            "Region is the merchant's state, or its city for online merchants (which become 'ONLINE').",
            "A transaction with a non-empty 'Errors?' value is 'failed'; the failed-payment ratio is real signal here.",
            "Refunds (negative amounts) are dropped.",
            "account_created is the card's first day in the data (tenure observed here).",
        ],
        "inert_features": ["new_device_flag"],
    }
    return out, notes


# ------------------------------------------------------------------ PaySim

PAYSIM_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
PAYSIM_START = pd.Timestamp("2017-11-01")             # PaySim has only an hour counter; this date is arbitrary


def load_paysim(path, max_rows=None):
    """Kaggle 'ealaxi/paysim1'. Loadable, but almost every origin account appears once (see profile())."""
    _check_columns(path, PAYSIM_COLUMNS)
    df = pd.read_csv(path, usecols=PAYSIM_COLUMNS, nrows=max_rows)
    bad = ~(df["amount"] > 0)
    dropped = int(bad.sum())
    df = df[~bad]
    out = _finalize(pd.DataFrame({
        "transaction_id": "", "customer_id": df["nameOrig"].astype(str), "merchant_id": df["nameDest"].astype(str),
        "merchant_category": df["type"].astype(str), "timestamp": PAYSIM_START + pd.to_timedelta(df["step"], unit="h"),
        "amount": df["amount"], "device_id": UNKNOWN_DEVICE, "geo_region": UNKNOWN_GEO,
        "payment_method": df["type"].astype(str).str.lower(), "status": "success",
        "account_created": pd.NaT, "is_fraud": df["isFraud"],
    }))
    notes = {
        "dataset": "paysim", "rows": len(out), "currency": "unspecified", "dropped_non_positive_amounts": dropped,
        "assumptions": [
            "Timestamps are PAYSIM_START plus the hour counter; the real calendar date is unknown.",
            "No device, region or payment status: new_device_flag, new_geo_flag and failed_ratio_trailing10 are inert.",
            "Balance columns, which carry most of PaySim's fraud signal, are not part of MerchantShield's features.",
        ],
        "inert_features": ["new_device_flag", "new_geo_flag", "failed_ratio_trailing10"],
    }
    return out, notes


# ---------------------------------------------------------------- IEEE-CIS

def load_ieee(data_dir, max_rows=None):
    """IEEE-CIS via the existing adapter (documented degradations live in ml/external/ieee/ieee_adapter.py)."""
    d = Path(data_dir)
    txn, ident = d / "train_transaction.csv", d / "train_identity.csv"
    if not txn.exists():
        raise DatasetError(f"train_transaction.csv not found in {d}")
    adapted, meta = _ieee_load_and_adapt(str(txn), str(ident) if ident.exists() else None, verbose=False)
    df = adapted.drop(columns=["TransactionDT"], errors="ignore").copy()
    for col in ("timestamp", "account_created"):
        s = pd.to_datetime(df[col])
        df[col] = s.dt.tz_convert("UTC").dt.tz_localize(None) if s.dt.tz is not None else s
    if max_rows:
        df = df.sort_values("timestamp", kind="stable").head(max_rows)
    out = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    out["transaction_id"] = [f"txn_{i:09d}" for i in range(1, len(out) + 1)]
    out["is_fraud"] = out["is_fraud"].astype(int)
    out = out[RAW_COLUMNS]
    return out, {"dataset": "ieee", "rows": len(out), "currency": "USD", "adapter": meta,
                 "assumptions": ["See ml/external/ieee/ieee_adapter.py for the documented feature degradations."],
                 "inert_features": ["new_device_flag", "failed_ratio_trailing10"]}
